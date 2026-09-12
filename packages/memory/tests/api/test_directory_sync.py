"""Configured sync routes preserve roles, space scope and control revisions."""
import asyncio
import json

import httpx
import pytest

from scone_memory.api.app import create_app
from tests.ingestion.test_directory_sync import env, runner
from tests.ingestion.test_directory_service import service, settled


def auth(key='writer'):
    return {'authorization': 'Bearer ' + key}


@pytest.fixture
async def setup(env):
    memory, root, _ = env
    (root / 'note.txt').write_text('A document to synchronize')
    sync = runner(env)
    entered, release = asyncio.Event(), asyncio.Event()
    parse = sync.parser.parse
    async def gated(*args):
        entered.set()
        await release.wait()
        return await parse(*args)
    sync.parser.parse = gated
    host = service(env, sync=sync)
    app = create_app(memory, {'writer': 'alpha', 'reader': 'alpha', 'other': 'bravo'},
        roles={'writer': 'write', 'reader': 'read'}, directory_sync_service=host)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local.test') as client:
        try:
            yield client, app, host, entered, release, memory, root
        finally:
            release.set()
            await host.aclose()


async def test_catalog_admission_history_results_and_source_access(setup):
    client, _, host, entered, release, memory, root = setup
    body = {'run_id': 'scan', 'collection_id': 'notes'}
    assert (await client.get('/v1/capabilities', headers=auth())).json()['features']['documents.sync']
    assert (await client.get('/v1/sync-collections')).status_code == 401
    catalog = await client.get('/v1/sync-collections', headers=auth('reader'))
    assert catalog.status_code == 200 and str(root) not in catalog.text
    assert catalog.json()['items'][0]['collection_id'] == 'notes'
    assert (await client.get('/v1/sync-collections', headers=auth('other'))).json()['items'] == []
    assert (await client.post('/v1/sync-runs', json=body, headers=auth('reader'))).status_code == 403
    admitted = await client.post('/v1/sync-runs', json=body, headers=auth())
    assert admitted.status_code == 202, admitted.text
    await asyncio.wait_for(entered.wait(), 3)
    assert (await client.get('/v1/sync-runs/scan/result', headers=auth())).status_code == 409
    assert (await client.get('/v1/sync-runs/scan', headers=auth('other'))).status_code == 404
    assert (await client.get('/v1/sync-runs', headers=auth('other'))).json()['items'] == []
    saved = await client.get('/v1/sync-runs/scan/request', headers=auth('reader'))
    assert saved.json()['spec']['delete_missing'] is False
    busy = await client.post('/v1/sync-runs', json={**body, 'run_id': 'second'}, headers=auth())
    assert busy.status_code == 429 and busy.headers['retry-after'] == '1'
    release.set()
    await settled(host)
    history = await client.get('/v1/sync-runs', headers=auth('reader'))
    assert history.json()['items'][0]['status'] == 'completed'
    result = await client.get('/v1/sync-runs/scan/result', headers=auth('reader'))
    assert result.status_code == 200 and result.headers['cache-control'] == 'no-store'
    episode = result.json()['items'][0]['source']['episode_id']
    await memory.forget('alpha', episode)
    # A historical receipt remains historical; opening its episode observes Gone.
    assert (await client.get('/v1/sync-runs/scan/result', headers=auth())).json() == result.json()
    assert (await client.get(f'/v1/episodes/{episode}', headers=auth())).status_code == 410


async def test_controls_require_current_revision_and_explicit_resume(setup):
    client, _, host, entered, release, _, _ = setup
    admitted = await client.post('/v1/sync-runs', json={'run_id': 'scan', 'collection_id': 'notes'}, headers=auth())
    revision = admitted.json()['record']['revision']
    await asyncio.wait_for(entered.wait(), 3)
    assert (await client.post('/v1/sync-runs/scan/cancel', json={'expected_revision': revision - 1}, headers=auth())).status_code == 409
    cancelled = await client.post('/v1/sync-runs/scan/cancel', json={'expected_revision': revision}, headers=auth())
    assert cancelled.status_code == 202
    status = await settled(host)
    assert status.status == 'cancelled'
    assert (await client.post('/v1/sync-runs/scan/resume', json={'expected_revision': revision}, headers=auth())).status_code == 409
    release.set()
    resumed = await client.post('/v1/sync-runs/scan/resume',
        json={'expected_revision': status.record.revision}, headers=auth())
    assert resumed.status_code == 202
    assert (await settled(host)).status == 'completed'


@pytest.mark.parametrize('body', [
    {'run_id': 'scan', 'collection_id': 'notes', 'root': '/etc'},
    {'run_id': 'scan', 'collection_id': 'notes', 'space': 'bravo'},
    {'run_id': 'scan', 'collection_id': 'notes', 'delete_missing': 1},
    {'run_id': '../scan', 'collection_id': 'notes'},
])
async def test_no_http_roots_spaces_or_coerced_flags(setup, body):
    client, _, host, _, _, _, _ = setup
    response = await client.post('/v1/sync-runs', json=body, headers=auth())
    assert response.status_code == 422, response.text
    assert await host.request('alpha', 'scan') is None


async def test_stream_limits_duplicates_and_auth_recheck(setup, monkeypatch):
    client, app, host, _, _, memory, _ = setup
    oversized = await client.post('/v1/sync-runs', content=b'x'*8193, headers=auth())
    assert oversized.status_code == 413
    duplicate = await client.post('/v1/sync-runs', content=b'{"run_id":"scan","run_id":"other","collection_id":"notes"}', headers=auth())
    assert duplicate.status_code == 422
    body = json.dumps({'run_id': 'scan', 'collection_id': 'notes'}).encode()
    async def changed_body():
        yield body[:8]
        app.state.keys['writer'] = 'bravo'
        yield body[8:]
    assert (await client.post('/v1/sync-runs', content=changed_body(), headers=auth())).status_code == 401
    app.state.keys['writer'] = 'alpha'
    before = memory.space_deleted
    calls = 0
    async def changed_space(space):
        nonlocal calls
        calls += 1
        if calls == 2:
            app.state.roles['writer'] = 'read'
        return await before(space)
    monkeypatch.setattr(memory, 'space_deleted', changed_space)
    response = await client.post('/v1/sync-runs', content=body, headers=auth())
    assert response.status_code == 403
    assert await host.request('alpha', 'scan') is None


async def test_unconfigured_app_omits_routes(env):
    memory, _, _ = env
    app = create_app(memory, {'writer': 'alpha'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local.test') as client:
        assert not (await client.get('/v1/capabilities', headers=auth())).json()['features'].get('documents.sync', False)
        assert (await client.get('/v1/sync-collections', headers=auth())).status_code == 404


async def test_revoked_cancel_does_not_signal_worker(setup, monkeypatch):
    client, app, host, entered, release, memory, _ = setup
    await client.post('/v1/sync-runs', json={'run_id': 'scan', 'collection_id': 'notes'}, headers=auth())
    await asyncio.wait_for(entered.wait(), 3)
    before = (await host.status('alpha', 'scan')).record
    original = memory.space_deleted
    calls = 0
    async def revoked(space):
        nonlocal calls
        calls += 1
        if calls == 2:
            app.state.roles['writer'] = 'read'
        return await original(space)
    monkeypatch.setattr(memory, 'space_deleted', revoked)
    response = await client.post('/v1/sync-runs/scan/cancel', json={'expected_revision': before.revision}, headers=auth())
    assert response.status_code == 403
    assert (await host.status('alpha', 'scan')).record == before
    assert (await host.status('alpha', 'scan')).active_local
    release.set()
    assert (await settled(host)).status == 'completed'


@pytest.mark.parametrize('composed', [False, True])
async def test_host_lifespan_joins_sync_before_returning(env, composed):
    from scone_memory.api.conversations import create_conversation_app
    memory, _, parent = env
    host = service(env)
    if composed:
        app = create_conversation_app(memory, {'writer': 'alpha'}, parent / 'conversations.sqlite', None,
                                      directory_sync_service=host)
    else:
        app = create_app(memory, {'writer': 'alpha'}, directory_sync_service=host)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local.test') as client:
            assert (await client.get('/v1/sync-collections', headers=auth())).status_code == 200
            response = await client.post('/v1/sync-runs', json={'run_id': 'scan', 'collection_id': 'notes'}, headers=auth())
            assert response.status_code == 202
    assert host._closed and not host._owners and not host._tasks
    assert await memory.space_deleted('alpha') is None


async def test_failed_worker_start_releases_sync_registry(env):
    memory, _, _ = env
    host = service(env)
    class BrokenWorker:
        stopped = False
        def start(self):
            raise RuntimeError('startup failed')
        async def stop(self):
            self.stopped = True
    worker = BrokenWorker()
    app = create_app(memory, {'writer': 'alpha'}, worker=worker, directory_sync_service=host)
    try:
        with pytest.raises(RuntimeError, match='startup failed'):
            async with app.router.lifespan_context(app):
                pass
        assert host._closed and worker.stopped
    finally:
        await host.aclose()


async def test_cancelled_composed_shutdown_releases_conversation_owner(env, monkeypatch):
    import fcntl
    import os
    from scone_memory.api import conversations
    memory, root, parent = env
    (root / 'note.txt').write_text('Wait during provider cleanup')
    sync = runner(env)
    entered, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def gated(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            await release.wait()
            raise
    sync.parser.parse = gated
    host = service(env, sync=sync)
    path = parent / 'conversation-cleanup.sqlite'
    app = conversations.create_conversation_app(memory, {'writer': 'alpha'}, path, None,
                                                directory_sync_service=host)
    opened, journals = [], []
    original_open, original_journal = os.open, conversations.SessionJournal
    def capture_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if str(path).endswith('conversation-cleanup.sqlite.owner.lock'):
            opened.append(descriptor)
        return descriptor
    def capture_journal(*args, **kwargs):
        value = original_journal(*args, **kwargs)
        journals.append(value)
        return value
    monkeypatch.setattr(os, 'open', capture_open)
    monkeypatch.setattr(conversations, 'SessionJournal', capture_journal)
    context = app.router.lifespan_context(app)
    await context.__aenter__()
    await host.start('alpha', 'scan', collection_id='notes')
    await asyncio.wait_for(entered.wait(), 3)
    ending = asyncio.create_task(context.__aexit__(None, None, None))
    leaked = False
    try:
        await asyncio.wait_for(draining.wait(), 3)
        ending.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await ending
        assert host._closed
        probe = original_open(str(path) + '.owner.lock', os.O_RDWR)
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                leaked = True
        finally:
            os.close(probe)
        assert not leaked, 'conversation owner survived cancelled shutdown'
    finally:
        release.set()
        await asyncio.gather(ending, return_exceptions=True)
        if leaked:
            for journal in journals:
                journal.close()
            os.close(opened[0])
        await host.aclose()


@pytest.mark.parametrize('failure', ['missing_parent', 'hardlink'])
async def test_early_composed_journal_refusal_closes_injected_service(env, failure):
    import os
    from scone_memory.api.conversations import create_conversation_app
    from scone_memory.core.errors import InvalidInput
    memory, _, parent = env
    host = service(env)
    path = parent / 'missing-parent' / 'conversation.sqlite'
    if failure == 'hardlink':
        path = parent / 'conversation.sqlite'
        path.touch()
        os.link(path, parent / 'alias.sqlite')
    app = create_conversation_app(memory, {'writer': 'alpha'}, path, None, directory_sync_service=host)
    try:
        with pytest.raises((OSError, InvalidInput)):
            async with app.router.lifespan_context(app):
                pass
        assert host._closed and host._runs._storage._closed
    finally:
        await host.aclose()


async def test_collection_labels_follow_ledger_unicode_response_contract(env):
    from scone_memory.ingestion.directory_service import DirectoryCollection, DirectorySyncService
    memory, _, parent = env
    label = 'Notes\ud800'
    host = DirectorySyncService(parent / 'unicode-service', key=b'd' * 32, memory=memory,
        collections=(DirectoryCollection('notes', label, runner(env)),))
    app = create_app(memory, {'writer': 'alpha'}, directory_sync_service=host)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local.test') as client:
            response = await client.get('/v1/sync-collections', headers=auth())
            assert response.status_code == 200
            assert response.json()['items'][0]['label'] == label
    finally:
        await host.aclose()


async def test_start_binds_discovered_configuration_before_admission(setup):
    client, _, host, entered, release, _, _ = setup
    catalog = (await client.get('/v1/sync-collections', headers=auth())).json()['items'][0]
    body = {'run_id': 'scan', 'collection_id': 'notes', 'expected_configuration': '0' * 64}
    refused = await client.post('/v1/sync-runs', json=body, headers=auth())
    assert refused.status_code == 409, refused.text
    assert refused.json()['code'] == 'sync_configuration_changed'
    assert await host.request('alpha', 'scan') is None
    assert not entered.is_set()
    body['expected_configuration'] = catalog['configuration']
    admitted = await client.post('/v1/sync-runs', json=body, headers=auth())
    assert admitted.status_code == 202, admitted.text
    await asyncio.wait_for(entered.wait(), 3)
    release.set()
    await settled(host)
    assert (await client.post('/v1/sync-runs', json=body, headers=auth())).status_code == 202


@pytest.mark.parametrize('value', [True, 'short', 'A' * 64, 42])
async def test_start_rejects_malformed_expected_configuration(setup, value):
    client, _, host, _, _, _, _ = setup
    response = await client.post('/v1/sync-runs', headers=auth(), json={
        'run_id': 'scan', 'collection_id': 'notes', 'expected_configuration': value})
    assert response.status_code == 422
    assert await host.request('alpha', 'scan') is None


async def test_result_envelope_binds_authenticated_space_and_run(setup):
    client, _, host, _, release, _, _ = setup
    release.set()
    await client.post('/v1/sync-runs', headers=auth(), json={'run_id': 'scan', 'collection_id': 'notes'})
    await settled(host)
    result = (await client.get('/v1/sync-runs/scan/result', headers=auth())).json()
    assert result['space'] == 'alpha'
    assert result['run_id'] == 'scan'
