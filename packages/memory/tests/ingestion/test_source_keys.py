"""Stable-key reads resolve uncertain replacements without writing memory."""
from __future__ import annotations

import io
import json

import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record, SyncMemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import Conflict, Gone, InvalidInput, NotFound
from scone_memory.runtime import cli


@pytest.fixture(params=['memory', 'sqlite'])
async def keyed_engine(request, tmp_path):
    if request.param == 'sqlite':
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
        documents, vectors = SqliteDocumentStore(tmp_path / 'key.db'), SqliteVectorIndex(tmp_path / 'key.db')
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    yield engine
    await engine.close()


async def test_key_lifecycle_and_scope(keyed_engine):
    engine = keyed_engine
    key = 'file:reports/Q3 ?#% 🥐'
    with pytest.raises(NotFound) as missing:
        await engine.episode_by_key('alpha', key)
    assert not isinstance(missing.value, Gone)
    original = await engine.remember('alpha', 'Original observatory report', dedup_key=key)
    other = await engine.remember('beta', 'Another space', dedup_key=key)
    assert (await engine.episode_by_key('alpha', key)).episode_id == original.episode_id
    assert (await engine.episode_by_key('beta', key)).episode_id == other.episode_id
    updated = await engine.replace('alpha', Record('Revised observatory report', dedup_key=key))
    assert (await engine.episode_by_key('alpha', key)).episode_id == updated.added.episode_id
    receipt = await engine.forget('alpha', updated.added.episode_id)
    with pytest.raises(Gone) as forgotten:
        await engine.episode_by_key('alpha', key)
    assert forgotten.value.forgotten_at == receipt.forgotten_at
    restored = await engine.replace('alpha', Record('Intentionally restored', dedup_key=key))
    assert (await engine.episode_by_key('alpha', key)).episode_id == restored.added.episode_id


async def test_exact_key_identity_and_read_only_attachments(keyed_engine):
    engine = keyed_engine
    attachment = await engine.attach('alpha', b'original bytes', 'text/plain', 'report.txt')
    added = await engine.remember('alpha', 'A source', dedup_key=' key ', attachment_ids=[attachment.attachment_id])
    await engine.remember('alpha', 'A different source', dedup_key='key')
    revision = await engine.documents.revision('alpha')
    before = await engine.status('alpha')
    for _ in range(2):
        current = await engine.episode_by_key('alpha', ' key ')
        assert current.episode_id == added.episode_id
        assert current.attachments == (attachment,)
    assert await engine.documents.revision('alpha') == revision
    assert await engine.status('alpha') == before
    with pytest.raises(NotFound):
        await engine.episode_by_key('alpha', 'KEY')
    # A content-deduplicated source is not accidentally exposed under its text.
    await engine.remember('alpha', 'unkeyed source')
    with pytest.raises(NotFound):
        await engine.episode_by_key('alpha', 'unkeyed source')


@pytest.mark.parametrize('key', [None, False, 12, [], {}, '', 'x' * 257, '\ud800'])
async def test_invalid_keys_are_input_errors(keyed_engine, key):
    with pytest.raises(InvalidInput):
        await keyed_engine.episode_by_key('alpha', key)


async def test_read_retries_when_key_changes_while_loading_attachments(keyed_engine, monkeypatch):
    engine = keyed_engine
    old = await engine.remember('alpha', 'Old source', dedup_key='doc')
    normal = engine.blobs.for_episode
    changed = None
    async def replace_during_read(space, episode_id):
        nonlocal changed
        if changed is None:
            changed = True
            changed = await engine.replace(space, Record('New source', dedup_key='doc'))
        return await normal(space, episode_id)
    monkeypatch.setattr(engine.blobs, 'for_episode', replace_during_read)
    current = await engine.episode_by_key('alpha', 'doc')
    assert current.episode_id != old.episode_id
    assert current.content == 'New source'


async def test_missing_read_retries_when_a_key_is_created(keyed_engine, monkeypatch):
    engine = keyed_engine
    normal = engine.documents.tombstone_by_hash
    created = False
    async def create_during_read(space, digest):
        nonlocal created
        if not created:
            created = True
            await engine.remember(space, 'Newly created source', dedup_key='doc')
        return await normal(space, digest)
    monkeypatch.setattr(engine.documents, 'tombstone_by_hash', create_during_read)
    assert (await engine.episode_by_key('alpha', 'doc')).content == 'Newly created source'


async def test_busy_key_is_a_bounded_conflict(keyed_engine, monkeypatch):
    engine = keyed_engine
    await engine.remember('alpha', 'Initial source', dedup_key='doc')
    normal = engine.blobs.for_episode
    changes = 0
    changing = False
    async def always_replace(space, episode_id):
        nonlocal changes, changing
        if not changing:
            changing = True
            changes += 1
            await engine.replace(space, Record(f'Changed source {changes}', dedup_key='doc'))
            changing = False
        return await normal(space, episode_id)
    monkeypatch.setattr(engine.blobs, 'for_episode', always_replace)
    with pytest.raises(Conflict):
        await engine.episode_by_key('alpha', 'doc')
    assert 1 <= changes <= 3


@pytest.mark.parametrize('field,value', [('space', 'beta'), ('content_hash', 'different-key')])
@pytest.mark.parametrize('read', [1, 2])
@pytest.mark.parametrize('deleted', [False, True])
async def test_faulty_adapter_identity_fails_closed(keyed_engine, monkeypatch, field, value, read, deleted):
    engine = keyed_engine
    added = await engine.remember('alpha', 'Private fixture source', dedup_key='doc')
    if deleted:
        await engine.forget('alpha', added.episode_id)
    method = 'tombstone_by_hash' if deleted else 'episode_by_hash'
    normal = getattr(engine.documents, method)
    calls = 0
    async def mismatched(space, digest):
        nonlocal calls
        calls += 1
        result = await normal(space, digest)
        return result.model_copy(update={field: value}) if calls == read else result
    monkeypatch.setattr(engine.documents, method, mismatched)
    transport = ASGITransport(app=create_app(engine, {'read-key': 'alpha'}), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url='http://fixture') as client:
        response = await client.get('/v1/episodes/by-key', params={'dedup_key': 'doc'},
                                    headers={'Authorization': 'Bearer read-key'})
    assert response.status_code == 500
    assert 'Private fixture source' not in response.text
    assert 'forgotten_at' not in response.text


async def test_http_lookup_contract_and_existing_id_route(keyed_engine):
    engine = keyed_engine
    key = 'local:file?query=# 🥐'
    added = await engine.remember('alpha', 'Scoped source', dedup_key=key)
    headers = {'Authorization': 'Bearer read-key'}
    transport = ASGITransport(app=create_app(engine, {'read-key': 'alpha', 'other-key': 'beta'}, roles={'read-key': 'read'}))
    async with AsyncClient(transport=transport, base_url='http://fixture') as client:
        response = await client.get('/v1/episodes/by-key', params={'dedup_key': key}, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()['episode_id'] == added.episode_id
        numeric = await client.get(f'/v1/episodes/{added.episode_id}', headers=headers)
        assert response.json() == numeric.json()
        assert (await client.get('/v1/episodes/by-key', params={'dedup_key': key})).status_code == 401
        foreign = await client.get('/v1/episodes/by-key', params={'dedup_key': key}, headers={'Authorization': 'Bearer other-key'})
        assert foreign.status_code == 404 and 'Scoped source' not in foreign.text
        for params in ({}, {'dedup_key': ''}, {'dedup_key': 'x' * 257}):
            assert (await client.get('/v1/episodes/by-key', params=params, headers=headers)).status_code == 422
        caps = await client.get('/v1/capabilities', headers=headers)
        assert caps.json()['features']['episodes.by_key'] is True
        await engine.forget('alpha', added.episode_id)
        forgotten = await client.get('/v1/episodes/by-key', params={'dedup_key': key}, headers=headers)
        assert forgotten.status_code == 410 and forgotten.json()['forgotten_at']


def test_sync_lookup():
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())) as engine:
        added = engine.remember('alpha', 'Blocking client source', dedup_key='doc')
        assert engine.episode_by_key('alpha', 'doc').episode_id == added.episode_id
        with pytest.raises(NotFound):
            engine.episode_by_key('beta', 'doc')


def test_cli_lookup_survives_restart_and_reports_missing(tmp_path, capsys):
    env = {'SCONE_SQLITE_PATH': str(tmp_path / 'cli.db'), 'SCONE_EMBEDDER': 'hash', 'SCONE_BLOBS': 'memory'}
    out = io.StringIO()
    assert cli.main(['remember', '--space', 'alpha', '--key', 'doc', '--json'], env=env,
                    stdin=io.StringIO('Durable keyed source'), out=out) == 0
    added = json.loads(out.getvalue())
    out = io.StringIO()
    assert cli.main(['source-key', 'doc', '--space', 'alpha', '--json'], env=env, out=out) == 0
    assert json.loads(out.getvalue())['episode_id'] == added['episode_id']
    assert cli.main(['source-key', 'doc', '--space', 'beta'], env=env, out=io.StringIO()) == 2
    assert 'not found' in capsys.readouterr().err.lower()
