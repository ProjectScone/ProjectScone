"""Durable document controls retain authentication and tenant boundaries."""
import asyncio
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.ingestion.import_service import DocumentImportService, ImportParserBinding
from ..ingestion.test_import_service import HeldParser


def auth(key='writer'):
    return {'authorization': 'Bearer ' + key}


@pytest.fixture
async def setup(tmp_path):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    original = await engine.attach('alpha', b'Ada studies stars.', 'text/plain', filename='note.md')
    parser = HeldParser()
    service = DocumentImportService(tmp_path / 'imports', key=b'k' * 32, memory=engine,
        parser_for=lambda _: ImportParserBinding('parser-v1', parser), max_active=1)
    app = create_app(engine, {'writer': 'alpha', 'reader': 'alpha', 'other': 'beta'},
        roles={'writer': 'write', 'reader': 'read'}, document_import_service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        yield client, app, service, parser, engine, {'import_id': 'one',
            'attachment_id': original.attachment_id, 'filename': 'note.md'}
    await service.aclose(); await engine.close()


async def test_authenticated_admission_history_and_verified_result(setup):
    client, app, service, parser, engine, body = setup
    assert (await client.get('/v1/capabilities', headers=auth())).json()['features']['documents.jobs']
    assert (await client.post('/v1/document-jobs', json=body)).status_code == 401
    assert (await client.post('/v1/document-jobs', json=body, headers=auth('reader'))).status_code == 403
    started = await client.post('/v1/document-jobs', json=body, headers=auth())
    assert started.status_code == 202, started.text
    await asyncio.wait_for(parser.entered.wait(), 3)
    assert (await client.get('/v1/document-jobs/one/result', headers=auth())).status_code == 409
    assert (await client.get('/v1/document-jobs/one', headers=auth('other'))).status_code == 404
    assert (await client.get('/v1/document-jobs', headers=auth('other'))).json()['items'] == []
    overloaded = await client.post('/v1/document-jobs', json={**body, 'import_id': 'two'}, headers=auth())
    assert overloaded.status_code == 429 and overloaded.headers['retry-after'] == '1'
    saved = await client.get('/v1/document-jobs/one/request', headers=auth('reader'))
    assert saved.json()['spec']['filename'] == 'note.md'
    parser.release.set(); await service.wait('alpha', 'one')
    page = await client.get('/v1/document-jobs', headers=auth('reader'))
    assert page.json()['items'][0]['status'] == 'completed'
    result = await client.get('/v1/document-jobs/one/result', headers=auth('reader'))
    assert result.status_code == 200 and result.json()['space'] == 'alpha'
    assert result.headers['cache-control'] == 'no-store'
    await engine.forget('alpha', result.json()['added']['episode_id'])
    assert (await client.get('/v1/document-jobs/one/result', headers=auth())).status_code == 409
    assert parser.calls == 1


async def test_control_revision_and_request_limits_are_strict(setup):
    client, _, service, parser, _, body = setup
    assert (await client.post('/v1/document-jobs', content=b'x' * 8193, headers=auth())).status_code == 413
    started = (await client.post('/v1/document-jobs', json=body, headers=auth())).json()
    await asyncio.wait_for(parser.entered.wait(), 3)
    for value in [True, '1', 1.0, -1]:
        response = await client.post('/v1/document-jobs/one/cancel', json={'expected_revision': value}, headers=auth())
        assert response.status_code == 422, response.text
    for key in ['reader', 'other']:
        response = await client.post('/v1/document-jobs/one/cancel', json={'expected_revision': 1}, headers=auth(key))
        assert response.status_code in (403, 404)
    cancelled = await client.post('/v1/document-jobs/one/cancel', json={'expected_revision': started['revision']}, headers=auth())
    assert cancelled.json()['status'] == 'cancelled'
    stale = await client.post('/v1/document-jobs/one/resume', json={'expected_revision': started['revision']}, headers=auth())
    assert stale.status_code == 409
    parser.release.set()
    resumed = await client.post('/v1/document-jobs/one/resume', json={'expected_revision': cancelled.json()['revision']}, headers=auth())
    assert resumed.status_code == 202
    assert (await service.wait('alpha', 'one')).status == 'completed'


async def test_streamed_body_and_attachment_wait_recheck_authorization(setup, monkeypatch):
    client, app, service, parser, engine, body = setup
    encoded = json.dumps(body).encode()
    async def stream():
        yield encoded[:10]; app.state.keys['writer'] = 'beta'; yield encoded[10:]
    response = await client.post('/v1/document-jobs', content=stream(), headers=auth())
    assert response.status_code in (401, 403)
    app.state.keys['writer'] = 'alpha'
    original = engine.attachment
    async def changed(*args, **kwargs):
        result = await original(*args, **kwargs)
        app.state.roles['writer'] = 'read'
        return result
    monkeypatch.setattr(engine, 'attachment', changed)
    response = await client.post('/v1/document-jobs', json=body, headers=auth())
    assert response.status_code == 403
    assert await service.request('alpha', 'one') is None and parser.calls == 0


async def test_result_verification_follows_invocation_read(setup, monkeypatch):
    client, _, service, parser, engine, body = setup
    parser.release.set()
    await client.post('/v1/document-jobs', json=body, headers=auth())
    await service.wait('alpha', 'one')
    result = await service.result('alpha', 'one')
    original = service.request
    reads = 0
    async def forgotten(*args, **kwargs):
        nonlocal reads
        saved = await original(*args, **kwargs)
        reads += 1
        if reads == 2:
            await engine.forget('alpha', result.added.episode_id)
        return saved
    monkeypatch.setattr(service, 'request', forgotten)
    response = await client.get('/v1/document-jobs/one/result', headers=auth())
    assert response.status_code == 409, response.text
