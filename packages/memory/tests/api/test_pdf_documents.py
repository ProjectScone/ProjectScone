"""PDF ingestion must be reachable through the authorized, bounded HTTP lane."""
import asyncio

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from ..ingestion.test_pdf_ingestion import pdf_bytes


@pytest.fixture
async def service():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {'writer': 'alpha', 'reader': 'alpha', 'other': 'beta'},
                     roles={'reader': 'read'}, ingest_concurrency=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                 headers={'authorization': 'Bearer writer'}) as client:
        yield client, engine
    await engine.close()


async def upload(client, raw=None, media_type='application/pdf'):
    response = await client.post('/v1/attachments', content=pdf_bytes() if raw is None else raw,
                                 headers={'content-type': media_type})
    assert response.status_code == 200, response.text
    return {'attachment_id': response.json()['attachment_id']}


async def test_pdf_http_ingests_recalls_resolves_pages_and_forgets(service):
    client, _ = service
    body = await upload(client)
    saved = await client.post('/v1/documents/pdf', json=body)
    assert saved.status_code == 200, saved.text
    episode = saved.json()['added']['episode_id']
    repeat = (await client.post('/v1/documents/pdf', json=body)).json()
    assert repeat['added']['episode_id'] == episode and repeat['added']['deduplicated']
    recalled = (await client.get('/v1/recall', params={'q': 'Polaris'})).json()
    chunk = next(item for item in recalled['items'] if item['episode_id'] == episode)
    path = f'/v1/episodes/{episode}/pdf'
    pages = await client.get(path, params={'chunk_id': chunk['chunk_id']},
                             headers={'authorization': 'Bearer reader'})
    assert pages.status_code == 200, pages.text
    assert pages.json()['pages'][0]['number'] == 1
    assert (await client.get(pages.json()['download_path'])).content == pdf_bytes()
    assert (await client.get(path, headers={'authorization': 'Bearer other'})).status_code == 404
    assert (await client.delete(f'/v1/episodes/{episode}')).status_code == 200
    assert (await client.get(path)).status_code == 410
    assert not (await client.get('/v1/recall', params={'q': 'Polaris'})).json()['items']


async def test_pdf_http_enforces_authorization_and_input_bounds(service):
    client, _ = service
    body = await upload(client)
    for headers, status in [({'authorization': ''}, 401),
                            ({'authorization': 'Bearer reader'}, 403),
                            ({'authorization': 'Bearer other'}, 404)]:
        assert (await client.post('/v1/documents/pdf', json=body, headers=headers)).status_code == status
    assert (await client.post('/v1/documents/pdf', content=b'x' * 4097)).status_code == 413
    for payload in [{'attachment_id': '../file'}, {**body, 'url': 'https://example.invalid'},
                    {**body, 'limits': {'max_pages': 100000}}]:
        assert (await client.post('/v1/documents/pdf', json=payload)).status_code == 400
    wrong = await upload(client, b'plain text', 'text/plain')
    assert (await client.post('/v1/documents/pdf', json=wrong)).status_code == 422


async def test_pdf_http_failure_leaves_memory_healthy_without_an_episode(service):
    client, engine = service
    for raw in [b'%PDF-1.7 corrupt', pdf_bytes(pages=('',))]:
        body = await upload(client, raw)
        response = await client.post('/v1/documents/pdf', json=body)
        assert response.status_code == 422, response.text
    assert (await engine.documents.counts('alpha')).episodes == 0
    assert (await client.get('/healthz')).status_code == 200


async def test_pdf_parser_shares_admission_with_other_ingestion(service, monkeypatch):
    client, _ = service
    from scone_memory.ingestion.pdf import PypdfParser
    original = PypdfParser.parse
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_parse(self, *args, **kwargs):
        entered.set()
        await release.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(PypdfParser, 'parse', held_parse)
    body = await upload(client)
    task = asyncio.create_task(client.post('/v1/documents/pdf', json=body))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        busy = await client.post('/v1/episodes', json={'content': 'Must wait for admission.'})
        assert busy.status_code == 429 and busy.headers['retry-after'] == '1'
    finally:
        release.set()
        response = await task
    assert response.status_code == 200, response.text


async def test_pdf_missing_dependency_is_reported_as_feature_unavailable(service, monkeypatch):
    client, _ = service
    import scone_memory.api.pdf_documents as module
    monkeypatch.setattr(module, 'pdf_available', lambda: False)
    capabilities = (await client.get('/v1/capabilities')).json()['features']
    assert capabilities['documents.pdf'] is False
    assert capabilities['documents.pdf.provenance'] is True
    response = await client.post('/v1/documents/pdf', json=await upload(client))
    assert response.status_code == 501
    assert (await client.get('/healthz')).status_code == 200
