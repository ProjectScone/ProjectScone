"""Actual parser children feed authenticated indexing and source retrieval."""
import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def service():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {'write': 'alpha', 'read': 'alpha', 'other': 'beta'}, roles={'read': 'read'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                 headers={'authorization': 'Bearer write'}) as client:
        yield client, engine
    await engine.close()


@pytest.mark.parametrize(('filename', 'raw', 'locator'), [
    ('launch.json', b'{"schedule":{"launch":"Friday"}}', '/schedule/launch'),
    ('launch.csv', b'task,day\nlaunch,Friday\n', 'row:2'),
    ('launch.html', b'<title>Launch</title><p>Friday launch</p>', 'line:'),
    ('launch.md', b'# Launch\nFriday\n', 'line:'),
])
async def test_upload_index_recall_cite_repeat_and_delete(service, filename, raw, locator):
    client, engine = service
    upload = await client.post('/v1/attachments', content=raw,
                               headers={'content-type': 'application/octet-stream', 'x-filename': filename})
    assert upload.status_code == 200, upload.text
    body = {'attachment_id': upload.json()['attachment_id']}
    indexed = await client.post('/v1/documents', json=body)
    assert indexed.status_code == 200, indexed.text
    episode_id = indexed.json()['added']['episode_id']
    repeated = await client.post('/v1/documents', json=body)
    assert repeated.json()['added']['episode_id'] == episode_id
    assert repeated.json()['added']['deduplicated'] is True
    recalled = (await client.get('/v1/recall', params={'q': 'Friday'})).json()['items']
    chunk = next(item for item in recalled if item['episode_id'] == episode_id)
    evidence_path = f'/v1/episodes/{episode_id}/document'
    evidence = await client.get(evidence_path, params={'chunk_id': chunk['chunk_id']},
                                 headers={'authorization': 'Bearer read'})
    assert evidence.status_code == 200, evidence.text
    assert any(locator in item['locator'] for item in evidence.json()['segments'])
    assert (await client.get(evidence.json()['download_path'])).content == raw
    assert (await client.get(evidence_path, headers={'authorization': 'Bearer other'})).status_code == 404
    assert (await client.post('/v1/documents', json=body, headers={'authorization': 'Bearer read'})).status_code == 403
    assert (await client.delete(f'/v1/episodes/{episode_id}')).status_code == 200
    assert (await client.get(evidence_path)).status_code == 410
    assert (await engine.documents.counts('alpha')).episodes == 0


async def test_document_boundary_rejects_oversize_extra_fields_and_reports_formats(service):
    client, _ = service
    response = await client.get('/v1/documents/formats')
    assert response.status_code == 200
    formats = response.json()['formats']
    for suffix in ('.docx', '.xlsx', '.pptx', '.json', '.csv', '.eml', '.epub'):
        assert suffix in formats
    assert (await client.post('/v1/documents', content=b'x'*4097)).status_code == 413
    assert (await client.post('/v1/documents', json={'attachment_id': 'a'*64, 'url': 'http://example.com'})).status_code == 400
    assert (await client.post('/v1/documents', json={'attachment_id': 'a'*64})).status_code == 404
