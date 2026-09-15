"""A stored document's Markdown, rebuilt from its retained manifest, traced to the episode's text."""
import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

PAGE = b'<h2>Launch</h2><ol><li>Build</li><li>Ship on Friday</li></ol><table><tr><th>Day</th></tr><tr><td>Friday</td></tr></table>'


@pytest.fixture
async def service():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {'write': 'alpha', 'read': 'alpha', 'other': 'beta'}, roles={'read': 'read'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                 headers={'authorization': 'Bearer write'}) as client:
        yield client, engine
    await engine.close()


async def test_stored_document_markdown_traces_every_span_to_the_episode_text(service):
    client, engine = service
    upload = await client.post('/v1/attachments', content=PAGE,
                               headers={'content-type': 'application/octet-stream', 'x-filename': 'launch.html'})
    indexed = await client.post('/v1/documents', json={'attachment_id': upload.json()['attachment_id']})
    assert indexed.status_code == 200, indexed.text
    episode_id = indexed.json()['added']['episode_id']
    path = f'/v1/episodes/{episode_id}/document/markdown'
    response = await client.get(path, headers={'authorization': 'Bearer read'})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['markdown'] == '## Launch\n\n1. Build\n2. Ship on Friday\n\n| Day |\n| --- |\n| Friday |'
    assert (body['episode_id'], body['filename']) == (episode_id, 'launch.html')
    assert body['original_sha256'] == upload.json()['attachment_id']
    content = (await engine.episode('alpha', episode_id)).content.encode()
    markdown = body['markdown'].encode()
    for span in body['spans']:
        for source in span['sources']:
            assert content[source['start']:source['end']]
        if span['kind'] == 'list_item':
            [source] = span['sources']
            quoted = content[source['start']:source['end']].decode()
            assert markdown[span['markdown_start']:span['markdown_end']].decode().endswith(quoted)
    cut = (await client.get(path, params={'max_bytes': 12})).json()
    assert cut['markdown'] == '## Launch' and cut['bound']['cut'] is True
    assert (await client.get(path, params={'max_bytes': 0})).status_code == 422
    assert (await client.get(path, headers={'authorization': 'Bearer other'})).status_code == 404
