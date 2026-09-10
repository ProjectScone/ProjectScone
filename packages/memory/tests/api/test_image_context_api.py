import pytest
from fastapi.testclient import TestClient
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from ..ingestion.test_image_context import picture, context


@pytest.fixture
async def client():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with TestClient(create_app(engine, {'writer': 'alpha', 'reader': 'alpha', 'other': 'beta'},
        roles={'reader': 'read'})) as client:
        yield client
    await engine.close()


def test_image_context_search_returns_downloadable_original_with_entity_evidence(client):
    auth = {'authorization': 'Bearer writer'}
    image = client.post('/v1/attachments', content=picture(), headers={**auth, 'content-type': 'image/png'}).json()
    body = {'attachment_id': image['attachment_id'], 'context': context().model_dump(mode='json')}
    saved = client.post('/v1/images', json=body, headers=auth)
    assert saved.status_code == 200, saved.text
    found = client.get('/v1/images/search', params={'query': 'Who is Pikachu?', 'entity_id': 'pokemon:25'},
        headers={'authorization': 'Bearer reader'})
    assert found.status_code == 200
    match = found.json()['matches'][0]
    assert match['context']['entities'][0]['attribute_indexes'] == [0]
    assert client.get(match['download_path'], headers=auth).content == picture()
    assert client.get(match['download_path'], headers={'authorization': 'Bearer other'}).status_code == 404
    assert client.post('/v1/images', json=body, headers={'authorization': 'Bearer reader'}).status_code == 403
    assert client.post('/v1/images', json=body, headers={'authorization': 'Bearer other'}).status_code == 404
    assert client.get('/v1/images/search', params={'query': 'Pikachu'}).status_code == 401


def test_bad_context_and_large_body_are_rejected(client):
    auth = {'authorization': 'Bearer writer'}
    assert client.post('/v1/images', content=b'x' * 140_000, headers=auth).status_code == 413
    assert client.post('/v1/images', json={'attachment_id': '../invalid', 'context': {}}, headers=auth).status_code == 400
    assert client.get('/v1/images/search', params={'query': 'Pikachu', 'limit': 26}, headers=auth).status_code == 422
