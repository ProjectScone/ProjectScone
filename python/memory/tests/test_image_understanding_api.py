import hashlib

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.image_understanding import mount_image_understanding_routes
from scone_memory.providers.llm import ChatError
from scone_memory.providers.vision import ImageUnderstanding

PNG = b'\x89PNG\r\n\x1a\nretained test bytes'


class FixtureVision:
    def __init__(self):
        self.calls = []
        self.failure = None

    async def describe(self, data, media_type, *, prompt, source=None, attachment_id=None):
        self.calls.append((data, media_type, prompt, source, attachment_id))
        if self.failure:
            raise self.failure
        return ImageUnderstanding('A model interpretation.', hashlib.sha256(data).hexdigest(), source, media_type, 'fixture-model', 3, 2)


@pytest.fixture
async def setup():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    image = await engine.attach('alpha', PNG, media_type='image/png')
    episode = await engine.remember('alpha', 'Retained screenshot', source='local-slide.png', attachment_ids=[image.attachment_id])
    other = await engine.remember('alpha', 'Unrelated episode')
    vision = FixtureVision()

    def authorize(authorization: str | None = Header(default=None)):
        if authorization not in ('Bearer alpha', 'Bearer beta'):
            raise HTTPException(401)
        return authorization.split(' ')[1]

    app = FastAPI()
    mount_image_understanding_routes(app, engine, authorize, lambda: vision)
    with TestClient(app) as client:
        yield engine, client, image, episode, other, vision


async def test_generates_unsaved_interpretation_with_exact_source_and_image(setup):
    engine, client, image, episode, other, vision = setup
    response = client.post(f'/v1/episodes/{episode.episode_id}/attachments/{image.attachment_id}/understand',
                           json={'prompt': 'Describe this image'}, headers={'authorization': 'Bearer alpha'})
    assert response.status_code == 200
    body = response.json()
    assert body['episode_id'] == episode.episode_id
    assert body['attachment_id'] == image.attachment_id
    assert body['persisted'] is False
    assert body['understanding']['origin'] == 'model_generated'
    assert body['understanding']['source'] == 'local-slide.png'
    assert vision.calls == [(PNG, 'image/png', 'Describe this image', 'local-slide.png', image.attachment_id)]
    assert (await engine.episode('alpha', episode.episode_id)).content == 'Retained screenshot'
    assert (await engine.episode('alpha', other.episode_id)).content == 'Unrelated episode'


async def test_auth_space_and_episode_image_association_are_checked_before_inference(setup):
    _, client, image, episode, other, vision = setup
    path = f'/v1/episodes/{episode.episode_id}/attachments/{image.attachment_id}/understand'
    assert client.post(path, json={'prompt': 'Describe'}).status_code == 401
    assert client.post(path, json={'prompt': 'Describe'}, headers={'authorization': 'Bearer beta'}).status_code == 404
    wrong = f'/v1/episodes/{other.episode_id}/attachments/{image.attachment_id}/understand'
    assert client.post(wrong, json={'prompt': 'Describe'}, headers={'authorization': 'Bearer alpha'}).status_code == 404
    assert vision.calls == []


@pytest.mark.parametrize('body', [{'prompt': ''}, {'prompt': ' '}, {'prompt': 'x' * 16001},
                                  {'prompt': 'Describe', 'approved': True}, {'prompt': 3}])
async def test_invalid_tasks_rejected_without_inference(setup, body):
    _, client, image, episode, _, vision = setup
    response = client.post(f'/v1/episodes/{episode.episode_id}/attachments/{image.attachment_id}/understand',
                           json=body, headers={'authorization': 'Bearer alpha'})
    assert response.status_code == 400
    assert vision.calls == []


async def test_provider_failures_are_sanitized(setup, caplog):
    _, client, image, episode, _, vision = setup
    vision.failure = ChatError('private model response')
    with caplog.at_level('INFO', logger='scone_memory.api.image_understanding'):
        response = client.post(f'/v1/episodes/{episode.episode_id}/attachments/{image.attachment_id}/understand',
                               json={'prompt': 'private task'}, headers={'authorization': 'Bearer alpha'})
    assert response.status_code == 502
    assert 'private model response' not in response.text
    records = [record for record in caplog.records if hasattr(record, 'event')]
    assert [record.event for record in records] == ['image_understanding.started', 'image_understanding.finished']
    assert records[-1].exception_type == 'ChatError'
    assert records[-1].outcome == 'failed'
    assert records[-1].elapsed_ms >= 0
    assert 'private task' not in caplog.text and 'private model response' not in caplog.text


async def test_unconfigured_model_does_not_fabricate_a_description():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    image = await engine.attach('alpha', PNG, media_type='image/png')
    episode = await engine.remember('alpha', 'Image', attachment_ids=[image.attachment_id])
    app = FastAPI()
    mount_image_understanding_routes(app, engine, lambda: 'alpha', lambda: None)
    with TestClient(app) as client:
        response = client.post(f'/v1/episodes/{episode.episode_id}/attachments/{image.attachment_id}/understand', json={'prompt': 'Describe'})
    assert response.status_code == 503
