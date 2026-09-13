"""Explicit generated interpretations of exact retained video frame pixels."""
import hashlib

import httpx
import pytest

from scone_memory.api.__main__ import build_app
from scone_memory.ingestion.files import ingest_document
from scone_memory.providers.vision import ImageUnderstanding
from scone_memory.runtime.config import Settings
from tests.api.test_video_documents import video_host


class FrameVision:
    model = 'selected-local-model'

    def __init__(self):
        self.calls = []
        self.after = None

    async def describe(self, data, media_type, *, prompt, source=None, attachment_id=None):
        self.calls.append((data, media_type, prompt, source, attachment_id))
        if self.after is not None:
            await self.after()
        return ImageUnderstanding('A generated frame interpretation.', hashlib.sha256(data).hexdigest(),
                                  source, media_type, self.model, 64, 32)


@pytest.fixture(params=[False, True], ids=['memory', 'composed'])
async def understanding_host(video_host, tmp_path, monkeypatch, request):
    from scone_memory.runtime import model_runtime
    engine, video, recognizer, original, env = video_host
    raw = (await engine.attachment('alpha', original.attachment_id))[1]
    saved = await ingest_document(engine, 'alpha', raw, filename='slides.mp4', parser=video.parser)
    vision = FrameVision()
    monkeypatch.setattr(model_runtime, 'local_vision_factory', lambda store: lambda: vision)
    env['SCONE_INGEST_CONCURRENCY'] = '1'
    env['SCONE_MODEL_CONNECTIONS'] = str(tmp_path / 'models.json')
    if request.param:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversations.sqlite')
    app = build_app(Settings.from_env(env), engine, document_video=video)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
        yield engine, video, recognizer, saved, vision, getattr(app.state, 'memory_app', app), client


async def test_interpretation_uses_verified_frame_and_never_persists_or_repeats_ocr(understanding_host):
    engine, _, ocr, saved, vision, _, client = understanding_host
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0'
    before = await engine.documents.counts('alpha')
    response = await client.post(path + '/understand', json={'prompt': 'Describe the visible scene.'})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['schema_version'] == 1 and body['space'] == 'alpha'
    assert body['episode_id'] == str(saved.added.episode_id)
    assert body['original_sha256'] == saved.original.attachment_id
    assert body['manifest_sha256'] == saved.manifest.attachment_id
    assert body['frame']['ordinal'] == 0 and body['frame']['presentation_timestamp'] == '0'
    assert body['frame']['time_base'] == '1/16384'
    assert body['persisted'] is False and body['understanding']['origin'] == 'model_generated'
    assert body['understanding']['model'] == vision.model
    png = await client.get(path)
    assert vision.calls[0][0] == png.content
    assert body['frame']['png_sha256'] == hashlib.sha256(png.content).hexdigest()
    assert ocr.calls == 2 and len(vision.calls) == 1
    assert await engine.documents.counts('alpha') == before


@pytest.mark.parametrize('token,status', [('reader', 403), ('other', 404), ('missing', 401)])
async def test_interpretation_requires_source_and_inference_authorization(understanding_host, token, status):
    _, _, _, saved, vision, _, client = understanding_host
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
        json={'prompt': 'Describe'}, headers={'authorization': 'Bearer ' + token})
    assert response.status_code == status, response.text
    assert not vision.calls


@pytest.mark.parametrize('phase', ['decoded', 'inferred'])
@pytest.mark.parametrize('change,status', [('forget', 410), ('unlink', 404), ('revoke', 401),
                                         ('scope', 401), ('role', 403)])
async def test_source_or_authority_changes_suppress_interpretation(understanding_host, monkeypatch, phase, change, status):
    engine, video, _, saved, vision, app, client = understanding_host
    async def mutate():
        if change == 'forget':
            await engine.forget('alpha', saved.added.episode_id)
        elif change == 'unlink':
            await engine.blobs.unlink('alpha', saved.added.episode_id)
        elif change == 'revoke':
            app.state.keys.pop('writer')
        elif change == 'scope':
            app.state.keys['writer'] = 'beta'
        else:
            app.state.roles['writer'] = 'read'
    if phase == 'inferred':
        vision.after = mutate
    else:
        read_frame = video.parser.read_frame
        async def changed(*args, **kwargs):
            frame = await read_frame(*args, **kwargs)
            await mutate()
            return frame
        monkeypatch.setattr(video.parser, 'read_frame', changed)
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
                                 json={'prompt': 'Describe'})
    assert response.status_code == status, response.text
    assert 'understanding' not in response.json()
    assert len(vision.calls) == (1 if phase == 'inferred' else 0)


@pytest.mark.parametrize('patch', [{'origin': 'approved'}, {'width': 65}, {'height': True},
    {'attachment_id': 'a' * 64}, {'source': 'unrelated'}, {'model': 'unselected'},
    {'media_type': 'image/jpeg'}, {'text': ' '}, {'text': 'x' * 64001}, {'text': '\ud800'}])
async def test_invalid_or_substituted_provider_result_is_not_exposed(understanding_host, monkeypatch, patch):
    from dataclasses import replace
    _, _, _, saved, vision, _, client = understanding_host
    describe = vision.describe
    async def substituted(*args, **kwargs):
        return replace(await describe(*args, **kwargs), **patch)
    monkeypatch.setattr(vision, 'describe', substituted)
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
                                 json={'prompt': 'Describe'})
    assert response.status_code == 502, response.text
    assert 'understanding' not in response.json()


@pytest.mark.parametrize('body', [{'prompt': ''}, {'prompt': ' '}, {'prompt': 'x' * 16001},
    {'prompt': '\ud800'}, {'prompt': 'Describe', 'persist': True}, {'prompt': 1}])
async def test_invalid_task_never_calls_vision(understanding_host, body):
    _, _, _, saved, vision, _, client = understanding_host
    import json
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
                                 content=json.dumps(body).encode(), headers={'content-type': 'application/json'})
    assert response.status_code == 400, response.text
    assert not vision.calls


async def test_cancel_releases_bounded_slot_without_automatic_retry(understanding_host):
    import asyncio
    _, _, _, saved, vision, _, client = understanding_host
    entered = asyncio.Event()
    async def pending():
        entered.set()
        await asyncio.Event().wait()
    vision.after = pending
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand'
    task = asyncio.create_task(client.post(path, json={'prompt': 'Describe'}))
    await asyncio.wait_for(entered.wait(), 10)
    busy = await client.post(path, json={'prompt': 'Describe'})
    assert busy.status_code == 429
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(vision.calls) == 1
    vision.after = None
    assert (await client.post(path, json={'prompt': 'Describe again'})).status_code == 200
    assert len(vision.calls) == 2


async def test_selected_model_is_read_for_each_explicit_request(understanding_host):
    _, _, _, saved, vision, _, client = understanding_host
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand'
    first = await client.post(path, json={'prompt': 'Describe'})
    vision.model = 'another-selected-local-model'
    second = await client.post(path, json={'prompt': 'Describe'})
    assert first.json()['understanding']['model'] == 'selected-local-model'
    assert second.json()['understanding']['model'] == 'another-selected-local-model'


async def test_video_interpretation_advertises_decoder_and_model_requirements(video_host):
    from scone_memory.api.app import create_app
    engine, video, _, _, _ = video_host
    for decoder, available, expected in [(video, True, True), (video, False, False), (None, True, False)]:
        app = create_app(engine, keys={'writer': 'alpha'}, document_video=decoder,
                         vision_factory=lambda: FrameVision(), vision_available=lambda: available)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture',
                                     headers={'authorization': 'Bearer writer'}) as client:
            response = await client.get('/v1/capabilities')
            assert response.json()['features'].get('documents.video.understand', False) is expected


async def test_excessively_nested_task_refuses_without_inference(understanding_host, monkeypatch):
    from types import SimpleNamespace
    from scone_memory.api import video_understanding
    def limited_decoder(_body):
        raise RecursionError("JSON nesting limit exceeded")
    monkeypatch.setattr(video_understanding, "json", SimpleNamespace(loads=limited_decoder))
    _, _, _, saved, vision, _, client = understanding_host
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
        content=b'{"prompt":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}')
    assert response.status_code == 400
    assert not vision.calls


@pytest.mark.parametrize('phase', ['decoded', 'inferred'])
@pytest.mark.parametrize('behavior', ['blocks', 'swallows_timeout'])
async def test_expired_work_cannot_start_inference_or_publish(understanding_host, monkeypatch, phase, behavior):
    import asyncio
    import time
    from types import SimpleNamespace
    from scone_memory.api import video_understanding
    from scone_memory.ingestion.video_source import read_video_source
    from scone_memory.ingestion.formats.types import DocumentLimits
    engine, video, _, saved, vision, _, client = understanding_host
    source = await read_video_source(engine, 'alpha', saved.added.episode_id, 0)
    frame = await video.parser.read_frame(source.raw, source.evidence.filename,
        source.evidence.video, 0, limits=DocumentLimits())
    real_timeout = asyncio.timeout
    monkeypatch.setattr(video_understanding, 'asyncio', SimpleNamespace(timeout=lambda _: real_timeout(0.01)))
    async def late():
        if behavior == 'blocks':
            time.sleep(0.04)
        else:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
    async def cached(*args, **kwargs):
        if phase == 'decoded':
            await late()
        return frame
    monkeypatch.setattr(video.parser, 'read_frame', cached)
    if phase == 'inferred':
        vision.after = late
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
                                 json={'prompt': 'Describe'})
    assert response.status_code == 504, response.text
    assert len(vision.calls) == (1 if phase == 'inferred' else 0)


async def test_provider_cannot_swallow_request_cancellation(understanding_host):
    import asyncio
    _, _, _, saved, vision, _, client = understanding_host
    entered = asyncio.Event()
    async def swallow():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
    vision.after = swallow
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand'
    task = asyncio.create_task(client.post(path, json={'prompt': 'Describe'}))
    await asyncio.wait_for(entered.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    vision.after = None
    assert (await client.post(path, json={'prompt': 'Try again'})).status_code == 200


async def test_unicode_prompt_limit_accepts_escaped_json(understanding_host):
    import json
    _, _, _, saved, vision, _, client = understanding_host
    prompt = '😀' * 16000
    response = await client.post(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0/understand',
                                 content=json.dumps({'prompt': prompt}).encode())
    assert response.status_code == 200, response.text
    assert vision.calls[0][2] == prompt
