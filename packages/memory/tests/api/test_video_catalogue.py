"""Browser video citations preserve int64 clocks and current source access."""
import httpx
import pytest

from scone_memory.api.__main__ import build_app
from scone_memory.ingestion.files import prepare_document, store_document
from scone_memory.ingestion.formats.types import DocumentLimits
from scone_memory.runtime.config import Settings
from tests.api.test_video_documents import video_host
from tests.api.test_video_frames_api import frame_host


async def test_catalogue_binds_full_evidence_without_decoding(frame_host, monkeypatch):
    _, video, recognizer, saved, _, client = frame_host
    async def unexpected(*args, **kwargs):
        raise AssertionError('catalogue must not decode or run OCR')
    monkeypatch.setattr(video.parser._decoder, 'sample', unexpected)
    path = f'/v1/episodes/{saved.added.episode_id}/document'
    existing = (await client.get(path)).json()
    response = await client.get(path + '/video/catalogue')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['schema_version'] == 1
    assert body['timestamp_encoding'] == 'decimal-string'
    assert body['episode_id'] == str(saved.added.episode_id)
    assert body['space'] == 'alpha'
    evidence = body['evidence']
    assert evidence['original'] == existing['original']
    assert evidence['manifest'] == existing['manifest']
    assert evidence['segments'] == existing['segments']
    assert evidence['video']['start_timestamp'] == '0'
    assert evidence['video']['frames'][0]['presentation_timestamp'] == '0'
    assert isinstance(evidence['video']['duration_ticks'], str)
    assert isinstance(existing['video']['start_timestamp'], int)
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert recognizer.calls == 2


@pytest.mark.parametrize('start', [9007199254740993, -9007199254740993, -(2**63)])
async def test_catalogue_preserves_unsafe_signed_timestamps_without_decoder(video_host, start):
    engine, video, recognizer, original, env = video_host
    raw = (await engine.attachment('alpha', original.attachment_id))[1]
    manifest = await prepare_document(raw, 'slides.mp4', parser=video.parser, limits=DocumentLimits())
    retained = manifest.parsed.video
    frames = tuple(frame.model_copy(update={'presentation_timestamp': frame.presentation_timestamp + start})
                   for frame in retained.frames)
    shifted = retained.model_copy(update={'start_timestamp': start, 'frames': frames})
    manifest = manifest.model_copy(update={'parsed': manifest.parsed.model_copy(update={'video': shifted})})
    saved = await store_document(engine, 'alpha', original, manifest)
    app = build_app(Settings.from_env(env), engine)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer reader'}) as client:
        response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/catalogue')
    assert response.status_code == 200, response.text
    result = response.json()['evidence']['video']
    assert result['start_timestamp'] == str(start)
    assert result['frames'][0]['presentation_timestamp'] == str(start)
    assert int(result['frames'][1]['presentation_timestamp']) - int(result['start_timestamp']) == 16384
    assert recognizer.calls == 2


async def test_catalogue_refuses_other_space_and_forgotten_source(frame_host):
    engine, _, _, saved, _, client = frame_host
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/catalogue'
    assert (await client.get(path, headers={'authorization': 'Bearer other'})).status_code == 404
    assert (await client.get(path, headers={'authorization': 'Bearer missing'})).status_code == 401
    await engine.forget('alpha', saved.added.episode_id)
    assert (await client.get(path)).status_code == 410


async def test_invalid_native_attachment_text_refuses_without_encoding_crash(frame_host):
    engine, video, _, saved, _, client = frame_host
    raw = (await engine.attachment('alpha', saved.original.attachment_id))[1]
    original = await engine.attach('beta', raw, 'application/octet-stream', filename='bad\ud800.mp4')
    manifest = await prepare_document(raw, 'slides.mp4', parser=video.parser, limits=DocumentLimits())
    other = await store_document(engine, 'beta', original, manifest)
    response = await client.get(f'/v1/episodes/{other.added.episode_id}/document/video/catalogue',
                                headers={'authorization': 'Bearer other'})
    assert response.status_code == 422, response.text
    assert 'evidence' not in response.json()


@pytest.mark.parametrize('change', ['scope', 'revoked', 'forgotten', 'unlinked', 'text', 'manifest'])
@pytest.mark.parametrize('read_number', [1, 2])
async def test_catalogue_revalidates_after_attachment_reads(frame_host, monkeypatch, change, read_number):
    engine, _, _, saved, app, client = frame_host
    attachment = engine.attachment
    triggered = False
    reads = 0
    async def changed(space, attachment_id):
        nonlocal triggered, reads
        result, raw = await attachment(space, attachment_id)
        if attachment_id == saved.manifest.attachment_id:
            reads += 1
        if not triggered and attachment_id == saved.manifest.attachment_id and reads == read_number:
            triggered = True
            if change == 'scope':
                app.state.keys['reader'] = 'beta'
            elif change == 'revoked':
                del app.state.keys['reader']
            elif change == 'forgotten':
                await engine.forget('alpha', saved.added.episode_id)
            elif change == 'unlinked':
                await engine.blobs.unlink('alpha', saved.added.episode_id)
            elif change == 'text':
                source = await engine.documents.get_episode('alpha', saved.added.episode_id)
                engine.documents._episodes[source.episode_id] = source.model_copy(update={'content': 'Changed'})
            else:
                return result, raw + b'changed'
        return result, raw
    monkeypatch.setattr(engine, 'attachment', changed)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/catalogue')
    assert triggered
    assert response.status_code in (401, 403, 404, 410, 422), response.text
    assert 'evidence' not in response.json()
