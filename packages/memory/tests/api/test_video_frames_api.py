"""Retained frame pixels are checked against provenance and current access."""
import hashlib

import httpx
import pytest

from scone_memory.api.__main__ import build_app
from scone_memory.ingestion.files import ingest_document
from scone_memory.runtime.config import Settings
from tests.api.test_video_documents import video_host


@pytest.fixture(params=[False, True], ids=['memory', 'composed'])
async def frame_host(video_host, request, tmp_path):
    engine, video, recognizer, original, env = video_host
    raw = (await engine.attachment('alpha', original.attachment_id))[1]
    env['SCONE_INGEST_CONCURRENCY'] = '1'
    saved = await ingest_document(engine, 'alpha', raw, filename='slides.mp4', parser=video.parser)
    if request.param:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversation.sqlite')
    app = build_app(Settings.from_env(env), engine, document_video=video)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer reader'}) as client:
        yield engine, video, recognizer, saved, getattr(app.state, 'memory_app', app), client


async def test_frame_read_returns_retained_pixels_and_no_ocr(frame_host):
    _, _, recognizer, saved, _, client = frame_host
    episode_id = saved.added.episode_id
    evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
    frame = evidence['video']['frames'][1]
    response = await client.get(f'/v1/episodes/{episode_id}/document/video/frames/{frame["ordinal"]}')
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'image/png'
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert hashlib.sha256(response.content).hexdigest() == frame['png_sha256']
    assert len(response.content) == frame['png_bytes']
    assert response.headers['x-scone-video-pts'] == str(frame['presentation_timestamp'])
    assert response.headers['x-scone-video-time-base'] == evidence['video']['time_base']
    assert recognizer.calls == 2


@pytest.mark.parametrize('ordinal', [-1, 100000, 7])
async def test_unretained_frame_refuses_before_decode(frame_host, monkeypatch, ordinal):
    _, video, recognizer, saved, _, client = frame_host
    async def unexpected(*args, **kwargs):
        raise AssertionError('invalid frame must be rejected before decoding')
    monkeypatch.setattr(video.parser._decoder, 'sample', unexpected)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/{ordinal}')
    assert response.status_code == 422, response.text
    assert recognizer.calls == 2


async def test_other_space_cannot_read_or_decode_frame(frame_host, monkeypatch):
    _, video, _, saved, _, client = frame_host
    async def unexpected(*args, **kwargs):
        raise AssertionError('unauthorized frame must not be decoded')
    monkeypatch.setattr(video.parser._decoder, 'sample', unexpected)
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0'
    response = await client.get(path, headers={'authorization': 'Bearer other'})
    assert response.status_code == 404
    assert (await client.get(path, headers={'authorization': 'Bearer missing'})).status_code == 401


@pytest.mark.parametrize('change', ['scope', 'revoked', 'forgotten', 'unlinked'])
async def test_source_or_access_changes_during_decode_refuse(frame_host, monkeypatch, change):
    engine, video, _, saved, app, client = frame_host
    sample = video.parser._decoder.sample
    called = []
    async def changed(*args, **kwargs):
        frames = await sample(*args, **kwargs)
        called.append(True)
        if change == 'scope':
            app.state.keys['reader'] = 'beta'
        elif change == 'revoked':
            del app.state.keys['reader']
        elif change == 'forgotten':
            await engine.forget('alpha', saved.added.episode_id)
        else:
            await engine.blobs.unlink('alpha', saved.added.episode_id)
        return frames
    monkeypatch.setattr(video.parser._decoder, 'sample', changed)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0')
    assert response.status_code in (401, 403, 404, 410, 422), response.text
    assert called == [True]
    assert response.headers['content-type'] != 'image/png'


async def test_changed_decoded_pixels_refuse(frame_host, monkeypatch):
    from dataclasses import replace
    _, video, _, saved, _, client = frame_host
    sample = video.parser._decoder.sample
    async def changed(*args, **kwargs):
        frames = await sample(*args, **kwargs)
        altered = replace(frames.frames[0], png=frames.frames[1].png, sha256=frames.frames[1].sha256)
        return replace(frames, frames=(altered, *frames.frames[1:]))
    monkeypatch.setattr(video.parser._decoder, 'sample', changed)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0')
    assert response.status_code == 422, response.text


@pytest.mark.parametrize('change', ['timestamp', 'decoder', 'dimensions', 'png_bytes'])
async def test_changed_frame_identity_refuses(frame_host, monkeypatch, change):
    from dataclasses import replace
    _, video, _, saved, _, client = frame_host
    sample = video.parser._decoder.sample
    async def changed(*args, **kwargs):
        frames = await sample(*args, **kwargs)
        if change == 'decoder':
            return replace(frames, decoder_revision='a' * 64)
        first = frames.frames[0]
        if change == 'timestamp':
            first = replace(first, selection=replace(first.selection, presentation_timestamp=1))
        elif change == 'dimensions':
            first = replace(first, width=first.width + 1)
        else:
            first = replace(first, png=first.png + b'changed')
        return replace(frames, frames=(first, *frames.frames[1:]))
    monkeypatch.setattr(video.parser._decoder, 'sample', changed)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0')
    assert response.status_code == 422, response.text


async def test_forget_during_final_hydration_cannot_return_stale_frame(frame_host, monkeypatch):
    engine, video, _, saved, _, client = frame_host
    sample, episode = video.parser._decoder.sample, engine.episode
    decoded = False
    checked = []
    async def sampled(*args, **kwargs):
        nonlocal decoded
        result = await sample(*args, **kwargs)
        decoded = True
        return result
    async def hydrated(*args, **kwargs):
        result = await episode(*args, **kwargs)
        if decoded:
            checked.append(True)
            if len(checked) == 2:
                await engine.forget('alpha', saved.added.episode_id)
        return result
    monkeypatch.setattr(video.parser._decoder, 'sample', sampled)
    monkeypatch.setattr(engine, 'episode', hydrated)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0')
    assert len(checked) == 2
    assert response.status_code == 410, response.text


@pytest.mark.parametrize('change', ['original', 'manifest', 'scope', 'unlink'])
async def test_final_attachment_reads_revalidate_source_and_access(frame_host, monkeypatch, change):
    engine, video, _, saved, app, client = frame_host
    sample, attachment = video.parser._decoder.sample, engine.attachment
    decoded = False
    changed = []
    async def sampled(*args, **kwargs):
        nonlocal decoded
        result = await sample(*args, **kwargs)
        decoded = True
        return result
    async def read(space, attachment_id):
        result, data = await attachment(space, attachment_id)
        if decoded and attachment_id == saved.manifest.attachment_id:
            changed.append(True)
            if change == 'manifest':
                return result, data + b'changed'
            if change == 'scope':
                app.state.keys['reader'] = 'beta'
            elif change == 'unlink':
                await engine.blobs.unlink('alpha', saved.added.episode_id)
        if decoded and attachment_id == saved.original.attachment_id and change == 'original':
            changed.append(True)
            return result, data + b'changed'
        return result, data
    monkeypatch.setattr(video.parser._decoder, 'sample', sampled)
    monkeypatch.setattr(engine, 'attachment', read)
    response = await client.get(f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0')
    assert changed
    assert response.status_code in (401, 403, 422), response.text
    assert response.headers['content-type'] != 'image/png'


async def test_busy_frame_reads_refuse_and_cancel_releases_capacity(frame_host, monkeypatch):
    import asyncio
    _, video, recognizer, saved, _, client = frame_host
    entered = asyncio.Event()
    sample = video.parser._decoder.sample
    async def paused(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        return await sample(*args, **kwargs)
    monkeypatch.setattr(video.parser._decoder, 'sample', paused)
    path = f'/v1/episodes/{saved.added.episode_id}/document/video/frames/0'
    pending = asyncio.create_task(client.get(path))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        busy = await client.get(path)
        assert busy.status_code == 429, busy.text
    finally:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    monkeypatch.setattr(video.parser._decoder, 'sample', sample)
    assert (await client.get(path)).status_code == 200
    assert recognizer.calls == 2
