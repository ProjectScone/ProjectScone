"""Local decoding and scripted timestamps verify ingestion, not recognition quality."""
import io
import shutil
import wave

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.ingestion.formats.media import MediaDocumentParser, TranscriptionSegment


def audio_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b'\0' * 6400)
    return buffer.getvalue()


@pytest.fixture
async def media_service():
    from scone_memory.ingestion.document_media import DocumentMedia

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        pytest.skip('explicit local ffmpeg unavailable')
    calls = []
    async def transcribe(data):
        calls.append(data)
        return (TranscriptionSegment(text='Café launch is Friday.', start_seconds=0.0, end_seconds=0.15),)
    media = DocumentMedia(MediaDocumentParser(transcribe, ffmpeg_executable=ffmpeg), revision='fixture-v1')
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {'writer': 'alpha', 'reader': 'alpha', 'other': 'beta'},
        roles={'reader': 'read'}, document_media=media)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture',
                                 headers={'authorization': 'Bearer writer'}) as client:
        yield client, engine, media, calls, app
    await engine.close()


async def test_media_upload_recall_timestamp_evidence_and_forgetting(media_service):
    client, engine, _, calls, _ = media_service
    formats = (await client.get('/v1/documents/formats')).json()['formats']
    assert formats['.wav']['available'] is True
    assert formats['.mp4']['extraction'] == 'audio-only'
    assert formats['.ts']['parser'] == 'text', 'TypeScript must not become an audio container'
    raw = audio_bytes()
    upload = await client.post('/v1/attachments', content=raw, headers={'content-type': 'application/octet-stream'})
    assert upload.status_code == 200, upload.text
    body = {'attachment_id': upload.json()['attachment_id'], 'filename': 'launch.wav'}
    for key, status in [('reader', 403), ('other', 404)]:
        response = await client.post('/v1/documents', json=body, headers={'authorization': 'Bearer '+key})
        assert response.status_code == status
    assert calls == []
    response = await client.post('/v1/documents', json=body)
    assert response.status_code == 200, response.text
    episode_id = response.json()['added']['episode_id']
    assert len(calls) == 1
    recalled = (await client.get('/v1/recall', params={'q': 'Café launch Friday'})).json()['items']
    chunk = next(item for item in recalled if item['episode_id'] == episode_id)
    path = f'/v1/episodes/{episode_id}/document'
    evidence = await client.get(path, params={'chunk_id': chunk['chunk_id']}, headers={'authorization': 'Bearer reader'})
    assert evidence.status_code == 200, evidence.text
    data = evidence.json()
    assert data['metadata']['extraction'] == 'audio-only'
    assert data['metadata']['transcriber_revision'] == 'fixture-v1'
    assert data['segments'][0]['text'] == 'Café launch is Friday.'
    assert data['segments'][0]['metadata']['end_seconds'] == '0.15'
    assert (await client.get(data['download_path'])).content == raw
    assert (await client.get(path, headers={'authorization': 'Bearer other'})).status_code == 404
    await engine.forget('alpha', episode_id)
    assert (await client.get(path)).status_code == 410
    assert len(calls) == 1, 'reading evidence never retranscribes'


async def test_media_is_not_advertised_without_host_configuration():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine, {'writer':'alpha'})),
            base_url='http://fixture', headers={'authorization':'Bearer writer'}) as client:
            formats = (await client.get('/v1/documents/formats')).json()['formats']
            assert '.wav' not in formats and '.mp4' not in formats
    finally:
        await engine.close()


@pytest.mark.parametrize('composed', [False, True])
async def test_host_media_jobs_survive_restart_and_bind_parser_revision(media_service, tmp_path, monkeypatch, composed):
    import json
    from dataclasses import replace
    from scone_memory.api.__main__ import build_app
    from scone_memory.runtime.config import Settings

    _, engine, media, calls, _ = media_service
    path = tmp_path / 'jobs.json'
    path.write_text(json.dumps({'schema_version':1, 'state_dir':'jobs', 'key_env':'TEST_MEDIA_JOB_KEY', 'parser_revision':'parsers-v1'}))
    path.chmod(0o600)
    monkeypatch.setenv('TEST_MEDIA_JOB_KEY', 'ab'*32)
    env = {'SCONE_API_KEYS':'writer:alpha:write,reader:alpha:read,other:beta:write', 'SCONE_DOCUMENT_JOBS_CONFIG':str(path)}
    if composed:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversations.sqlite')
    settings = Settings.from_env(env)
    raw = audio_bytes()
    original = await engine.attach('alpha', raw, 'application/octet-stream', filename='launch.wav')
    for first in [True, False]:
        app = build_app(settings, engine, document_media=media)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
            base_url='http://fixture', headers={'authorization':'Bearer writer'}) as client:
            target = getattr(app.state, 'memory_app', app)
            service = target.state.document_import_service
            assert (await client.get('/v1/documents/formats')).json()['formats']['.wav']['available']
            if first:
                response = await client.post('/v1/document-jobs', json={'import_id':'audio-one', 'attachment_id':original.attachment_id, 'filename':'launch.wav'})
                assert response.status_code == 202, response.text
                assert (await service.wait('alpha', 'audio-one')).status == 'completed'
            result = await client.get('/v1/document-jobs/audio-one/result', headers={'authorization':'Bearer reader'})
            assert result.status_code == 200, result.text
            assert (await client.get('/v1/document-jobs/audio-one/result', headers={'authorization':'Bearer other'})).status_code == 404
            episode_id = result.json()['added']['episode_id']
            evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
            assert evidence['segments'][0]['metadata']['end_seconds'] == '0.15'
            assert len(calls) == 1, 'restart and result reads must not transcribe again'
    app = build_app(settings, engine, document_media=replace(media, revision='different-model-v2'))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
        base_url='http://fixture', headers={'authorization':'Bearer writer'}) as client:
        response = await client.get('/v1/document-jobs/audio-one/result')
        assert response.status_code == 409, response.text
        assert len(calls) == 1


@pytest.mark.parametrize('change', ['scope', 'role'])
async def test_media_rechecks_authorization_after_transcription_before_storage(media_service, monkeypatch, change):
    client, engine, media, calls, app = media_service
    original = media.media_parser._transcribe
    async def changed(data):
        result = await original(data)
        if change == 'scope':
            app.state.keys['writer'] = 'beta'
        else:
            app.state.roles['writer'] = 'read'
        return result
    monkeypatch.setattr(media.media_parser, '_transcribe', changed)
    upload = await client.post('/v1/attachments', content=audio_bytes(), headers={'content-type':'application/octet-stream'})
    response = await client.post('/v1/documents', json={'attachment_id':upload.json()['attachment_id'], 'filename':'launch.wav'})
    assert response.status_code in (401, 403), response.text
    assert len(calls) == 1
    assert (await engine.documents.counts('alpha')).episodes == 0
    assert (await engine.documents.counts('beta')).episodes == 0


async def test_configured_media_does_not_intercept_typescript(media_service):
    client, _, _, calls, _ = media_service
    upload = await client.post('/v1/attachments', content=b'export const launch = "Friday";', headers={'content-type':'application/octet-stream'})
    response = await client.post('/v1/documents', json={'attachment_id':upload.json()['attachment_id'], 'filename':'launch.ts'})
    assert response.status_code == 200, response.text
    assert response.json()['format'] == 'ts'
    assert calls == []


async def test_video_import_keeps_audio_only_source_times(media_service, tmp_path):
    import subprocess
    client, _, media, calls, _ = media_service
    path = tmp_path / 'clip.mp4'
    subprocess.run([media.media_parser._ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'color=c=black:s=16x16:r=10', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
        '-t', '0.2', '-c:v', 'mpeg4', '-c:a', 'aac', str(path)], check=True, timeout=10)
    raw = path.read_bytes()
    uploaded = await client.post('/v1/attachments', content=raw, headers={'content-type':'application/octet-stream'})
    response = await client.post('/v1/documents', json={'attachment_id':uploaded.json()['attachment_id'], 'filename':'clip.mp4'})
    assert response.status_code == 200, response.text
    evidence = (await client.get(f"/v1/episodes/{response.json()['added']['episode_id']}/document")).json()
    assert evidence['metadata']['extraction'] == 'audio-only'
    assert evidence['segments'][0]['metadata']['audio_stream'] == '0'
    assert evidence['segments'][0]['metadata']['end_seconds'] == '0.15'
    assert len(calls) == 1
    assert (await client.get(evidence['download_path'])).content == raw


async def test_normalized_playback_matches_transcriber_input_without_inference(media_service):
    from hashlib import sha256
    client, _, _, calls, _ = media_service
    uploaded = await client.post('/v1/attachments', content=audio_bytes(), headers={'content-type':'application/octet-stream'})
    indexed = await client.post('/v1/documents', json={'attachment_id':uploaded.json()['attachment_id'], 'filename':'clip.wav'})
    assert indexed.status_code == 200, indexed.text
    episode_id = indexed.json()['added']['episode_id']
    provenance = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
    expected = calls[0]
    assert provenance['metadata']['audio_wav_sha256'] == sha256(expected).hexdigest()
    assert provenance['metadata']['audio_wav_bytes'] == str(len(expected))
    path = f'/v1/episodes/{episode_id}/document/audio'
    audio = await client.get(path, headers={'authorization':'Bearer reader'})
    assert audio.status_code == 200, audio.text
    assert audio.headers['content-type'] == 'audio/wav'
    assert audio.headers['cache-control'] == 'no-store'
    assert audio.content == expected
    assert len(calls) == 1, 'playback must never transcribe'
    assert (await client.get(path, headers={'authorization':'Bearer other'})).status_code == 404
    await client.delete(f'/v1/episodes/{episode_id}')
    assert (await client.get(path)).status_code == 410


@pytest.mark.parametrize('change', ['bytes', 'forgotten', 'scope'])
async def test_normalized_playback_refuses_changed_audio_or_final_source(media_service, monkeypatch, change):
    client, engine, media, _, app = media_service
    uploaded = await client.post('/v1/attachments', content=audio_bytes(), headers={'content-type':'application/octet-stream'})
    indexed = await client.post('/v1/documents', json={'attachment_id':uploaded.json()['attachment_id'], 'filename':'clip.wav'})
    episode_id = indexed.json()['added']['episode_id']
    original = media.media_parser.decode_audio
    decoded = []
    async def changed(*args, **kwargs):
        audio = await original(*args, **kwargs)
        decoded.append(True)
        if change == 'bytes':
            return audio[:-1] + bytes([audio[-1] ^ 1])
        if change == 'forgotten':
            await engine.forget('alpha', episode_id)
        else:
            app.state.keys['writer'] = 'beta'
        return audio
    monkeypatch.setattr(media.media_parser, 'decode_audio', changed)
    response = await client.get(f'/v1/episodes/{episode_id}/document/audio')
    assert decoded == [True]
    assert response.status_code == {'bytes':422, 'forgotten':410, 'scope':401}[change], response.text
    assert response.headers['content-type'].startswith('application/json')


async def test_media_format_discovery_rechecks_the_configured_decoder(tmp_path):
    from scone_memory.ingestion.document_media import DocumentMedia
    executable=tmp_path/'ffmpeg-fixture'
    executable.write_text('#!/bin/sh\nexit 1\n')
    executable.chmod(0o700)
    async def unused(audio): raise AssertionError('format discovery must not transcribe')
    media=DocumentMedia(MediaDocumentParser(unused,ffmpeg_executable=str(executable)),revision='fixture-v1')
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine,{'reader':'alpha'},
            roles={'reader':'read'},document_media=media)),base_url='http://fixture',headers={'authorization':'Bearer reader'}) as client:
            async def formats(): return (await client.get('/v1/documents/formats')).json()['formats']
            first=await formats()
            assert first['.wav']['available'] is True
            executable.chmod(0o600)
            assert (await formats())['.wav']['available'] is False
            executable.chmod(0o700)
            assert (await formats())['.wav']['available'] is True
            executable.unlink()
            latest=await formats()
            assert latest['.wav']['available'] is False
            assert latest['.txt']['available'] is True
            assert latest['.wav']['requires']=='configured ffmpeg and timestamped transcriber'
    finally: await engine.close()
