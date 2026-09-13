"""Native decoding and scripted OCR verify explicit host dispatch and recovery."""
from dataclasses import replace
import json
import shutil
import subprocess

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.ingestion.document_video import DocumentVideo
from scone_memory.ingestion.video_frames import VideoFrameDecoder, VideoFramePolicy
from scone_memory.ingestion.video_ocr import VideoDocumentParser
from scone_memory.ocr.types import OcrRegion, OcrResult
from scone_memory.runtime.config import Settings


class Recognizer:
    def __init__(self):
        self.calls = 0
        self.after = None
        self.fail_at = None

    async def recognize(self, image, **options):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('interrupted OCR')
        if self.after is not None:
            self.after()
        return OcrResult(engine='fixture', width=64, height=32,
            regions=(OcrRegion(text='Café launch Friday', box=(0.1, 0.1, 0.9, 0.8), score=0.8),))


@pytest.fixture
async def video_host(tmp_path, monkeypatch):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('installed ffmpeg and ffprobe required')
    pytest.importorskip('PIL')
    path = tmp_path / 'slides.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x32:rate=1:duration=2',
                    '-an', '-c:v', 'libx264', str(path)], check=True)
    decoder = VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe)
    recognizer = Recognizer()
    video = DocumentVideo(VideoDocumentParser(decoder, recognizer, model_revision='fixture-v1',
        policy=VideoFramePolicy(interval_seconds=1)), revision='host-v1')
    jobs = tmp_path / 'jobs.json'
    jobs.write_text(json.dumps({'schema_version': 1, 'state_dir': 'jobs',
        'key_env': 'TEST_VIDEO_KEY', 'parser_revision': 'parser-v1'}))
    jobs.chmod(0o600)
    monkeypatch.setenv('TEST_VIDEO_KEY', 'ab' * 32)
    env = {'SCONE_API_KEYS': 'writer:alpha:write,reader:alpha:read,other:beta:write',
           'SCONE_DOCUMENT_JOBS_CONFIG': str(jobs)}
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        original = await engine.attach('alpha', path.read_bytes(), 'application/octet-stream', filename='slides.mp4')
        yield engine, video, recognizer, original, env
    finally:
        await engine.close()


@pytest.mark.parametrize('composed', [False, True])
async def test_explicit_video_import_survives_host_restart(video_host, tmp_path, composed):
    engine, video, recognizer, original, env = video_host
    if composed:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversation.sqlite')
    settings = Settings.from_env(env)
    for first in [True, False]:
        app = build_app(settings, engine, document_video=video)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
            base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
            formats = (await client.get('/v1/documents/formats')).json()
            assert formats['video_ocr']['available'] is True
            assert formats['video_ocr']['includes_audio'] is False
            assert '.ts' not in formats['video_ocr']['extensions']
            if first:
                body = {'import_id': 'slides-one', 'attachment_id': original.attachment_id,
                        'filename': 'slides.mp4', 'video_ocr': True}
                for key, code in [('reader', 403), ('other', 404)]:
                    response = await client.post('/v1/document-jobs', json=body,
                        headers={'authorization': 'Bearer ' + key})
                    assert response.status_code == code, response.text
                assert recognizer.calls == 0
                response = await client.post('/v1/document-jobs', json=body)
                assert response.status_code == 202, response.text
                target = getattr(app.state, 'memory_app', app)
                assert (await target.state.document_import_service.wait('alpha', 'slides-one')).status == 'completed'
                changed = await client.post('/v1/document-jobs', json={**body, 'video_ocr': False})
                assert changed.status_code == 409
            response = await client.get('/v1/document-jobs/slides-one/result', headers={'authorization': 'Bearer reader'})
            assert response.status_code == 200, response.text
            assert response.json()['video_ocr'] is True
            saved = (await client.get('/v1/document-jobs/slides-one/request')).json()
            assert saved['spec']['video_ocr'] is True
            episode_id = response.json()['added']['episode_id']
            evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
            assert evidence['parser'] == 'video-frame-ocr'
            assert len(evidence['video']['frames']) == 2
            assert recognizer.calls == 2, 'restart and evidence reads must not rerun OCR'
    app = build_app(settings, engine, document_video=replace(video, revision='changed-v2'))
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
        assert (await client.get('/v1/document-jobs/slides-one/result')).status_code == 409
        assert recognizer.calls == 2


async def test_video_ocr_does_not_replace_existing_transcription(video_host):
    from scone_memory.ingestion.document_media import DocumentMedia
    from scone_memory.ingestion.formats.media import MediaDocumentParser
    engine, video, recognizer, original, env = video_host
    async def transcribe(data):
        raise AssertionError('visual extraction must not dispatch to transcriber')
    media = DocumentMedia(MediaDocumentParser(transcribe, ffmpeg_executable=shutil.which('ffmpeg')), revision='audio-v1')
    app = build_app(Settings.from_env(env), engine, document_media=media, document_video=video)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
        formats = (await client.get('/v1/documents/formats')).json()
        assert formats['formats']['.mp4']['extraction'] == 'audio-only'
        body = {'attachment_id': original.attachment_id, 'filename': 'slides.mp4', 'video_ocr': True}
        response = await client.post('/v1/documents', json=body)
        assert response.status_code == 200, response.text
        assert response.json()['video_ocr'] is True
        assert recognizer.calls == 2


@pytest.mark.parametrize('choice', [1, 'true', None])
async def test_video_choice_requires_a_boolean(video_host, choice):
    engine, video, recognizer, original, env = video_host
    app = build_app(Settings.from_env(env), engine, document_video=video)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
        response = await client.post('/v1/documents', json={'attachment_id': original.attachment_id,
            'filename': 'slides.mp4', 'video_ocr': choice})
        assert response.status_code == 400
        assert recognizer.calls == 0


@pytest.mark.parametrize('change', ['role', 'scope'])
async def test_changed_authorization_during_ocr_refuses_storage(video_host, change):
    engine, video, recognizer, original, env = video_host
    app = build_app(Settings.from_env(env), engine, document_video=video)
    def changed():
        if change == 'role':
            app.state.roles['writer'] = 'read'
        else:
            app.state.keys['writer'] = 'beta'
    recognizer.after = changed
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
        response = await client.post('/v1/documents', json={'attachment_id': original.attachment_id,
            'filename': 'slides.mp4', 'video_ocr': True})
        assert response.status_code in (401, 403), response.text
        assert (await engine.documents.counts('alpha')).episodes == 0
        assert (await engine.documents.counts('beta')).episodes == 0


async def test_directory_video_choice_is_explicit_and_bound(video_host, tmp_path, monkeypatch):
    from scone_memory.runtime.directory_sync import load_directory_sync
    from tests.runtime.test_directory_sync_runtime import wait_service
    engine, video, recognizer, original, _ = video_host
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'slides.mp4').write_bytes((await engine.attachment('alpha', original.attachment_id))[1])
    path = tmp_path / 'sync.json'
    body = {'schema_version': 1, 'state_dir': 'sync-state', 'key_env': 'TEST_VIDEO_KEY',
        'store_id': 'fixture', 'collections': [{'collection_id': 'slides', 'label': 'Slides',
            'space': 'alpha', 'root': str(source), 'parser_revision': 'parser-v1'}]}
    path.write_text(json.dumps(body))
    path.chmod(0o600)
    host = load_directory_sync(str(path), engine, document_video=video)
    try:
        assert '.mp4' not in host._collections[('alpha', 'slides')].sync.scanner.extensions
    finally:
        await host.aclose()

    body['collections'][0]['video_ocr'] = True
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='video OCR'):
        load_directory_sync(str(path), engine)
    host = load_directory_sync(str(path), engine, document_video=video)
    try:
        await host.start('alpha', 'scan', collection_id='slides')
        status = await wait_service(host)
        assert status.status == 'completed', status
        assert recognizer.calls == 2
        first_revision = host._collections[('alpha', 'slides')].sync.parser_revision
    finally:
        await host.aclose()
    host = load_directory_sync(str(path), engine, document_video=replace(video, revision='changed-v2'))
    try:
        assert host._collections[('alpha', 'slides')].sync.parser_revision != first_revision
        assert recognizer.calls == 2
    finally:
        await host.aclose()


async def test_partial_video_job_reuses_receipts_after_host_restart(video_host):
    engine, video, recognizer, original, env = video_host
    recognizer.fail_at = 2
    settings = Settings.from_env(env)
    for first in [True, False]:
        app = build_app(settings, engine, document_video=video)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
            base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
            service = app.state.document_import_service
            if first:
                response = await client.post('/v1/document-jobs', json={'import_id': 'partial',
                    'attachment_id': original.attachment_id, 'filename': 'slides.mp4', 'video_ocr': True})
                assert response.status_code == 202
                assert (await service.wait('alpha', 'partial')).status == 'failed'
                assert recognizer.calls == 2
            else:
                recognizer.fail_at = None
                status = (await client.get('/v1/document-jobs/partial')).json()
                response = await client.post('/v1/document-jobs/partial/resume',
                    json={'expected_revision': status['revision']})
                assert response.status_code == 202, response.text
                assert (await service.wait('alpha', 'partial')).status == 'completed'
                assert recognizer.calls == 3, 'only the missing frame should run after restart'


@pytest.mark.parametrize('mode', ['unconfigured', 'conflicting', 'wrong_extension'])
async def test_invalid_video_choices_refuse_before_ocr(video_host, mode):
    engine, video, recognizer, original, env = video_host
    app = build_app(Settings.from_env(env), engine, document_video=None if mode == 'unconfigured' else video)
    body = {'attachment_id': original.attachment_id, 'filename': 'slides.mp4', 'video_ocr': True}
    if mode == 'conflicting':
        body['pdf_ocr'] = {'mode': 'all_pages', 'reading_order': 'provider'}
    elif mode == 'wrong_extension':
        body['filename'] = 'slides.ts'
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
        base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
        response = await client.post('/v1/documents', json=body)
        assert response.status_code == 422, response.text
        response = await client.post('/v1/document-jobs', json={**body, 'import_id': 'invalid'})
        assert response.status_code == 422, response.text
        assert recognizer.calls == 0


async def test_visual_only_video_durable_result_and_frame_reads_survive_restart(video_host, monkeypatch):
    engine, video, recognizer, original, env = video_host
    async def empty(image, **options):
        recognizer.calls += 1
        return OcrResult(engine='fixture', width=64, height=32, regions=())
    monkeypatch.setattr(recognizer, 'recognize', empty)
    settings = Settings.from_env(env)
    for first in (True, False):
        app = build_app(settings, engine, document_video=video)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
            base_url='http://fixture', headers={'authorization': 'Bearer writer'}) as client:
            if first:
                response = await client.post('/v1/document-jobs', json={'import_id': 'empty-video',
                    'attachment_id': original.attachment_id, 'filename': 'slides.mp4', 'video_ocr': True})
                assert response.status_code == 202, response.text
                status = await app.state.document_import_service.wait('alpha', 'empty-video')
                assert status.status == 'completed', status
            result = await client.get('/v1/document-jobs/empty-video/result')
            assert result.status_code == 200, result.text
            body = result.json()
            assert body['segments'] == 0 and body['added']['chunks'] == 0
            episode_id = body['added']['episode_id']
            assert (await client.get(f'/v1/episodes/{episode_id}')).json()['content'] == ''
            path = f'/v1/episodes/{episode_id}/document'
            catalogue = await client.get(path + '/video/catalogue')
            assert catalogue.status_code == 200, catalogue.text
            evidence = catalogue.json()['evidence']
            assert evidence['segments'] == [] and all(f['empty'] for f in evidence['video']['frames'])
            png = await client.get(path + '/video/frames/0')
            assert png.status_code == 200 and png.content.startswith(b'\x89PNG')
            assert recognizer.calls == 2
