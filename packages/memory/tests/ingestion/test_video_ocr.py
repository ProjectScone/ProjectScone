"""Video OCR retains sampled frame evidence, including frames without text."""
import hashlib
import shutil
import subprocess

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.types import DocumentLimits
from scone_memory.ingestion.video_frames import VideoFrameDecoder, VideoFramePolicy
from scone_memory.ingestion.video_ocr import VideoDocumentParser
from scone_memory.ocr.types import OcrRegion, OcrResult


@pytest.fixture
async def video(tmp_path):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('local ffmpeg and ffprobe required')
    pytest.importorskip('PIL')
    path = tmp_path / 'slides.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x32:rate=1:duration=3',
                    '-an', '-c:v', 'libx264', str(path)], check=True)
    return path.read_bytes(), VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe)


class ScriptedOcr:
    def __init__(self, *, empty=False, fail_at=None):
        self.calls = []
        self.empty, self.fail_at = empty, fail_at

    async def recognize(self, image, **options):
        index = len(self.calls)
        self.calls.append(hashlib.sha256(image).hexdigest())
        if index == self.fail_at:
            raise RuntimeError('OCR interrupted')
        regions = () if self.empty or index == 1 else (
            OcrRegion(text='Café ☕', box=(0.1, 0.1, 0.9, 0.8), score=0.8),)
        return OcrResult(engine='scripted-ocr', width=64, height=32, regions=regions)


class Receipts:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value


async def test_video_ocr_keeps_empty_frame_and_exact_utf8_geometry(video):
    data, decoder = video
    ocr = ScriptedOcr()
    parser = VideoDocumentParser(decoder, ocr, model_revision='fixture-v1',
                                 policy=VideoFramePolicy(interval_seconds=1))
    parsed = await parser.parse(data, 'slides.mp4', DocumentLimits())
    assert len(parsed.segments) == 2
    assert parsed.video.source_sha256 == hashlib.sha256(data).hexdigest()
    assert [frame.empty for frame in parsed.video.frames] == [False, True, False]
    assert [frame.presentation_timestamp for frame in parsed.video.frames] == [0, 16384, 32768]
    assert [frame.png_sha256 for frame in parsed.video.frames] == ocr.calls
    for segment in parsed.segments:
        region = segment.regions[0]
        assert segment.text.encode()[region.start:region.end] == 'Café ☕'.encode()
        assert region.coordinate_space == 'normalized_displayed_frame_top_left'
    assert parsed.video.model_revision == 'fixture-v1'


async def test_empty_video_ocr_refuses_without_fabricated_text(video):
    data, decoder = video
    with pytest.raises(InvalidInput, match='no recognized text'):
        await VideoDocumentParser(decoder, ScriptedOcr(empty=True), model_revision='fixture-v1').parse(
            data, 'slides.mp4', DocumentLimits())


async def test_recovery_reuses_completed_frames_including_empty(video):
    data, decoder = video
    receipts, ocr = Receipts(), ScriptedOcr(fail_at=2)
    parser = VideoDocumentParser(decoder, ocr, model_revision='fixture-v1',
                                 policy=VideoFramePolicy(interval_seconds=1))
    with pytest.raises(RuntimeError, match='interrupted'):
        await parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts)
    ocr.fail_at = None
    parsed = await parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts)
    assert len(ocr.calls) == 4
    assert [frame.empty for frame in parsed.video.frames] == [False, True, False]
    again = await parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts)
    assert again == parsed and len(ocr.calls) == 4


async def test_video_manifest_retains_full_frame_inventory_and_original_binding(video):
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.ingestion.files import prepare_document, encode_manifest, store_document, document_provenance
    data, decoder = video
    parser = VideoDocumentParser(decoder, ScriptedOcr(), model_revision='fixture-v1',
                                 policy=VideoFramePolicy(interval_seconds=1))
    manifest = await prepare_document(data, 'slides.mp4', parser=parser, limits=DocumentLimits())
    assert manifest.schema_version == 6
    assert b'"video"' in encode_manifest(manifest)
    with pytest.raises(ValueError, match='version six'):
        type(manifest).model_validate({**manifest.model_dump(), 'schema_version': 5})
    with pytest.raises(ValueError, match='original'):
        type(manifest).model_validate({**manifest.model_dump(), 'original_sha256': 'a' * 64})
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        original = await engine.attach('alpha', data, media_type='video/mp4')
        stored = await store_document(engine, 'alpha', original, manifest)
        evidence = await document_provenance(engine, 'alpha', stored.added.episode_id)
        assert evidence.video == manifest.parsed.video
        assert [frame.empty for frame in evidence.video.frames] == [False, True, False]
        assert (await engine.episode('alpha', stored.added.episode_id)).metadata['evidence_origin'] == 'extracted_text'
    finally:
        await engine.close()


def test_nonvideo_extraction_omits_new_field_for_legacy_identity():
    from scone_memory.ingestion.files import DocumentManifest, encode_manifest
    from scone_memory.ingestion.formats.types import DocumentSegment, ParsedDocument
    parsed = ParsedDocument(format='txt', parser='plain', segments=(DocumentSegment(text='Text', locator='line:1'),))
    manifest = DocumentManifest(original_sha256='a' * 64, filename='source.txt', parsed=parsed)
    assert encode_manifest(manifest) == (
        b'{"schema_version":1,"offset_unit":"extracted_text_utf8_bytes","original_sha256":"' + b'a' * 64 +
        b'","filename":"source.txt","parsed":{"format":"txt","parser":"plain","segments":[{"text":"Text",'
        b'"locator":"line:1","metadata":{}}],"metadata":{}}}')


@pytest.mark.parametrize('damage', ['missing', 'png', 'text', 'duplicate', 'header', 'revision'])
async def test_completed_receipt_changes_refuse_before_any_more_ocr(video, damage):
    import json
    data, decoder = video
    receipts, ocr = Receipts(), ScriptedOcr()
    parser = VideoDocumentParser(decoder, ocr, model_revision='fixture-v1',
                                 policy=VideoFramePolicy(interval_seconds=1))
    await parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts)
    key = 'video-ocr-frame-001'
    if damage == 'missing':
        del receipts.values[key]
    elif damage in ('png', 'text'):
        value = json.loads(receipts.values['video-ocr-frame-000'])
        if damage == 'png':
            value['png_sha256'] = 'a' * 64
        else:
            value['result']['regions'][0]['text'] = 'Altered observation'
        receipts.values['video-ocr-frame-000'] = json.dumps(value).encode()
    elif damage == 'duplicate':
        receipts.values[key] = receipts.values[key].replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')
    elif damage == 'header':
        receipts.values['video-ocr-header'] += b' '
    else:
        parser = VideoDocumentParser(decoder, ocr, model_revision='fixture-v2',
                                     policy=VideoFramePolicy(interval_seconds=1))
    with pytest.raises(InvalidInput):
        await parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts)
    assert len(ocr.calls) == 3


async def test_copied_ocr_result_is_revalidated_before_checkpoint(video):
    data, decoder = video
    receipts = Receipts()

    class WrongDimensions(ScriptedOcr):
        async def recognize(self, image, **options):
            result = await super().recognize(image, **options)
            return result.model_copy(update={'width': 0})

    with pytest.raises(InvalidInput, match='observation'):
        await VideoDocumentParser(decoder, WrongDimensions(), model_revision='fixture').parse_checkpointed(
            data, 'slides.mp4', DocumentLimits(), receipts)
    assert 'video-ocr-frame-000' not in receipts.values


async def test_orphan_receipt_is_not_adopted_under_new_configuration(video):
    data, decoder = video
    receipts, ocr = Receipts(), ScriptedOcr()
    receipts.values['video-ocr-frame-255'] = b'orphan'
    with pytest.raises(InvalidInput, match='header'):
        await VideoDocumentParser(decoder, ocr, model_revision='fixture').parse_checkpointed(
            data, 'slides.mp4', DocumentLimits(), receipts)
    assert not ocr.calls


async def test_sigkill_resumes_only_unfinished_frame_in_encrypted_workflow(video, tmp_path):
    import asyncio
    import json
    import os
    from pathlib import Path
    import sys
    import scone_memory
    data, decoder = video
    (tmp_path / 'input.mp4').write_bytes(data)
    worker = Path(__file__).parent / 'fixtures' / 'video_ocr_worker.py'
    env = {key: value for key, value in os.environ.items() if key in {'PATH', 'TMPDIR', 'LANG'}}
    env['PYTHONPATH'] = str(Path(scone_memory.__file__).resolve().parent.parent)
    child = None

    async def launch(phase):
        return await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path),
            decoder.ffmpeg, decoder.ffprobe, phase, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)

    try:
        child = await launch('hold')

        async def entered():
            while not (tmp_path / 'ocr-entered').exists():
                if child.returncode is not None:
                    raise AssertionError((await child.stderr.read()).decode())
                await asyncio.sleep(0.02)

        await asyncio.wait_for(entered(), 20)
        before = (tmp_path / 'ocr-calls').read_text().splitlines()
        assert len(before) == 3
        child.kill()
        await asyncio.wait_for(child.wait(), 10)
        child = await launch('resume')
        await asyncio.wait_for(child.wait(), 25)
        assert child.returncode == 0, (await child.stderr.read()).decode()
        calls = (tmp_path / 'ocr-calls').read_text().splitlines()
        assert len(calls) == 4
        assert calls[-1] == before[-1].replace('hold:', 'resume:')
        result = json.loads((tmp_path / 'result.json').read_text())
        assert result['empty'] == [False, True, False]
        assert len(result['text']) == 2
        assert result['frame_hashes'] == [line.split(':')[1] for line in before]
        assert b'Retained Caf' not in (tmp_path / 'journal.db').read_bytes()
        child = await launch('read')
        await asyncio.wait_for(child.wait(), 20)
        assert child.returncode == 0, (await child.stderr.read()).decode()
        assert (tmp_path / 'ocr-calls').read_text().splitlines() == calls
        assert json.loads((tmp_path / 'result.json').read_text()) == result
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()


async def test_video_evidence_rejects_inconsistent_frame_to_text_mapping(video):
    from scone_memory.ingestion.formats.types import validate_document
    data, decoder = video
    parsed = await VideoDocumentParser(decoder, ScriptedOcr(), model_revision='fixture',
        policy=VideoFramePolicy(interval_seconds=1)).parse(data, 'slides.mp4', DocumentLimits())
    first = parsed.video.frames[0]
    for changed in (
        parsed.video.model_copy(update={'frames': (first.model_copy(update={'empty': True}), *parsed.video.frames[1:])}),
        parsed.video.model_copy(update={'source_sha256': 'invalid'}),
        parsed.video.model_copy(update={'unavailable_requests': 50}),
        parsed.video.model_copy(update={'frames': (first.model_copy(update={'requested_seconds': (5,)}), *parsed.video.frames[1:])}),
    ):
        with pytest.raises(InvalidInput):
            validate_document(parsed.model_copy(update={'video': changed}), DocumentLimits())


async def test_ocr_timeout_does_not_checkpoint_unfinished_observation(video, monkeypatch):
    import asyncio
    data, decoder = video
    sampled = await decoder.sample(data, 'slides.mp4')

    async def sample(*args, **kwargs):
        return sampled

    class SlowOcr:
        async def recognize(self, image, **options):
            await asyncio.sleep(10)
            pytest.fail('recognizer must be cancelled')

    monkeypatch.setattr(decoder, 'sample', sample)
    receipts = Receipts()
    with pytest.raises(InvalidInput, match='wall time'):
        await VideoDocumentParser(decoder, SlowOcr(), model_revision='fixture').parse_checkpointed(
            data, 'slides.mp4', DocumentLimits(timeout_seconds=0.05), receipts)
    assert 'video-ocr-frame-000' not in receipts.values


async def test_ocr_cancellation_keeps_only_completed_observations(video, monkeypatch):
    import asyncio
    data, decoder = video
    sampled = await decoder.sample(data, 'slides.mp4', policy=VideoFramePolicy(interval_seconds=1))

    async def sample(*args, **kwargs):
        return sampled

    entered = asyncio.Event()

    class HoldingOcr(ScriptedOcr):
        async def recognize(self, image, **options):
            if self.calls:
                entered.set()
                await asyncio.Event().wait()
            return await super().recognize(image, **options)

    monkeypatch.setattr(decoder, 'sample', sample)
    receipts = Receipts()
    parser = VideoDocumentParser(decoder, HoldingOcr(), model_revision='fixture',
                                 policy=VideoFramePolicy(interval_seconds=1))
    task = asyncio.create_task(parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 'video-ocr-frame-000' in receipts.values
    assert 'video-ocr-frame-001' not in receipts.values
    assert 'video-ocr-complete' not in receipts.values


async def test_installed_tesseract_reads_visible_video_text(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    from scone_memory.ocr.tesseract import TesseractOcr
    ffmpeg, ffprobe, tesseract = (shutil.which(name) for name in ('ffmpeg', 'ffprobe', 'tesseract'))
    if not all((ffmpeg, ffprobe, tesseract)):
        pytest.skip('installed local ffmpeg, ffprobe and Tesseract required')
    slide = tmp_path / 'slide.png'
    image = Image.new('RGB', (640, 120), 'white')
    ImageDraw.Draw(image).text((25, 30), 'CHECKPOINT FRAME', fill='black', font=ImageFont.load_default(size=44))
    image.save(slide)
    path = tmp_path / 'visible.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-loop', '1', '-i', str(slide), '-t', '1', '-r', '1',
                    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    parser = VideoDocumentParser(VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe),
        TesseractOcr(executable=tesseract, language='eng', page_segmentation=6), model_revision='installed-tesseract-eng-psm6')
    parsed = await parser.parse(path.read_bytes(), 'visible.mp4', DocumentLimits())
    words = ' '.join(segment.text for segment in parsed.segments).split()
    assert 'CHECKPOINT' in words and 'FRAME' in words
    assert len(parsed.video.frames) == 1 and not parsed.video.frames[0].empty
    assert all(region.box[2] > region.box[0] for segment in parsed.segments for region in segment.regions)


async def test_total_region_budget_is_shared_across_frames(video):
    data, decoder = video
    receipts, budgets = Receipts(), []
    region = OcrRegion(text='x', box=(0.1, 0.1, 0.9, 0.8))

    class DenseOcr:
        async def recognize(self, image, **options):
            budgets.append(options['max_regions'])
            return OcrResult(engine='dense', width=64, height=32, regions=(region,) * 7500)

    parser = VideoDocumentParser(decoder, DenseOcr(), model_revision='fixture',
                                 policy=VideoFramePolicy(interval_seconds=1))
    with pytest.raises(InvalidInput, match='total region'):
        await parser.parse_checkpointed(data, 'slides.mp4', DocumentLimits(), receipts)
    assert budgets == [10_000, 10_000, 5000]
    assert 'video-ocr-frame-002' not in receipts.values
    assert 'video-ocr-complete' not in receipts.values
