"""Structural adapter tests using generated pixels/silence, not quality benchmarks."""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
import shutil
import subprocess
import wave

from PIL import Image
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.media import ImageDocumentParser, MediaDocumentParser, TranscriptionSegment
from scone_memory.ingestion.formats.types import DocumentLimits
from scone_memory.ocr.types import OcrRegion, OcrResult


class StructuralOcr:
    def __init__(self) -> None:
        self.calls = 0

    async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                        max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult:
        self.calls += 1
        with Image.open(io.BytesIO(image)) as frame:
            assert frame.format == 'PNG'
            assert frame.width * frame.height <= max_pixels
            return OcrResult(engine='structural-fixture-only', width=frame.width, height=frame.height,
                regions=(OcrRegion(text='fixture text', box=(0.1, 0.2, 0.8, 0.9), score=0.7),))


def image_bytes(format: str = 'PNG', *, animated: bool = False) -> bytes:
    output = io.BytesIO()
    first = Image.new('RGB', (20, 12), 'white')
    if animated:
        first.save(output, format=format, save_all=True,
            append_images=[Image.new('RGB', (20, 12), 'black')], duration=100, loop=0)
    else:
        first.save(output, format=format)
    return output.getvalue()


def silent_wav(seconds: float = 0.1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b'\x00' * int(seconds * 8_000) * 4)
    return output.getvalue()


@pytest.fixture
def ffmpeg() -> str:
    executable = shutil.which('ffmpeg')
    if executable is None:
        pytest.skip('explicit local ffmpeg is unavailable')
    return executable


async def structural_transcriber(wav_bytes: bytes) -> tuple[TranscriptionSegment, ...]:
    """Fabricated transcript verifies the contract only; silence has no words."""
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
        assert wav.getnframes() > 0
    return (TranscriptionSegment(text='structural fixture only', start_seconds=0.0, end_seconds=0.05),)


@pytest.mark.parametrize(('format', 'suffix'), [('PNG', 'png'), ('JPEG', 'jpg'), ('JPEG', 'jpeg'),
    ('GIF', 'gif'), ('WEBP', 'webp'), ('TIFF', 'tiff'), ('TIFF', 'tif'), ('BMP', 'bmp')])
async def test_image_normalization_and_region_provenance(format: str, suffix: str) -> None:
    result = await ImageDocumentParser(StructuralOcr()).parse(image_bytes(format), f'image.{suffix}', DocumentLimits())
    assert result.segments[0].text == 'fixture text'
    assert result.segments[0].locator == 'frame:1/region:1'
    assert result.segments[0].metadata['box'] == '[0.1, 0.2, 0.8, 0.9]'
    assert result.metadata['frame_policy'] == 'first'


async def test_animated_first_and_all_frames_are_explicit_and_bounded() -> None:
    data = image_bytes('GIF', animated=True)
    engine = StructuralOcr()
    first = await ImageDocumentParser(engine).parse(data, 'image.gif')
    assert len(first.segments) == engine.calls == 1
    all_frames = await ImageDocumentParser(engine, all_frames=True).parse(data, 'image.gif')
    assert all_frames.segments[1].locator == 'frame:2/region:1'
    with pytest.raises(InvalidInput):
        await ImageDocumentParser(engine, all_frames=True, max_frames=1).parse(data, 'image.gif')


async def test_image_input_pixel_and_text_limits() -> None:
    engine = StructuralOcr()
    parser = ImageDocumentParser(engine)
    with pytest.raises(InvalidInput):
        await parser.parse(b'not an image', 'image.png')
    with pytest.raises(InvalidInput):
        await parser.parse(image_bytes(), 'image.png', DocumentLimits(max_input_bytes=1))
    with pytest.raises(InvalidInput):
        await ImageDocumentParser(engine, max_pixels=100).parse(image_bytes(), 'image.png')
    assert engine.calls == 0
    with pytest.raises(InvalidInput):
        await parser.parse(image_bytes(), 'image.png', DocumentLimits(max_text_bytes=1))


async def test_image_provider_failure_propagates() -> None:
    class FailingOcr(StructuralOcr):
        async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                            max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult:
            raise RuntimeError('provider failed')

    with pytest.raises(RuntimeError, match='provider failed'):
        await ImageDocumentParser(FailingOcr()).parse(image_bytes(), 'image.png')


async def test_wav_conversion_and_timestamps(ffmpeg: str) -> None:
    parsed = await MediaDocumentParser(structural_transcriber, ffmpeg_executable=ffmpeg).parse(silent_wav(), 'audio.wav')
    assert parsed.metadata['extraction'] == 'audio-only'
    assert parsed.segments[0].metadata['start_seconds'] == '0.0'
    assert parsed.segments[0].metadata['end_seconds'] == '0.05'
    assert parsed.segments[0].locator.startswith('audio:0/segment:1/seconds:')


async def test_video_only_transcribes_audio(ffmpeg: str, tmp_path: Path) -> None:
    video = tmp_path / 'fixture.mp4'
    subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
        '-i', 'color=c=black:s=16x16:r=10', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
        '-t', '0.2', '-c:v', 'mpeg4', '-c:a', 'aac', str(video)], check=True, timeout=10)
    parsed = await MediaDocumentParser(structural_transcriber, ffmpeg_executable=ffmpeg).parse(video.read_bytes(), 'video.mp4')
    assert parsed.format == 'mp4'
    assert parsed.metadata['extraction'] == 'audio-only'
    assert all(segment.metadata['extraction'] == 'transcription' for segment in parsed.segments)


async def test_media_rejects_duration_input_and_playlists(ffmpeg: str) -> None:
    parser = MediaDocumentParser(structural_transcriber, ffmpeg_executable=ffmpeg, max_duration_seconds=0.05)
    with pytest.raises(InvalidInput):
        await parser.parse(silent_wav(0.1), 'audio.wav')
    with pytest.raises(InvalidInput):
        await parser.parse(silent_wav(), 'audio.wav', DocumentLimits(max_input_bytes=1))
    with pytest.raises(InvalidInput):
        await parser.parse(b'#EXTM3U\nhttp://127.0.0.1:9999/private.wav\n', 'audio.wav')
    with pytest.raises(InvalidInput):
        await parser.parse(b'http://127.0.0.1:9999/private.wav', 'audio.wav')


async def test_media_provider_failure_and_invalid_times(ffmpeg: str) -> None:
    async def failure(data: bytes) -> tuple[TranscriptionSegment, ...]:
        raise RuntimeError('transcriber failed')

    with pytest.raises(RuntimeError, match='transcriber failed'):
        await MediaDocumentParser(failure, ffmpeg_executable=ffmpeg).parse(silent_wav(), 'audio.wav')

    async def outside(data: bytes) -> tuple[TranscriptionSegment, ...]:
        return (TranscriptionSegment(text='outside', start_seconds=0.0, end_seconds=10.0),)

    with pytest.raises(InvalidInput, match='timestamps'):
        await MediaDocumentParser(outside, ffmpeg_executable=ffmpeg).parse(silent_wav(), 'audio.wav')
    with pytest.raises(InvalidInput, match='text byte'):
        await MediaDocumentParser(structural_transcriber, ffmpeg_executable=ffmpeg).parse(
            silent_wav(), 'audio.wav', DocumentLimits(max_text_bytes=1))


async def test_media_provider_cancellation_and_timeout(ffmpeg: str) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def waiting(data: bytes) -> tuple[TranscriptionSegment, ...]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return ()

    parser = MediaDocumentParser(waiting, ffmpeg_executable=ffmpeg)
    task = asyncio.create_task(parser.parse(silent_wav(), 'audio.wav'))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    started.clear()
    stopped.clear()
    with pytest.raises((TimeoutError, InvalidInput)):
        await parser.parse(silent_wav(), 'audio.wav', DocumentLimits(timeout_seconds=0.5))
    assert stopped.is_set()


async def test_oriented_image_dimensions_and_segment_limit() -> None:
    buffer = io.BytesIO()
    image = Image.new('RGB', (20, 12), 'white')
    exif = image.getexif()
    exif[274] = 6
    image.save(buffer, format='JPEG', exif=exif)
    parsed = await ImageDocumentParser(StructuralOcr()).parse(buffer.getvalue(), 'rotated.jpg')
    assert parsed.segments[0].metadata['width'] == '12'
    assert parsed.segments[0].metadata['height'] == '20'
    with pytest.raises(InvalidInput, match='segment'):
        await ImageDocumentParser(StructuralOcr(), all_frames=True).parse(
            image_bytes('GIF', animated=True), 'image.gif', DocumentLimits(max_segments=1))


async def test_image_provider_cancellation() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class WaitingOcr(StructuralOcr):
        async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                            max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError('unreachable')

    task = asyncio.create_task(ImageDocumentParser(WaitingOcr()).parse(image_bytes(), 'image.png'))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


async def test_media_transcriber_object_and_empty_results(ffmpeg: str) -> None:
    class Provider:
        async def transcribe(self, audio_wav: bytes) -> tuple[TranscriptionSegment, ...]:
            return await structural_transcriber(audio_wav)

    parsed = await MediaDocumentParser(Provider(), ffmpeg_executable=ffmpeg).parse(silent_wav(), 'audio.wav')
    assert parsed.segments

    async def empty(data: bytes) -> tuple[TranscriptionSegment, ...]:
        return ()

    with pytest.raises(InvalidInput, match='nonempty'):
        await MediaDocumentParser(empty, ffmpeg_executable=ffmpeg).parse(silent_wav(), 'audio.wav')
