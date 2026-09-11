"""Opt-in OCR and timestamped transcription; no models or providers auto-load.

Image recognition extracts visible text, not visual semantics. Video parsing
transcribes the first audio stream only; it does not inspect video frames.
"""
from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from time import monotonic
from typing import Protocol
import wave

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...core.errors import InvalidInput
from ...ocr.process import python_worker, run_bounded
from ...ocr.types import OcrEngine
from .registry import extension
from .types import DocumentLimits, DocumentSegment, DocumentTextRegion, ParsedDocument, validate_document

IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp', '.tif', '.tiff', '.bmp'})
AUDIO_EXTENSIONS = frozenset({'.wav', '.mp3', '.flac', '.ogg', '.oga', '.opus', '.aac', '.m4a'})
VIDEO_EXTENSIONS = frozenset({'.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi', '.mpeg', '.mpg', '.ts'})
_DEMUXERS = 'wav,mp3,flac,ogg,aac,mov,matroska,webm,avi,mpeg,mpegts'
_SAMPLE_RATE = 16_000


class TranscriptionSegment(BaseModel):
    """Provider-observed source times in seconds, relative to audio start."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    text: str = Field(min_length=1, max_length=2_000_000)
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered(self) -> TranscriptionSegment:
        if self.end_seconds <= self.start_seconds or not self.text.strip():
            raise ValueError('transcription requires text and an increasing time interval')
        return self


class MediaTranscriber(Protocol):
    """Caller-managed provider accepting mono 16 kHz signed 16-bit PCM WAV.

    Return timestamped segments relative to the WAV start. Provider exceptions
    and cancellation propagate. The parser bounds execution time and validates
    results; the caller configures any local model or self-managed endpoint.
    """
    async def transcribe(self, audio_wav: bytes) -> tuple[TranscriptionSegment, ...]: ...


TranscriptionCallback = Callable[[bytes], Awaitable[tuple[TranscriptionSegment, ...]]]


def _input(data: bytes, filename: str, limits: DocumentLimits, supported: frozenset[str]) -> str:
    suffix = extension(filename)
    if suffix not in supported:
        raise InvalidInput('unsupported media document extension')
    if not isinstance(data, bytes) or not data or len(data) > limits.max_input_bytes:
        raise InvalidInput('media document is empty or exceeds its input byte limit')
    return suffix.lstrip('.')


def _remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise InvalidInput('media document exceeded its wall time limit')
    return remaining


class ImageDocumentParser:
    """OCR the first frame by default; all-frame mode rejects excess frames.

    Pixel and encoded PNG byte budgets apply across all selected frames.
    Regions use normalized coordinates of EXIF-oriented displayed frames.
    """
    def __init__(self, engine: OcrEngine, *, all_frames: bool = False,
                 max_frames: int = 32, max_pixels: int = 20_000_000) -> None:
        if (type(max_frames) is not int or not 1 <= max_frames <= 256
                or type(max_pixels) is not int or not 1 <= max_pixels <= 20_000_000
                or type(all_frames) is not bool):
            raise InvalidInput('invalid image frame or pixel limits')
        self._engine = engine
        self._all_frames = all_frames
        self._max_frames = max_frames
        self._max_pixels = max_pixels

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits = DocumentLimits()) -> ParsedDocument:
        suffix = _input(data, filename, limits, IMAGE_EXTENSIONS)
        deadline = monotonic() + limits.timeout_seconds
        raw = await run_bounded(python_worker(__name__, 'image',
            str(limits.max_input_bytes), str(self._max_frames if self._all_frames else 1),
            str(self._max_pixels), str(int(self._all_frames))), data,
            timeout=_remaining(deadline), max_output=34_000_000)
        payload = json.loads(raw)
        if isinstance(payload, dict) and 'error' in payload:
            raise InvalidInput(str(payload['error']))
        frames = _Frames.model_validate_json(raw)
        segments: list[DocumentSegment] = []
        size = 0
        for number, frame in enumerate(frames.frames, 1):
            png = base64.b64decode(frame.png, validate=True)
            timeout = _remaining(deadline)
            result = await asyncio.wait_for(self._engine.recognize(png,
                max_pixels=self._max_pixels, max_regions=min(limits.max_segments, 10_000),
                timeout_seconds=timeout), timeout)
            if (result.width, result.height) != (frame.width, frame.height):
                raise InvalidInput('OCR result dimensions do not match the decoded frame')
            for region_number, region in enumerate(result.regions, 1):
                if not region.text.strip():
                    continue
                size += len(region.text.encode()) + (2 if segments else 0)
                if len(segments) >= limits.max_segments or size > limits.max_text_bytes:
                    raise InvalidInput('image OCR exceeds its segment or text byte limit')
                metadata = {'frame': str(number), 'region': str(region_number),
                    'box': json.dumps(region.box), 'coordinate_space': 'normalized-displayed-frame',
                    'width': str(frame.width), 'height': str(frame.height), 'engine': result.engine,
                    'block': str(region.block), 'line': str(region.line), 'extraction': 'ocr'}
                if region.score is not None:
                    metadata['recognizer_score'] = str(region.score)
                segments.append(DocumentSegment(text=region.text,
                    locator=f'frame:{number}/region:{region_number}', metadata=metadata,
                    regions=(DocumentTextRegion(text=region.text, box=region.box, score=region.score,
                        block=region.block, line=region.line, start=0, end=len(region.text.encode()),
                        coordinate_space='normalized_displayed_frame_top_left'),)))
        if not segments:
            raise InvalidInput('image contains no recognized text')
        parsed = ParsedDocument(format=suffix, parser='image-ocr', segments=tuple(segments),
            metadata={'extraction': 'ocr', 'frame_policy': 'all' if self._all_frames else 'first',
                      'frames_processed': str(len(frames.frames))})
        validate_document(parsed, limits)
        return parsed


class MediaDocumentParser:
    """Bounded audio decoding with explicit ffmpeg and transcription provider.

    Input must be native media bytes, never a URL or playlist. Only native
    demuxers and local file/pipe protocols are permitted. Audio beyond the
    duration limit is rejected rather than silently truncated.
    """
    def __init__(self, transcriber: MediaTranscriber | TranscriptionCallback, *,
                 ffmpeg_executable: str, max_duration_seconds: float = 60.0) -> None:
        if (not Path(ffmpeg_executable).is_absolute() or not Path(ffmpeg_executable).is_file()
                or not math.isfinite(max_duration_seconds) or not 0 < max_duration_seconds <= 600):
            raise InvalidInput('configure an existing absolute ffmpeg path and duration in (0, 600]')
        self._transcribe = transcriber if callable(transcriber) else transcriber.transcribe
        self._ffmpeg = ffmpeg_executable
        self._max_duration = max_duration_seconds

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits = DocumentLimits()) -> ParsedDocument:
        suffix = _input(data, filename, limits, AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
        deadline = monotonic() + limits.timeout_seconds
        max_samples = int(self._max_duration * _SAMPLE_RATE)
        with tempfile.TemporaryDirectory(prefix='scone-media-') as directory:
            source = Path(directory) / 'input.media'
            source.write_bytes(data)
            pcm = await run_bounded([self._ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-threads', '1', '-protocol_whitelist', 'file,pipe', '-format_whitelist', _DEMUXERS,
                '-i', str(source), '-map', '0:a:0', '-vn', '-sn', '-dn',
                '-t', str((max_samples + 1) / _SAMPLE_RATE), '-ac', '1', '-ar', str(_SAMPLE_RATE),
                '-threads', '1', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'], b'',
                timeout=_remaining(deadline), max_output=(max_samples + 1) * 2)
        if len(pcm) > max_samples * 2:
            raise InvalidInput('media exceeds its decoded audio duration limit')
        if not pcm or len(pcm) % 2:
            raise InvalidInput('media contains no decodable audio')
        duration = len(pcm) / (2 * _SAMPLE_RATE)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_SAMPLE_RATE)
            wav.writeframes(pcm)
        transcript = await asyncio.wait_for(self._transcribe(buffer.getvalue()), _remaining(deadline))
        if not isinstance(transcript, tuple) or not transcript or len(transcript) > limits.max_segments:
            raise InvalidInput('transcription requires a bounded nonempty tuple of segments')
        segments: list[DocumentSegment] = []
        size = 0
        previous_start = 0.0
        for number, segment in enumerate(transcript, 1):
            if not isinstance(segment, TranscriptionSegment):
                raise InvalidInput('transcription provider returned an invalid segment')
            if segment.start_seconds < previous_start or segment.end_seconds > duration + 1 / _SAMPLE_RATE:
                raise InvalidInput('transcription timestamps are unordered or outside decoded audio')
            previous_start = segment.start_seconds
            size += len(segment.text.encode()) + (2 if segments else 0)
            if size > limits.max_text_bytes:
                raise InvalidInput('transcription exceeds its extracted text byte limit')
            segments.append(DocumentSegment(text=segment.text,
                locator=f'audio:0/segment:{number}/seconds:{segment.start_seconds}-{segment.end_seconds}',
                metadata={'start_seconds': str(segment.start_seconds), 'end_seconds': str(segment.end_seconds),
                          'audio_stream': '0', 'extraction': 'transcription'}))
        parsed = ParsedDocument(format=suffix, parser='media-transcription', segments=tuple(segments),
            metadata={'extraction': 'audio-only', 'duration_seconds': str(duration), 'sample_rate': str(_SAMPLE_RATE)})
        validate_document(parsed, limits)
        return parsed


class _Frame(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    png: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class _Frames(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    frames: tuple[_Frame, ...] = Field(min_length=1, max_length=256)


def _image_worker() -> None:
    # Optional Pillow import stays in a bounded child, including native decode.
    from PIL import Image, ImageOps

    input_limit, frame_limit, pixel_limit, all_frames = map(int, sys.argv[2:])
    data = sys.stdin.buffer.read(input_limit + 1)
    if not data or len(data) > input_limit:
        raise InvalidInput('image exceeds its input byte limit')
    allowed = ['PNG', 'JPEG', 'GIF', 'WEBP', 'TIFF', 'BMP']
    with Image.open(io.BytesIO(data), formats=allowed) as check:
        check.verify()
    frames: list[_Frame] = []
    pixels = encoded_bytes = 0
    with Image.open(io.BytesIO(data), formats=allowed) as image:
        for number in range(frame_limit + 1 if all_frames else 1):
            try:
                image.seek(number)
            except EOFError:
                break
            if number >= frame_limit:
                raise InvalidInput('image exceeds its frame limit')
            pixels += image.width * image.height
            if pixels > pixel_limit:
                raise InvalidInput('image exceeds its total decoded pixel limit')
            displayed = ImageOps.exif_transpose(image).convert('RGB')
            buffer = io.BytesIO()
            displayed.save(buffer, format='PNG')
            png = buffer.getvalue()
            encoded_bytes += len(png)
            if encoded_bytes > 25_000_000:
                raise InvalidInput('image exceeds its normalized PNG byte limit')
            frames.append(_Frame(png=base64.b64encode(png).decode('ascii'),
                                 width=displayed.width, height=displayed.height))
    sys.stdout.write(_Frames(frames=tuple(frames)).model_dump_json())


if __name__ == '__main__':
    try:
        _image_worker()
    except ImportError:
        sys.stdout.write(json.dumps({'error': 'image OCR requires the optional Pillow dependency'}))
    except Exception as error:
        message = str(error) if isinstance(error, InvalidInput) else 'image decoding failed; invalid or unsupported image'
        sys.stdout.write(json.dumps({'error': message}))
