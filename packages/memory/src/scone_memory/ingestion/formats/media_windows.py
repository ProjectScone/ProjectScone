"""Deterministic audio windows and durable, source-bound completed observations."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import struct
import wave

from pydantic import BaseModel, ConfigDict, Field

from ...core.errors import InvalidInput
from ..extraction_checkpoint import ExtractionCheckpoints
from .media import TranscriptionCallback, TranscriptionSegment, _remaining, _transcription_document
from .media_checkpoint import TranscriptionCoverage, _unique
from .types import DocumentLimits

WINDOW_IMPLEMENTATION = 'quiet-audio-windows-v1'
WINDOW_BINDING_KEY = 'media-window-binding'
_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 320
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024


class WindowReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    binding: str = Field(pattern=r'^[a-f0-9]{64}$')
    start_sample: int = Field(ge=0, lt=9_600_000)
    end_sample: int = Field(gt=0, le=9_600_000)
    audio_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    segments: tuple[TranscriptionSegment, ...] = Field(max_length=20_000)


def window_ranges(pcm: bytes, seconds: int) -> tuple[tuple[int, int], ...]:
    """Cover every sample once, preferring a quiet cut near the window end.

    Quiet means at least eight 20 ms frames with RMS <= 1% of full scale.
    Search only the last 20% (at most two seconds). This is a boundary
    heuristic, not voice detection: every window still reaches the model.
    """
    total, maximum = len(pcm) // 2, seconds * _SAMPLE_RATE
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + maximum)
        if end < total:
            search = end - min(2 * _SAMPLE_RATE, maximum // 5)
            quiet_start: int | None = None
            candidate: int | None = None
            for frame in range(search, end - _FRAME_SAMPLES + 1, _FRAME_SAMPLES):
                samples = struct.unpack('<320h', pcm[frame * 2:(frame + _FRAME_SAMPLES) * 2])
                quiet = sum(sample * sample for sample in samples) <= _FRAME_SAMPLES * (327.68 ** 2)
                if quiet:
                    if quiet_start is None:
                        quiet_start = frame
                    if frame + _FRAME_SAMPLES - quiet_start >= 8 * _FRAME_SAMPLES:
                        candidate = (quiet_start + frame + _FRAME_SAMPLES) // 2
                else:
                    quiet_start = None
            # A quiet run continuing to the boundary needs no earlier cut.
            if candidate is not None and quiet_start is None:
                end = candidate
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _wav(pcm: bytes, start: int, end: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_SAMPLE_RATE)
        output.writeframes(pcm[start * 2:end * 2])
    return buffer.getvalue()


def _read(checkpoints: ExtractionCheckpoints, index: int, binding: str,
          start: int, end: int, digest: str) -> WindowReceipt | None:
    raw = checkpoints.get(f'media-window-{index:04d}')
    if raw is None:
        return None
    try:
        if type(raw) is not bytes or len(raw) > _MAX_RECEIPT_BYTES:
            raise ValueError('checkpoint size or type')
        json.loads(raw, object_pairs_hook=_unique)
        receipt = WindowReceipt.model_validate_json(raw)
        if (receipt.binding != binding or receipt.start_sample != start
                or receipt.end_sample != end or receipt.audio_sha256 != digest):
            raise ValueError('checkpoint binding')
        return receipt
    except (ValueError, RecursionError):
        raise InvalidInput('media window checkpoint is invalid or does not match this extraction') from None


def _validate(segments: tuple[TranscriptionSegment, ...], duration: float, digest: str,
              audio_bytes: int, limits: DocumentLimits) -> tuple[TranscriptionSegment, ...]:
    if isinstance(segments, tuple) and not segments:
        return ()
    return _transcription_document('wav', segments, duration, digest, audio_bytes, limits)[1]


async def transcribe_windows(audio: bytes, *, seconds: int, binding: str,
                             transcribe: TranscriptionCallback, limits: DocumentLimits,
                             deadline: float, checkpoints: ExtractionCheckpoints | None
                             ) -> tuple[tuple[TranscriptionSegment, ...], TranscriptionCoverage]:
    pcm = audio[44:]  # The decoder emits canonical mono 16 kHz signed 16-bit WAV.
    ranges = window_ranges(pcm, seconds)
    marker = json.dumps({'binding': binding, 'audio_sha256': hashlib.sha256(audio).hexdigest()},
                        sort_keys=True, separators=(',', ':')).encode()
    saved: list[WindowReceipt | None] = []
    missing = False
    if checkpoints is not None:
        previous = checkpoints.get(WINDOW_BINDING_KEY)
        if previous is not None and previous != marker:
            raise InvalidInput('media window checkpoints do not match this extraction or decoded audio')
        for index, (start, end) in enumerate(ranges):
            _remaining(deadline)
            digest = hashlib.sha256(_wav(pcm, start, end)).hexdigest()
            receipt = _read(checkpoints, index, binding, start, end, digest)
            if receipt is not None:
                if previous is None or missing:
                    raise InvalidInput('media window checkpoint has no binding or completed prefix')
                try:
                    _validate(receipt.segments, (end - start) / _SAMPLE_RATE,
                              digest, (end - start) * 2 + 44, limits)
                except InvalidInput:
                    raise InvalidInput('media window checkpoint contains invalid observations') from None
            else:
                missing = True
            saved.append(receipt)
        if previous is None:
            checkpoints.put(WINDOW_BINDING_KEY, marker)
    else:
        saved = [None] * len(ranges)
    result: list[TranscriptionSegment] = []
    size = empty_windows = 0
    for index, ((start, end), receipt) in enumerate(zip(ranges, saved, strict=True)):
        remaining = _remaining(deadline)
        wav = _wav(pcm, start, end)
        digest = hashlib.sha256(wav).hexdigest()
        observed = (receipt.segments if receipt is not None else
                    await asyncio.wait_for(transcribe(wav), remaining))
        validated = _validate(observed, (end - start) / _SAMPLE_RATE, digest, len(wav), limits)
        if not validated:
            empty_windows += 1
        for segment in validated:
            size += len(segment.text.encode()) + (2 if result else 0)
            if len(result) >= limits.max_segments or size > limits.max_text_bytes:
                raise InvalidInput('transcription exceeds its aggregate segment or text byte limit')
            offset = start / _SAMPLE_RATE
            result.append(TranscriptionSegment(text=segment.text,
                start_seconds=segment.start_seconds + offset, end_seconds=segment.end_seconds + offset))
        _remaining(deadline)
        if checkpoints is not None and receipt is None:
            raw = WindowReceipt(binding=binding, start_sample=start, end_sample=end,
                                audio_sha256=digest, segments=validated).model_dump_json().encode()
            if len(raw) > _MAX_RECEIPT_BYTES:
                raise InvalidInput('media window checkpoint exceeds its byte limit')
            checkpoints.put(f'media-window-{index:04d}', raw)
    _remaining(deadline)
    if not result:
        raise InvalidInput('transcriber must return a nonempty tuple of timestamped segments')
    return tuple(result), TranscriptionCoverage(windows=len(ranges), empty_windows=empty_windows)
