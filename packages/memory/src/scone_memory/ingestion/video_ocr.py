"""Explicit video OCR with source-bound reusable per-frame observations."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import re
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput
from ..ocr.types import OcrEngine, OcrResult
from .extraction_checkpoint import ExtractionCheckpoints
from .formats.media import _remaining
from .formats.registry import extension
from .formats.types import DocumentLimits, DocumentSegment, DocumentTextRegion, ParsedDocument, validate_document
from .video_evidence import DocumentVideoEvidence, VideoOcrFrame
from .video_frames import VideoFrame, VideoFrameDecoder, VideoFramePolicy, VideoFrames, _validated_limits, _validated_policy

_HEADER = 'video-ocr-header'
_COMPLETE = 'video-ocr-complete'
_MAX_RECEIPT = 16_000_000
_MAX_TOTAL_REGIONS = 20_000


class _Receipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1] = 1
    binding: str = Field(pattern=r'^[a-f0-9]{64}$')
    png_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    result: OcrResult


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate receipt key')
        result[key] = value
    return result


def _key(index: int) -> str:
    return f'video-ocr-frame-{index:03}'


def _observation(result: OcrResult, frame: VideoFrame, limits: DocumentLimits) -> OcrResult:
    try:
        checked = OcrResult.model_validate(result.model_dump())
        size = sum(len(region.text.encode()) for region in checked.regions)
    except (ValueError, AttributeError) as error:
        raise InvalidInput('video OCR observation is invalid') from error
    if ((checked.width, checked.height) != (frame.width, frame.height)
            or len(checked.regions) > 10_000 or size > limits.max_text_bytes):
        raise InvalidInput('video OCR observation exceeds its limits or has incorrect dimensions')
    return checked


def _receipt(raw: bytes, binding: str, frame: VideoFrame, limits: DocumentLimits) -> _Receipt:
    try:
        if type(raw) is not bytes or not 0 < len(raw) <= _MAX_RECEIPT:
            raise ValueError('receipt byte limit')
        value = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(value, dict) or type(value.get('schema_version')) is not int:
            raise ValueError('receipt version')
        saved = _Receipt.model_validate_json(raw)
        if saved.binding != binding or saved.png_sha256 != frame.sha256:
            raise ValueError('receipt frame binding')
        _observation(saved.result, frame, limits)
        return saved
    except (ValueError, RecursionError) as error:
        raise InvalidInput('video OCR receipt is invalid or belongs to different work') from error


def _completion(receipts: list[_Receipt]) -> bytes:
    return hashlib.sha256(b'\n'.join(receipt.model_dump_json().encode() for receipt in receipts)).hexdigest().encode()


def _segment(frame: VideoFrame, result: OcrResult, stream: int) -> DocumentSegment | None:
    parts: list[str] = []
    regions: list[DocumentTextRegion] = []
    offset = 0
    for region in result.regions:
        if not region.text.strip():
            continue
        if parts:
            offset += 1
        end = offset + len(region.text.encode())
        regions.append(DocumentTextRegion(**region.model_dump(), start=offset, end=end,
                                          coordinate_space='normalized_displayed_frame_top_left'))
        parts.append(region.text)
        offset = end
    if not parts:
        return None
    return DocumentSegment(text='\n'.join(parts), locator=f'video:stream:{stream}/frame:{frame.selection.ordinal}',
        metadata={'extraction': 'ocr', 'engine': result.engine, 'video_frame_ordinal': str(frame.selection.ordinal)},
        regions=tuple(regions))


class VideoDocumentParser:
    """Caller-owned decoder/OCR; change model_revision when OCR behavior changes.

    Checkpoint storage must scope and authenticate receipts to source, space and
    run. Frames are re-decoded and verified on replay; completed OCR is reused.
    """
    def __init__(self, decoder: VideoFrameDecoder, engine: OcrEngine, *, model_revision: str,
                 policy: VideoFramePolicy | None = None) -> None:
        if not isinstance(decoder, VideoFrameDecoder) or not callable(getattr(engine, 'recognize', None)):
            raise InvalidInput('video OCR requires an explicit decoder and OCR engine')
        if not isinstance(model_revision, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', model_revision):
            raise InvalidInput('video OCR requires a bounded model revision')
        self._decoder, self._engine, self._revision = decoder, engine, model_revision
        self._policy = _validated_policy(VideoFramePolicy() if policy is None else policy)

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits = DocumentLimits()) -> ParsedDocument:
        return await self._parse(data, filename, limits, None)

    async def parse_checkpointed(self, data: bytes, filename: str, limits: DocumentLimits,
                                 checkpoints: ExtractionCheckpoints) -> ParsedDocument:
        return await self._parse(data, filename, limits, checkpoints)

    async def _parse(self, data: bytes, filename: str, limits: DocumentLimits,
                     checkpoints: ExtractionCheckpoints | None) -> ParsedDocument:
        limits = _validated_limits(limits)
        deadline = monotonic() + limits.timeout_seconds
        sampled = await self._decoder.sample(data, filename, policy=self._policy, limits=limits)
        header = _canonical({'implementation': 'video-frame-ocr-v1', 'source': sampled.source_sha256,
            'filename': filename, 'limits': limits.model_dump(), 'decoder': sampled.decoder_revision,
            'policy_revision': sampled.policy_revision, 'policy': sampled.policy.model_dump(),
            'model_revision': self._revision, 'max_total_regions': _MAX_TOTAL_REGIONS, 'plan': {
                'stream': sampled.plan.stream_index, 'time_base': str(sampled.plan.time_base),
                'start': sampled.plan.start_timestamp, 'duration': str(sampled.plan.duration),
                'decoded_frames': sampled.plan.decoded_frames, 'unavailable': sampled.plan.unavailable_requests},
            'frames': [{'selection': asdict(frame.selection), 'sha256': frame.sha256,
                        'width': frame.width, 'height': frame.height, 'bytes': len(frame.png)} for frame in sampled.frames]})
        binding = hashlib.sha256(header).hexdigest()
        saved: list[_Receipt | None] = []
        complete = None
        if checkpoints is not None:
            previous = checkpoints.get(_HEADER)
            if previous is None:
                if checkpoints.get(_COMPLETE) is not None or any(checkpoints.get(_key(index)) is not None for index in range(256)):
                    raise InvalidInput('video OCR receipts are missing their header')
                checkpoints.put(_HEADER, header)
            elif previous != header:
                raise InvalidInput('video OCR configuration or decoded frames changed')
            complete = checkpoints.get(_COMPLETE)
        for index, frame in enumerate(sampled.frames):
            raw = checkpoints.get(_key(index)) if checkpoints is not None else None
            saved.append(_receipt(raw, binding, frame, limits) if raw is not None else None)
        known = [receipt for receipt in saved if receipt is not None]
        if complete is not None and (len(known) != len(saved) or complete != _completion(known)):
            raise InvalidInput('completed video OCR receipts are incomplete or changed')
        raw_text_bytes = sum(len(region.text.encode()) for receipt in known for region in receipt.result.regions)
        if raw_text_bytes > limits.max_text_bytes:
            raise InvalidInput('video OCR exceeds its text byte limit')
        region_count = sum(len(receipt.result.regions) for receipt in known)
        if region_count > _MAX_TOTAL_REGIONS:
            raise InvalidInput('video OCR exceeds its total region limit')
        results: list[_Receipt] = []
        for index, frame in enumerate(sampled.frames):
            receipt = saved[index]
            if receipt is None:
                if region_count >= _MAX_TOTAL_REGIONS:
                    raise InvalidInput('video OCR exceeds its total region limit')
                remaining = _remaining(deadline)
                try:
                    observation = await asyncio.wait_for(self._engine.recognize(frame.png,
                        max_pixels=self._policy.max_pixels, max_regions=min(10_000, _MAX_TOTAL_REGIONS - region_count), timeout_seconds=remaining), remaining)
                except asyncio.TimeoutError:
                    raise InvalidInput('video OCR exceeded its wall time limit') from None
                result = _observation(observation, frame, limits)
                region_count += len(result.regions)
                if region_count > _MAX_TOTAL_REGIONS:
                    raise InvalidInput('video OCR exceeds its total region limit')
                raw_text_bytes += sum(len(region.text.encode()) for region in result.regions)
                if raw_text_bytes > limits.max_text_bytes:
                    raise InvalidInput('video OCR exceeds its text byte limit')
                receipt = _Receipt(binding=binding, png_sha256=frame.sha256, result=result)
                encoded = receipt.model_dump_json().encode()
                if len(encoded) > _MAX_RECEIPT:
                    raise InvalidInput('video OCR receipt exceeds its byte limit')
                if checkpoints is not None:
                    checkpoints.put(_key(index), encoded)
            results.append(receipt)
        parsed = self._document(sampled, results, filename)
        validate_document(parsed, limits)
        _remaining(deadline)
        if checkpoints is not None:
            checkpoints.put(_COMPLETE, _completion(results))
        return parsed

    def _document(self, sampled: VideoFrames, receipts: list[_Receipt], filename: str) -> ParsedDocument:
        segments: list[DocumentSegment] = []
        frames: list[VideoOcrFrame] = []
        for frame, receipt in zip(sampled.frames, receipts, strict=True):
            result = receipt.result
            segment = _segment(frame, result, sampled.plan.stream_index)
            if segment is not None:
                segments.append(segment)
            frames.append(VideoOcrFrame(ordinal=frame.selection.ordinal,
                presentation_timestamp=frame.selection.presentation_timestamp,
                requested_seconds=frame.selection.requested_seconds, width=frame.width, height=frame.height,
                png_sha256=frame.sha256, png_bytes=len(frame.png), ocr_engine=result.engine, empty=segment is None))
        if not segments:
            raise InvalidInput('video contains no recognized text')
        base = sampled.plan.time_base
        evidence = DocumentVideoEvidence(source_sha256=sampled.source_sha256, decoder_revision=sampled.decoder_revision,
            policy_revision=sampled.policy_revision, model_revision=self._revision, policy=sampled.policy,
            stream_index=sampled.plan.stream_index, time_base=f'{base.numerator}/{base.denominator}',
            start_timestamp=sampled.plan.start_timestamp, duration_ticks=int(sampled.plan.duration / base),
            decoded_frames=sampled.plan.decoded_frames, unavailable_requests=sampled.plan.unavailable_requests,
            frames=tuple(frames))
        return ParsedDocument(format=extension(filename).lstrip('.'), parser='video-frame-ocr', segments=tuple(segments),
                              metadata={'extraction': 'ocr', 'coverage': 'sampled-video-frames'}, video=evidence)
