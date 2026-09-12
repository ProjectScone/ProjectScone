"""Compact completed transcription receipts; never store decoded audio here."""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...core.errors import InvalidInput
from ..extraction_checkpoint import ExtractionCheckpoints
from .media import TranscriptionSegment
from .types import DocumentLimits

_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
TRANSCRIPTION_CHECKPOINT_KEY = 'media-transcript'


class TranscriptionCoverage(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    windows: int = Field(ge=1, le=1000)
    empty_windows: int = Field(ge=0, lt=1000)

    @model_validator(mode='after')
    def has_observations(self) -> Self:
        if self.empty_windows >= self.windows:
            raise ValueError('completed transcription requires a window with observations')
        return self


class CompletedTranscription(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1] = 1
    binding: str = Field(pattern=r'^[a-f0-9]{64}$')
    audio_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    audio_bytes: int = Field(ge=46, le=19_200_044)
    duration_seconds: float = Field(gt=0, le=600, allow_inf_nan=False)
    segments: tuple[TranscriptionSegment, ...] = Field(min_length=1, max_length=20_000)
    coverage: TranscriptionCoverage | None = None

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('checkpoint version must be an integer')
        return value


def transcription_binding(data: bytes, filename: str, limits: DocumentLimits,
                          decoder: str, duration: float, chunk_seconds: int | None = None) -> str:
    settings = {'implementation': 'media-transcription-receipt-v1',
                'source': hashlib.sha256(data).hexdigest(), 'filename': filename,
                'limits': limits.model_dump(), 'decoder': decoder, 'max_duration': duration}
    if chunk_seconds is not None:
        from .media_windows import WINDOW_IMPLEMENTATION
        settings.update(chunk_seconds=chunk_seconds, window_implementation=WINDOW_IMPLEMENTATION)
    raw = json.dumps(settings, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate checkpoint key')
        result[key] = value
    return result


def read_transcription(checkpoints: ExtractionCheckpoints, binding: str) -> CompletedTranscription | None:
    raw = checkpoints.get(TRANSCRIPTION_CHECKPOINT_KEY)
    if raw is None:
        return None
    try:
        if type(raw) is not bytes or len(raw) > _MAX_RECEIPT_BYTES:
            raise ValueError('checkpoint size or type')
        json.loads(raw, object_pairs_hook=_unique)
        receipt = CompletedTranscription.model_validate_json(raw)
        if receipt.binding != binding:
            raise ValueError('checkpoint binding')
        return receipt
    except (ValueError, RecursionError):
        raise InvalidInput('media transcription checkpoint is invalid or does not match this extraction') from None


def save_transcription(checkpoints: ExtractionCheckpoints, receipt: CompletedTranscription) -> None:
    raw = receipt.model_dump_json(exclude_none=True).encode()
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise InvalidInput('media transcription checkpoint exceeds its byte limit')
    checkpoints.put(TRANSCRIPTION_CHECKPOINT_KEY, raw)
