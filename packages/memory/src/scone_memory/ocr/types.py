"""Provider-neutral OCR observations in displayed image coordinates."""
from __future__ import annotations

import math
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator, model_serializer, model_validator


class OcrRegion(BaseModel):
    """A word or line; score is a recognizer score, not factual confidence."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    text: str = Field(min_length=1, max_length=100_000)
    box: tuple[float, float, float, float]
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    block: int = Field(default=0, ge=0)
    line: int = Field(default=0, ge=0)

    @field_validator('box')
    @classmethod
    def valid_box(cls, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if (not all(math.isfinite(value) and 0 <= value <= 1 for value in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('OCR box must be a nonempty normalized rectangle')
        return box


class OcrResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    engine: str = Field(min_length=1, max_length=96)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    regions: tuple[OcrRegion, ...] = Field(max_length=50_000)


class OrderedOcrRegion(OcrRegion):
    """Optional inferred column membership and original provider position."""
    provider_index: int | None = Field(default=None, ge=0, lt=50_000)
    reading_column: int | None = Field(default=None, ge=0, le=8)

    @model_validator(mode='after')
    def paired_order(self) -> OrderedOcrRegion:
        if (self.provider_index is None) != (self.reading_column is None):
            raise ValueError('OCR reading order requires both provider index and column')
        return self

    @model_serializer(mode='wrap')
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        value: dict[str, object] = handler(self)
        if self.provider_index is None:
            value.pop('provider_index', None)
            value.pop('reading_column', None)
        return value


class OcrEngine(Protocol):
    async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000,
                        max_regions: int = 10_000, timeout_seconds: float = 30.0) -> OcrResult: ...
