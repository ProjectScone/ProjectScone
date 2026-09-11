"""Format-neutral extraction results with source-local evidence locators."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ...core.errors import InvalidInput


class DocumentLimits(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    max_input_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=25 * 1024 * 1024)
    max_text_bytes: int = Field(default=2_000_000, ge=1, le=2_000_000)
    max_segments: int = Field(default=20_000, ge=1, le=20_000)
    max_archive_entries: int = Field(default=10_000, ge=1, le=10_000)
    max_archive_bytes: int = Field(default=100_000_000, ge=1, le=100_000_000)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120, allow_inf_nan=False)


class DocumentSegment(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    text: str = Field(min_length=1)
    locator: str = Field(min_length=1, max_length=4096)
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    format: str = Field(min_length=1, max_length=64)
    parser: str = Field(min_length=1, max_length=128)
    segments: tuple[DocumentSegment, ...] = Field(min_length=1, max_length=20_000)
    metadata: dict[str, str] = Field(default_factory=dict)


def validate_document(parsed: ParsedDocument, limits: DocumentLimits) -> None:
    if len(parsed.segments) > limits.max_segments:
        raise InvalidInput('document exceeds its segment limit')
    total = 0
    for segment in parsed.segments:
        if not segment.text.strip():
            raise InvalidInput('document contains an empty text segment')
        total += len(segment.text.encode('utf-8')) + 2
        if total - 2 > limits.max_text_bytes:
            raise InvalidInput('document exceeds its extracted text byte limit')
        _metadata(segment.metadata)
    _metadata(parsed.metadata)


def _metadata(values: dict[str, str]) -> None:
    if len(values) > 32 or any(len(k) > 128 or len(v.encode('utf-8')) > 4096 for k, v in values.items()):
        raise InvalidInput('document metadata exceeds its limit')
