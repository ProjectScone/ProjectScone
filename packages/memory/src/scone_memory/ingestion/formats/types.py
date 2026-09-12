"""Format-neutral extraction results with source-local evidence locators."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, SerializerFunctionWrapHandler, model_serializer

from ...core.errors import InvalidInput
from ...ocr.types import OrderedOcrRegion
from ...ocr.layout import ReadingOrderReceipt, validate_reading_order
from .table_types import DocumentTableCell, validate_tables


class DocumentTextRegion(OrderedOcrRegion):
    """Recognized geometry with half-open, segment-relative UTF-8 byte spans."""
    start: int = Field(ge=0, le=2_000_000)
    end: int = Field(ge=0, le=2_000_000)
    coordinate_space: Literal['normalized_displayed_page_top_left',
                              'normalized_displayed_frame_top_left'] = 'normalized_displayed_page_top_left'


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
    regions: tuple[DocumentTextRegion, ...] = Field(default=(), max_length=50_000)
    table_cells: tuple[DocumentTableCell, ...] = Field(default=(), max_length=20_000)

    @model_serializer(mode='wrap')
    def serialize_evidence(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if not self.table_cells:
            result.pop('table_cells', None)
        return result


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
        _regions(segment)
    _metadata(parsed.metadata)
    validate_tables(parsed.segments)


def _regions(segment: DocumentSegment) -> None:
    try:
        raw = segment.metadata.get('ocr_reading_order')
        receipt = ReadingOrderReceipt.model_validate_json(raw) if raw is not None else None
        validate_reading_order(segment.regions, receipt)
    except ValueError as error:
        raise InvalidInput('document OCR reading order is invalid') from error
    if not segment.regions:
        return
    if len(segment.regions) > 50_000:
        raise InvalidInput('document region count exceeds its limit')
    encoded = segment.text.encode('utf-8')
    previous_end = 0
    for observed in segment.regions:
        # Extension parsers may use model_copy/model_construct, bypassing validation.
        try:
            region = DocumentTextRegion.model_validate(observed.model_dump())
        except (ValidationError, AttributeError):
            raise InvalidInput('document region geometry is invalid') from None
        if (not previous_end <= region.start < region.end <= len(encoded)
                or encoded[region.start:region.end] != region.text.encode('utf-8')
                or encoded[previous_end:region.start].strip()):
            raise InvalidInput('document region does not match its extracted text span')
        previous_end = region.end
    if encoded[previous_end:].strip():
        raise InvalidInput('document regions do not cover the extracted text')


def _metadata(values: dict[str, str]) -> None:
    if len(values) > 32 or any(len(k) > 128 or len(v.encode('utf-8')) > 4096 for k, v in values.items()):
        raise InvalidInput('document metadata exceeds its limit')
