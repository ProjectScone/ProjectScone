"""Optional PDF parsing with a cancellable subprocess and explicit page spans."""
from __future__ import annotations

import asyncio
from importlib.util import find_spec
import json
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput
from ..ocr.types import OcrRegion
from ..ocr.process import python_worker, run_bounded


class PdfLimits(BaseModel):
    """Output/input bounds and a hard subprocess wall deadline, not a RAM quota."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    max_input_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=25 * 1024 * 1024)
    max_pages: int = Field(default=100, ge=1, le=1000)
    max_text_bytes: int = Field(default=2_000_000, ge=1, le=2_000_000)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120, allow_inf_nan=False)


class PdfTextRegion(OcrRegion):
    """Recognized text with absolute extracted UTF-8 offsets and raster geometry."""
    start: int = Field(ge=0, le=2_000_000)
    end: int = Field(ge=0, le=2_000_000)


class PdfPage(BaseModel):
    """One-based page and half-open spans in the extracted text's UTF-8 bytes.

    Dimensions describe the unrotated PDF media box; rotation is clockwise.
    These are page geometry, not estimated glyph/paragraph bounding boxes.
    """
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    number: int = Field(ge=1, le=1000)
    start: int = Field(ge=0, le=2_000_000)
    end: int = Field(ge=0, le=2_000_000)
    width_points: float = Field(gt=0, allow_inf_nan=False)
    height_points: float = Field(gt=0, allow_inf_nan=False)
    rotation: int = Field(ge=0, le=270, multiple_of=90)
    empty: bool
    extraction: Literal['text_layer', 'ocr'] = 'text_layer'
    region_geometry: Literal['normalized_displayed_page_top_left'] = 'normalized_displayed_page_top_left'
    regions: tuple[PdfTextRegion, ...] = Field(default=(), max_length=50_000)
    ocr_engine: str | None = Field(default=None, min_length=1, max_length=96)


class ParsedPdf(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    text: str
    parser: str = Field(min_length=1, max_length=128)
    pages: tuple[PdfPage, ...] = Field(min_length=1, max_length=1000)


class PdfParser(Protocol):
    async def parse(self, data: bytes, limits: PdfLimits) -> ParsedPdf: ...


class PypdfParser:
    """Extract a PDF text layer without OCR, networking, or model inference."""

    async def parse(self, data: bytes, limits: PdfLimits = PdfLimits()) -> ParsedPdf:
        return await self._parse(data, limits, allow_empty=False)

    async def _parse(self, data: bytes, limits: PdfLimits, *, allow_empty: bool, metadata_only: bool = False) -> ParsedPdf:
        if not isinstance(data, bytes) or not data or len(data) > limits.max_input_bytes:
            raise InvalidInput('PDF input exceeds its byte limit or has no bytes')
        if not data.startswith(b'%PDF-'):
            raise InvalidInput('input does not have a PDF header')
        if find_spec('pypdf') is None:
            raise InvalidInput('PDF parsing requires the optional scone-memory[pdf] extra')
        output = await run_bounded(python_worker(
            'scone_memory.ingestion._pdf_worker', limits.model_dump_json(),
            *(['--allow-empty'] if allow_empty else []),
            *(['--metadata-only'] if metadata_only else []),
        ), data, timeout=limits.timeout_seconds, label='PDF parser',
            max_output=limits.max_text_bytes * 6 + limits.max_pages * 2048 + 4096)
        try:
            envelope = json.loads(output)
            if 'error' in envelope:
                raise InvalidInput(str(envelope['error']))
            parsed = ParsedPdf.model_validate_json(output)
        except (ValueError, TypeError) as error:
            raise InvalidInput('PDF parser returned invalid output') from error
        validate_pdf(parsed, limits, allow_empty=allow_empty)
        return parsed


def validate_pdf(parsed: ParsedPdf, limits: PdfLimits, *, allow_empty: bool = False) -> None:
    """Validate extension parser output before any source is stored."""
    encoded = parsed.text.encode('utf-8')
    if (not allow_empty and not parsed.text.strip()) or len(encoded) > limits.max_text_bytes:
        raise InvalidInput('PDF extracted text is empty or exceeds its byte limit')
    if len(parsed.pages) > limits.max_pages:
        raise InvalidInput('PDF exceeds its page limit')
    previous_end = 0
    for number, page in enumerate(parsed.pages, 1):
        separator = b'' if number == 1 else b'\n\n'
        if (page.number != number or page.start != previous_end + len(separator)
                or not page.start <= page.end <= len(encoded)
                or encoded[previous_end:page.start] != separator):
            raise InvalidInput('PDF page spans do not match the extracted text')
        try:
            page_text = encoded[page.start:page.end].decode('utf-8')
        except UnicodeDecodeError as error:
            raise InvalidInput('PDF page span splits a UTF-8 character') from error
        if page.empty != (not page_text.strip()):
            raise InvalidInput('PDF page coverage does not match the extracted text')
        if page.extraction == 'text_layer' and (page.regions or page.ocr_engine):
            raise InvalidInput('PDF text-layer page cannot contain OCR regions')
        if page.extraction == 'ocr' and (not page.ocr_engine or (not page.empty and not page.regions)):
            raise InvalidInput('PDF OCR page must identify its engine and recognized regions')
        region_end = page.start
        for region in page.regions:
            if (not region_end <= region.start < region.end <= page.end
                    or encoded[region.start:region.end] != region.text.encode('utf-8')
                    or encoded[region_end:region.start].strip()):
                raise InvalidInput('PDF OCR region does not match its extracted text span')
            region_end = region.end
        if page.extraction == 'ocr' and encoded[region_end:page.end].strip():
            raise InvalidInput('PDF OCR regions do not cover the extracted text')
        previous_end = page.end
    if previous_end != len(encoded):
        raise InvalidInput('PDF page spans do not cover the extracted text')
