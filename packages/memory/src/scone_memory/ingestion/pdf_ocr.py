"""Opt-in scanned PDF extraction through a caller-selected OCR engine."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from importlib.util import find_spec
import json
import hashlib
from importlib.metadata import PackageNotFoundError, version
import logging
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, ValidationError, model_serializer

from ..core.errors import InvalidInput
from ..ocr.process import python_worker, run_bounded
from ..ocr.tesseract import png_dimensions
from ..ocr.types import OcrEngine, OcrResult
from ..ocr.layout import ReadingMode, OrderedRegions, order_columns
from .extraction_checkpoint import ExtractionCheckpoints
from .pdf import ParsedPdf, PdfLimits, PdfPage, PdfTextRegion, PypdfParser, validate_pdf


logger = logging.getLogger(__name__)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InvalidInput('PDF OCR exceeded its wall time limit')
    return remaining


class OcrPdfOptions(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    mode: Literal['missing_text', 'all_pages'] = 'missing_text'
    dpi: int = Field(default=150, ge=72, le=300)
    max_pixels: int = Field(default=20_000_000, ge=1, le=20_000_000)
    max_regions: int = Field(default=10_000, ge=1, le=50_000)
    reading_order: ReadingMode = 'provider'

    @model_serializer(mode='wrap')
    def preserve_legacy(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        value: dict[str, object] = handler(self)
        if self.reading_order == 'provider':
            value.pop('reading_order', None)
        return value


def _page_text(result: OcrResult, offset: int, max_bytes: int,
               order: OrderedRegions | None = None) -> tuple[str, tuple[PdfTextRegion, ...]]:
    parts: list[str] = []
    regions: list[PdfTextRegion] = []
    previous: tuple[int, int] | None = None
    indices = range(len(result.regions)) if order is None else order.indices
    previous_column: int | None = None
    for position, index in enumerate(indices):
        region = result.regions[index]
        column = order.columns[position] if order else None
        key = (region.block, region.line)
        separator = '' if previous is None else (' ' if previous == key else '\n')
        if previous is not None and column != previous_column:
            separator = '\n'
        offset += len(separator)
        end = offset + len(region.text.encode('utf-8'))
        if end > max_bytes:
            raise InvalidInput('PDF OCR text exceeds its byte limit')
        parts.extend((separator, region.text))
        regions.append(PdfTextRegion(**region.model_dump(), start=offset, end=end,
            provider_index=index if order else None, reading_column=column))
        offset = end
        previous = key
        previous_column = column
    return ''.join(parts), tuple(regions)


def assemble_ocr_pdf(parsed: ParsedPdf, recognized: Mapping[int, OcrResult], limits: PdfLimits, *,
                     reading_order: ReadingMode = 'provider') -> ParsedPdf:
    """Rebuild absolute UTF-8 spans from native text and validated page results."""
    if reading_order not in ('provider', 'columns_ltr', 'columns_rtl'):
        raise InvalidInput('unsupported OCR reading order')
    encoded = parsed.text.encode('utf-8')
    pages: list[PdfPage] = []
    texts: list[str] = []
    offset = 0
    for page in parsed.pages:
        if page.number > 1:
            offset += 2
        text = encoded[page.start:page.end].decode('utf-8')
        updates: dict[str, object] = {}
        if page.number in recognized:
            result = recognized[page.number]
            order = None if reading_order == 'provider' else order_columns(result.regions,
                direction='rtl' if reading_order == 'columns_rtl' else 'ltr')
            text, regions = _page_text(result, offset, limits.max_text_bytes, order)
            updates = {'extraction': 'ocr', 'ocr_engine': result.engine, 'regions': regions,
                       'reading_order': order.receipt if order else None}
        end = offset + len(text.encode('utf-8'))
        if end > limits.max_text_bytes:
            raise InvalidInput('PDF OCR text exceeds its byte limit')
        pages.append(page.model_copy(update={**updates, 'start': offset, 'end': end, 'empty': not text.strip()}))
        texts.append(text)
        offset = end
    suffix = '' if reading_order == 'provider' else f'+{reading_order}-v1'
    output = ParsedPdf(text='\n\n'.join(texts), parser=f'{parsed.parser}+scone-ocr-v1{suffix}', pages=tuple(pages))
    validate_pdf(output, limits)
    return output


class _PageReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1] = 1
    binding: str = Field(pattern=r'^[a-f0-9]{64}$')
    page: int = Field(ge=1, le=1000)
    result: OcrResult


def _checkpoint_binding(data: bytes, parsed: ParsedPdf, options: OcrPdfOptions,
                        limits: PdfLimits) -> str:
    dependencies: dict[str, str] = {}
    for package in ('pypdf', 'pypdfium2'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = 'unavailable'
    payload = {'strategy': 'ocr-page-observations-v1',
        'original': hashlib.sha256(data).hexdigest(),
        'inspection': hashlib.sha256(parsed.model_dump_json().encode()).hexdigest(),
        'options': options.model_dump(), 'limits': limits.model_dump(),
        'dependencies': dependencies}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _page_receipt(raw: bytes, binding: str, page: int, options: OcrPdfOptions) -> OcrResult:
    try:
        if type(raw) is not bytes or len(raw) > 16 * 1024 * 1024:
            raise ValueError('checkpoint bytes')
        receipt = _PageReceipt.model_validate_json(raw)
        if (receipt.binding != binding or receipt.page != page
                or receipt.result.width * receipt.result.height > options.max_pixels
                or len(receipt.result.regions) > options.max_regions):
            raise ValueError('checkpoint binding or bounds')
        return receipt.result
    except (ValueError, TypeError):
        raise InvalidInput('PDF OCR page checkpoint is invalid or does not match this extraction') from None


class OcrPdfParser:
    """OCR missing text pages, or explicitly all pages, within one wall deadline.

    Processing is sequential to bound concurrent native raster memory. No models
    are downloaded; the caller explicitly supplies the recognition implementation.
    """
    def __init__(self, engine: OcrEngine, *, options: OcrPdfOptions = OcrPdfOptions()):
        self.engine = engine
        self.options = options

    async def parse(self, data: bytes, limits: PdfLimits = PdfLimits()) -> ParsedPdf:
        try:
            return await asyncio.wait_for(self._parse(data, limits), limits.timeout_seconds)
        except asyncio.TimeoutError as error:
            raise InvalidInput('PDF OCR exceeded its wall time limit') from error

    async def parse_checkpointed(self, data: bytes, limits: PdfLimits,
                                 checkpoints: ExtractionCheckpoints) -> ParsedPdf:
        """Reuse completed page observations within the existing whole-call deadline."""
        try:
            return await asyncio.wait_for(self._parse(data, limits, checkpoints), limits.timeout_seconds)
        except asyncio.TimeoutError as error:
            raise InvalidInput('PDF OCR exceeded its wall time limit') from error

    async def _parse(self, data: bytes, limits: PdfLimits,
                     checkpoints: ExtractionCheckpoints | None = None) -> ParsedPdf:
        deadline = time.monotonic() + limits.timeout_seconds
        parsed = await self.inspect(data, limits)
        binding = _checkpoint_binding(data, parsed, self.options, limits) if checkpoints is not None else ""
        if checkpoints is not None:
            saved_binding = checkpoints.get('ocr-binding')
            encoded_binding = binding.encode('ascii')
            if saved_binding is None:
                checkpoints.put('ocr-binding', encoded_binding)
            elif saved_binding != encoded_binding:
                raise InvalidInput('PDF OCR checkpoints do not match this extraction')
        recognized: dict[int, OcrResult] = {}
        text_bytes = len(parsed.text.encode())
        for page in parsed.pages:
            if page.empty or self.options.mode == 'all_pages':
                key = f'ocr-page:{page.number}'
                cached = checkpoints.get(key) if checkpoints is not None else None
                if cached is None:
                    result = await self._recognize(data, page.number, deadline)
                else:
                    result = _page_receipt(cached, binding, page.number, self.options)
                _remaining(deadline)
                text_bytes += sum(len(r.text.encode()) for r in result.regions) + max(0, len(result.regions) - 1) - (page.end - page.start)
                if text_bytes > limits.max_text_bytes:
                    raise InvalidInput('PDF OCR text exceeds its byte limit')
                if cached is None and checkpoints is not None:
                    receipt = _PageReceipt(binding=binding, page=page.number, result=result).model_dump_json().encode()
                    result = _page_receipt(receipt, binding, page.number, self.options)
                    _remaining(deadline)
                    checkpoints.put(key, receipt)
                recognized[page.number] = result
        output = assemble_ocr_pdf(parsed, recognized, limits, reading_order=self.options.reading_order)
        _remaining(deadline)
        return output

    async def inspect(self, data: bytes, limits: PdfLimits = PdfLimits()) -> ParsedPdf:
        """Read native page geometry/text without recognizing missing pages."""
        return await PypdfParser()._parse(data, limits, allow_empty=True,
            metadata_only=self.options.mode == 'all_pages', allow_text_errors=True)

    async def recognize_page(self, data: bytes, page: int, *, timeout_seconds: float = 30.0) -> OcrResult:
        """Render and recognize one page, including a deadline for the provider."""
        budget = PdfLimits(timeout_seconds=timeout_seconds).timeout_seconds
        try:
            return await asyncio.wait_for(self._recognize(data, page, time.monotonic() + budget), budget)
        except asyncio.TimeoutError as error:
            raise InvalidInput('PDF OCR page exceeded its wall time limit') from error

    async def _recognize(self, data: bytes, page: int, deadline: float) -> OcrResult:
        if find_spec('pypdfium2') is None:
            raise InvalidInput('PDF OCR rendering requires scone-memory[pdf-ocr]')
        payload = json.dumps({'page': page, 'dpi': self.options.dpi, 'max_pixels': self.options.max_pixels})
        image = await run_bounded(python_worker('scone_memory.ingestion._pdf_render_worker', payload),
            data, timeout=_remaining(deadline), max_output=80_000_000, label='PDF renderer')
        if image.startswith(b'{'):
            raise InvalidInput(str(json.loads(image).get('error', 'PDF rendering failed')))
        dimensions = png_dimensions(image, self.options.max_pixels)
        started = time.monotonic()
        try:
            result = await self.engine.recognize(image, max_pixels=self.options.max_pixels,
                max_regions=self.options.max_regions, timeout_seconds=_remaining(deadline))
        except TimeoutError as error:
            raise InvalidInput('PDF OCR recognition provider timed out') from error
        if not isinstance(result, OcrResult):
            raise InvalidInput('OCR recognizer returned invalid output')
        try:
            result = OcrResult.model_validate_json(result.model_dump_json(warnings=False))
        except (ValidationError, ValueError, TypeError) as error:
            raise InvalidInput('OCR recognizer returned invalid output') from error
        logger.debug('PDF OCR page=%d engine=%s regions=%d recognition_ms=%.1f',
            page, result.engine, len(result.regions), (time.monotonic() - started) * 1000.)
        if (result.width, result.height) != dimensions or len(result.regions) > self.options.max_regions:
            raise InvalidInput('OCR result dimensions or region count do not match the rendered page')
        return result
