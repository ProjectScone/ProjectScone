"""Opt-in scanned PDF extraction through a caller-selected OCR engine."""
from __future__ import annotations

import asyncio
from importlib.util import find_spec
import json
import logging
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput
from ..ocr.process import run_bounded
from ..ocr.tesseract import png_dimensions
from ..ocr.types import OcrEngine, OcrResult
from .pdf import ParsedPdf, PdfLimits, PdfPage, PdfTextRegion, PypdfParser, validate_pdf


logger = logging.getLogger(__name__)


class OcrPdfOptions(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    mode: Literal['missing_text', 'all_pages'] = 'missing_text'
    dpi: int = Field(default=150, ge=72, le=300)
    max_pixels: int = Field(default=20_000_000, ge=1, le=20_000_000)
    max_regions: int = Field(default=10_000, ge=1, le=50_000)


def _page_text(result: OcrResult, offset: int, max_bytes: int) -> tuple[str, tuple[PdfTextRegion, ...]]:
    parts: list[str] = []
    regions: list[PdfTextRegion] = []
    previous: tuple[int, int] | None = None
    for region in result.regions:
        key = (region.block, region.line)
        separator = '' if previous is None else (' ' if previous == key else '\n')
        offset += len(separator)
        end = offset + len(region.text.encode('utf-8'))
        if end > max_bytes:
            raise InvalidInput('PDF OCR text exceeds its byte limit')
        parts.extend((separator, region.text))
        regions.append(PdfTextRegion(**region.model_dump(), start=offset, end=end))
        offset = end
        previous = key
    return ''.join(parts), tuple(regions)


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

    async def _parse(self, data: bytes, limits: PdfLimits) -> ParsedPdf:
        deadline = time.monotonic() + limits.timeout_seconds
        parsed = await PypdfParser()._parse(data, limits, allow_empty=True,
            metadata_only=self.options.mode == 'all_pages')
        encoded = parsed.text.encode('utf-8')
        pages: list[PdfPage] = []
        texts: list[str] = []
        offset = 0
        for page in parsed.pages:
            if page.number > 1:
                offset += 2
            text = encoded[page.start:page.end].decode('utf-8')
            updates: dict[str, object] = {}
            if page.empty or self.options.mode == 'all_pages':
                result = await self._recognize(data, page.number, deadline)
                text, regions = _page_text(result, offset, limits.max_text_bytes)
                updates = {'extraction': 'ocr', 'ocr_engine': result.engine, 'regions': regions}
            end = offset + len(text.encode('utf-8'))
            if end > limits.max_text_bytes:
                raise InvalidInput('PDF OCR text exceeds its byte limit')
            pages.append(page.model_copy(update={**updates, 'start': offset, 'end': end, 'empty': not text.strip()}))
            texts.append(text)
            offset = end
        output = ParsedPdf(text='\n\n'.join(texts), parser=f'{parsed.parser}+scone-ocr-v1', pages=tuple(pages))
        validate_pdf(output, limits)
        return output

    async def _recognize(self, data: bytes, page: int, deadline: float) -> OcrResult:
        if find_spec('pypdfium2') is None:
            raise InvalidInput('PDF OCR rendering requires scone-memory[pdf-ocr]')
        payload = json.dumps({'page': page, 'dpi': self.options.dpi, 'max_pixels': self.options.max_pixels})
        image = await run_bounded([sys.executable, '-m', 'scone_memory.ingestion._pdf_render_worker', payload],
            data, timeout=deadline - time.monotonic(), max_output=80_000_000)
        if image.startswith(b'{'):
            raise InvalidInput(str(json.loads(image).get('error', 'PDF rendering failed')))
        dimensions = png_dimensions(image, self.options.max_pixels)
        started = time.monotonic()
        result = await self.engine.recognize(image, max_pixels=self.options.max_pixels,
            max_regions=self.options.max_regions, timeout_seconds=max(.000001, deadline - time.monotonic()))
        result = OcrResult.model_validate_json(result.model_dump_json())
        logger.debug('PDF OCR page=%d engine=%s regions=%d recognition_ms=%.1f',
            page, result.engine, len(result.regions), (time.monotonic() - started) * 1000.)
        if (result.width, result.height) != dimensions or len(result.regions) > self.options.max_regions:
            raise InvalidInput('OCR result dimensions or region count do not match the rendered page')
        return result
