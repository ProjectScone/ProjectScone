"""Explicit format dispatch; native parsers run in cancellable bounded children."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Protocol

from ...core.errors import InvalidInput
from ...ocr.process import python_worker, run_bounded
from ..pdf import PdfLimits, PdfParser, PypdfParser, validate_pdf
from .types import DocumentLimits, DocumentSegment, DocumentTextRegion, ParsedDocument, validate_document


class DocumentParser(Protocol):
    async def parse(self, data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument: ...


def extension(filename: str) -> str:
    if (not isinstance(filename, str) or not filename.strip() or len(filename.encode()) > 1024
            or '\x00' in filename):
        raise InvalidInput('document requires a bounded filename with its format extension')
    return PurePosixPath(filename).suffix.lower()


class BuiltinDocumentParser:
    """Text, structured data, Office and PDF. OCR is an explicit PDF parser option."""
    def __init__(self, *, pdf_parser: PdfParser | None = None,
                 parsers: Mapping[str, DocumentParser] | None = None):
        self._pdf = pdf_parser or PypdfParser()
        self._parsers = dict(parsers or {})
        if any(not key.startswith('.') or key.lower() != key or '/' in key for key in self._parsers):
            raise InvalidInput('parser extensions must be lowercase dotted suffixes')

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits = DocumentLimits()) -> ParsedDocument:
        suffix = extension(filename)
        if not isinstance(data, bytes) or not data or len(data) > limits.max_input_bytes:
            raise InvalidInput('document exceeds its input byte limit or is empty')
        if suffix in self._parsers:
            parsed = await self._parsers[suffix].parse(data, filename, limits)
        elif suffix == '.pdf':
            pdf_limits = PdfLimits(max_input_bytes=limits.max_input_bytes,
                max_text_bytes=limits.max_text_bytes, max_pages=min(1000, limits.max_segments),
                timeout_seconds=limits.timeout_seconds)
            pdf = await self._pdf.parse(data, pdf_limits)
            validate_pdf(pdf, pdf_limits)
            encoded = pdf.text.encode()
            parsed = ParsedDocument(format='pdf', parser=pdf.parser, segments=tuple(
                DocumentSegment(text=encoded[p.start:p.end].decode(), locator=f'page:{p.number}',
                    metadata={'page': str(p.number), 'extraction': p.extraction,
                              'width_points': str(p.width_points), 'height_points': str(p.height_points),
                              'rotation': str(p.rotation),
                              **({'ocr_reading_order': p.reading_order.model_dump_json()} if p.reading_order else {}),
                              **({'ocr_engine': p.ocr_engine} if p.ocr_engine else {})},
                    regions=tuple(DocumentTextRegion(text=r.text, box=r.box, score=r.score,
                        block=r.block, line=r.line, start=r.start - p.start, end=r.end - p.start,
                        provider_index=r.provider_index, reading_column=r.reading_column,
                        coordinate_space=p.region_geometry) for r in p.regions))
                for p in pdf.pages if not p.empty),
                metadata={'empty_pages': ','.join(str(p.number) for p in pdf.pages if p.empty)})
        else:
            raw = await run_bounded(python_worker('scone_memory.ingestion.formats.worker',
                filename, limits.model_dump_json()), data, timeout=limits.timeout_seconds,
                max_output=min(25_000_000, limits.max_text_bytes * 6 + limits.max_segments * 8192))
            payload = json.loads(raw)
            if isinstance(payload, dict) and 'error' in payload:
                raise InvalidInput(str(payload['error']))
            parsed = ParsedDocument.model_validate_json(raw)
        validate_document(parsed, limits)
        return parsed
