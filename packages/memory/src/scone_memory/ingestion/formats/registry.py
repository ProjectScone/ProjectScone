"""Explicit format dispatch; native parsers run in cancellable bounded children."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Protocol

from ...core.errors import InvalidInput
from ...ocr.process import python_worker, run_bounded
from ..extraction_checkpoint import CheckpointedDocumentParser, CheckpointedPdfParser, ExtractionCheckpoints, checkpoint_dispatch_allowed
from ...ocr.table_cells import page_table_cells
from ..pdf import PdfLimits, PdfParser, PypdfParser, validate_pdf
from ..text_layer import unreadable
from .types import DocumentLimits, DocumentSegment, DocumentTextRegion, ParsedDocument, validate_document


class DocumentParser(Protocol):
    async def parse(self, data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument: ...


def extension(filename: str) -> str:
    if (not isinstance(filename, str) or not filename.strip() or len(filename.encode()) > 1024
            or '\x00' in filename):
        raise InvalidInput('document requires a bounded filename with its format extension')
    return PurePosixPath(filename).suffix.lower()


def _section(titles: tuple[str, ...], limit: int = 4_096) -> str:
    """The titles as one metadata value, dropping the outermost until it fits: the innermost
    section is the one a passage is about."""
    kept = list(titles)
    while len(kept) > 1 and len(' > '.join(kept).encode('utf-8')) + (len('… > '.encode('utf-8')) if len(kept) < len(titles) else 0) > limit:
        kept.pop(0)
    joined = ' > '.join(kept)
    return joined if len(kept) == len(titles) else '… > ' + joined


class BuiltinDocumentParser:
    """Text, structured data, Office and PDF. OCR is an explicit PDF parser option."""
    def __init__(self, *, pdf_parser: PdfParser | None = None,
                 parsers: Mapping[str, DocumentParser] | None = None):
        self._pdf = pdf_parser or PypdfParser()
        self._parsers = dict(parsers or {})
        if any(not key.startswith('.') or key.lower() != key or '/' in key for key in self._parsers):
            raise InvalidInput('parser extensions must be lowercase dotted suffixes')

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits = DocumentLimits()) -> ParsedDocument:
        return await self._parse(data, filename, limits)

    async def parse_checkpointed(self, data: bytes, filename: str, limits: DocumentLimits,
                                 checkpoints: ExtractionCheckpoints) -> ParsedDocument:
        return await self._parse(data, filename, limits, checkpoints)

    async def _parse(self, data: bytes, filename: str, limits: DocumentLimits,
                     checkpoints: ExtractionCheckpoints | None = None) -> ParsedDocument:
        suffix = extension(filename)
        if not isinstance(data, bytes) or not data or len(data) > limits.max_input_bytes:
            raise InvalidInput('document exceeds its input byte limit or is empty')
        if suffix in self._parsers:
            parser = self._parsers[suffix]
            if checkpoints is not None and isinstance(parser, CheckpointedDocumentParser) and checkpoint_dispatch_allowed(parser):
                parsed = await parser.parse_checkpointed(data, filename, limits, checkpoints)
            else:
                parsed = await parser.parse(data, filename, limits)
        elif suffix == '.pdf':
            pdf_limits = PdfLimits(max_input_bytes=limits.max_input_bytes,
                max_text_bytes=limits.max_text_bytes, max_pages=min(1000, limits.max_segments),
                timeout_seconds=limits.timeout_seconds)
            if checkpoints is not None and isinstance(self._pdf, CheckpointedPdfParser) and checkpoint_dispatch_allowed(self._pdf):
                pdf = await self._pdf.parse_checkpointed(data, pdf_limits, checkpoints)
            else:
                pdf = await self._pdf.parse(data, pdf_limits)
            validate_pdf(pdf, pdf_limits)
            encoded = pdf.text.encode()
            # Kept, never dropped: named, so a reader knows the page's text is not the page.
            unreadable_pages = [p.number for p in pdf.pages if unreadable(encoded[p.start:p.end].decode())]
            tables = {p.number: page_table_cells(p.regions, [(r.start - p.start, r.end - p.start) for r in p.regions],
                                                 encoded[p.start:p.end], f'page:{p.number}')
                      for p in pdf.pages if p.regions}
            parsed = ParsedDocument(format='pdf', parser=pdf.parser, segments=tuple(
                DocumentSegment(text=encoded[p.start:p.end].decode(), locator=f'page:{p.number}',
                    metadata={'page': str(p.number), 'extraction': p.extraction,
                              **({'unreadable': 'true'} if p.number in unreadable_pages else {}),
                              'width_points': str(p.width_points), 'height_points': str(p.height_points),
                              'rotation': str(p.rotation),
                              **({'ocr_reading_order': p.reading_order.model_dump_json()} if p.reading_order else {}),
                              **({'ocr_engine': p.ocr_engine} if p.ocr_engine else {}),
                              **({'layout_labels': p.labels.model_dump_json()} if p.labels else {}),
                              **({'tables': str(tables[p.number].proposed),
                                  'tables_unreadable': str(tables[p.number].unreadable)}
                                 if p.number in tables and tables[p.number].proposed else {}),
                              **({'section': _section(p.section)} if p.section else {})},
                    regions=tuple(DocumentTextRegion(text=r.text, box=r.box, score=r.score,
                        block=r.block, paragraph=r.paragraph, line=r.line, label=r.label,
                        start=r.start - p.start, end=r.end - p.start,
                        provider_index=r.provider_index, reading_column=r.reading_column,
                        coordinate_space=p.region_geometry) for r in p.regions),
                    table_cells=tables[p.number].cells if p.number in tables else ())
                for p in pdf.pages if not p.empty),
                metadata={'empty_pages': ','.join(str(p.number) for p in pdf.pages if p.empty),
                          **({'unreadable_pages': ','.join(map(str, unreadable_pages))} if unreadable_pages else {}),
                          **({'outline': pdf.outline} if pdf.outline != 'none' else {})})
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
