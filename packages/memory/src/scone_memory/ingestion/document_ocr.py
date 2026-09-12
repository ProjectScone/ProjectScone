"""Host-selected OCR with bounded, per-import PDF extraction choices."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..core.errors import InvalidInput
from ..ocr.layout import ReadingMode
from ..ocr.types import OcrEngine
from .extraction_checkpoint import ExtractionCheckpoints, checkpoint_dispatch_allowed
from .formats.registry import BuiltinDocumentParser, extension
from .formats.types import DocumentLimits, ParsedDocument
from .pdf_ocr import OcrPdfOptions, OcrPdfParser


class PdfOcrSelection(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    mode: Literal['missing_text', 'all_pages']
    reading_order: ReadingMode


@dataclass(frozen=True)
class DocumentOcr:
    """Trusted host recognizer; clients can select extraction behavior only."""
    engine: OcrEngine = field(repr=False)
    dpi: int = 150

    def __post_init__(self) -> None:
        OcrPdfOptions(dpi=self.dpi)
        if not callable(getattr(self.engine, 'recognize', None)):
            raise ValueError('document OCR requires a recognizer')

    def available(self) -> bool:
        return find_spec('pypdf') is not None and find_spec('pypdfium2') is not None

    def parser(self, selection: PdfOcrSelection) -> SelectedDocumentOcr:
        choice = PdfOcrSelection.model_validate(selection.model_dump())
        if not self.available():
            raise InvalidInput('document OCR requires the installed pdf-ocr extra')
        options = OcrPdfOptions(mode=choice.mode, reading_order=choice.reading_order, dpi=self.dpi)
        return SelectedDocumentOcr(BuiltinDocumentParser(pdf_parser=OcrPdfParser(self.engine, options=options)),
                                   choice, self.dpi)


@dataclass(frozen=True)
class SelectedDocumentOcr:
    parser: BuiltinDocumentParser
    selection: PdfOcrSelection
    dpi: int

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
        return await self._parse(data, filename, limits)

    async def parse_checkpointed(self, data: bytes, filename: str, limits: DocumentLimits,
                                 checkpoints: ExtractionCheckpoints) -> ParsedDocument:
        return await self._parse(data, filename, limits, checkpoints)

    async def _parse(self, data: bytes, filename: str, limits: DocumentLimits,
                     checkpoints: ExtractionCheckpoints | None = None) -> ParsedDocument:
        if extension(filename) != '.pdf':
            raise InvalidInput('PDF OCR can only be selected for a PDF document')
        parsed = (await self.parser.parse_checkpointed(data, filename, limits, checkpoints)
                  if checkpoints is not None and checkpoint_dispatch_allowed(self.parser) else await self.parser.parse(data, filename, limits))
        metadata = {**parsed.metadata, 'pdf_ocr': json.dumps(
            {**self.selection.model_dump(), 'dpi': self.dpi}, sort_keys=True, separators=(',', ':'))}
        return ParsedDocument.model_validate({**parsed.model_dump(), 'metadata': metadata})


def ocr_choices(config: DocumentOcr | None) -> dict[str, object]:
    return {'available': config is not None and config.available(),
            'modes': ['missing_text', 'all_pages'],
            'reading_orders': ['provider', 'columns_ltr', 'columns_rtl']}
