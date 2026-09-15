"""Host-selected OCR with bounded, per-import PDF extraction choices."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

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
    #: One of the languages the host offers, as Tesseract names installed data (``deu``,
    #: ``jpn+eng``). Unset reads with the host's own language, as every request did before.
    language: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_]{1,32}(\+[A-Za-z0-9_]{1,32}){0,7}$')

    @model_serializer(mode='wrap')
    def _omit_unset_language(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        # A selection that names no language keeps the identity every existing import and job has.
        result: dict[str, object] = handler(self)
        if self.language is None:
            result.pop('language', None)
        return result


@dataclass(frozen=True)
class DocumentOcr:
    """Trusted host recognizer; clients can select extraction behavior, and a language from the
    ones the host offers, only."""
    engine: OcrEngine = field(repr=False)
    dpi: int = 150
    #: The languages a request may choose, besides the engine's own.
    languages: tuple[str, ...] = ()
    #: How the host reads one of those languages.
    engine_for: Callable[[str], OcrEngine] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        OcrPdfOptions(dpi=self.dpi)
        if not callable(getattr(self.engine, 'recognize', None)):
            raise ValueError('document OCR requires a recognizer')
        if self.languages and self.engine_for is None:
            raise ValueError('document OCR that offers languages must say how to read them')

    def available(self) -> bool:
        return find_spec('pypdf') is not None and find_spec('pypdfium2') is not None

    def parser(self, selection: PdfOcrSelection) -> SelectedDocumentOcr:
        choice = PdfOcrSelection.model_validate(selection.model_dump())
        # A language the host does not offer is refused whatever else this host has installed.
        if choice.language is not None and (choice.language not in self.languages or self.engine_for is None):
            offered = ', '.join(self.languages) or 'none'
            raise InvalidInput(f'OCR language {choice.language!r} is not offered by this server; offered: {offered}')
        if not self.available():
            raise InvalidInput('document OCR requires the installed pdf-ocr extra')
        engine = self.engine if choice.language is None or self.engine_for is None else self.engine_for(choice.language)
        options = OcrPdfOptions(mode=choice.mode, reading_order=choice.reading_order, dpi=self.dpi)
        return SelectedDocumentOcr(BuiltinDocumentParser(pdf_parser=OcrPdfParser(engine, options=options)),
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
            'reading_orders': ['provider', 'columns_ltr', 'columns_rtl'],
            **({'languages': list(config.languages)} if config is not None and config.languages else {})}
