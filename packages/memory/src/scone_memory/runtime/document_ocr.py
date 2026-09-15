"""Explicit installed Tesseract configuration for the standard local hosts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import InvalidInput
from ..ingestion.document_ocr import DocumentOcr
from ..ocr.tesseract import TesseractOcr

if TYPE_CHECKING:
    from .config import Settings


def build_document_ocr(settings: Settings) -> DocumentOcr | None:
    if settings.document_ocr_executable is None:
        return None
    path = Path(settings.document_ocr_executable)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('SCONE_DOCUMENT_OCR_EXECUTABLE must name an installed absolute executable')
    def engine_for(language: str) -> TesseractOcr:
        return TesseractOcr(executable=str(path), language=language, page_segmentation=settings.document_ocr_psm,
                            orientation=settings.document_ocr_orientation)
    try:
        engine = engine_for(settings.document_ocr_language)
        for language in settings.document_ocr_languages:
            engine_for(language)
    except InvalidInput as error:
        raise ValueError(str(error)) from error
    configured = DocumentOcr(engine, dpi=settings.document_ocr_dpi, languages=settings.document_ocr_languages,
                             engine_for=engine_for if settings.document_ocr_languages else None)
    if not configured.available():
        raise ValueError('document OCR requires the installed scone-memory[pdf-ocr] extra')
    return configured
