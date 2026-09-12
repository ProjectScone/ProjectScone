"""Optional OCR, without implicit model downloads or generative inference."""
from .tables import TableCandidate, TableCell, TableLayout, infer_tables
from .tesseract import TesseractOcr
from .types import OcrEngine, OcrRegion, OcrResult

__all__ = ['TesseractOcr', 'OcrEngine', 'OcrRegion', 'OcrResult',
           'TableCandidate', 'TableCell', 'TableLayout', 'infer_tables']
