"""Optional OCR, without implicit model downloads or generative inference."""
from .tesseract import TesseractOcr
from .types import OcrEngine, OcrRegion, OcrResult

__all__ = ['TesseractOcr', 'OcrEngine', 'OcrRegion', 'OcrResult']
