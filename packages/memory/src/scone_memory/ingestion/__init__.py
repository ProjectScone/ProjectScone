"""Source processing: chunking, distillation, the consolidation worker."""

from .documents import PdfIngested, PdfProvenance, ingest_pdf, pdf_provenance
from .pdf_ocr import OcrPdfOptions, OcrPdfParser
from .pdf import ParsedPdf, PdfLimits, PdfPage, PdfParser, PdfTextRegion, PypdfParser

__all__ = ['OcrPdfOptions', 'OcrPdfParser', 'ParsedPdf', 'PdfIngested', 'PdfProvenance', 'PdfLimits', 'PdfPage', 'PdfParser', 'PdfTextRegion', 'PypdfParser',
           'ingest_pdf', 'pdf_provenance']

from .images import ImageAttribute, ImageContext, ImageEntity, ImageIngested, ImageMatch, ImageProvenance, ImageRecall, ingest_image, image_provenance, recall_images

__all__ += ["ImageAttribute", "ImageContext", "ImageEntity", "ImageIngested", "ImageMatch", "ImageProvenance", "ImageRecall", "ingest_image", "image_provenance", "recall_images"]

from .image_html import HtmlImageContext, image_contexts_from_html
__all__ += ["HtmlImageContext", "image_contexts_from_html"]

from .files import DocumentIngested, DocumentProvenance, document_provenance, ingest_document
from .document_source import DocumentSource
from .formats.registry import BuiltinDocumentParser, DocumentParser
from .formats.types import DocumentLimits, DocumentSegment, DocumentTextRegion, ParsedDocument
from .formats.table_types import DocumentTableCell, DocumentTableHeader, DocumentTableContext

__all__ += ['DocumentIngested', 'DocumentProvenance', 'DocumentSource', 'document_provenance', 'ingest_document',
            'BuiltinDocumentParser', 'DocumentParser', 'DocumentLimits', 'DocumentSegment', 'DocumentTextRegion', 'ParsedDocument',
            'DocumentIngestionWorkflow', 'PdfOcrWorkflow', 'PdfOcrIngested', 'DocumentTableCell', 'DocumentTableHeader', 'DocumentTableContext']


def __getattr__(name: str) -> object:
    if name == 'DocumentIngestionWorkflow':
        from .file_workflow import DocumentIngestionWorkflow
        return DocumentIngestionWorkflow
    if name in {'PdfOcrWorkflow', 'PdfOcrIngested'}:
        from .pdf_ocr_workflow import PdfOcrIngested, PdfOcrWorkflow
        return PdfOcrWorkflow if name == 'PdfOcrWorkflow' else PdfOcrIngested
    raise AttributeError(name)
