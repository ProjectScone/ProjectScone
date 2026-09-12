"""OCR can recover text extraction failures without relaxing PDF structure checks."""
from io import BytesIO

import pytest

pypdf = pytest.importorskip('pypdf')
pytest.importorskip('pypdfium2')

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion._pdf_worker import extract
from scone_memory.ingestion.pdf import PdfLimits, PypdfParser
from scone_memory.ingestion.pdf_ocr import OcrPdfParser
from .test_pdf_ingestion import pdf_bytes
from .test_pdf_ocr import ObservedOcr


def blank_pdf():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def test_valid_page_without_a_content_stream_reaches_ocr():
    engine = ObservedOcr()
    result = await OcrPdfParser(engine).parse(blank_pdf())
    assert engine.calls == 1
    assert result.pages[0].extraction == 'ocr'
    assert result.text == 'Café uses Polaris'


async def test_text_only_blank_pdf_is_reported_as_empty_instead_of_corrupt():
    with pytest.raises(InvalidInput, match='no extractable text'):
        await PypdfParser().parse(blank_pdf())


def test_only_the_failed_text_layer_is_replaced_with_an_empty_ocr_candidate(monkeypatch):
    original = pypdf.PageObject.extract_text

    def fail_second(self, *args, **kwargs):
        if self.page_number == 1:
            raise ValueError('Unsupported font layout fixture')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pypdf.PageObject, 'extract_text', fail_second)
    raw = pdf_bytes()
    with pytest.raises(InvalidInput, match='corrupt or uses unsupported'):
        extract(raw, PdfLimits(), allow_empty=True)
    result = extract(raw, PdfLimits(), allow_empty=True, allow_text_errors=True)
    assert result.pages[0].empty is False
    assert result.pages[1].empty is True
    assert result.text == 'Café calibration uses Polaris.\n\n'
    assert 'fallback' in result.parser


@pytest.mark.parametrize('data,limits,reason', [
    (pdf_bytes(password='locked'), PdfLimits(), 'encrypted'),
    (pdf_bytes(), PdfLimits(max_pages=1), 'page limit'),
    (pdf_bytes(), PdfLimits(max_text_bytes=5), 'byte limit'),
    (b'%PDF-1.7\ncorrupt', PdfLimits(), 'corrupt'),
], ids=['encrypted', 'page_limit', 'text_limit', 'corrupt'])
def test_ocr_fallback_keeps_document_validation_and_resource_limits(data, limits, reason):
    with pytest.raises(InvalidInput, match=reason):
        extract(data, limits, allow_empty=True, allow_text_errors=True)


@pytest.mark.parametrize('error', [MemoryError, RecursionError, pypdf.errors.LimitReachedError])
def test_ocr_fallback_does_not_suppress_extraction_resource_failures(monkeypatch, error):
    def exhaust(self, *args, **kwargs):
        raise error('bounded failure fixture')
    monkeypatch.setattr(pypdf.PageObject, 'extract_text', exhaust)
    with pytest.raises(InvalidInput):
        extract(pdf_bytes(), PdfLimits(), allow_empty=True, allow_text_errors=True)
