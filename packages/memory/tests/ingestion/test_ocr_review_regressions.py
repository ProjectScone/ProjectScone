"""Reproductions verified independently from the September 11 OCR review."""
from io import BytesIO
import hashlib
import json
import sys
import time

import pytest
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput, NotFound
from scone_memory.ingestion import ingest_pdf, ingest_document
from scone_memory.ingestion.pdf import ParsedPdf, PdfLimits, PdfPage, PdfTextRegion
from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser
from scone_memory.ocr.process import run_bounded
from scone_memory.ocr.tesseract import HEADER, parse_tsv
from .test_pdf_ingestion import pdf_bytes


async def test_child_environment_excludes_unrelated_server_credentials(monkeypatch):
    monkeypatch.setenv('SCONE_TEST_DUMMY_CREDENTIAL', 'synthetic-only')
    monkeypatch.setenv('TESSDATA_PREFIX', '/configured/local/languages')
    script = ('import os,json; print(json.dumps({'
              '"credential":os.environ.get("SCONE_TEST_DUMMY_CREDENTIAL"),'
              '"languages":os.environ.get("TESSDATA_PREFIX")}))')
    output = await run_bounded([sys.executable, '-c', script], b'', timeout=5., max_output=1000)
    assert json.loads(output) == {'credential': None, 'languages': '/configured/local/languages'}


async def test_pdf_renderer_ignores_modules_in_working_directory(tmp_path, monkeypatch):
    pytest.importorskip('pypdfium2')
    marker = tmp_path / 'untrusted-import'
    (tmp_path / 'pypdfium2.py').write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError("wrong import")\n')
    monkeypatch.chdir(tmp_path)
    class EmptyRecognizer:
        async def recognize(self, image, **kwargs):
            from scone_memory.ocr import OcrResult
            from scone_memory.ocr.tesseract import png_dimensions
            width, height = png_dimensions(image, kwargs['max_pixels'])
            return OcrResult(engine='test', width=width, height=height, regions=())
    parser = OcrPdfParser(EmptyRecognizer(), options=OcrPdfOptions(mode='all_pages'))
    await parser._recognize(pdf_bytes(pages=('Polaris',)), 1, time.monotonic() + 10)
    assert not marker.exists()


def test_pdf_rendering_records_actual_dpi_for_recognition():
    pytest.importorskip('pypdfium2')
    from scone_memory.ingestion._pdf_render_worker import RenderOptions, render
    png = render(pdf_bytes(pages=('Polaris',)), RenderOptions(page=1, dpi=150, max_pixels=20_000_000))
    with Image.open(BytesIO(png)) as image:
        assert image.info.get('dpi') == pytest.approx((150., 150.), abs=.02)


@pytest.mark.parametrize('separator', ['\u2028', '\u2029', '\x85', '\x0b', '\x0c'])
def test_tsv_unicode_inside_a_word_does_not_create_an_extra_row(separator):
    text = f'alpha{separator}beta'
    data = (HEADER + '\n5\t1\t1\t1\t1\t1\t1\t1\t10\t10\t90\t' + text + '\n').encode()
    parsed = parse_tsv(data, width=100, height=100, max_regions=10, engine='test')
    assert len(parsed.regions) == 1
    assert parsed.regions[0].text == text


async def test_provider_timeout_is_distinct_from_document_deadline():
    pytest.importorskip('pypdfium2')
    class ProviderTimeout:
        async def recognize(self, image, **kwargs):
            raise TimeoutError('provider exhausted its own budget')
    with pytest.raises(InvalidInput, match='recognition.*timed out'):
        await OcrPdfParser(ProviderTimeout(), options=OcrPdfOptions(mode='all_pages')).parse(pdf_bytes())


async def test_invalid_recognizer_output_is_a_classified_input_error():
    pytest.importorskip('pypdfium2')
    class InvalidRecognizer:
        async def recognize(self, image, **kwargs):
            return {'not': 'an OCR result'}
    with pytest.raises(InvalidInput, match='OCR.*invalid'):
        await OcrPdfParser(InvalidRecognizer(), options=OcrPdfOptions(mode='all_pages')).parse(pdf_bytes())


async def test_expired_render_budget_is_reported_as_deadline_exhaustion():
    pytest.importorskip('pypdfium2')
    with pytest.raises(InvalidInput, match='wall time'):
        await OcrPdfParser(None)._recognize(pdf_bytes(), 1, time.monotonic() - 1)


class DenseOcr:
    async def parse(self, data, limits):
        text = ' '.join(['word'] * 100)
        regions = tuple(PdfTextRegion(text='word', box=(.1, .1, .9, .9), start=i * 5, end=i * 5 + 4)
                        for i in range(100))
        return ParsedPdf(text=text, parser='test', pages=(PdfPage(number=1, start=0, end=len(text),
            width_points=100., height_points=100., rotation=0, empty=False, extraction='ocr',
            ocr_engine='test', regions=regions),))


@pytest.mark.parametrize('kind', ['pdf', 'json'])
async def test_oversized_manifest_is_rejected_before_retaining_original(kind):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    memory.max_attachment_bytes = 1024
    data = b'%PDF-synthetic' if kind == 'pdf' else json.dumps({str(i): 'word' for i in range(40)}).encode()
    try:
        with pytest.raises(InvalidInput, match='manifest.*byte limit'):
            if kind == 'pdf':
                await ingest_pdf(memory, 'review', data, parser=DenseOcr())
            else:
                await ingest_document(memory, 'review', data, filename='fixture.json')
        with pytest.raises(NotFound):
            await memory.attachment('review', hashlib.sha256(data).hexdigest())
    finally:
        await memory.close()
