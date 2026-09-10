from io import BytesIO
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip('pypdf')
pytest.importorskip('pypdfium2')

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import ingest_pdf, pdf_provenance
from scone_memory.ingestion.pdf import PdfLimits, validate_pdf
from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser
from scone_memory.ocr import OcrRegion, OcrResult
from scone_memory.ocr.tesseract import TesseractOcr, png_dimensions
from test_pdf_ingestion import pdf_bytes


def scanned_pdf():
    image = Image.new('RGB', (1200, 400), 'white')
    ImageDraw.Draw(image).text((50, 50), 'ProjectScone calibration uses Polaris 12345',
        font=ImageFont.load_default(size=36), fill='black')
    output = BytesIO()
    image.save(output, format='PDF', resolution=150.)
    return output.getvalue()


class ObservedOcr:
    def __init__(self, text='Café uses Polaris'):
        self.calls = 0
        self.text = text

    async def recognize(self, image, *, max_pixels, max_regions, timeout_seconds):
        self.calls += 1
        width, height = png_dimensions(image, max_pixels)
        return OcrResult(engine='fixture', width=width, height=height,
            regions=(OcrRegion(text=self.text, box=(.1, .1, .9, .2), score=.8),))


async def test_scanned_source_has_exact_unicode_spans_and_original_coordinates():
    parser = OcrPdfParser(ObservedOcr())
    parsed = await parser.parse(scanned_pdf(), PdfLimits())
    page = parsed.pages[0]
    assert parsed.text == 'Café uses Polaris' and page.extraction == 'ocr'
    assert page.regions[0].end == len(parsed.text.encode())
    assert page.regions[0].box == (.1, .1, .9, .2)
    assert page.region_geometry == 'normalized_displayed_page_top_left'
    validate_pdf(parsed, PdfLimits())


async def test_default_mode_preserves_native_text_and_only_recognizes_missing_pages():
    engine = ObservedOcr()
    parsed = await OcrPdfParser(engine).parse(pdf_bytes(pages=('Native Polaris.', '')), PdfLimits())
    assert engine.calls == 1
    assert [page.extraction for page in parsed.pages] == ['text_layer', 'ocr']
    assert parsed.text.startswith('Native Polaris.\n\nCafé')
    region = parsed.pages[1].regions[0]
    assert parsed.text.encode()[region.start:region.end].decode() == region.text


async def test_explicit_all_pages_mode_recognizes_even_existing_text_layers():
    engine = ObservedOcr()
    parsed = await OcrPdfParser(engine, options=OcrPdfOptions(mode='all_pages')).parse(pdf_bytes(), PdfLimits())
    assert engine.calls == 2 and all(page.extraction == 'ocr' for page in parsed.pages)


async def test_pixel_limit_is_checked_before_recognition():
    engine = ObservedOcr()
    with pytest.raises(InvalidInput, match='pixel'):
        await OcrPdfParser(engine, options=OcrPdfOptions(max_pixels=100)).parse(scanned_pdf(), PdfLimits())
    assert engine.calls == 0


async def test_corrupted_ocr_span_is_not_valid_provenance():
    parsed = await OcrPdfParser(ObservedOcr()).parse(scanned_pdf(), PdfLimits())
    page = parsed.pages[0]
    region = page.regions[0].model_copy(update={'text': 'fabricated'})
    altered = parsed.model_copy(update={'pages': (page.model_copy(update={'regions': (region,)}),)})
    with pytest.raises(InvalidInput, match='region'):
        validate_pdf(altered, PdfLimits())


@pytest.mark.skipif(shutil.which('tesseract') is None, reason='Tesseract executable is optional')
async def test_real_scan_recall_resolves_to_retained_original_and_recognized_region():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    raw = scanned_pdf()
    try:
        result = await ingest_pdf(memory, 'alpha', raw, parser=OcrPdfParser(TesseractOcr(page_segmentation=6)))
        episode = await memory.episode('alpha', result.added.episode_id)
        assert 'Polaris' in episode.content
        assert episode.metadata['pdf_coverage'] == 'ocr'
        recalled = await memory.recall('alpha', 'Polaris')
        item = next(item for item in recalled.items if item.episode_id == result.added.episode_id)
        provenance = await pdf_provenance(memory, 'alpha', result.added.episode_id, chunk_id=item.chunk_id)
        assert any(region.text == 'Polaris' for region in provenance.pages[0].regions)
        assert (await memory.attachment('alpha', provenance.original.attachment_id))[1] == raw
    finally:
        await memory.close()


async def test_native_pdf_manifest_keeps_its_original_v1_bytes_and_identity():
    import hashlib
    import json
    from scone_memory.ingestion import PypdfParser
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    raw = pdf_bytes()
    parsed = await PypdfParser().parse(raw, PdfLimits())
    old_pages = [page.model_dump(exclude={'extraction', 'region_geometry', 'regions', 'ocr_engine'})
        for page in parsed.pages]
    old_manifest = json.dumps({'schema_version': 1, 'offset_unit': 'extracted_text_utf8_bytes',
        'geometry': 'unrotated_media_box_points', 'original_sha256': hashlib.sha256(raw).hexdigest(),
        'text_sha256': hashlib.sha256(parsed.text.encode()).hexdigest(), 'parser': parsed.parser,
        'pages': old_pages}, separators=(',', ':'), ensure_ascii=False).encode()
    try:
        result = await ingest_pdf(memory, 'alpha', raw)
        assert (await memory.attachment('alpha', result.manifest.attachment_id))[1] == old_manifest
    finally:
        await memory.close()


async def test_ocr_document_deadline_cancels_a_slow_provider_before_writing():
    import asyncio
    class Slow(ObservedOcr):
        async def recognize(self, image, **kwargs):
            await asyncio.sleep(30)
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        with pytest.raises(InvalidInput, match='wall time'):
            await ingest_pdf(memory, 'alpha', scanned_pdf(), parser=OcrPdfParser(Slow()),
                limits=PdfLimits(timeout_seconds=1.5))
        assert (await memory.documents.counts('alpha')).episodes == 0
    finally:
        await memory.close()


async def test_all_pages_does_not_limit_discarded_native_text():
    engine = ObservedOcr()
    parsed = await OcrPdfParser(engine, options=OcrPdfOptions(mode='all_pages')).parse(
        pdf_bytes(pages=('discarded hidden text ' * 200,)), PdfLimits(max_text_bytes=100))
    assert parsed.text == 'Café uses Polaris' and engine.calls == 1


async def test_region_frame_uses_rotated_cropped_displayed_page():
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(BytesIO(scanned_pdf()))
    page = reader.pages[0]
    page.cropbox.upper_right = (288, 192)
    page.rotate(90)
    writer = PdfWriter()
    writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    sizes = []
    class ObserveSize(ObservedOcr):
        async def recognize(self, image, **kwargs):
            sizes.append(png_dimensions(image, kwargs['max_pixels']))
            return await super().recognize(image, **kwargs)
    parsed = await OcrPdfParser(ObserveSize()).parse(output.getvalue(), PdfLimits())
    assert sizes == [(400, 600)]
    assert parsed.pages[0].rotation == 90
    assert parsed.pages[0].width_points == 576
    assert parsed.pages[0].regions[0].box == (.1, .1, .9, .2)


async def test_nonempty_native_pages_and_blank_ocr_pages_remain_partial():
    class Blank:
        async def recognize(self, image, **kwargs):
            width, height = png_dimensions(image, kwargs['max_pixels'])
            return OcrResult(engine='blank-fixture', width=width, height=height, regions=())
    parsed = await OcrPdfParser(Blank()).parse(pdf_bytes(pages=('Polaris.', '')), PdfLimits())
    assert parsed.pages[1].empty and parsed.pages[1].extraction == 'ocr'
    with pytest.raises(InvalidInput, match='empty'):
        await OcrPdfParser(Blank()).parse(scanned_pdf(), PdfLimits())
