"""Column recovery keeps original word observations and exact UTF-8 evidence."""
import pytest
import json
import asyncio

from scone_memory.ocr.types import OcrRegion, OcrResult


def columns():
    regions = [OcrRegion(text='Shared heading', box=(.1, .04, .9, .08), line=0)]
    for row in range(4):
        y = .2 + row * .08
        regions.extend((OcrRegion(text=f'Café left {row}', box=(.1, y, .43, y + .03), line=row + 1),
                        OcrRegion(text=f'Right {row}', box=(.57, y, .9, y + .03), line=row + 1)))
    regions.append(OcrRegion(text='Shared footer', box=(.1, .8, .9, .84), line=5))
    return tuple(regions)


@pytest.mark.parametrize('direction', ['ltr', 'rtl'])
def test_spanning_heading_and_footer_surround_column_order(direction):
    from scone_memory.ocr.layout import order_columns
    regions = columns()
    result = order_columns(regions, direction=direction)
    left, right = (1, 3, 5, 7), (2, 4, 6, 8)
    expected = (0, *(left + right if direction == 'ltr' else right + left), 9)
    assert result.indices == expected
    assert result.columns == (0, 1, 1, 1, 1, 2, 2, 2, 2, 0)
    assert result.receipt.columns == 2
    assert result.receipt.notes == ('geometry_inferred',)


def test_middle_spanning_text_prevents_unjustified_column_assignment():
    from scone_memory.ocr.layout import order_columns
    regions = (*columns(), OcrRegion(text='Middle spanning content', box=(.1, .3, .9, .35)))
    result = order_columns(regions)
    assert result.indices == tuple(range(len(regions)))
    assert result.receipt.columns == 0
    assert 'no_separating_gutter' in result.receipt.notes


def test_short_rows_or_word_spaces_are_not_enough_to_infer_columns():
    from scone_memory.ocr.layout import order_columns
    for regions in [columns()[1:5], (OcrRegion(text='one', box=(.1, .1, .4, .2)),)]:
        result = order_columns(regions)
        assert result.indices == tuple(range(len(regions)))
        assert result.receipt.columns == 0


def test_word_level_spanning_lines_do_not_block_body_column_detection():
    from scone_memory.ocr.layout import order_columns
    words = []
    for line in columns():
        tokens = line.text.split()
        x, top, end, bottom = line.box
        width = (end - x) / len(tokens)
        words.extend(OcrRegion(text=word, box=(x + n * width, top, x + (n + 1) * width - .01, bottom),
                               line=line.line) for n, word in enumerate(tokens))
    result = order_columns(words)
    text = ' '.join(words[i].text for i in result.indices)
    assert result.receipt.columns == 2
    assert text.index('Café left 3') < text.index('Right 0')
    assert sorted(result.indices) == list(range(len(words)))


def test_pdf_ordering_records_provider_indices_and_new_utf8_spans():
    from scone_memory.ingestion.pdf import ParsedPdf, PdfPage, PdfLimits
    from scone_memory.ingestion.pdf_ocr import assemble_ocr_pdf
    empty = ParsedPdf(text='', parser='fixture', pages=(PdfPage(number=1, start=0, end=0,
        width_points=600., height_points=800., rotation=0, empty=True),))
    result = OcrResult(engine='fixture', width=600, height=800, regions=columns())
    parsed = assemble_ocr_pdf(empty, {1: result}, PdfLimits(), reading_order='columns_ltr')
    assert parsed.pages[0].reading_order.columns == 2
    assert [r.provider_index for r in parsed.pages[0].regions] == [0, 1, 3, 5, 7, 2, 4, 6, 8, 9]
    assert parsed.text.index('Café left 3') < parsed.text.index('Right 0')
    for region in parsed.pages[0].regions:
        assert parsed.text.encode()[region.start:region.end] == region.text.encode()
        assert region.box == result.regions[region.provider_index].box


def test_provider_default_preserves_old_serialized_receipts():
    from scone_memory.ingestion.pdf import PdfPage, PdfTextRegion
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions
    region = PdfTextRegion(text='word', box=(.1, .1, .2, .2), start=0, end=4)
    assert 'provider_index' not in region.model_dump()
    assert 'reading_column' not in region.model_dump()
    page = PdfPage(number=1, start=0, end=4, width_points=600., height_points=800., rotation=0, empty=False)
    assert 'reading_order' not in page.model_dump()
    assert 'reading_order' not in OcrPdfOptions().model_dump()


class ColumnOcr:
    def __init__(self):
        self.calls = 0

    async def recognize(self, image, *, max_pixels, **kwargs):
        from scone_memory.ocr.tesseract import png_dimensions
        self.calls += 1
        width, height = png_dimensions(image, max_pixels)
        return OcrResult(engine='fixture', width=width, height=height, regions=columns())


async def test_ordered_pdf_and_generic_document_keep_v3_provenance_over_http(tmp_path):
    pytest.importorskip('pypdfium2')
    import httpx
    from scone_memory.api import create_app
    from scone_memory.ingestion import ingest_pdf, ingest_document, document_provenance
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser
    from .test_document_workflow import open_memory
    from .test_pdf_ingestion import pdf_bytes
    memory = await open_memory(tmp_path)
    memory.chunk_target = 30
    parser = OcrPdfParser(ColumnOcr(), options=OcrPdfOptions(mode='all_pages', reading_order='columns_ltr'))
    raw = pdf_bytes(pages=('',))
    try:
        pdf = await ingest_pdf(memory, 'alpha', raw, parser=parser)
        _, encoded = await memory.attachment('alpha', pdf.manifest.attachment_id)
        manifest = json.loads(encoded)
        assert manifest['schema_version'] == 3
        assert manifest['pages'][0]['reading_order']['columns'] == 2
        generic = await ingest_document(memory, 'alpha', raw, filename='source.pdf',
                                        parser=BuiltinDocumentParser(pdf_parser=parser))
        _, encoded = await memory.attachment('alpha', generic.manifest.attachment_id)
        assert json.loads(encoded)['schema_version'] == 3
        chunks = await memory.documents.chunks_of('alpha', generic.added.episode_id)
        assert len(chunks) > 1
        selected = await document_provenance(memory, 'alpha', generic.added.episode_id, chunk_id=chunks[-1].chunk_id)
        assert len(selected.segments[0].regions) < len(columns())
        assert selected.segments[0].regions[-1].provider_index == 9
        app = create_app(memory, {'reader': 'alpha'}, roles={'reader': 'read'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                     headers={'Authorization': 'Bearer reader'}) as client:
            response = await client.get(f'/v1/episodes/{generic.added.episode_id}/document')
            assert response.status_code == 200
            value = response.json()
            segment = value['segments'][0]
            assert json.loads(segment['metadata']['ocr_reading_order'])['columns'] == 2
            assert [r['provider_index'] for r in segment['regions']] == [0, 1, 3, 5, 7, 2, 4, 6, 8, 9]
    finally:
        await memory.close()


async def test_column_strategy_binds_page_checkpoints_and_survives_restart(tmp_path):
    pytest.importorskip('pypdfium2')
    from scone_memory.agents import WorkflowError
    from scone_memory.ingestion import pdf_provenance
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions
    from .test_document_workflow import HeldEmbedder, open_memory
    from .test_pdf_ocr_workflow import retain, workflow
    held = HeldEmbedder()
    memory = await open_memory(tmp_path, held)
    original = await retain(memory)
    ocr = ColumnOcr()
    options = OcrPdfOptions(mode='all_pages', reading_order='columns_ltr')
    job = workflow(memory, tmp_path, ocr, options=options)
    args = dict(space='alpha', attachment_id=original.attachment_id)
    task = asyncio.create_task(job.run('scan', **args))
    try:
        await asyncio.wait_for(held.entered.wait(), 10)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()
    memory = await open_memory(tmp_path)
    wrong = workflow(memory, tmp_path, ocr)
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await wrong.run('scan', **args)
    finally:
        wrong.close()
    job = workflow(memory, tmp_path, ocr, options=options)
    try:
        result = await job.run('scan', **args)
        assert result.reused_pages == (1, 2) and ocr.calls == 2
        evidence = await pdf_provenance(memory, 'alpha', result.added.episode_id)
        assert all(page.reading_order.columns == 2 for page in evidence.pages)
        assert [r.provider_index for r in evidence.pages[0].regions] == [0, 1, 3, 5, 7, 2, 4, 6, 8, 9]
    finally:
        job.close()
        await memory.close()


@pytest.mark.parametrize('damage', ['duplicate', 'missing_receipt', 'wrong_column', 'bad_notes'])
def test_forged_reading_order_is_rejected(damage):
    from scone_memory.core.errors import InvalidInput
    from scone_memory.ingestion.pdf import ParsedPdf, PdfPage, PdfLimits, validate_pdf
    from scone_memory.ingestion.pdf_ocr import assemble_ocr_pdf
    empty = ParsedPdf(text='', parser='fixture', pages=(PdfPage(number=1, start=0, end=0,
        width_points=600., height_points=800., rotation=0, empty=True),))
    result = OcrResult(engine='fixture', width=600, height=800, regions=columns())
    parsed = assemble_ocr_pdf(empty, {1: result}, PdfLimits(), reading_order='columns_ltr')
    page = parsed.pages[0]
    if damage == 'missing_receipt':
        page = page.model_copy(update={'reading_order': None})
    elif damage == 'bad_notes':
        page = page.model_copy(update={'reading_order': page.reading_order.model_copy(update={'notes': ()})})
    else:
        regions = list(page.regions)
        regions[1] = regions[1].model_copy(update={'provider_index': 0} if damage == 'duplicate' else {'reading_column': 7})
        page = page.model_copy(update={'regions': tuple(regions)})
    with pytest.raises(InvalidInput, match='reading order'):
        validate_pdf(parsed.model_copy(update={'pages': (page,)}), PdfLimits())


async def test_synchronous_ordering_cannot_return_success_after_document_deadline(monkeypatch):
    from scone_memory.core.errors import InvalidInput
    from scone_memory.ingestion import pdf_ocr
    from scone_memory.ingestion.pdf import ParsedPdf, PdfPage, PdfLimits
    parser = pdf_ocr.OcrPdfParser(ColumnOcr(), options=pdf_ocr.OcrPdfOptions(reading_order='columns_ltr'))
    empty = ParsedPdf(text='', parser='fixture', pages=(PdfPage(number=1, start=0, end=0,
        width_points=600., height_points=800., rotation=0, empty=True),))
    async def inspect(*args):
        return empty
    async def recognize(*args):
        return OcrResult(engine='fixture', width=600, height=800, regions=columns())
    monkeypatch.setattr(parser, 'inspect', inspect)
    monkeypatch.setattr(parser, '_recognize', recognize)
    assemble = pdf_ocr.assemble_ocr_pdf
    clock = [0.]
    def delayed(*args, **kwargs):
        clock[0] = 2.
        return assemble(*args, **kwargs)
    monkeypatch.setattr(pdf_ocr, 'assemble_ocr_pdf', delayed)
    monkeypatch.setattr(pdf_ocr.time, 'monotonic', lambda: clock[0])
    with pytest.raises(InvalidInput, match='wall time'):
        await parser._parse(b'fixture', PdfLimits(timeout_seconds=1.))
