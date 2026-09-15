"""A layout engine beside OCR, or the rules without one: every recognized page's regions labelled."""
import json
import os
import struct
import sys
import zlib

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.document_ocr import DocumentOcr, ocr_choices
from scone_memory.ingestion.pdf import ParsedPdf, PdfLimits, PdfPage
from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser, assemble_ocr_pdf
from scone_memory.ocr.layout_json import JsonLayoutEngine
from scone_memory.ocr.types import LayoutRegion, LayoutResult, OcrRegion, OcrResult


def png(width=200, height=120) -> bytes:
    raw = b''.join(b'\x00' + b'\xff' * (width * 3) for _ in range(height))
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def words(text, top, height=0.03, key=0, left=0.1):
    made, x = [], left
    for word in text.split():
        w = 0.01 * len(word)
        made.append(OcrRegion(text=word, box=(x, top, x + w, top + height), block=0, paragraph=0, line=key))
        x += w + 0.006
    return made


def scanned(*, pages=1):
    parsed = ParsedPdf(text='\n\n'.join('' for _ in range(pages)) if pages > 1 else '', parser='fixture', pages=tuple(
        PdfPage(number=n, start=0, end=0, width_points=600., height_points=800., rotation=0, empty=True)
        for n in range(1, pages + 1)))
    regions = (*words("Annual Report", 0.05, height=0.05, key=1), *words("The year was long and full of robots.", 0.2, key=2),
               *words("They worked in every warehouse we own.", 0.24, key=3),
               *words("Acme confidential", 0.94, height=0.02, key=4), *words("7", 0.97, height=0.02, key=5))
    result = OcrResult(engine='fixture', width=600, height=800, regions=regions)
    return parsed, {n: result for n in range(1, pages + 1)}


def test_a_recognized_page_s_regions_are_labelled_by_the_rules_and_running_lines_are_judged_across_pages():
    parsed, recognized = scanned(pages=3)
    output = assemble_ocr_pdf(parsed, recognized, PdfLimits())
    first, third = output.pages[0], output.pages[2]
    assert first.labels is not None and first.labels.source == 'inferred' and first.labels.strategy == 'labels-v1'
    by_text = {r.text: r.label for r in first.regions}
    assert by_text['Annual'] == 'title' and by_text['robots.'] == 'paragraph' and by_text['7'] == 'page_number'
    assert by_text['confidential'] == 'paragraph', "the first page a line appears on is where it may be a title"
    assert set(first.labels.rules) == {'title', 'page_number'}
    on_third = {r.text: r.label for r in third.regions}
    assert on_third['confidential'] == 'footer', "a line at the bottom of three pages is a running footer after its first"
    assert on_third['Annual'] == 'header' and 'running' in third.labels.rules, "and so is one at the top"
    assert 'label' in first.regions[0].model_dump() and 'label' not in OcrRegion(text='x', box=(0, 0, 1, 1)).model_dump(), \
        "a region without a label serializes as it always did"


def test_an_engine_s_boxes_label_the_page_instead_of_the_rules():
    parsed, recognized = scanned()
    layout = LayoutResult(engine='pp-structure', width=600, height=800, dropped=1, regions=(
        LayoutRegion(label='caption', box=(0.0, 0.0, 1.0, 0.12), score=0.9, order=0),
        LayoutRegion(label='paragraph', box=(0.0, 0.15, 1.0, 0.3), order=1)))
    output = assemble_ocr_pdf(parsed, recognized, PdfLimits(), layouts={1: layout})
    page = output.pages[0]
    assert page.labels is not None and page.labels.source == 'engine' and page.labels.engine == 'pp-structure'
    by_text = {r.text: r.label for r in page.regions}
    assert by_text['Annual'] == 'caption' and by_text['robots.'] == 'paragraph' and by_text['7'] is None
    assert page.labels.unlabeled == 3 and page.labels.dropped == 1 and page.labels.rules == ()


def engine_script(tmp_path, body):
    script = tmp_path / 'layout.py'
    script.write_text(f"#!{sys.executable}\nimport json, struct, sys\nimage = sys.stdin.buffer.read()\n"
                      "width, height = struct.unpack('>II', image[16:24])\n" + body, encoding='utf-8')
    os.chmod(script, 0o755)
    return str(script)


async def test_a_json_lines_executable_is_a_layout_engine_and_nothing_it_says_is_trusted_beyond_the_contract(tmp_path):
    good = engine_script(tmp_path, "print(json.dumps({'width': width, 'height': height}))\n"
                         "print(json.dumps({'label': 'paragraph_title', 'box': [10, 10, 190, 30], 'score': 0.93, 'order': 0}))\n"
                         "print(json.dumps({'label': 'Text', 'box': [10, 40, 190, 110], 'score': 0.99, 'order': 1}))\n"
                         "print(json.dumps({'label': 'poster', 'box': [0, 0, 5, 5]}))\n")
    engine = JsonLayoutEngine(good, name='fixture-layout')
    seen = await engine.analyse(png(200, 120))
    assert seen.engine == 'fixture-layout' and (seen.width, seen.height) == (200, 120) and seen.dropped == 1
    assert [r.label for r in seen.regions] == ['heading', 'paragraph'] and seen.regions[0].box == (0.05, 10 / 120, 0.95, 0.25)
    assert seen.regions[0].score == 0.93 and seen.regions[1].order == 1
    for body, why in ((f"print(json.dumps({{'width': 7, 'height': height}}))\n", 'did not name the page'),
                      ("print(json.dumps({'width': width, 'height': height}))\nprint('not json')\n", 'not JSON'),
                      ("print(json.dumps({'width': width, 'height': height}))\nprint(json.dumps({'label': 'text', 'box': [0, 0, 500, 5]}))\n", 'not a rectangle'),
                      ("print(json.dumps({'width': width, 'height': height}))\nprint(json.dumps({'box': [0, 0, 5, 5]}))\n", 'no label'),
                      ("pass\n", 'answered nothing')):
        with pytest.raises(InvalidInput, match=why):
            await JsonLayoutEngine(engine_script(tmp_path, body)).analyse(png(200, 120))
    with pytest.raises(InvalidInput, match='absolute executable'):
        JsonLayoutEngine('layout.py')


class FakeLayout:
    name = 'fake-layout'

    def __init__(self):
        self.images = []

    async def analyse(self, image, *, max_pixels=20_000_000, max_regions=10_000, timeout_seconds=30.0):
        from scone_memory.ocr.tesseract import png_dimensions

        width, height = png_dimensions(image, max_pixels)
        self.images.append(len(image))
        return LayoutResult(engine=self.name, width=width, height=height,
                            regions=(LayoutRegion(label='paragraph', box=(0.0, 0.0, 1.0, 1.0)),))


class WordOcr:
    async def recognize(self, image, **kwargs):
        from scone_memory.ocr.tesseract import png_dimensions

        width, height = png_dimensions(image, kwargs['max_pixels'])
        return OcrResult(engine='fixture', width=width, height=height,
                         regions=(OcrRegion(text='Polaris', box=(0.1, 0.1, 0.4, 0.14)),))


async def test_the_parser_runs_the_layout_engine_beside_ocr_and_keeps_its_answer_in_the_page_receipts():
    pytest.importorskip('pypdfium2')
    from tests.ingestion.test_pdf_ingestion import pdf_bytes

    layout = FakeLayout()
    parser = OcrPdfParser(WordOcr(), options=OcrPdfOptions(mode='all_pages'), layout=layout)
    parsed = await parser.parse(pdf_bytes(pages=('Polaris',)), PdfLimits(timeout_seconds=60.))
    page = parsed.pages[0]
    assert page.labels is not None and page.labels.source == 'engine' and page.labels.engine == 'fake-layout'
    assert page.regions[0].label == 'paragraph' and layout.images, "the engine saw the rendered page"
    plain = OcrPdfParser(WordOcr(), options=OcrPdfOptions(mode='all_pages'))
    assert (await plain.parse(pdf_bytes(pages=('Polaris',)), PdfLimits(timeout_seconds=60.))).pages[0].labels.source == 'inferred'
    from scone_memory.ingestion.pdf_ocr import _checkpoint_binding

    empty = ParsedPdf(text='', parser='fixture', pages=(PdfPage(number=1, start=0, end=0, width_points=600.,
                                                               height_points=800., rotation=0, empty=True),))
    assert _checkpoint_binding(b'x', empty, parser.options, PdfLimits(), 'fake-layout') != \
        _checkpoint_binding(b'x', empty, parser.options, PdfLimits()), "a receipt made without an engine is not reused with one"


def test_the_host_says_whether_labels_come_from_an_engine_and_refuses_one_that_cannot_analyse():
    assert ocr_choices(None)['layout'] == 'inferred'
    assert ocr_choices(DocumentOcr(WordOcr()))['layout'] == 'inferred'
    assert ocr_choices(DocumentOcr(WordOcr(), layout=FakeLayout()))['layout'] == 'engine'
    with pytest.raises(ValueError, match='analyse'):
        DocumentOcr(WordOcr(), layout=object())  # type: ignore[arg-type]


def test_the_rules_labels_on_a_text_layer_do_not_move_the_checkpoint_binding():
    from scone_memory.ingestion.pdf_ocr import _checkpoint_binding
    from scone_memory.ocr.labels import LayoutLabels

    bare = PdfPage(number=1, start=0, end=5, width_points=600., height_points=800., rotation=0, empty=False)
    labelled = bare.model_copy(update={'labels': LayoutLabels(source='inferred', strategy='labels-v1', counts={'paragraph': 1})})
    one = ParsedPdf(text='hello', parser='fixture', pages=(bare,))
    two = ParsedPdf(text='hello', parser='fixture', pages=(labelled,))
    assert 'labels' not in bare.model_dump() and 'labels' in labelled.model_dump(), "an absent receipt is not written"
    assert _checkpoint_binding(b'x', one, OcrPdfOptions(), PdfLimits()) == _checkpoint_binding(b'x', two, OcrPdfOptions(), PdfLimits()), \
        "a receipt made before the rules read anything binds to the same inspection"
