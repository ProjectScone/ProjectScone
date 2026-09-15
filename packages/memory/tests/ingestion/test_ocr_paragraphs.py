"""Recognized words kept together as the lines and paragraphs they were read in.

Tesseract reports each word with its block, paragraph and line. The
paragraph number was read and dropped, and an image became one segment
per word: two short paragraphs from a real Tesseract run were stored as
31 segments, ``The\\n\\nharbour\\n\\ncrane``, so a chunker looking for
paragraphs found one per word and a sentence window found no sentence.
Here words on a line are joined by a space, lines by a line break and
paragraphs by a blank line, on a PDF page and in an image, and each word
keeps its own box and span. An engine that reports no layout at all is
read as before, one segment per region.
"""

from __future__ import annotations

import io
import json
import shutil

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.media import ImageDocumentParser
from scone_memory.ocr.tesseract import parse_tsv
from scone_memory.ocr.types import OcrRegion, OcrResult

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"


def word(text, block, paragraph, line, left, top):
    return OcrRegion(text=text, box=(left, top, left + 0.05, top + 0.02), score=0.9, block=block,
                     paragraph=paragraph, line=line)


# Two paragraphs in one block, the first over two lines, then a second block.
LAID_OUT = (word("The", 1, 1, 0, 0.1, 0.1), word("crane", 1, 1, 0, 0.2, 0.1),
            word("rusted.", 1, 1, 1, 0.1, 0.13),
            word("Paid", 1, 2, 2, 0.1, 0.3), word("in", 1, 2, 2, 0.2, 0.3), word("June.", 1, 2, 2, 0.3, 0.3),
            word("Footer", 2, 1, 3, 0.1, 0.9))


class Engine:
    def __init__(self, regions):
        self.regions = regions

    async def recognize(self, image, *, max_pixels=20_000_000, max_regions=10_000, timeout_seconds=30.0):
        with Image.open(io.BytesIO(image)) as frame:
            return OcrResult(engine="fixture", width=frame.width, height=frame.height, regions=self.regions)


def png(width=200, height=120):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_tesseract_rows_keep_their_paragraph_number():
    rows = ("5\t1\t1\t1\t1\t1\t5\t10\t20\t10\t85\tThe\n" "5\t1\t1\t2\t1\t1\t5\t30\t20\t10\t85\tPaid\n"
            "5\t1\t2\t1\t1\t1\t5\t45\t20\t4\t85\tFooter\n")
    result = parse_tsv((HEADER + rows).encode(), width=100, height=50, max_regions=10, engine="test")
    assert [(region.block, region.paragraph) for region in result.regions] == [(1, 1), (1, 2), (2, 1)]


async def test_an_image_is_one_segment_per_paragraph_with_every_word_as_a_region():
    parsed = await ImageDocumentParser(Engine(LAID_OUT)).parse(png(), "scan.png")
    assert [segment.text for segment in parsed.segments] == ["The crane\nrusted.", "Paid in June.", "Footer"]
    assert [segment.locator for segment in parsed.segments] == [
        "frame:1/paragraph:1", "frame:1/paragraph:2", "frame:1/paragraph:3"]
    first = parsed.segments[0]
    assert [region.text for region in first.regions] == ["The", "crane", "rusted."]
    for segment in parsed.segments:
        for region in segment.regions:
            assert segment.text.encode()[region.start:region.end].decode() == region.text
    assert json.loads(first.metadata["box"]) == [0.1, 0.1, 0.25, 0.15], "the paragraph's box holds its words"
    assert first.metadata["words"] == "3" and first.metadata["block"] == "1"


async def test_an_engine_that_reports_no_layout_is_read_one_region_at_a_time():
    flat = (OcrRegion(text="first line of text", box=(0.1, 0.1, 0.9, 0.2)),
            OcrRegion(text="second line of text", box=(0.1, 0.3, 0.9, 0.4)))
    parsed = await ImageDocumentParser(Engine(flat)).parse(png(), "scan.png")
    assert [segment.locator for segment in parsed.segments] == ["frame:1/region:1", "frame:1/region:2"]
    assert [segment.text for segment in parsed.segments] == ["first line of text", "second line of text"]


async def test_the_text_limit_still_counts_every_segment():
    with pytest.raises(InvalidInput):
        from scone_memory.ingestion.formats.types import DocumentLimits

        await ImageDocumentParser(Engine(LAID_OUT)).parse(png(), "scan.png", DocumentLimits(max_text_bytes=20))


def test_a_region_without_a_paragraph_serializes_as_it_did():
    from scone_memory.ingestion.formats.types import DocumentTextRegion

    plain = DocumentTextRegion(text="Café", box=(0.1, 0.2, 0.3, 0.4), start=0, end=5)
    assert "paragraph" not in plain.model_dump() and "paragraph" not in OcrRegion(text="x", box=(0.1, 0.1, 0.2, 0.2)).model_dump()
    laid = DocumentTextRegion(text="Café", box=(0.1, 0.2, 0.3, 0.4), start=0, end=5, paragraph=2)
    assert laid.model_dump()["paragraph"] == 2
    assert DocumentTextRegion.model_validate_json(laid.model_dump_json()) == laid


def test_a_pdf_page_puts_a_blank_line_between_paragraphs_and_a_break_between_lines():
    from scone_memory.ingestion.pdf_ocr import _page_text

    result = OcrResult(engine="fixture", width=100, height=100, regions=LAID_OUT)
    text, regions = _page_text(result, 0, 2_000_000)
    assert text == "The crane\nrusted.\n\nPaid in June.\n\nFooter"
    assert all(text.encode()[region.start:region.end].decode() == region.text for region in regions)


async def test_a_real_tesseract_reads_two_paragraphs_as_paragraphs():
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract is not installed")
    from scone_memory.ocr.tesseract import TesseractOcr

    image = Image.new("RGB", (1400, 520), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=34)
    draw.text((40, 40), "The harbour crane survey found rust on the jib.", font=font, fill="black")
    draw.text((40, 90), "The slew ring needed grease before June.", font=font, fill="black")
    draw.text((40, 260), "Invoice 20931 was paid by the harbour office.", font=font, fill="black")
    draw.text((40, 310), "Payment arrived on the ninth of June.", font=font, fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    parsed = await ImageDocumentParser(TesseractOcr()).parse(output.getvalue(), "scan.png")
    assert len(parsed.segments) == 2, [segment.text for segment in parsed.segments]
    assert parsed.segments[0].text == "The harbour crane survey found rust on the jib.\nThe slew ring needed grease before June."


def test_table_analysis_reads_regions_with_and_without_a_paragraph():
    from scone_memory.ocr.tables import infer_tables

    laid = infer_tables(LAID_OUT)
    plain = infer_tables(tuple(OcrRegion(text=region.text, box=region.box, score=region.score, block=region.block,
                                         line=region.line) for region in LAID_OUT))
    assert laid is not None and plain is not None


async def test_a_pdf_file_import_keeps_each_word_s_paragraph():
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser

    from .test_pdf_ocr import scanned_pdf

    parsed = await BuiltinDocumentParser(pdf_parser=OcrPdfParser(Engine(LAID_OUT))).parse(scanned_pdf(), "scan.pdf")
    [page] = parsed.segments
    assert page.text == "The crane\nrusted.\n\nPaid in June.\n\nFooter"
    assert [region.paragraph for region in page.regions] == [1, 1, 1, 2, 2, 2, 1]


def test_a_new_column_that_starts_a_new_paragraph_keeps_its_blank_line():
    from scone_memory.ingestion.pdf_ocr import _page_text
    from scone_memory.ocr.layout import order_columns

    left = [OcrRegion(text=f"Left {row}", box=(0.1, 0.1 + row * 0.05, 0.4, 0.13 + row * 0.05), block=1, paragraph=1,
                      line=row) for row in range(4)]
    right = [OcrRegion(text=f"Right {row}", box=(0.6, 0.1 + row * 0.05, 0.9, 0.13 + row * 0.05), block=2,
                       paragraph=1, line=4 + row) for row in range(4)]
    result = OcrResult(engine="fixture", width=100, height=100, regions=tuple(left + right))
    order = order_columns(result.regions, direction="ltr")
    text, _ = _page_text(result, 0, 2_000_000, order)
    assert "Left 3\n\nRight 0" in text and "Left 0\nLeft 1" in text


async def test_an_engine_that_reports_only_paragraphs_is_read_by_paragraph():
    regions = (OcrRegion(text="first", box=(0.1, 0.1, 0.2, 0.2), paragraph=1),
               OcrRegion(text="second", box=(0.1, 0.3, 0.2, 0.4), paragraph=2))
    parsed = await ImageDocumentParser(Engine(regions)).parse(png(), "scan.png")
    assert [segment.locator for segment in parsed.segments] == ["frame:1/paragraph:1", "frame:1/paragraph:2"]


async def test_a_blank_word_is_not_a_region_and_leaves_no_double_space():
    regions = (word("The", 1, 1, 0, 0.1, 0.1), OcrRegion(text="  ", box=(0.15, 0.1, 0.18, 0.12), block=1, paragraph=1),
               word("crane", 1, 1, 0, 0.2, 0.1))
    parsed = await ImageDocumentParser(Engine(regions)).parse(png(), "scan.png")
    assert [segment.text for segment in parsed.segments] == ["The crane"]
    assert [region.text for region in parsed.segments[0].regions] == ["The", "crane"]
