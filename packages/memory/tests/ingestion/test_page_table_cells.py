"""A page's inferred tables as document table cells, cited to the page's text."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.types import DocumentSegment, DocumentTextRegion
from scone_memory.ingestion.formats.table_types import validate_tables
from scone_memory.ingestion.pdf import ParsedPdf, PdfLimits, PdfPage
from scone_memory.ingestion.pdf_ocr import assemble_ocr_pdf
from scone_memory.ocr.table_cells import page_table_cells
from scone_memory.ocr.types import OcrRegion, OcrResult

from .test_pdf_layout import placed, table_page


def grid_regions(label='table'):
    made = []
    for row, texts in enumerate((("Item", "Q1", "Q2"), ("Widgets", "10", "12"), ("Gadgets", "7", "9"), ("Total", "38", "")), 1):
        top = 0.2 + row * 0.05
        for column, (text, left, right) in enumerate(zip(texts, (0.1, 0.4, 0.65), (0.28, 0.5, 0.75))):
            if text:
                width = right if not (row == 4 and column == 1) else 0.5  # the total's number fills its column
                made.append(OcrRegion(text=text, box=(left, top, width, top + 0.03), line=row, label=label))
    return made


def test_a_page_s_grid_becomes_cells_cited_to_its_text_and_validated_as_the_office_readers_are():
    parsed = ParsedPdf(text='', parser='fixture', pages=(PdfPage(number=1, start=0, end=0, width_points=600.,
                                                                  height_points=800., rotation=0, empty=True),))
    output = assemble_ocr_pdf(parsed, {1: OcrResult(engine='fixture', width=600, height=800, regions=tuple(grid_regions()))},
                              PdfLimits())
    page = output.pages[0]
    assert {r.label for r in page.regions} == {'table'}
    text = output.text.encode()[page.start:page.end]
    tables = page_table_cells(page.regions, [(r.start - page.start, r.end - page.start) for r in page.regions], text, 'page:1')
    assert tables.proposed == 1 and tables.unreadable == 0 and len(tables.cells) == 11
    by_place = {(c.row, c.column): c for c in tables.cells}
    assert by_place[(0, 0)].text == 'Item' and by_place[(3, 1)].text == '38' and by_place[(3, 1)].column_span == 1
    assert all(text[c.start:c.end].decode() == c.text for c in tables.cells), "every cell is the page's own text"
    assert by_place[(0, 0)].table_locator == 'page:1/table:1' and by_place[(0, 0)].locator == 'page:1/table:1/cell:0,0'
    segment = DocumentSegment(text=text.decode(), locator='page:1', table_cells=tables.cells,
                              regions=tuple(DocumentTextRegion(text=r.text, box=r.box, line=r.line, label=r.label,
                                                               start=r.start - page.start, end=r.end - page.start)
                                            for r in page.regions))
    validate_tables((segment,))


def test_cells_come_in_the_page_s_text_order_and_a_table_cited_wrongly_is_left_out():
    regions = grid_regions()
    # A recognizer that reads a table's columns as blocks (Tesseract does)
    # gives the page's text column by column.
    order = sorted(range(len(regions)), key=lambda i: (regions[i].box[0], regions[i].box[1]))
    spans, parts, offset = [(0, 0)] * len(regions), [], 0
    for index in order:
        parts.append(regions[index].text)
        spans[index] = (offset, offset + len(regions[index].text))
        offset += len(regions[index].text) + 1
    page_text = ' '.join(parts).encode()
    tables = page_table_cells(regions, spans, page_text, 'page:2')
    assert tables.proposed == 1 and tables.unreadable == 0 and len(tables.cells) == 11
    assert [c.start for c in tables.cells] == sorted(c.start for c in tables.cells), "cells in the text's order"
    assert [(c.row, c.column) for c in tables.cells[:4]] == [(0, 0), (1, 0), (2, 0), (3, 0)], \
        "the column read first comes first, each cell keeping its place"
    assert all(page_text[c.start:c.end].decode() == c.text for c in tables.cells)
    validate_tables((DocumentSegment(text=page_text.decode(), locator='page:2', table_cells=tables.cells),))
    # The same grid, but a cell's span would cite words that are not the cell's.
    shuffled = list(spans)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    tables = page_table_cells(regions, shuffled, page_text, 'page:2')
    assert tables.proposed == 1 and tables.unreadable == 1 and tables.cells == (), "a table cited wrongly is no table"
    assert page_table_cells((), (), b'', 'page:3').proposed == 0
    for label in ('paragraph', None):
        tables = page_table_cells(grid_regions(label), spans, page_text, 'page:2')
        assert tables.proposed == 0 and tables.cells == (), "a page whose labels name no table carries none, whatever the grid says"
    with pytest.raises(ValueError, match='span'):
        page_table_cells(regions, spans[:-1], page_text, 'page:2')


async def test_a_text_layer_page_s_table_reaches_the_document_as_cells():
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.formats.types import DocumentLimits
    rows = [("Revenue", "2,903", "6,854"), ("Cost of revenue", "1,650", "2,720"), ("Operations and support", "574", "718"),
            ("Sales and marketing", "1,263", "4,789"), ("Research and development", "587", "2,054"), ("Total costs", "4,074", "10,281")]
    parsed = await BuiltinDocumentParser().parse(placed(table_page(rows)), 'quarter.pdf', DocumentLimits())
    [segment] = parsed.segments
    assert segment.metadata['tables'] == '1' and segment.metadata['tables_unreadable'] == '0'
    encoded = segment.text.encode()
    assert all(encoded[c.start:c.end].decode() == c.text for c in segment.table_cells), "every cell is the page's own bytes"
    by_row = {}
    for cell in segment.table_cells:
        by_row.setdefault(cell.row, []).append(cell.text)
    assert by_row[max(by_row)] == ["Total costs", "4,074", "10,281"] and len(by_row) >= len(rows)
    assert [c.start for c in segment.table_cells] == sorted(c.start for c in segment.table_cells)
