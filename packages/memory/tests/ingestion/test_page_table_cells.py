"""A page's inferred tables as document table cells, cited to the page's text."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.types import DocumentSegment, DocumentTextRegion
from scone_memory.ingestion.formats.table_types import DocumentTableHeader, validate_tables
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
    # The first full row is the header: no number in it, numbers under Q1 and Q2.
    assert tables.headed == 1 and all(by_place[(0, c)].is_header for c in range(3)) and not by_place[(1, 0)].is_header
    assert by_place[(2, 1)].headers == (DocumentTableHeader(locator='page:1/table:1/cell:0,1', text='Q1', association='column'),)
    assert by_place[(3, 0)].headers[0].text == 'Item' and by_place[(0, 1)].headers == ()
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


def test_a_header_is_read_from_the_table_s_shape_and_only_then():
    from scone_memory.ocr.table_cells import header_row
    from scone_memory.ocr.tables import infer_tables

    def cell(text, row, left, right):
        top = 0.1 + row * 0.05
        return OcrRegion(text=text, box=(left, top, right, top + 0.03), label='table')
    def rows(*lines):
        return [cell(text, row, left, right) for row, line in enumerate(lines)
                for text, (left, right) in zip(line, ((0.1, 0.28), (0.4, 0.5), (0.65, 0.75)))]
    years = rows(("Item", "2021", "2022"), ("Revenue", "2,903", "6,854"), ("Costs", "1,650", "2,720"), ("Total", "4,553", "9,574"))
    assert header_row(infer_tables(years).tables[0]) == 0, "a bare year heads a column, as over a financial statement"
    continued = rows(("6", "Lakshmi Mittal", "60"), ("7", "Ingvar Kamprad", "81"), ("8", "Larry Page", "45"))
    assert header_row(infer_tables(continued).tables[0]) is None, "a first row of numbers is a row"
    words = rows(("Symbol", "Meaning", "Section"), ("RR", "Ridge Rider", "two"), ("IPD", "Prisoners' Dilemma", "three"), ("GAN", "Adversarial Network", "four"))
    assert header_row(infer_tables(words).tables[0]) is None, "words alone tell no header from a first row"
    titled = [cell("Results by quarter", 0, 0.1, 0.75)] + [cell(text, row + 1, left, right) for row, line in enumerate(
        (("Item", "Q1", "Q2"), ("Widgets", "10", "12"), ("Gadgets", "7", "9"), ("Total", "17", "21")))
        for text, (left, right) in zip(line, ((0.1, 0.28), (0.4, 0.5), (0.65, 0.75)))]
    [table] = infer_tables(titled).tables
    assert header_row(table) == 1, "the title across the table is not the header; the first full row is"


async def test_a_pdf_table_s_columns_are_named_for_the_table_query():
    from types import SimpleNamespace
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.formats.types import DocumentLimits
    from scone_memory.retrieval.table_query import Condition, TableQueryArgs, answer_from, tables_from
    ops = [(72, 740, "Results by quarter, in millions", 10), (72, 720, "Item", 10), (300, 720, "Q1", 10), (420, 720, "Q2", 10)]
    for i, (label, first, second) in enumerate((("Revenue", "2,903", "6,854"), ("Cost of revenue", "1,650", "2,720"),
                                                ("Sales and marketing", "1,263", "4,789"), ("Total costs", "4,074", "10,281"))):
        y = 700 - 14 * i
        ops += [(72, y, label, 10), (300, y, first, 10), (420, y, second, 10)]
    parsed = await BuiltinDocumentParser().parse(placed(ops), 'quarter.pdf', DocumentLimits())
    [segment] = parsed.segments
    assert segment.metadata['header_basis'] == 'pdf_first_row' and segment.metadata['tables_headed'] == '1'
    [table] = tables_from(SimpleNamespace(segments=parsed.segments))
    assert table.columns == ('Item', 'Q1', 'Q2') and table.basis == 'pdf_first_row', "the title above the header is not a row"
    assert len(table.rows) == 4 and [row.cells['Item'].text for row in table.rows][-1] == 'Total costs'
    answer = answer_from((table,), TableQueryArgs(operation='sum', column='Q2',
                                                  where=(Condition(column='Item', op='contains', value='revenue'),)))
    assert answer.value == '9574' and {q.text for q in answer.cells} >= {'6,854', '2,720'}
    # The page without a header row (the fixture from the layout tests) names no column.
    plain = await BuiltinDocumentParser().parse(placed(table_page([("Revenue", "2,903", "6,854"), ("Costs", "1,650", "2,720"),
                                                                   ("Total", "4,553", "9,574")])), 'plain.pdf', DocumentLimits())
    assert 'header_basis' not in plain.segments[0].metadata and not any(c.is_header for c in plain.segments[0].table_cells)


def test_a_statement_s_values_are_numbers_and_its_years_over_a_blank_column_are_the_header():
    from scone_memory.ocr.table_cells import _is_number, header_row
    from scone_memory.ocr.tables import infer_tables
    assert all(_is_number(t) for t in ('(1,133)', '$ (1,133)', '—', '-', '100.0 %', '12%', '$ 2,903', '(0.06)'))
    assert not any(_is_number(t) for t in ('2021', '(1,133', '()', 'Revenue', '', '— 5'))
    # Years over a blank label column head the grid; a row of values does not.
    def line(text, left, right, row):
        return OcrRegion(text=text, box=(left, .1 + row * .04, right, .125 + row * .04), label='table')
    rows = [('Revenue', '$ 2,903', '$ 6,854'), ('Loss from operations', '(1,524)', '(482)'), ('Other', '—', '1,710'),
            ('Margin', '52.0 %', '58.7 %')]
    regions = [line('2021', .55, .6, 0), line('2022', .8, .85, 0)]
    for number, (label, first, second) in enumerate(rows, 1):
        regions += [line(label, .02, .3, number), line(first, .5, .6, number), line(second, .75, .85, number)]
    [table] = infer_tables(regions).tables
    assert table.rows == 5 and header_row(table) == 0
    [table] = infer_tables(regions[2:]).tables
    assert header_row(table) is None, "a first row of values is no header"


async def test_a_statement_page_s_signs_captions_and_losses_reach_the_table_query():
    import re
    from types import SimpleNamespace
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.formats.types import DocumentLimits
    from scone_memory.retrieval.table_query import Condition, TableQueryArgs, answer_from, tables_from
    # The page as a filing sets it: a caption over the value columns, the
    # years over a blank label column, a currency sign at each column's
    # left edge on the first and last rows, far from its right-aligned
    # number, a section's name on a row of its own, losses in parentheses
    # and a dash where there is no amount.
    ops = [(72, 760, "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS", 10), (330, 740, "Three Months Ended March 31,", 10),
           (360, 726, "2021", 10), (470, 726, "2022", 10)]
    rows = [("Revenue", "2,903", "6,854", True), ("Costs and expenses", None, None, False), ("Cost of revenue", "1,710", "4,026", False),
            ("Operations and support", "423", "574", False), ("Loss from operations", "(1,524)", "(482)", False),
            ("Other income", "—", "1,710", False), ("Net loss", "(108)", "(5,930)", True)]
    for i, (label, first, second, signed) in enumerate(rows):
        y = 712 - 14 * i
        ops.append((72, y, label, 10))
        if first is not None:
            ops += [(345, y, first, 10), (455, y, second, 10)]
        if signed:
            ops += [(300, y, "$", 10), (410, y, "$", 10)]
    parsed = await BuiltinDocumentParser().parse(placed(ops), 'operations.pdf', DocumentLimits())
    [segment] = parsed.segments
    validate_tables(parsed.segments)
    assert segment.metadata['tables'] == '1' and segment.metadata['tables_headed'] == '1'
    encoded = segment.text.encode()
    assert all(encoded[c.start:c.end].decode() == c.text for c in segment.table_cells), "every cell is the page's own bytes"
    signed = sorted(c.text for c in segment.table_cells if c.text.startswith('$'))
    assert len(signed) == 4 and all(re.fullmatch(r'\$ {2,}[\d,()]+', text) for text in signed), signed
    assert [c.text for c in segment.table_cells if c.is_header] == ['2021', '2022']
    [table] = tables_from(SimpleNamespace(segments=parsed.segments))
    assert table.columns == ('column 1', '2021', '2022') and table.basis == 'pdf_first_row'
    assert [row.cells['column 1'].text for row in table.rows] == [label for label, *_ in rows], "the caption above the header is not a row"
    lost = answer_from((table,), TableQueryArgs(operation='sum', column='2022',
                                               where=(Condition(column='column 1', op='contains', value='loss'),)))
    assert lost.value == '-6412' and {' '.join(c.text.split()) for c in lost.cells} == {'(482)', '$ (5,930)'}
    revenue = answer_from((table,), TableQueryArgs(operation='sum', column='2021',
                                                  where=(Condition(column='column 1', op='==', value='Revenue'),)))
    assert revenue.value == '2903'
