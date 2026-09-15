"""A parsed document rebuilt as Markdown, each line traced to the bytes it came from."""
from __future__ import annotations

import json

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser
from scone_memory.ingestion.formats.markdown_assembly import MAX_MARKDOWN_BYTES, assemble_markdown
from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.text import parse_text
from scone_memory.ingestion.formats.types import DocumentLimits, DocumentSegment, ParsedDocument
from .test_declared_structure import NUMBERING, STYLES, paragraph, word
from .test_office_formats import W, ooxml_archive, REL, R


def cell(text: str, *, header: bool = False, span: int = 1) -> str:
    properties = f'<w:tcPr><w:gridSpan w:val="{span}"/></w:tcPr>' if span > 1 else ''
    return f'<w:tc>{properties}<w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>'


def row(*cells: str, header: bool = False) -> str:
    return f'<w:tr>{"<w:trPr><w:tblHeader/></w:trPr>" if header else ""}{"".join(cells)}</w:tr>'


REPORT = ''.join([
    paragraph('Revenue', style='Berschrift1'),
    paragraph('By region', outline=1),
    paragraph('Sales grew in every region.'),
    paragraph('Europe', number=1, level='0'),
    paragraph('France', number=1, level='1'),
    paragraph('Asia', number=1, level='0'),
    paragraph('Collect', number=2),
    paragraph('Reconcile', number=2),
    paragraph('Table 1: Sales by region', style='Beschriftung'),
    '<w:tbl>' + row(cell('Region'), cell('Sales'), header=True) + row(cell('Europe'), cell('10'))
    + row(cell('Asia'), cell('20')) + '</w:tbl>',
    paragraph('End.'),
])

REPORT_MARKDOWN = '''# Revenue

## By region

Sales grew in every region.

- Europe
  - France
- Asia

1. Collect
2. Reconcile

*Table 1: Sales by region*

| Region | Sales |
| --- | --- |
| Europe | 10 |
| Asia | 20 |

End.'''


def extracted(parsed: ParsedDocument) -> bytes:
    return '\n\n'.join(segment.text for segment in parsed.segments).encode()


def assert_traced(parsed: ParsedDocument, result) -> None:
    """Every line lies in exactly one span, and every source names a segment and bytes inside it."""
    markdown = result.markdown.encode()
    content = extracted(parsed)
    starts: dict[str, list[tuple[int, int]]] = {}
    offset = 0
    for segment in parsed.segments:
        size = len(segment.text.encode())
        starts.setdefault(segment.locator, []).append((offset, offset + size))
        offset += size + 2
    assert '\r' not in result.markdown, 'a Markdown reader would count a carriage return as a line end'
    lines = result.markdown.split('\n')
    covered = [0] * len(lines)
    previous_end = 0
    for span in result.spans:
        assert previous_end <= span.start < span.end <= len(markdown)
        previous_end = span.end
        text = markdown[span.start:span.end].decode()
        assert text.count('\n') == span.last_line - span.first_line
        assert text == '\n'.join(lines[span.first_line - 1:span.last_line])
        for number in range(span.first_line, span.last_line + 1):
            covered[number - 1] += 1
        assert span.sources
        for source in span.sources:
            assert any(start <= source.start < source.end <= end for start, end in starts[source.locator])
            if not span.generated and span.kind != 'code':
                assert content[source.start:source.end].strip() == content[source.start:source.end]
    assert all(count == 1 if line else count <= 1 for line, count in zip(lines, covered)), 'each line in one span'


def test_word_document_rebuilds_with_heading_levels_nested_lists_a_caption_and_a_pipe_table() -> None:
    parsed = parse_office(word(REPORT, styles=STYLES, numbering=NUMBERING), 'report.docx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == REPORT_MARKDOWN
    assert_traced(parsed, result)
    record = result.record()
    assert record['blocks'] == {'heading': 2, 'paragraph': 2, 'list_item': 5, 'caption': 1, 'table': 1}
    assert record['structure_declared'] is True
    [table] = record['tables']
    assert table == {'locator': 'table:1', 'rendering': 'pipe', 'rows': 3, 'rows_written': 3, 'columns': 2,
                     'header_basis': 'declared', 'extra_header_rows': 0, 'spanning_cells': 0, 'line_breaks': 0,
                     'notes': '', 'caption': None, 'interleaved_segments': 0}
    assert record['bound'] == {'max_bytes': MAX_MARKDOWN_BYTES, 'cut': False, 'cut_at': None, 'segments_omitted': 0}
    heading = result.spans[0]
    assert (heading.kind, heading.first_line, heading.sources[0].locator) == ('heading', 1, 'paragraph:1')
    body_row = next(span for span in result.spans if span.kind == 'table_row' and 'Asia' in result.markdown.encode()[span.start:span.end].decode())
    [source] = body_row.sources
    assert extracted(parsed)[source.start:source.end] == 'Region: Asia\nSales: 20'.encode()
    delimiter = next(span for span in result.spans if span.kind == 'table_delimiter')
    assert delimiter.generated and delimiter.sources[0].locator == 'table:1/row:1'


def html(raw: str) -> ParsedDocument:
    return parse_text(raw.encode(), 'page.html', DocumentLimits())


def test_html_page_rebuilds_in_reading_order_with_code_and_a_captioned_table() -> None:
    parsed = html('<h1>Install</h1><p>Run the <code>setup</code> step.</p>'
                  '<ol><li>Download<ol><li>Mirror</li></ol><li>Verify</ol><ul><li>Linux</ul>'
                  '<figure><figcaption>Figure 1: Layout</figcaption></figure>'
                  '<pre>x = 1\n  ```\n  y = 2</pre>'
                  '<table><caption>Ports</caption><tr><th>Name</th><th>Port</th></tr><tr><td>web</td><td>80</td></tr></table>')
    result = assemble_markdown(parsed)
    assert result.markdown == '''# Install

Run the setup step.

1. Download
   1. Mirror
2. Verify

- Linux

*Figure 1: Layout*

````
x = 1
  ```
  y = 2
````

*Ports*

| Name | Port |
| --- | --- |
| web | 80 |'''
    assert_traced(parsed, result)
    assert result.record()['tables'][0]['caption'] == 'table:1/caption'


def test_text_that_looks_like_markdown_stays_text() -> None:
    parsed = html('<p># not a heading</p><p>1. not a list</p><p>- nor this</p><p>&gt; nor a quote</p>'
                  '<p>a | b *c* _d_ snake_case [e](f) &lt;g&gt; `h` \\i &amp;amp; R&amp;D</p><p>===</p>'
                  '<h2>Ends in #</h2>'
                  '<table><tr><th>Key</th></tr><tr><td>a|b<br>c</td></tr></table>')
    result = assemble_markdown(parsed)
    assert result.markdown.split('\n\n') == [
        '\\# not a heading', '1\\. not a list', '\\- nor this', '\\> nor a quote',
        'a \\| b \\*c\\* \\_d\\_ snake_case \\[e\\](f) \\<g> \\`h\\` \\\\i \\&amp; R&D',
        '\\===', '## Ends in \\#',
        '| Key |\n| --- |\n| a\\|b<br>c |',
    ]
    assert result.record()['tables'][0]['line_breaks'] == 1


def test_a_line_break_inside_a_block_stays_a_line_break_and_a_blank_line_starts_a_new_paragraph() -> None:
    parsed = ParsedDocument(format='txt', parser='test', segments=(
        DocumentSegment(text='  Line one  \nLine two\n\n   \nNext paragraph', locator='line:1'),))
    result = assemble_markdown(parsed)
    assert result.markdown == 'Line one\\\nLine two\n\nNext paragraph'
    assert [(span.first_line, span.last_line) for span in result.spans] == [(1, 2), (4, 4)]
    content = extracted(parsed)
    assert [content[s.sources[0].start:s.sources[0].end] for s in result.spans] == [
        b'Line one  \nLine two', b'Next paragraph']


def test_spanning_cells_are_said_so_before_the_table_not_flattened_silently() -> None:
    parsed = html('<table><tr><th rowspan="2">Region</th><th colspan="2">Sales</th></tr>'
                  '<tr><th>2025</th><th>2026</th></tr><tr><td>Europe</td><td>10</td><td>20</td></tr></table>')
    result = assemble_markdown(parsed)
    note, table = result.markdown.split('\n\n')
    assert table == '| Region | Sales |  |\n| --- | --- | --- |\n|  | 2025 | 2026 |\n| Europe | 10 | 20 |'
    assert note == ('> table:1: 2 cells span more than one row or column, which a pipe table cannot show; '
                    'each is written in its first slot and the slots it covers are left empty. '
                    '1 further header row is written as a body row.')
    [record] = result.record()['tables']
    assert (record['spanning_cells'], record['extra_header_rows'], record['columns']) == (2, 1, 3)
    assert result.spans[0].kind == 'table_note' and result.spans[0].generated
    assert_traced(parsed, result)


def test_a_table_without_declared_headers_gets_an_empty_header_row_not_its_first_row() -> None:
    parsed = parse_office(word('<w:tbl>' + row(cell('a'), cell('b')) + row(cell('c'), cell('d')) + '</w:tbl>'),
                          'grid.docx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == '|  |  |\n| --- | --- |\n| a | b |\n| c | d |'
    assert result.record()['tables'][0]['header_basis'] == 'none_declared'
    header = result.spans[0]
    assert header.kind == 'table_header' and header.generated
    assert [s.locator for s in header.sources] == ['table:1/row:1', 'table:1/row:2']


def test_delimited_rows_take_their_header_from_the_columns_the_reader_named() -> None:
    parsed = parse_text(b'name,team\nAda,core\n', 'people.csv', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == '| name | team |\n| --- | --- |\n| Ada | core |'
    assert result.record()['tables'][0]['header_basis'] == 'table_columns'
    assert_traced(parsed, result)
    broken = parse_text(b'"first\nname",team\nAda,core\n', 'people.csv', DocumentLimits())
    assert assemble_markdown(broken).markdown.startswith('| first<br>name | team |\n| --- | --- |\n')


def test_a_table_read_as_text_says_why_and_keeps_its_rows_as_paragraphs() -> None:
    parsed = html('<table><tr><td>outer<table><tr><td>inner</td></tr></table></td></tr></table>')
    result = assemble_markdown(parsed)
    blocks = result.markdown.split('\n\n')
    assert blocks[0] == '> table:1 was read as text, not as a grid (nested_table); its lines follow as paragraphs.'
    assert blocks[1:] == ['outer', 'inner']
    [record] = result.record()['tables']
    assert (record['rendering'], record['notes']) == ('text', 'nested_table')
    assert_traced(parsed, result)


def test_word_heading_deeper_than_markdown_allows_is_clamped_and_counted() -> None:
    parsed = parse_office(word(paragraph('Deep', outline=7) + paragraph('Shallow', outline=5)), 'deep.docx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == '###### Deep\n\n###### Shallow'
    assert result.record()['headings_clamped'] == 1


def test_a_list_that_skips_a_level_is_nested_one_step_and_counted() -> None:
    parsed = parse_office(word(paragraph('Top', number=2, level='1') + paragraph('Child', number=2, level='3')
                               + paragraph('Unknown kind', number=9)), 'skip.docx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == '- Top\n  - Child\n\n* Unknown kind'
    record = result.record()
    assert (record['list_levels_clamped'], record['list_kinds_unsaid']) == (2, 3)


def test_ordered_markers_widen_the_indent_their_children_need() -> None:
    items = ''.join(paragraph(f'Step {n}', number=2) for n in range(1, 11)) + paragraph('Detail', number=1, level='1')
    parsed = parse_office(word(items, numbering=NUMBERING), 'steps.docx', DocumentLimits())
    lines = assemble_markdown(parsed).markdown.split('\n')
    assert lines[-2:] == ['10. Step 10', '    - Detail']


def test_side_content_is_quoted_with_its_role_after_the_body() -> None:
    from .test_docx_parts import document, relation

    data = document('<w:p><w:r><w:t>Estimate.</w:t><w:footnoteReference w:id="2"/></w:r></w:p>',
                    relationships=relation('footnotes', '../notes.xml'), extra={
        'notes.xml': f'<w:footnotes xmlns:w="{W}"><w:footnote w:id="2"><w:p><w:r><w:t>Unaudited.</w:t></w:r></w:p></w:footnote></w:footnotes>'})
    parsed = parse_office(data, 'notes.docx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == 'Estimate.\n\n> footnote 2: Unaudited.'
    assert result.spans[1].kind == 'quote'


def test_the_byte_bound_cuts_between_blocks_and_between_table_rows_and_says_where() -> None:
    parsed = parse_office(word(REPORT, styles=STYLES, numbering=NUMBERING), 'report.docx', DocumentLimits())
    whole = assemble_markdown(parsed).markdown.encode()
    cut = whole.index(b'| Asia')
    result = assemble_markdown(parsed, max_bytes=cut + 3)
    assert result.markdown.encode() == whole[:cut - 1]
    record = result.record()
    content = extracted(parsed)
    assert record['bound']['cut'] is True
    assert content[record['bound']['cut_at']:].startswith('Region: Asia\nSales: 20'.encode())
    assert record['bound']['segments_omitted'] == 2
    assert record['tables'][0]['rows_written'] == 2
    tiny = assemble_markdown(parsed, max_bytes=10)
    assert tiny.markdown == '# Revenue' and tiny.record()['bound']['segments_omitted'] == len(parsed.segments) - 1
    nothing = assemble_markdown(parsed, max_bytes=1)
    assert nothing.markdown == '' and nothing.record()['bound']['cut_at'] == 0


@pytest.mark.parametrize('bound', [0, MAX_MARKDOWN_BYTES + 1, True])
def test_a_bound_outside_its_range_is_refused(bound) -> None:
    with pytest.raises(InvalidInput, match='markdown byte bound'):
        assemble_markdown(html('<p>x</p>'), max_bytes=bound)


def test_role_metadata_a_reader_cannot_have_meant_is_read_as_a_paragraph_and_counted() -> None:
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(
        DocumentSegment(text='Seven', locator='a', metadata={'block_role': 'heading', 'heading_level': 'seven'}),
        DocumentSegment(text='Item', locator='b', metadata={'block_role': 'list_item', 'list_level': '-1'}),
        DocumentSegment(text='Other', locator='c', metadata={'block_role': 'sidebar'}),
        DocumentSegment(text='Zero', locator='d', metadata={'block_role': 'heading', 'heading_level': '0'}),
    ))
    result = assemble_markdown(parsed)
    assert result.markdown == 'Seven\n\nItem\n\nOther\n\nZero'
    assert result.record()['roles_unreadable'] == 4


async def test_pdf_pages_declare_no_structure_and_become_paragraphs_traced_to_their_page() -> None:
    pytest.importorskip('pypdf')
    from .test_pdf_ingestion import pdf_bytes

    parsed = await BuiltinDocumentParser().parse(pdf_bytes(('# Revenue grew', 'Juniper ships on Friday.')), 'report.pdf')
    result = assemble_markdown(parsed)
    assert result.markdown == '\\# Revenue grew\n\nJuniper ships on Friday.'
    assert [s.sources[0].locator for s in result.spans] == ['page:1', 'page:2']
    record = result.record()
    assert record['structure_declared'] is False and record['blocks'] == {'paragraph': 2}
    assert_traced(parsed, result)


def test_record_is_json_and_names_the_offset_unit() -> None:
    record = assemble_markdown(html('<h2>A</h2><p>b</p>')).record()
    assert json.loads(json.dumps(record)) == record
    assert record['offset_unit'] == 'extracted_text_utf8_bytes'
    assert record['structure_declared'] is True, 'a heading is declared structure without any table'
    assert record['spans'][0] == {'kind': 'heading', 'markdown_start': 0, 'markdown_end': 4, 'first_line': 1,
                                  'last_line': 1, 'generated': False,
                                  'sources': [{'locator': 'line:1', 'start': 0, 'end': 1}]}


def test_a_carriage_return_is_a_line_break_in_text_and_in_a_cell() -> None:
    parsed = ParsedDocument(format='txt', parser='test', segments=(
        DocumentSegment(text='one\rtwo\r\n\r\nthree', locator='a'),))
    assert assemble_markdown(parsed).markdown == 'one\\\ntwo\n\nthree'
    table = html('<table><tr><th>K</th></tr><tr><td>a</td></tr></table>')
    last = table.segments[-1]
    [value] = last.table_cells
    changed = last.model_copy(update={'text': 'K: a\rb', 'table_cells': (value.model_copy(update={'text': 'a\rb', 'end': value.end + 2}),)})
    result = assemble_markdown(table.model_copy(update={'segments': (*table.segments[:-1], changed)}))
    assert result.markdown.endswith('| a<br>b |') and result.record()['tables'][0]['line_breaks'] == 1


def test_a_numbered_list_the_document_interrupts_carries_on_counting() -> None:
    body = (paragraph('One', number=2) + paragraph('Two', number=2) + paragraph('Aside.')
            + paragraph('Three', number=2) + paragraph('Other list', number=4))
    parsed = parse_office(word(body, numbering=NUMBERING), 'resume.docx', DocumentLimits())
    assert assemble_markdown(parsed).markdown == '1. One\n2. Two\n\nAside.\n\n3. Three\n\n1) Other list'


def test_a_table_note_header_and_delimiter_are_written_together_or_not_at_all() -> None:
    parsed = html('<p>Intro</p><table><tr><th colspan="2">Wide</th></tr><tr><td>a</td><td>b</td></tr></table>')
    whole = assemble_markdown(parsed).markdown
    delimiter_end = whole.index('| --- | --- |') + len('| --- | --- |')
    cut = assemble_markdown(parsed, max_bytes=delimiter_end - 1)
    assert cut.markdown == 'Intro'
    assert cut.record()['bound']['cut_at'] == len(b'Intro\n\n')
    assert cut.record()['tables'][0]['rows_written'] == 0
    fits = assemble_markdown(parsed, max_bytes=delimiter_end)
    assert fits.markdown == whole[:delimiter_end] and fits.record()['tables'][0]['rows_written'] == 1
    grid = assemble_markdown(parse_office(word('<w:tbl>' + row(cell('a')) + '</w:tbl>'), 'g.docx', DocumentLimits()))
    assert (grid.record()['tables'][0]['rows'], grid.record()['tables'][0]['rows_written']) == (1, 1)


def test_a_parsed_document_whose_cells_do_not_match_their_text_is_refused() -> None:
    table = html('<table><tr><th>K</th></tr><tr><td>a</td></tr></table>')
    last = table.segments[-1]
    [value] = last.table_cells
    forged = last.model_copy(update={'table_cells': (value.model_copy(update={'text': 'z'}),)})
    with pytest.raises(InvalidInput, match='does not match its text span'):
        assemble_markdown(table.model_copy(update={'segments': (*table.segments[:-1], forged)}))


def test_a_line_break_inside_a_list_item_is_indented_under_its_marker() -> None:
    result = assemble_markdown(html('<ol><li>Mix<br>then wait</li></ol>'))
    assert result.markdown == '1. Mix\\\n   then wait'


def test_a_nested_numbered_list_starts_again_under_each_parent() -> None:
    result = assemble_markdown(html('<ol><li>a<ol><li>x<li>y</ol><li>b<ol><li>z</ol></ol>'))
    assert result.markdown == '1. a\n   1. x\n   2. y\n2. b\n   1. z'


def test_column_names_wider_than_the_cells_widen_the_table() -> None:
    from scone_memory.ingestion.formats.table_types import DocumentTableCell

    cells = (DocumentTableCell(table_locator='t', locator='t/1', row=0, column=0, text='1', start=0, end=1),
             DocumentTableCell(table_locator='t', locator='t/2', row=0, column=1, text='2', start=2, end=3))
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(DocumentSegment(
        text='1 2', locator='row:1', table_cells=cells,
        metadata={'table_locator': 't', 'table_status': 'structured', 'table_columns': '["a", "b", "c"]'}),))
    assert assemble_markdown(parsed).markdown == '| a | b | c |\n| --- | --- | --- |\n| 1 | 2 |  |'


def test_a_paragraph_that_continues_an_html_list_item_stays_inside_that_item() -> None:
    parsed = html('<ul><li><p>First</p><p>More about<br>first</p></li><li>Second<ul><li>Child</li></ul>'
                  '<p>Tail of second</p></li></ul><p>After</p>')
    result = assemble_markdown(parsed)
    assert result.markdown == '- First\n\n  More about\\\n  first\n- Second\n  - Child\n\n  Tail of second\n\nAfter'
    assert [span.kind for span in result.spans] == [
        'list_item', 'list_continuation', 'list_item', 'list_item', 'list_continuation', 'paragraph']
    assert result.record()['blocks']['list_item'] == 3
    assert_traced(parsed, result)


def test_a_nested_list_after_a_continued_item_counts_from_one_again() -> None:
    result = assemble_markdown(html('<ol><li>a<ol><li>x</li></ol>more<ol><li>y</li></ol></li></ol>'))
    assert result.markdown == '1. a\n   1. x\n\n   more\n   1. y'


def test_a_table_the_document_interleaves_with_other_segments_is_written_whole_and_says_so() -> None:
    from .test_xlsx_tables import DEFINITION, workbook
    from .test_xlsx_tables import cell as sheet_cell, parse as parse_sheet, row as sheet_row

    rows = (sheet_row(4, sheet_cell('B4', 'Region') + sheet_cell('C4', 'Revenue') + sheet_cell('E4', 'note: audited'))
            + sheet_row(5, sheet_cell('B5', 'West') + sheet_cell('C5', '20') + sheet_cell('E5', 'q1'))
            + sheet_row(6, sheet_cell('B6', 'Total') + sheet_cell('C6', '20')))
    parsed = parse_sheet(workbook(rows, DEFINITION))
    result = assemble_markdown(parsed)
    assert result.markdown == ('> sheet:Sales/table:1: 2 segments the document puts between its rows are written after it.'
                               '\n\n| Region | Revenue |\n| --- | --- |\n| West | 20 |\n| Total | 20 |\n\nnote: audited\n\nq1')
    [table] = result.record()['tables']
    assert (table['rows'], table['rows_written'], table['header_basis'], table['interleaved_segments']) == (3, 3, 'declared', 2)
    assert_traced(parsed, result)
    west = result.markdown.index('| West | 20 |') + len('| West | 20 |')
    cut = assemble_markdown(parsed, max_bytes=west)
    assert cut.markdown == result.markdown[:west]
    bound = cut.record()['bound']
    assert bound['segments_omitted'] == 4, 'the side cells were never written, though they come before the cut row'
    assert extracted(parsed)[bound['cut_at']:].startswith(b'note: audited')
    one = html('<table><tr><th>A</th></tr><tr><td>1</td></tr></table><p>x</p><p>y</p>')
    assert assemble_markdown(one).record()['tables'][0]['interleaved_segments'] == 0


def test_a_caption_between_a_tables_rows_is_written_before_the_table() -> None:
    parsed = html('<table><tr><th>A</th></tr><caption>C</caption><tr><td>1</td></tr></table>')
    result = assemble_markdown(parsed)
    assert result.markdown == '*C*\n\n| A |\n| --- |\n| 1 |'
    [table] = result.record()['tables']
    assert (table['rows'], table['caption'], table['interleaved_segments']) == (2, 'table:1/caption', 0)
    assert_traced(parsed, result)


def test_a_bound_that_cuts_inside_a_paragraph_run_names_the_first_byte_not_written() -> None:
    parsed = ParsedDocument(format='txt', parser='test', segments=(
        DocumentSegment(text='First\n\nSecond', locator='a'), DocumentSegment(text='Third', locator='b')))
    bound = len('First\n\nSec')
    result = assemble_markdown(parsed, max_bytes=bound)
    assert result.markdown == 'First'
    assert result.record()['bound'] == {'max_bytes': bound, 'cut': True, 'cut_at': len(b'First\n\n'), 'segments_omitted': 2}


def test_code_a_heading_a_table_and_a_caption_inside_a_list_item_stay_inside_it() -> None:
    parsed = html('<ol>\n<li>\n<p>Install the package:</p>\n<pre><code>pip install x\n</code></pre>\n'
                  '<p>then check the version.</p>\n</li>\n<li>\n<p>Run it.</p>\n</li>\n</ol>\n')
    result = assemble_markdown(parsed)
    assert result.markdown == ('1. Install the package:\n\n   ```\n   pip install x\n   ```\n\n'
                               '   then check the version.\n2. Run it.')
    assert [span.kind for span in result.spans] == ['list_item', 'code', 'list_continuation', 'list_item']
    assert result.record()['blocks']['list_item'] == 2
    assert_traced(parsed, result)
    opened = html('<ul><li><pre>x\n\ny</pre></li><li><h2>T</h2></li>'
                  '<li><table><tr><th colspan="2">A</th></tr><tr><td>1</td><td>2</td></tr></table></li>'
                  '<li><figure><figcaption>F</figcaption></figure></li></ul><p>After</p>')
    result = assemble_markdown(opened)
    assert result.markdown == ('- ```\n  x\n\n  y\n  ```\n- ## T\n'
                               '- > table:1: 1 cell spans more than one row or column, which a pipe table cannot show; '
                               'each is written in its first slot and the slots it covers are left empty.\n\n'
                               '  | A |  |\n  | --- | --- |\n  | 1 | 2 |\n- *F*\n\nAfter')
    assert result.record()['blocks'] == {'code': 1, 'heading': 1, 'table': 1, 'caption': 1, 'list_item': 4,
                                         'paragraph': 1}
    assert_traced(opened, result)


def test_a_table_that_opens_a_list_item_is_written_with_its_marker_or_not_at_all() -> None:
    parsed = html('<ul><li>x</li><li><table><tr><th>A</th></tr><tr><td>1</td></tr></table></li></ul>')
    whole = assemble_markdown(parsed).markdown
    assert whole == '- x\n- | A |\n  | --- |\n  | 1 |'
    delimiter_end = whole.index('| --- |') + len('| --- |')
    assert assemble_markdown(parsed, max_bytes=delimiter_end).markdown == whole[:delimiter_end]
    cut = assemble_markdown(parsed, max_bytes=delimiter_end - 1)
    assert cut.markdown == '- x' and cut.record()['blocks'] == {'list_item': 1}


def test_side_content_in_a_list_item_role_is_quoted_not_listed() -> None:
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(DocumentSegment(
        text='Boxed', locator='a', metadata={'block_role': 'list_item', 'content_role': 'textbox'}),))
    assert assemble_markdown(parsed).markdown == '> textbox: Boxed'


def test_a_segment_holding_cells_of_two_rows_is_written_only_when_its_last_row_is() -> None:
    from scone_memory.ingestion.formats.table_types import DocumentTableCell

    cells = (DocumentTableCell(table_locator='t', locator='t/b', row=1, column=0, text='b', start=0, end=1),
             DocumentTableCell(table_locator='t', locator='t/a', row=0, column=0, text='a', start=2, end=3,
                               is_header=True))
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(
        DocumentSegment(text='b a', locator='rows', table_cells=cells, metadata={'table_locator': 't'}),
        DocumentSegment(text='End', locator='end')))
    whole = assemble_markdown(parsed).markdown
    assert whole == '| a |\n| --- |\n| b |\n\nEnd'
    cut = assemble_markdown(parsed, max_bytes=len('| a |\n| --- |'))
    assert cut.record()['bound'] == {'max_bytes': 13, 'cut': True, 'cut_at': 0, 'segments_omitted': 2}


def test_lists_in_separate_mail_parts_are_separate_lists() -> None:
    def message(subject: str, body: str) -> str:
        return (f'From sender@example.com Mon Sep 14 10:00:00 2026\nFrom: a@example.com\nSubject: {subject}\n'
                f'MIME-Version: 1.0\nContent-Type: text/html; charset=utf-8\n\n{body}\n\n')

    box = (message('One', '<ol><li>Collect</li><li>Reconcile</li></ol>')
           + message('Two', '<ol><li>Pack</li><li>Ship</li></ol>')).encode()
    markdown = assemble_markdown(parse_text(box, 'box.mbox', DocumentLimits())).markdown
    assert '1. Collect\n2. Reconcile' in markdown and '1. Pack\n2. Ship' in markdown
    mail = (b'From: a@example.com\nSubject: x\nMIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=B\n\n'
            b'--B\nContent-Type: text/html\n\n<ul><li>first</li></ul>\n'
            b'--B\nContent-Type: text/html\n\n<ul><li>second<p>more</p></li></ul>\n--B--\n')
    parsed = parse_text(mail, 'm.eml', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown.endswith('- first\n\n* second\n\n  more')
    assert [span.kind for span in result.spans][-3:] == ['list_item', 'list_item', 'list_continuation']


def test_a_code_block_keeps_its_content_and_counts_lines_as_a_markdown_reader_does() -> None:
    assert assemble_markdown(html('<pre><code>x = 1\n</code></pre>')).markdown == '```\nx = 1\n```'
    assert assemble_markdown(html('<pre><code>x = 1\n\n</code></pre>')).markdown == '```\nx = 1\n\n```'
    parsed = html('<pre>a\rb\r\nc</pre><p>after</p>')
    result = assemble_markdown(parsed)
    assert result.markdown == '```\na\nb\nc\n```\n\nafter'
    assert [(span.first_line, span.last_line) for span in result.spans] == [(1, 5), (7, 7)]
    assert_traced(parsed, result)


def test_a_header_row_between_body_rows_is_counted_and_said() -> None:
    parsed = html('<table><tr><td>x</td><td>1</td></tr><tr><th>Section</th><th>Total</th></tr>'
                  '<tr><td>y</td><td>2</td></tr></table>')
    result = assemble_markdown(parsed)
    assert result.markdown == ('> table:1: 1 header row is written as a body row.\n\n'
                               '|  |  |\n| --- | --- |\n| x | 1 |\n| Section | Total |\n| y | 2 |')
    [table] = result.record()['tables']
    assert (table['header_basis'], table['extra_header_rows']) == ('none_declared', 1)
    declared = assemble_markdown(html('<table><tr><th>K</th></tr><tr><td>a</td></tr><tr><th>Sub</th></tr></table>'))
    assert declared.markdown.startswith('> table:1: 1 further header row is written as a body row.\n\n| K |')
    assert declared.record()['tables'][0]['extra_header_rows'] == 1


def test_an_ordered_list_starts_where_the_source_says() -> None:
    assert assemble_markdown(html('<ol start="5"><li>five</li><li>six</li></ol>')).markdown == '5. five\n6. six'
    assert assemble_markdown(html('<ol start="0"><li>zero</li></ol>')).markdown == '0. zero'
    assert assemble_markdown(html('<ol><li>a<ol start="3"><li>c</li></ol></li></ol>')).markdown == '1. a\n   3. c'
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(DocumentSegment(
        text='Item', locator='a', metadata={'block_role': 'list_item', 'list_kind': 'ordered', 'list_start': 'x'}),))
    assert assemble_markdown(parsed).markdown == '1. Item'
    unnamed = ParsedDocument(format='custom', parser='plugin', segments=(
        DocumentSegment(text='a', locator='a', metadata={'block_role': 'list_item', 'list_kind': 'ordered'}),
        DocumentSegment(text='Aside', locator='b'),
        DocumentSegment(text='b', locator='c', metadata={'block_role': 'list_item', 'list_kind': 'ordered'})))
    assert assemble_markdown(unnamed).markdown == '1. a\n\nAside\n\n1. b', 'only a named list resumes its count'


def test_a_nested_numbered_list_after_a_nested_bullet_list_counts_from_its_own_start() -> None:
    result = assemble_markdown(html('<ol><li>a<ul><li>x</li></ul><ol><li>y</li></ol></li></ol>'))
    assert result.markdown == '1. a\n   - x\n   1. y'


def test_separate_lists_that_would_touch_are_told_apart_by_their_marker() -> None:
    result = assemble_markdown(html('<ul><li>a</li></ul><ul><li>b<ul><li>b1</li></ul></li></ul><ul><li>c</li></ul>'
                                    '<ol><li>d</li></ol><ol><li>e</li></ol><ol><li>f</li></ol>'))
    assert result.markdown == '- a\n\n* b\n  * b1\n\n- c\n\n1. d\n\n1) e\n\n1. f'
    apart = assemble_markdown(html('<ul><li>a</li></ul><p>p</p><ul><li>b</li></ul>'))
    assert apart.markdown == '- a\n\np\n\n- b'


def test_a_table_segment_without_cells_is_written_after_its_grid_not_dropped() -> None:
    from scone_memory.ingestion.formats.table_types import DocumentTableCell

    def grid_row(text: str, row: int) -> DocumentSegment:
        cell = DocumentTableCell(table_locator='t', locator=f't/{row}', row=row, column=0, text=text, start=0,
                                 end=len(text), is_header=row == 0)
        return DocumentSegment(text=text, locator=f'row:{row}', table_cells=(cell,),
                               metadata={'table_locator': 't', 'table_status': 'structured'})

    parsed = ParsedDocument(format='custom', parser='plugin', segments=(
        grid_row('K', 0), DocumentSegment(text='loose', locator='loose', metadata={'table_locator': 't'}),
        grid_row('v', 1)))
    result = assemble_markdown(parsed)
    assert result.markdown == ('> t: 1 segment the document puts between its rows is written after it.\n\n'
                               '| K |\n| --- |\n| v |\n\nloose')
    assert_traced(parsed, result)
    text = ParsedDocument(format='custom', parser='plugin', segments=(
        DocumentSegment(text='a', locator='a', metadata={'table_locator': 't', 'table_status': 'text_fallback'}),
        DocumentSegment(text='aside', locator='aside'),
        DocumentSegment(text='b', locator='b', metadata={'table_locator': 't', 'table_status': 'text_fallback'})))
    assert assemble_markdown(text).markdown == (
        '> t was read as text, not as a grid; its lines follow as paragraphs. 1 segment the document puts between '
        'its rows is written after it.\n\na\n\nb\n\naside')


# What readers merged from main put in a segment, and what the Markdown does with it.


def test_an_open_document_heading_names_only_its_level_and_is_still_a_heading() -> None:
    from .test_office_formats import O, T, archive

    body = ('<office:text><text:h text:outline-level="2">Refunds</text:h><text:p>Within 30 days.</text:p>'
            '<text:h text:outline-level="10">Deepest</text:h></office:text>')
    data = archive({'content.xml': f'<office:document-content xmlns:office="{O}" xmlns:text="{T}"><office:body>{body}'
                                   '</office:body></office:document-content>'})
    parsed = parse_office(data, 'terms.odt', DocumentLimits())
    assert 'block_role' not in parsed.segments[0].metadata, 'the reader gives a level and no role'
    result = assemble_markdown(parsed)
    assert result.markdown == '## Refunds\n\nWithin 30 days.\n\n###### Deepest'
    record = result.record()
    assert record['structure_declared'] is True and record['blocks'] == {'heading': 2, 'paragraph': 1}
    assert record['headings_clamped'] == 1 and record['roles_unreadable'] == 0
    assert_traced(parsed, result)


def test_a_level_alone_that_is_not_a_level_is_a_paragraph_and_counted() -> None:
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(
        DocumentSegment(text='Not a level', locator='a', metadata={'heading_level': 'x'}),
        DocumentSegment(text='Zero', locator='b', metadata={'heading_level': '0'}),
        DocumentSegment(text='Listed', locator='c', metadata={'block_role': 'list_item', 'heading_level': '2'}),
    ))
    result = assemble_markdown(parsed)
    assert result.markdown == 'Not a level\n\nZero\n\n- Listed'
    assert result.record()['roles_unreadable'] == 2


def test_a_word_heading_whose_style_chain_ran_short_is_a_paragraph_and_counted() -> None:
    parsed = parse_office(word(paragraph('Circular', style='Loop') + paragraph('Overview', style='Berschrift1'),
                               styles=STYLES), 'report.docx', DocumentLimits())
    assert parsed.segments[0].metadata['heading_level_unresolved'] == 'style_chain'
    result = assemble_markdown(parsed)
    assert result.markdown == 'Circular\n\n# Overview'
    assert result.record()['headings_unresolved'] == 1


def test_a_chart_is_quoted_after_its_slide_text_with_one_line_per_series() -> None:
    from .test_office_charts import BAR, deck

    parsed = parse_office(deck(BAR), 'results.pptx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == ('Results\n\n> chart: Revenue (bar chart)\\\n> 2025: Q1 10; Q2 12.5\\\n'
                               '> 2024: Q1 8; Q2 9')
    assert [(span.kind, span.sources[0].locator) for span in result.spans] == [
        ('paragraph', 'slide:1/paragraph:1'), ('quote', 'slide:1/chart:1')]
    assert_traced(parsed, result)


def test_a_chart_in_a_word_list_item_is_quoted_after_the_item_it_sits_in() -> None:
    from .test_docx_parts import document
    from .test_office_charts import BAR, C

    inline = f'<w:drawing><c:chart xmlns:c="{C}" r:id="chart"/></w:drawing>'
    item = '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr>'
    body = (f'<w:p>{item}<w:r><w:t>Collect</w:t>{inline}</w:r></w:p>'
            f'<w:p>{item}<w:r><w:t>Reconcile</w:t></w:r></w:p>')
    data = document(body, relationships=f'<Relationship Id="chart" Target="charts/chart1.xml" Type="{R}/chart"/>',
                    extra={'content/charts/chart1.xml': BAR})
    parsed = parse_office(data, 'report.docx', DocumentLimits())
    assert [s.locator for s in parsed.segments] == ['paragraph:1', 'paragraph:1/chart:1', 'paragraph:2']
    result = assemble_markdown(parsed)
    assert result.markdown.split('\n\n')[0] == '- Collect'
    assert result.markdown.split('\n\n')[1].startswith('> chart: Revenue (bar chart)')
    assert result.markdown.split('\n\n')[2] == '- Reconcile', 'the quote ends the list; nothing touches it'
    assert_traced(parsed, result)


def test_link_targets_are_counted_not_written_and_the_link_text_stays() -> None:
    from .test_office_links import docx, external, run

    body = (f'<w:p>{run("See the ")}<w:hyperlink r:id="rId7">{run("refund policy")}</w:hyperlink>'
            f'{run(" or ")}<w:hyperlink w:anchor="returns">{run("returns")}</w:hyperlink></w:p>'
            f'<w:p>{run("No links.")}</w:p>')
    parsed = parse_office(docx(body, external('rId7', 'https://example.com/refunds')), 'terms.docx', DocumentLimits())
    assert len(json.loads(parsed.segments[0].metadata['links'])) == 2
    result = assemble_markdown(parsed)
    assert result.markdown == 'See the refund policy or returns\n\nNo links.'
    assert result.record()['link_targets_unwritten'] == 2
    cut = assemble_markdown(parsed, max_bytes=1)
    assert cut.markdown == '' and cut.record()['link_targets_unwritten'] == 0, 'only written segments count'


def test_an_unreadable_link_list_counts_nothing_and_writes_the_text() -> None:
    parsed = ParsedDocument(format='custom', parser='plugin', segments=(
        DocumentSegment(text='See here', locator='a', metadata={'links': 'not json'}),
        DocumentSegment(text='And there', locator='b', metadata={'links': '{"text": "there"}'}),
    ))
    result = assemble_markdown(parsed)
    assert result.markdown == 'See here\n\nAnd there' and result.record()['link_targets_unwritten'] == 0


async def test_pdf_bookmark_sections_are_counted_not_made_into_headings() -> None:
    pytest.importorskip('pypdf')
    from .test_pdf_outline import chapters, with_outline

    parsed = await BuiltinDocumentParser().parse(with_outline(chapters), 'survey.pdf')
    assert parsed.segments[2].metadata['section'] == 'Chapter 2 > Refunds'
    result = assemble_markdown(parsed)
    assert result.markdown == ('Introduction to the survey.\n\nScope of the crane survey.\n\nRefunds within 30 days.'
                               '\n\nReturns by post.\n\nAppendix tables.')
    record = result.record()
    assert record['sections_unwritten'] == 5 and record['structure_declared'] is False
    assert record['blocks'] == {'paragraph': 5}


async def test_an_unreadable_pdf_page_is_written_as_extracted_and_counted() -> None:
    pytest.importorskip('pypdf')
    from .test_unreadable_text_layer import PAGES, READABLE, mixed_pdf

    parsed = await BuiltinDocumentParser().parse(mixed_pdf(PAGES), 'survey.pdf')
    result = assemble_markdown(parsed)
    assert result.markdown.startswith(READABLE + '\n\n')
    assert result.record()['unreadable_segments'] == 1
    assert [span.sources[0].locator for span in result.spans] == ['page:1', 'page:2']
    readable = await BuiltinDocumentParser().parse(mixed_pdf(((READABLE, False),)), 'survey.pdf')
    assert assemble_markdown(readable).record()['unreadable_segments'] == 0
    said_otherwise = ParsedDocument(format='pdf', parser='plugin', segments=(
        DocumentSegment(text='Fine.', locator='page:1', metadata={'unreadable': 'false'}),))
    assert assemble_markdown(said_otherwise).record()['unreadable_segments'] == 0


async def test_ocr_paragraphs_of_an_image_are_paragraphs_traced_to_each() -> None:
    pytest.importorskip('PIL')
    from scone_memory.ingestion.formats.media import ImageDocumentParser

    from .test_ocr_paragraphs import LAID_OUT, Engine, png

    parsed = await ImageDocumentParser(Engine(LAID_OUT)).parse(png(), 'scan.png')
    result = assemble_markdown(parsed)
    assert result.markdown == 'The crane\\\nrusted.\n\nPaid in June.\n\nFooter'
    assert [span.sources[0].locator for span in result.spans] == [
        'frame:1/paragraph:1', 'frame:1/paragraph:2', 'frame:1/paragraph:3']
    assert result.record()['structure_declared'] is False
    assert_traced(parsed, result)


def test_a_deck_with_sections_rebuilds_its_slides_in_the_presentations_order() -> None:
    from .test_office_formats import P
    from .test_pptx_sections import P14, slide

    sections = (f'<p:extLst><p:ext uri="{{521415D9-36F7-43E2-AB2F-B90AF26B5E84}}"><p14:sectionLst xmlns:p14="{P14}">'
                '<p14:section name="Opening" id="{A}"><p14:sldIdLst><p14:sldId id="257"/></p14:sldIdLst></p14:section>'
                '</p14:sectionLst></p:ext></p:extLst>')
    data = ooxml_archive({
        'ppt/presentation.xml': (f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst>'
                                 f'<p:sldId id="257" r:id="r2"/><p:sldId id="256" r:id="r1"/></p:sldIdLst>{sections}'
                                 '</p:presentation>'),
        'ppt/_rels/presentation.xml.rels': (f'<Relationships xmlns="{REL}"><Relationship Id="r1" '
                                            f'Target="slides/slide1.xml" Type="{R}/slide"/><Relationship '
                                            f'Id="r2" Target="slides/slide2.xml" Type="{R}/slide"/></Relationships>'),
        'ppt/slides/slide1.xml': slide('Body slide'),
        'ppt/slides/slide2.xml': slide('# Opening slide'),
    }, main_part='ppt/presentation.xml')
    parsed = parse_office(data, 'deck.pptx', DocumentLimits())
    result = assemble_markdown(parsed)
    assert result.markdown == '\\# Opening slide\n\nBody slide', 'a section name is not text and adds no heading'
    assert result.record()['structure_declared'] is False
    assert_traced(parsed, result)
