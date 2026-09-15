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
                     'notes': '', 'caption': None}
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
    assert result.markdown == '- Top\n  - Child\n\n- Unknown kind'
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
    assert assemble_markdown(parsed).markdown == '1. One\n2. Two\n\nAside.\n\n3. Three\n\n1. Other list'


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
