"""Declared spreadsheet tables preserve the source labels behind stored values."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.types import DocumentLimits
from .test_office_formats import S, R, REL, ooxml_archive


def workbook(rows: str, definition: str | None = None, *, extra: str = '', namespace: str = S) -> bytes:
    table = '' if definition is None else '<tableParts count="1"><tablePart r:id="t1"/></tableParts>'
    parts = {'xl/workbook.xml': f'<workbook xmlns="{namespace}" xmlns:r="{R}"><sheets><sheet name="Sales" sheetId="1" r:id="s1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="s1" Target="worksheets/sheet1.xml" Type="{R}/worksheet"/></Relationships>',
        'xl/worksheets/sheet1.xml': f'<worksheet xmlns="{namespace}" xmlns:r="{R}"><sheetData>{rows}</sheetData>{table}{extra}</worksheet>'}
    if definition is not None:
        parts['xl/worksheets/_rels/sheet1.xml.rels'] = f'<Relationships xmlns="{REL}"><Relationship Id="t1" Target="../tables/table1.xml" Type="{R}/table"/></Relationships>'
        parts['xl/tables/table1.xml'] = f'<table xmlns="{namespace}" {definition}</table>'
    return ooxml_archive(parts, main_part='xl/workbook.xml')


def cell(ref: str, value: str, *, formula: str = '') -> str:
    if formula:
        return f'<c r="{ref}" t="str">{formula}<v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'


def row(number: int, cells: str) -> str:
    return f'<row r="{number}">{cells}</row>'


DEFINITION = ('id="1" name="RevenueTable" displayName="RevenueTable" ref="B4:C6" totalsRowCount="1">'
    '<tableColumns count="2"><tableColumn id="1" name="Region"/><tableColumn id="2" name="Revenue"/></tableColumns>')
ROWS = (row(1, cell('A1', 'Notes')) + row(4, cell('B4', 'Region') + cell('C4', 'Revenue'))
    + row(5, cell('B5', 'West') + cell('C5', '€20')) + row(6, cell('B6', 'Total') + cell('C6', '€20', formula='<f>SUM(C5:C5)</f>')))


def parse(raw: bytes, limits: DocumentLimits = DocumentLimits()):
    return parse_office(raw, 'sales.xlsx', limits)


@pytest.mark.parametrize('namespace', [S, 'http://purl.oclc.org/ooxml/spreadsheetml/main'])
def test_declared_table_headers_and_totals_keep_exact_cell_sources(namespace):
    parsed = parse(workbook(ROWS, DEFINITION, namespace=namespace))
    value = next(s for s in parsed.segments if s.locator == 'sheet:Sales/cell:C5')
    assert value.text == 'Revenue: €20'
    evidence, = value.table_cells
    assert (evidence.row, evidence.column, evidence.is_header) == (1, 1, False)
    assert [(h.text, h.locator, h.association) for h in evidence.headers] == [('Revenue', 'sheet:Sales/cell:C4', 'column')]
    assert value.text.encode()[evidence.start:evidence.end] == '€20'.encode()
    assert value.metadata['table_range'] == 'B4:C6'
    assert value.metadata['member'] == 'xl/worksheets/sheet1.xml'
    total = parsed.segments[-1]
    assert total.metadata['table_role'] == 'totals' and total.metadata['formula'] == 'cached-value'
    assert parsed.segments[0].text == 'Notes' and not parsed.segments[0].table_cells


def test_headerless_table_does_not_promote_its_first_data_row():
    definition = DEFINITION.replace('totalsRowCount="1"', 'headerRowCount="0" totalsRowCount="0"')
    parsed = parse(workbook(ROWS, definition))
    cells = [c for s in parsed.segments for c in s.table_cells]
    assert len(cells) == 6 and not any(c.is_header or c.headers for c in cells)
    assert any('header_row_absent' in s.metadata.get('table_notes', '') for s in parsed.segments)


def test_header_names_come_from_retained_cells_and_disclose_definition_disagreement():
    parsed = parse(workbook(ROWS, DEFINITION.replace('name="Revenue"', 'name="Different"')))
    value = next(s for s in parsed.segments if s.locator.endswith('/cell:C5'))
    assert value.text == 'Revenue: €20'
    assert 'column_name_mismatch' in value.metadata['table_notes']


@pytest.mark.parametrize('definition', [DEFINITION.replace('B4:C6', 'B4:A6'), DEFINITION.replace('count="2"', 'count="3"'), DEFINITION.replace('totalsRowCount="1"', 'headerRowCount="2"')])
def test_invalid_declaration_preserves_text_with_explicit_fallback(definition):
    parsed = parse(workbook(ROWS, definition))
    assert [s.text for s in parsed.segments] == ['Notes', 'Region', 'Revenue', 'West', '€20', 'Total', '€20']
    assert not any(s.table_cells for s in parsed.segments)
    assert all(s.metadata['table_status'] == 'text_fallback' for s in parsed.segments)


def test_merge_intersecting_declared_table_is_not_silently_flattened_as_structured():
    parsed = parse(workbook(ROWS, DEFINITION, extra='<mergeCells><mergeCell ref="B5:C5"/></mergeCells>'))
    assert not any(s.table_cells for s in parsed.segments)
    assert 'merged_table_cells' in parsed.segments[0].metadata['table_notes']


def test_sparse_header_does_not_create_an_unretained_source_cell():
    parsed = parse(workbook(ROWS.replace(cell('C4', 'Revenue'), ''), DEFINITION))
    value = next(s for s in parsed.segments if s.locator.endswith('/cell:C5'))
    assert value.text == '€20' and not value.table_cells[0].headers
    assert 'missing_header_cell' in value.metadata['table_notes']


def test_table_coordinates_are_relative_even_at_the_bottom_of_a_sheet():
    rows = row(1048575, cell('XFC1048575', 'Label')) + row(1048576, cell('XFC1048576', 'Value'))
    definition = 'id="1" name="Last" ref="XFC1048575:XFC1048576"><tableColumns count="1"><tableColumn id="1" name="Label"/></tableColumns>'
    parsed = parse(workbook(rows, definition))
    value = parsed.segments[-1].table_cells[0]
    assert (value.row, value.column) == (1, 0)
    assert parsed.segments[-1].text == 'Label: Value'


def test_header_amplification_respects_total_text_limit():
    rows = row(1, cell('A1', 'H' * 500)) + ''.join(row(n, cell(f'A{n}', '1')) for n in range(2, 20))
    definition = 'id="1" name="Budget" ref="A1:A20"><tableColumns count="1"><tableColumn id="1" name="H"/></tableColumns>'
    with pytest.raises(InvalidInput, match='text byte limit'):
        parse(workbook(rows, definition), DocumentLimits(max_text_bytes=1000))


def test_sparse_implicit_cells_continue_after_the_previous_explicit_column():
    rows = row(4, cell('B4', 'Region') + '<c t="inlineStr"><is><t>Revenue</t></is></c>') + row(5, cell('B5', 'West') + cell('C5', '€20'))
    parsed = parse(workbook(rows, DEFINITION))
    value = parsed.segments[-1]
    assert value.text == 'Revenue: €20'
    assert value.table_cells[0].headers[0].locator == 'sheet:Sales/cell:C4'


def test_foreign_namespace_cell_text_does_not_become_a_declared_header():
    rows = ROWS.replace(cell('C4', 'Revenue'), '<alien:c xmlns:alien="urn:other" r="C4" t="inlineStr"><alien:is><alien:t>Revenue</alien:t></alien:is></alien:c>')
    parsed = parse(workbook(rows, DEFINITION))
    assert not any(s.table_cells for s in parsed.segments)
    assert 'unsupported_worksheet_structure' in parsed.segments[0].metadata['table_notes']


def test_declared_large_column_id_is_a_valid_uint32():
    parsed = parse(workbook(ROWS, DEFINITION.replace('tableColumn id="2"', 'tableColumn id="4294967295"')))
    assert parsed.segments[-1].table_cells


def test_missing_formula_cache_is_disclosed_without_executing_or_inventing_a_value():
    rows = ROWS.replace(cell('C5', '€20'), '<c r="C5"><f>1+1</f><v/></c>')
    parsed = parse(workbook(rows, DEFINITION))
    assert not any(s.locator.endswith('/cell:C5') for s in parsed.segments)
    assert all('missing_cached_formula' in s.metadata['table_notes'] for s in parsed.segments if s.table_cells)


async def test_declared_table_worker_http_and_citation_keep_header_provenance():
    import httpx
    from scone_memory import MemoryEngine, HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex
    from scone_memory.api import create_app
    from scone_memory.core.ports import NewChunk
    from scone_memory.ingestion import ingest_document, document_provenance
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        added = await ingest_document(memory, 'alpha', workbook(ROWS, DEFINITION), filename='sales.xlsx')
        episode = await memory.episode('alpha', added.added.episode_id)
        start = episode.content.encode().index('€20'.encode())
        chunk, = await memory.documents.insert_chunks([NewChunk(space='alpha', episode_id=episode.episode_id,
            ordinal=99, start=start, end=start+5, text='€20', created_at=episode.created_at)])
        evidence = await document_provenance(memory, 'alpha', episode.episode_id, chunk_id=chunk.chunk_id)
        value, = evidence.segments[0].table_cells
        assert value.headers[0].locator == 'sheet:Sales/cell:C4' and value.text == '€20'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(memory, {'reader': 'alpha'}, roles={'reader': 'read'})), base_url='http://test') as client:
            response = await client.get(f'/v1/episodes/{episode.episode_id}/document', headers={'Authorization': 'Bearer reader'})
        assert response.status_code == 200
        assert response.json()['parser'] == 'native-xml-xlsx-tables-v1'
        assert response.json()['segments'][-1]['metadata']['table_role'] == 'totals'
    finally:
        await memory.close()


async def test_table_headers_survive_extraction_checkpoint_restart(tmp_path):
    import asyncio
    from .test_document_workflow import CountingParser, HeldEmbedder, open_memory, workflow
    from scone_memory.ingestion import document_provenance
    parser, held = CountingParser(), HeldEmbedder()
    memory = await open_memory(tmp_path, held)
    original = await memory.attach('alpha', workbook(ROWS, DEFINITION), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename='sales.xlsx')
    job = workflow(memory, tmp_path, parser)
    task = asyncio.create_task(job.run('xlsx', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(held.entered.wait(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()
    memory = await open_memory(tmp_path)
    job = workflow(memory, tmp_path, parser)
    try:
        result = await job.run('xlsx', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_steps == ('extract',) and parser.calls == 1
        evidence = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        assert evidence.segments[-1].table_cells[0].headers[0].text == 'Revenue'
    finally:
        job.close()
        await memory.close()


@pytest.mark.parametrize('rows', [ROWS.replace('<row r="4">', '<row r="3">'), ROWS.replace(cell('C4', 'Revenue'), cell('C4', 'Revenue') + cell('C4', ''))])
def test_all_cell_coordinates_are_validated_including_empty_cells(rows):
    parsed = parse(workbook(rows, DEFINITION))
    assert not any(s.table_cells for s in parsed.segments)
    assert parsed.segments[0].metadata['table_status'] == 'text_fallback'


def test_foreign_shared_strings_never_become_structured_headers():
    from io import BytesIO
    from zipfile import ZipFile
    raw = workbook(ROWS.replace(cell('C4', 'Revenue'), '<c r="C4" t="s"><v>0</v></c>'), DEFINITION)
    with ZipFile(BytesIO(raw)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    relation = f'<Relationship Id="strings" Target="sharedStrings.xml" Type="{R}/sharedStrings"/>'
    parts['xl/_rels/workbook.xml.rels'] = parts['xl/_rels/workbook.xml.rels'].replace(b'</Relationships>', relation.encode() + b'</Relationships>')
    parts['xl/sharedStrings.xml'] = f'<sst xmlns="{S}"><si><alien:t xmlns:alien="urn:other">Revenue</alien:t></si></sst>'.encode()
    output = BytesIO()
    with ZipFile(output, 'w') as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    parsed = parse(output.getvalue())
    assert not any(s.table_cells for s in parsed.segments)
    assert parsed.segments[0].metadata['table_notes'] == 'unsupported_shared_string_structure'


@pytest.mark.parametrize('extra', ['<t>Injected</t>', '<is><t>Injected</t></is>'])
def test_all_extracted_inline_text_must_belong_to_the_single_string_container(extra):
    modified = cell('C4', 'Revenue').replace('</c>', extra + '</c>')
    parsed = parse(workbook(ROWS.replace(cell('C4', 'Revenue'), modified), DEFINITION))
    assert not any(s.table_cells for s in parsed.segments)
    assert parsed.segments[0].metadata['table_notes'] == 'unsupported_worksheet_structure'
