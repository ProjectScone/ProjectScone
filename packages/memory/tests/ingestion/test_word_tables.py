"""Word table structure must retain the source that gives each value its context."""
import pytest
import json
import httpx
import asyncio

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.types import DocumentLimits
from scone_memory.ingestion.formats.types import validate_document
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.ports import NewChunk
from scone_memory.ingestion import BuiltinDocumentParser, document_provenance, ingest_document
from scone_memory.ingestion.files import DocumentManifest, encode_manifest, prepare_document
from .test_office_formats import W, ooxml_archive


def cell(text: str, properties: str = '') -> str:
    return f'<w:tc><w:tcPr>{properties}</w:tcPr><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>'


def row(cells: str, properties: str = '') -> str:
    return f'<w:tr><w:trPr>{properties}</w:trPr>{cells}</w:tr>'


def document(rows: str, namespace: str = W) -> bytes:
    return ooxml_archive({'content/main.xml': f'<w:document xmlns:w="{namespace}"><w:body>'
        f'<w:tbl>{rows}</w:tbl></w:body></w:document>'}, main_part='content/main.xml')


def parse(rows: str, namespace: str = W):
    return parse_office(document(rows, namespace), 'table.docx', DocumentLimits())


def cells(parsed):
    return [c for s in parsed.segments for c in s.table_cells]


def merged_rows():
    return (row(cell('Region') + cell('Revenue'), '<w:tblHeader/>')
        + row(cell('West', '<w:vMerge w:val="restart"/>') + cell('10'))
        + row(cell('', '<w:vMerge/>') + cell('€20')))


@pytest.mark.parametrize('namespace', [W, 'http://purl.oclc.org/ooxml/wordprocessingml/main'])
def test_declared_multilevel_headers_and_horizontal_spans(namespace):
    parsed = parse(row(cell('Sales', '<w:gridSpan w:val="2"/>'), '<w:tblHeader/>')
        + row(cell('2025') + cell('2026'), '<w:tblHeader/>')
        + row(cell('€10') + cell('€20')), namespace)
    value = cells(parsed)[-1]
    assert (value.row, value.column) == (2, 1)
    assert [h.text for h in value.headers] == ['Sales', '2026']
    assert parsed.segments[-1].text == 'Sales / 2025: €10\nSales / 2026: €20'
    assert parsed.segments[-1].text.encode()[value.start:value.end] == '€20'.encode()
    assert cells(parsed)[0].column_span == 2
    assert parsed.segments[-1].metadata['member'] == 'content/main.xml'


def test_vertical_merge_retains_data_context_without_claiming_it_is_a_header():
    parsed = parse(row(cell('Region') + cell('Revenue'), '<w:tblHeader/>')
        + row(cell('West', '<w:vMerge w:val="restart"/>') + cell('10'))
        + row(cell('', '<w:vMerge/>') + cell('20')))
    west = next(c for c in cells(parsed) if c.text == 'West')
    value = cells(parsed)[-1]
    assert west.row_span == 2 and not west.is_header
    assert west.merged_locators == ('table:1/row:3/cell:1',)
    assert [(c.text, c.locator, c.association) for c in value.context] == [('West', west.locator, 'row_span')]
    assert [h.text for h in value.headers] == ['Revenue']
    assert parsed.segments[-1].text == 'West / Revenue: 20'


def test_legacy_horizontal_merge_records_every_source_cell():
    parsed = parse(row(cell('Annual', '<w:hMerge w:val="restart"/>')
                       + cell('sales', '<w:hMerge/>'), '<w:tblHeader/>')
                   + row(cell('10') + cell('20')))
    header = cells(parsed)[0]
    assert header.text == 'Annual\nsales' and header.column_span == 2
    assert header.merged_locators == ('table:1/row:1/cell:2',)
    assert cells(parsed)[-1].headers[0].locator == header.locator


def test_late_header_marker_and_first_row_styling_do_not_invent_headers():
    parsed = parse(row(cell('ordinary')) + row(cell('later'), '<w:tblHeader/>') + row(cell('value')))
    assert not any(c.is_header or c.headers for c in cells(parsed))
    assert 'noncontiguous_header' in parsed.segments[-1].metadata['table_notes']


def test_grid_before_preserves_omitted_leading_columns():
    parsed = parse(row(cell('Left') + cell('Right'), '<w:tblHeader/>')
                   + row(cell('42'), '<w:gridBefore w:val="1"/>'))
    assert cells(parsed)[-1].column == 1
    assert [h.text for h in cells(parsed)[-1].headers] == ['Right']


@pytest.mark.parametrize('properties', ['<w:vMerge/>', '<w:hMerge/>'])
def test_unmatched_merge_keeps_text_with_explicit_structure_fallback(properties):
    parsed = parse(row(cell('Keep me', properties)))
    assert parsed.segments[0].text == 'Keep me'
    assert not cells(parsed)
    assert parsed.segments[0].metadata['table_status'] == 'text_fallback'


def test_cell_grid_limits_reject_before_allocating_unbounded_spans():
    with pytest.raises(InvalidInput, match='table.*limit'):
        parse(row(cell('value', '<w:gridSpan w:val="999999999"/>')))


def test_deleted_cells_and_old_property_snapshots_do_not_supply_table_values():
    parsed = parse(row(cell('removed', '<w:cellDel w:id="1"/>') + cell('current',
        '<w:tcPrChange w:id="2"><w:tcPr><w:gridSpan w:val="9000"/></w:tcPr></w:tcPrChange>')))
    assert [c.text for c in cells(parsed)] == ['current']
    assert cells(parsed)[0].column_span == 1


def test_tracked_row_structure_is_disclosed_without_deleting_independently_live_text():
    parsed = parse(row(cell('Independently live'), '<w:del w:id="1"/>') + row(cell('Next')))
    assert [s.text for s in parsed.segments] == ['Independently live', 'Next']
    assert not cells(parsed)
    assert parsed.segments[0].metadata['table_notes'] == 'tracked_row_structure'


async def test_merged_sources_and_context_survive_worker_and_filtered_http_citation():
    raw = document(merged_rows())
    manifest = await prepare_document(raw, 'table.docx', parser=BuiltinDocumentParser(), limits=DocumentLimits())
    assert manifest.schema_version == 5
    assert DocumentManifest.model_validate_json(encode_manifest(manifest)) == manifest
    payload = json.loads(encode_manifest(manifest))
    payload['schema_version'] = 4
    with pytest.raises(ValueError, match='version five'):
        DocumentManifest.model_validate_json(json.dumps(payload))
    with pytest.raises(InvalidInput, match='version five'):
        encode_manifest(manifest.model_copy(update={'schema_version': 4}))
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        added = await ingest_document(memory, 'alpha', raw, filename='table.docx')
        episode = await memory.episode('alpha', added.added.episode_id)
        start = episode.content.encode().index('€20'.encode())
        chunk, = await memory.documents.insert_chunks([NewChunk(space='alpha', episode_id=episode.episode_id,
            ordinal=99, start=start, end=start + 5, text='€20', created_at=episode.created_at)])
        citation = await document_provenance(memory, 'alpha', episode.episode_id, chunk_id=chunk.chunk_id)
        value, = cells(citation)
        assert value.context[0].text == 'West' and value.headers[0].text == 'Revenue'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(memory, {'reader': 'alpha'},
                roles={'reader': 'read'})), base_url='http://test') as client:
            response = await client.get(f'/v1/episodes/{episode.episode_id}/document',
                params={'chunk_id': chunk.chunk_id}, headers={'Authorization': 'Bearer reader'})
        assert response.status_code == 200
        assert response.json()['segments'][0]['table_cells'][0]['context'][0]['locator'] == value.context[0].locator
    finally:
        await memory.close()


async def test_word_merge_evidence_resumes_from_retained_extraction_after_restart(tmp_path):
    from .test_document_workflow import CountingParser, HeldEmbedder, open_memory, workflow
    parser, embedder = CountingParser(), HeldEmbedder()
    memory = await open_memory(tmp_path, embedder)
    original = await memory.attach('alpha', document(merged_rows()),
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document', filename='table.docx')
    job = workflow(memory, tmp_path, parser)
    task = asyncio.create_task(job.run('word', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(embedder.entered.wait(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()
    memory = await open_memory(tmp_path)
    resumed = workflow(memory, tmp_path, parser)
    try:
        result = await resumed.run('word', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_steps == ('extract',) and parser.calls == 1
        citation = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        assert cells(citation)[-1].context[0].text == 'West'
        west = next(c for c in cells(citation) if c.text == 'West')
        assert west.row_span == 2 and len(west.merged_locators) == 1
    finally:
        resumed.close()
        await memory.close()


def test_forged_merged_source_and_context_rejected_at_extension_boundary():
    parsed = parse(merged_rows())
    value = parsed.segments[-1].table_cells[0]
    context = value.context[0]
    for changes in ({'context': (context.model_copy(update={'text': 'Invented'}),)},
                    {'context': (context, context)}, {'merged_locators': (value.locator,)},
                    {'merged_locators': ('x' * 4097,)}):
        segment = parsed.segments[-1].model_copy(update={'table_cells': (value.model_copy(update=changes),)})
        with pytest.raises(InvalidInput, match='table'):
            validate_document(parsed.model_copy(update={'segments': (*parsed.segments[:-1], segment)}), DocumentLimits())


def test_empty_rows_and_merge_continuations_do_not_overwrite_prior_evidence():
    parsed = parse(row(cell('Anchor', '<w:vMerge w:val="restart"/>'))
                   + row(cell('', '<w:vMerge/>')) + row(cell(' ')) + row(cell('Last')))
    assert [c.text for c in cells(parsed)] == ['Anchor', 'Last']
    assert cells(parsed)[0].row_span == 2
    assert cells(parsed)[-1].row == 3


def test_repeated_header_context_is_bounded_during_row_construction(monkeypatch):
    from scone_memory.ingestion.formats import word_tables
    created = 0
    original = word_tables.DocumentSegment
    def record(**kwargs):
        nonlocal created
        created += 1
        return original(**kwargs)
    monkeypatch.setattr(word_tables, 'DocumentSegment', record)
    raw = document(row(cell('H' * 500), '<w:tblHeader/>') + row(cell('1')) * 100)
    with pytest.raises(InvalidInput, match='text byte limit'):
        parse_office(raw, 'table.docx', DocumentLimits(max_text_bytes=1000))
    assert created <= 1


@pytest.mark.parametrize('foreign', ['table', 'row', 'cell'])
def test_foreign_structural_names_preserve_text_without_claiming_word_headers(foreign):
    rows = row(cell('Foreign'), '<w:tblHeader/>') + row(cell('10'))
    body = '<w:tbl>' + rows + '</w:tbl>'
    tag = {'table': 'tbl', 'row': 'tr', 'cell': 'tc'}[foreign]
    body = body.replace(f'<w:{tag}>', f'<x:{tag}>').replace(f'</w:{tag}>', f'</x:{tag}>')
    data = ooxml_archive({'word/document.xml': f'<w:document xmlns:w="{W}" xmlns:x="urn:not-word">'
                         f'<w:body>{body}</w:body></w:document>'}, main_part='word/document.xml')
    parsed = parse_office(data, 'table.docx', DocumentLimits())
    assert [s.text for s in parsed.segments] == ['Foreign', '10']
    assert not cells(parsed)
    assert parsed.segments[0].metadata['table_notes'] == 'unsupported_table_namespace'
