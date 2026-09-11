"""Source table evidence must preserve the relationships that give values meaning."""
import pytest
import json
import httpx
import asyncio

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.text import parse_text
from scone_memory.ingestion.formats.types import DocumentLimits, validate_document
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.ports import NewChunk
from scone_memory.ingestion import BuiltinDocumentParser, document_provenance, ingest_document
from scone_memory.ingestion.files import DocumentManifest, encode_manifest, prepare_document


def parse(raw: str):
    return parse_text(raw.encode(), 'table.html', DocumentLimits())


def cells(document):
    return [cell for segment in document.segments for cell in segment.table_cells]


def test_multilevel_headers_and_row_labels_are_citable_context() -> None:
    document = parse('<table><thead><tr><th rowspan="2">Region</th><th colspan="2">Sales</th>'
                     '<tr><th>2025</th><th>2026</th></thead><tbody>'
                     '<tr><th scope="row">Europe</th><td>€10</td><td>€20</td></tbody></table>')
    value = next(cell for cell in cells(document) if cell.text == '€20')
    assert (value.row, value.column) == (2, 2)
    assert {header.text for header in value.headers} == {'Europe', '2026', 'Sales'}
    segment = next(s for s in document.segments if value in s.table_cells)
    assert 'Europe / 2026 / Sales: €20' in segment.text
    assert segment.text.encode()[value.start:value.end] == '€20'.encode()
    indexed = {cell.locator: cell for cell in cells(document)}
    assert all(indexed[h.locator].text == h.text for h in value.headers)
    assert next(c for c in cells(document) if c.text == 'Sales').column_span == 2


def test_explicit_headers_override_position_and_empty_headers_disable_inference() -> None:
    document = parse('<table><tr><th id="a">Alpha</th><th id="b">Beta</th>'
                     '<tr><td headers="b b">one</td><td headers="">two</td></table>')
    values = {cell.text: cell for cell in cells(document)}
    assert [(h.text, h.association) for h in values['one'].headers] == [('Beta', 'explicit')]
    assert values['two'].headers == ()


def test_hidden_headers_never_leak_and_missing_references_are_disclosed() -> None:
    document = parse('<table><tr><th hidden id="private">secret<th id="public">Visible'
                     '<tr><td headers="private absent">one<td>two</table>')
    assert 'secret' not in document.model_dump_json()
    assert next(c for c in cells(document) if c.text == 'one').headers == ()
    assert any('unresolved_headers' in s.metadata.get('table_notes', '') for s in document.segments)


def test_rowspan_zero_stays_in_its_row_group() -> None:
    document = parse('<table><tbody><tr><th rowspan="0" scope="rowgroup">West<td>10'
                     '<tr><td>20<tbody><tr><th scope="row">East<td>30</table>')
    values = {cell.text: cell for cell in cells(document)}
    assert values['West'].row_span == 2
    assert [h.text for h in values['20'].headers] == ['West']
    assert [h.text for h in values['30'].headers] == ['East']


def test_omitting_tbody_preserves_row_group_header_context() -> None:
    rows = '<tr><th scope="rowgroup" rowspan="2">West<td>10<tr><td>20'
    implicit = parse('<table>' + rows + '</table>')
    explicit = parse('<table><tbody>' + rows + '</tbody></table>')
    assert [[h.text for h in c.headers] for c in cells(implicit)] == [[], ['West'], ['West']]
    assert cells(implicit) == cells(explicit)


def test_nested_tables_preserve_text_with_explicit_fallback() -> None:
    document = parse('<table><tr><td>outer<table><tr><td>inner</table>tail</table>')
    assert all(value in '\n'.join(s.text for s in document.segments) for value in ('outer', 'inner', 'tail'))
    assert not cells(document)
    assert all(s.metadata.get('table_notes') == 'nested_table' for s in document.segments)


def test_table_spans_cannot_allocate_an_unbounded_grid() -> None:
    with pytest.raises(InvalidInput, match='table.*limit'):
        parse('<table><tr><td rowspan="65534" colspan="1000">value</table>')


def test_non_table_serialization_does_not_gain_an_empty_field() -> None:
    document = parse('<p>plain</p>')
    assert 'table_cells' not in document.model_dump_json()


def test_header_id_budget_does_not_reject_unrelated_html_or_guess_late_ids() -> None:
    prefix = '<p>' + ''.join(f'<i id="n{i}"></i>' for i in range(20_001)) + 'Visible</p>'
    assert parse(prefix).segments[0].text == 'Visible'
    document = parse(prefix + '<table><tr><th id="late">Late<tr><td headers="late">value</table>')
    assert cells(document)[-1].headers == ()
    assert 'header_id_limit' in document.segments[-1].metadata['table_notes']


def test_long_table_reuses_header_structure_without_quadratic_scanning() -> None:
    document = parse('<table><tr><th>Index<th>Value' + ''.join(
        f'<tr><td>{row}<td>{row * 10}' for row in range(2000)) + '</table>')
    assert len(cells(document)) == 4002
    assert [h.text for h in cells(document)[-1].headers] == ['Value']


def test_extension_parser_cannot_forge_header_evidence_or_spans() -> None:
    document = parse('<table><tr><th>Price<tr><td>€20</table>')
    row = document.segments[-1]
    value = row.table_cells[0]
    bad_header = value.headers[0].model_copy(update={'text': 'Fake'})
    wrong_axis = value.headers[0].model_copy(update={'association': 'row'})
    for change in ({'headers': (bad_header,)}, {'headers': (wrong_axis,)}, {'start': value.start + 1}):
        invalid = row.model_copy(update={'table_cells': (value.model_copy(update=change),)})
        with pytest.raises(InvalidInput, match='table'):
            validate_document(document.model_copy(update={'segments': (*document.segments[:-1], invalid)}),
                              DocumentLimits())


async def test_table_evidence_survives_worker_storage_and_filtered_http_citation() -> None:
    raw = '<table><tr><th>Region<th>Revenue<tr><th scope="row">Europe<td>€20</table>'.encode()
    manifest = await prepare_document(raw, 'report.html', parser=BuiltinDocumentParser(), limits=DocumentLimits())
    assert manifest.schema_version == 4
    assert DocumentManifest.model_validate_json(encode_manifest(manifest)) == manifest
    for version in (1, 2, 3):
        value = json.loads(encode_manifest(manifest))
        value['schema_version'] = version
        with pytest.raises(ValueError, match='version four'):
            DocumentManifest.model_validate_json(json.dumps(value))
        with pytest.raises(InvalidInput, match='version four'):
            encode_manifest(manifest.model_copy(update={'schema_version': version}))
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        result = await ingest_document(memory, 'alpha', raw, filename='report.html')
        episode = await memory.episode('alpha', result.added.episode_id)
        start = episode.content.encode().index('€20'.encode())
        chunk, = await memory.documents.insert_chunks([NewChunk(episode_id=episode.episode_id, space='alpha',
            ordinal=99, start=start, end=start + 5, text='€20', created_at=episode.created_at)])
        citation = await document_provenance(memory, 'alpha', episode.episode_id, chunk_id=chunk.chunk_id)
        value, = cells(citation)
        assert value.text == '€20'
        assert {h.text for h in value.headers} == {'Europe', 'Revenue'}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(
                memory, {'reader': 'alpha'}, roles={'reader': 'read'})), base_url='http://test') as client:
            response = await client.get(f'/v1/episodes/{episode.episode_id}/document',
                params={'chunk_id': chunk.chunk_id}, headers={'Authorization': 'Bearer reader'})
        assert response.status_code == 200
        assert response.json()['segments'][0]['table_cells'][0]['headers'][0]['text'] == 'Europe'
    finally:
        await memory.close()


async def test_table_headers_survive_cancelled_indexing_and_journal_restart(tmp_path) -> None:
    from .test_document_workflow import CountingParser, HeldEmbedder, open_memory, workflow
    embedder = HeldEmbedder()
    parser = CountingParser()
    memory = await open_memory(tmp_path, embedder)
    original = await memory.attach('alpha', b'<table><tr><th>Revenue<tr><td>20</table>',
                                   'text/html', filename='table.html')
    job = workflow(memory, tmp_path, parser)
    task = asyncio.create_task(job.run('table', space='alpha', attachment_id=original.attachment_id))
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
        result = await resumed.run('table', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_steps == ('extract',)
        assert parser.calls == 1
        citation = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        assert cells(citation)[-1].headers[0].text == 'Revenue'
        replay = await resumed.run('table', space='alpha', attachment_id=original.attachment_id)
        assert replay.reused_steps == ('extract', 'index')
    finally:
        resumed.close()
        await memory.close()


def test_caption_inline_markup_and_preformatted_cell_text_are_retained() -> None:
    document = parse('<p>Before</p><table><caption>Annual <b>sales</b></caption>'
                     '<tr><th>Notes<tr><td>A<br>B<pre> x  y\n z</pre>C</table><p>After</p>')
    assert document.segments[0].text == 'Before'
    assert document.segments[1].text == 'Annual sales'
    assert document.segments[-1].text == 'After'
    assert ' x  y\n z' in cells(document)[-1].text
    assert 'A\nB' in cells(document)[-1].text


def test_inline_whitespace_collapses_across_tags_without_trimming_pre() -> None:
    document = parse('<table><tr><td>A <b> B</b>  C<td><pre>  x\n y  </pre></table>')
    assert [c.text for c in cells(document)] == ['A B C', '  x\n y  ']


def test_first_document_id_and_table_scope_control_explicit_headers() -> None:
    document = parse('<div id="x">Outside</div><table><tr><th id="x">Wrong<th id="y">Right'
                     '<tr><td headers="x y">one</table><table><tr><td headers="y">two</table>')
    values = {c.text: c for c in cells(document)}
    assert [h.text for h in values['one'].headers] == ['Right']
    assert values['two'].headers == ()


def test_column_group_headers_apply_only_inside_declared_group() -> None:
    document = parse('<table><colgroup span="2"></colgroup><colgroup></colgroup>'
                     '<tr><th scope="colgroup" colspan="2">Shared<th scope="col">Other'
                     '<tr><td>a<td>b<td>c</table>')
    values = {c.text: c for c in cells(document)}
    assert [h.text for h in values['b'].headers] == ['Shared']
    assert [h.text for h in values['c'].headers] == ['Other']


@pytest.mark.parametrize('groups', [
    '<colgroup span="2"><colgroup>',
    '<col span="2"><colgroup>',
])
def test_omitted_column_group_tags_preserve_group_header_associations(groups: str) -> None:
    rows = '<tr><th scope="colgroup" colspan="2">Shared<th scope="col">Other<tr><td>a<td>b<td>c'
    document = parse('<table>' + groups + rows + '</table>')
    values = {c.text: c for c in cells(document)}
    assert [h.text for h in values['a'].headers] == ['Shared']
    assert [h.text for h in values['b'].headers] == ['Shared']
    assert [h.text for h in values['c'].headers] == ['Other']


def test_intervening_header_blocks_stop_older_headers_of_same_shape() -> None:
    document = parse('<table><tr><th>Old<tr><td>one<tr><th>New<tr><td>two</table>')
    values = {c.text: c for c in cells(document)}
    assert [h.text for h in values['two'].headers] == ['New']


def test_source_text_and_header_amplification_limits_apply() -> None:
    raw = '<table><tr><th>' + 'H' * 200 + ''.join(f'<tr><td>{n}' for n in range(50)) + '</table>'
    with pytest.raises(InvalidInput, match='text byte limit'):
        parse_text(raw.encode(), 'table.html', DocumentLimits(max_text_bytes=1000))


def test_empty_cells_keep_their_column_and_nonempty_cell_offsets() -> None:
    document = parse('<table><tr><td><td>é<td></table>')
    assert [(c.text, c.column) for c in cells(document)] == [('', 0), ('é', 1), ('', 2)]
    assert cells(document)[1].end - cells(document)[1].start == 2


@pytest.mark.parametrize('empty', ['<pre>  \n  </pre>', '\u00a0', ' '])
def test_whitespace_only_rows_do_not_overwrite_prior_segments(empty: str) -> None:
    document = parse('<p>Before</p><table><tr><td>' + empty + '<tr><td>value</table>')
    assert document.segments[0].text == 'Before'
    assert document.segments[0].table_cells == ()
    assert cells(document)[0].row == 1
    header = parse('<table><tr><th>' + empty + '<tr><td>value</table>')
    assert cells(header)[0].headers == ()
    with pytest.raises(InvalidInput, match='no extractable text'):
        parse('<table><tr><td>' + empty + '</table>')


def test_zero_span_cannot_cross_separate_implicit_row_groups() -> None:
    document = parse('<table><tr><th rowspan="0">First<td>one'
                     '<tbody><tr><td>middle</tbody><tr><td>last</table>')
    assert cells(document)[0].row_span == 1


def test_footer_before_body_is_positioned_after_body_without_inventing_headers() -> None:
    document = parse('<table><tfoot><tr><th>Total<th>30</tfoot>'
                     '<tbody><tr><td>A<td>10<tr><td>B<td>20</tbody></table>')
    assert [c.text for c in cells(document)] == ['A', '10', 'B', '20', 'Total', '30']
    assert cells(document)[0].row == 0
    assert cells(document)[-1].row == 2
    assert cells(document)[0].headers == ()
    assert cells(document)[-1].locator == 'table:1/cell:2'
