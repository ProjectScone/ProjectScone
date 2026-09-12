"""Notebook sources and saved observations retain separate, exact cell locators."""
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser, DocumentLimits, document_provenance, ingest_document
from scone_memory.ingestion.formats.capabilities import document_formats


def notebook(cells):
    return json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': cells}, ensure_ascii=False).encode()


def cell(source='Café Polaris', kind='markdown', **kwargs):
    return {'cell_type': kind, 'metadata': {}, 'source': source, **kwargs}


async def parse(raw, limits=DocumentLimits()):
    return await BuiltinDocumentParser().parse(raw, 'analysis.ipynb', limits)


async def test_notebook_sources_outputs_and_binary_omissions_have_distinct_evidence():
    parsed = await parse(notebook([
        cell(['# Café\n', 'Measurements'], id='intro'),
        cell('print(2)', 'code', execution_count=7, outputs=[
            {'output_type': 'stream', 'name': 'stdout', 'text': ['2', '\n']},
            {'output_type': 'execute_result', 'execution_count': 7, 'metadata': {},
             'data': {'text/plain': ['value = 2'], 'text/html': '<b>duplicate representation</b>'}},
            {'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': 'aW1hZ2U='}},
            {'output_type': 'display_data', 'metadata': {}, 'data': {'text/html': '<p>Visible <b>report</b></p><script>hidden()</script>'}},
            {'output_type': 'error', 'ename': 'ValueError', 'evalue': 'bad input', 'traceback': ['Trace line', 'ValueError: bad input']},
        ]),
        cell('literal raw cell', 'raw'),
    ]))
    assert parsed.format == 'ipynb'
    assert [s.text for s in parsed.segments] == ['# Café\nMeasurements', 'print(2)', '2\n',
        'value = 2', 'Visible report', 'Trace line\nValueError: bad input', 'literal raw cell']
    assert parsed.segments[0].locator == '#/cells/0/source'
    assert parsed.segments[0].metadata['cell_id'] == 'intro'
    assert parsed.segments[2].locator == '#/cells/1/outputs/0/text'
    assert parsed.segments[2].metadata['evidence_kind'] == 'saved_output'
    assert parsed.segments[1].metadata['evidence_kind'] == 'cell_source'
    assert parsed.metadata['outputs_without_text'] == '1'
    assert document_formats()['.ipynb']['available'] is True


@pytest.mark.parametrize('raw', [
    b'{"nbformat":4,"nbformat":3,"cells":[]}',
    b'{"nbformat":3,"worksheets":[]}',
    b'{"nbformat":true,"cells":[]}',
    b'{"nbformat":4,"cells":{}}',
    notebook([cell(['valid', 12])]),
    notebook([cell('source', 'unknown')]),
    notebook([cell('source', 'code', outputs='not a list')]),
    notebook([cell('source', 'code', outputs=[{'output_type': 'stream', 'text': 4}])]),
    notebook([cell('source', 'code', outputs=[{'output_type': 'unknown'}])]),
    b'{"nbformat":4,"cells":[{"cell_type":"raw","source":"\\ud800"}]}',
])
async def test_malformed_notebook_is_rejected(raw):
    with pytest.raises(InvalidInput):
        await parse(raw)


async def test_notebook_limits_include_empty_cells_and_saved_output_text():
    with pytest.raises(InvalidInput, match='cell|segment'):
        await parse(notebook([cell(''), cell(''), cell('source')]), DocumentLimits(max_segments=2))
    with pytest.raises(InvalidInput, match='byte'):
        await parse(notebook([cell('x', 'code', outputs=[{'output_type': 'stream', 'name': 'stdout', 'text': 'é' * 10}])]),
                    DocumentLimits(max_text_bytes=20))


async def test_missing_traceback_uses_actual_fields_and_empty_plain_output_can_fall_back_to_html():
    parsed = await parse(notebook([cell('x', 'code', outputs=[
        {'output_type': 'error', 'ename': 'ValueError', 'evalue': 'bad input', 'traceback': []},
        {'output_type': 'display_data', 'data': {'text/plain': '', 'text/html': '<p>Report</p>'}},
        {'output_type': 'stream', 'name': 'stdout', 'text': ''},
    ])]))
    assert [(s.locator, s.text) for s in parsed.segments[1:]] == [
        ('#/cells/0/outputs/0/ename', 'ValueError'),
        ('#/cells/0/outputs/0/evalue', 'bad input'),
        ('#/cells/0/outputs/1/data/text~1html', 'Report'),
    ]
    assert parsed.metadata['outputs_without_text'] == '1'


async def test_notebook_code_is_retained_without_execution_and_recall_cites_its_cell(tmp_path):
    marker = tmp_path / 'must-not-exist'
    code = f'open({str(marker)!r}, "w").write("executed")'
    raw = notebook([cell(code, 'code', outputs=[]), cell('Café calibration uses Polaris')])
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        result = await ingest_document(memory, 'alpha', raw, filename='analysis.ipynb')
        assert not marker.exists()
        recalled = await memory.recall('alpha', 'Polaris')
        chunk = next(item for item in recalled.items if item.episode_id == result.added.episode_id)
        evidence = await document_provenance(memory, 'alpha', result.added.episode_id, chunk_id=chunk.chunk_id)
        assert any(s.locator == '#/cells/1/source' for s in evidence.segments)
        assert (await memory.attachment('alpha', result.original.attachment_id))[1] == raw
    finally:
        await memory.close()
