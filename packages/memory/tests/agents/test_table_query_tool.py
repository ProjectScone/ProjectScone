"""The table tool in the agent loop: words become arguments, the computation stays exact.

An agent that can search can find a spreadsheet; what it could not do
is add a column up without a model doing arithmetic in prose. With the
tool on, it lists a document's tables and asks a structured question,
and what comes back is the same exact answer as over HTTP, every cell
quoted and tied to the chunk that holds it, so the final answer's
sources can be checked like any other tool evidence.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.files import ingest_document
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope

CSV = b'region,revenue\r\nWest,"1,250.50"\r\nEast,35\r\nWest,20\r\n'


async def memory_with_csv():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    saved = await ingest_document(memory, 'alpha', CSV, filename='sales.csv')
    return memory, saved.added.episode_id


def tools(memory, **kwargs) -> ScopedMemoryTools:
    return ScopedMemoryTools(memory, 'alpha', scope=RecallScope.validated(), **kwargs)


async def test_the_tools_are_offered_only_when_enabled():
    memory, _ = await memory_with_csv()
    assert {row['name'] for row in tools(memory).anthropic()} == {'search_memory', 'trace_memory', 'read_memory'}
    offered = {row['name'] for row in tools(memory, enable_tables=True).anthropic()}
    assert {'list_tables', 'query_table'} <= offered
    denied = await tools(memory).run('query_table', {'episode_id': 1, 'operation': 'count'})
    assert denied['ok'] is False and denied['error'] == 'unknown_tool'
    with pytest.raises(ValueError):
        tools(memory, enable_tables='yes')  # type: ignore[arg-type]


async def test_listing_then_querying_returns_the_exact_answer_tied_to_chunks():
    memory, episode_id = await memory_with_csv()
    bound = tools(memory, enable_tables=True)
    listed = await bound.run('list_tables', {'episode_id': episode_id})
    assert listed['ok'] is True and listed['tables'][0]['columns'] == ['region', 'revenue']
    asked = await bound.run('query_table', {'episode_id': episode_id, 'operation': 'sum', 'column': 'revenue',
                                            'where': [{'column': 'region', 'op': '==', 'value': 'West'}]})
    assert asked['ok'] is True and asked['status'] == 'prepared' and asked['verified_accuracy'] is False
    table = asked['table']
    assert table['value'] == '1270.5' and [c['text'] for c in table['cells']] == ['1,250.50', '20']
    chunk_ids = {row['chunk_id'] for row in asked['items']}
    assert chunk_ids and all(c['chunk_id'] in chunk_ids for c in table['cells']), 'every quoted cell names the chunk that holds it'
    for row in asked['items']:
        for cell in table['cells']:
            if cell['chunk_id'] == row['chunk_id']:
                assert cell['text'] in row['text']
    assert asked['coverage']['mode'] == 'table_cells'


async def test_refusals_come_back_as_tool_errors_not_exceptions():
    memory, episode_id = await memory_with_csv()
    bound = tools(memory, enable_tables=True)
    refused = await bound.run('query_table', {'episode_id': episode_id, 'operation': 'sum', 'column': 'profit'})
    assert refused['ok'] is False and refused['error'] == 'column_not_found'
    missing = await bound.run('query_table', {'episode_id': episode_id + 99, 'operation': 'count'})
    assert missing['ok'] is False and missing['error'] == 'evidence_unavailable'
    malformed = await bound.run('query_table', {'episode_id': episode_id, 'operation': 'total'})
    assert malformed['ok'] is False and malformed['error'] == 'invalid_arguments'


async def test_scope_keeps_other_kinds_and_sessions_out():
    memory, episode_id = await memory_with_csv()
    narrowed = ScopedMemoryTools(memory, 'alpha', scope=RecallScope.validated(kind='conversation'), enable_tables=True)
    out_of_scope = await narrowed.run('query_table', {'episode_id': episode_id, 'operation': 'count'})
    assert out_of_scope['ok'] is False and out_of_scope['error'] == 'evidence_unavailable'
    assert (await narrowed.run('list_tables', {'episode_id': episode_id}))['ok'] is False


async def test_prepared_evidence_is_revalidated_after_the_source_is_forgotten():
    memory, episode_id = await memory_with_csv()
    bound = tools(memory, enable_tables=True)
    prepared = await bound.prepare('query_table', {'episode_id': episode_id, 'operation': 'count'})
    assert await prepared.validate() is True
    await memory.forget('alpha', episode_id)
    assert await prepared.validate() is False
