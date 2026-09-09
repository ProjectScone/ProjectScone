"""Graph fact reads apply exact source/space filtering before bounded output."""
from unittest.mock import AsyncMock
import asyncio

import pytest

from scone_memory.core.ports import NewFact

STAMP = '2026-01-01T00:00:00.000Z'


async def test_graph_fact_port_keeps_history_and_filters_before_limit(engine):
    from scone_memory.core.graph_read import GraphFactReader
    store = engine.documents
    assert isinstance(store, GraphFactReader)
    facts = []
    for space, source, status in [('alpha', 8, 'active'), ('foreign', 9, 'active'),
        ('alpha', 9, 'closed'), ('alpha', 9, 'proposed'), ('alpha', None, 'active')]:
        facts.append(await store.insert_fact(NewFact(space, 'subject', 'uses', 'object', STAMP,
            status=status, source_episode_id=source)))
    assert await store.facts_for_graph('alpha', 9, 1) == [facts[2]]
    assert await store.facts_for_graph('alpha', 9, 2001) == facts[2:4]
    assert await store.facts_for_graph('alpha', None, 2001) == [facts[0], *facts[2:]]
    assert await store.facts_for_graph('alpha', 10, 2001) == []
    assert await store.facts_for_graph('alpha', None, 0) == []
    assert await store.facts_for_graph('alpha', None, -1) == []


@pytest.mark.parametrize('source,limit', [(True, 10), (0, 10), (-1, 10), ('8', 10), (None, True), (None, 1.5)])
async def test_graph_fact_port_rejects_invalid_inputs(engine, source, limit):
    with pytest.raises(ValueError):
        await engine.documents.facts_for_graph('alpha', source, limit)


async def test_graph_never_calls_full_ledger_scan_and_reports_fact_budget(engine):
    for index in range(6):
        await engine.documents.insert_fact(NewFact('alpha', f's{index}', 'uses', 'object', STAMP))
    engine.documents.list_facts = AsyncMock(side_effect=AssertionError('unbounded ledger scan'))
    graph = await engine.graph('alpha', fact_limit=2)
    assert sum(node.kind == 'claim' for node in graph.nodes.values()) == 2
    assert graph.truncated and graph.facts_truncated
    assert graph.fact_read_status == 'bounded'
    assert graph.as_dict()['facts_truncated'] is True


async def test_focused_graph_finds_old_source_among_unrelated_facts(engine):
    episode = await engine.remember('alpha', 'Target source')
    for index in range(12):
        await engine.documents.insert_fact(NewFact('alpha', f's{index}', 'uses', 'object', STAMP))
    wanted = await engine.documents.insert_fact(NewFact('alpha', 'target', 'uses', 'Polaris', STAMP,
        source_episode_id=episode.episode_id))
    engine.documents.list_facts = AsyncMock(side_effect=AssertionError('unbounded ledger scan'))
    graph = await engine.graph('alpha', episode_id=episode.episode_id, fact_limit=2)
    assert [node.id for node in graph.nodes.values() if node.kind == 'claim'] == [f'claim:{wanted.fact_id}']
    assert not graph.facts_truncated


async def test_unsupported_store_reports_claims_unavailable_without_legacy_scan(engine):
    engine.documents.facts_for_graph = None
    engine.documents.list_facts = AsyncMock(side_effect=AssertionError('unbounded fallback'))
    graph = await engine.graph('alpha')
    assert graph.fact_read_status == 'unavailable' and graph.truncated
    assert not any(node.kind == 'claim' for node in graph.nodes.values())


async def test_invalid_cross_space_results_are_not_drawn(engine):
    foreign = await engine.documents.insert_fact(NewFact('foreign', 'PRIVATE', 'is', 'private', STAMP))
    engine.documents.facts_for_graph = AsyncMock(return_value=[foreign])
    graph = await engine.graph('alpha')
    assert graph.fact_read_status == 'failed' and graph.truncated
    assert 'PRIVATE' not in str(graph.as_dict())


async def test_memory_source_index_tracks_updates_and_space_deletion():
    from scone_memory import InMemoryDocumentStore
    store = InMemoryDocumentStore()
    original = await store.insert_fact(NewFact('alpha', 'subject', 'uses', 'object', STAMP, source_episode_id=8))
    changed = original.model_copy(update={'source_episode_id':9})
    await store.update_fact(changed)
    assert await store.facts_for_graph('alpha', 8, 10) == []
    assert await store.facts_for_graph('alpha', 9, 10) == [changed]
    assert await store.facts_for_graph('alpha', None, 10) == [changed]
    await store.delete_space('alpha', STAMP)
    assert await store.facts_for_graph('alpha', 9, 10) == []
    assert await store.facts_for_graph('alpha', None, 10) == []


async def test_sqlite_actual_fact_queries_use_source_and_space_indexes():
    from scone_memory.backends import SqliteDocumentStore
    store = SqliteDocumentStore(':memory:')
    try:
        await store.insert_fact(NewFact('alpha', 'subject', 'uses', 'object', STAMP, source_episode_id=8))
        statements = []
        store.conn.set_trace_callback(statements.append)
        for source in (8, None):
            statements.clear()
            await store.facts_for_graph('alpha', source, 10)
            [sql] = [text for text in statements if text.startswith('SELECT * FROM facts')]
            plan = ' '.join(row[3] for row in store.conn.execute('EXPLAIN QUERY PLAN ' + sql))
            assert 'SEARCH facts USING INDEX' in plan
            assert ('facts_source_id' if source else 'facts_space_id') in plan
            assert 'SCAN' not in plan and 'TEMP B-TREE' not in plan
    finally:
        await store.close()


async def test_reader_shares_fact_budget_across_source_groups_and_flags_unvisited_groups():
    from scone_memory import InMemoryDocumentStore
    from scone_memory.retrieval.activity_facts import read_activity_facts
    store = InMemoryDocumentStore()
    for source in (9, 8, 8):
        await store.insert_fact(NewFact('alpha', 'subject', 'uses', 'object', STAMP, source_episode_id=source))
    original = store.facts_for_graph
    store.facts_for_graph = AsyncMock(wraps=original)
    result = await read_activity_facts(store, 'alpha', {8, 9}, 2)
    assert len(result.facts) == 2 and {fact.source_episode_id for fact in result.facts} == {8}
    assert result.truncated and result.status == 'bounded'
    store.facts_for_graph.assert_awaited_once_with('alpha', 8, 3)


@pytest.mark.parametrize('kind', ['duplicate', 'wrong_source', 'oversized', 'provider_failure'])
async def test_reader_discards_invalid_or_failed_snapshots(kind):
    from scone_memory import InMemoryDocumentStore
    from scone_memory.retrieval.activity_facts import read_activity_facts
    store = InMemoryDocumentStore()
    one = await store.insert_fact(NewFact('alpha', 'subject', 'uses', 'object', STAMP, source_episode_id=8))
    rows = {'duplicate':[one, one], 'wrong_source':[one.model_copy(update={'source_episode_id':9})],
        'oversized':[one] * 4, 'provider_failure':[]}[kind]
    store.facts_for_graph = AsyncMock(return_value=rows,
        side_effect=RuntimeError('PRIVATE backend detail') if kind == 'provider_failure' else None)
    result = await read_activity_facts(store, 'alpha', {8}, 2)
    assert result.facts == [] and result.truncated and result.status == 'failed'
    assert 'PRIVATE' not in str(result)


async def test_reader_propagates_cancellation():
    from scone_memory import InMemoryDocumentStore
    from scone_memory.retrieval.activity_facts import read_activity_facts
    store = InMemoryDocumentStore()
    entered = asyncio.Event()
    async def wait(*args):
        entered.set()
        await asyncio.Event().wait()
    store.facts_for_graph = wait
    task = asyncio.create_task(read_activity_facts(store, 'alpha', None, 2))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_reader_rejects_duplicate_fact_ids_across_source_groups():
    from scone_memory import InMemoryDocumentStore
    from scone_memory.retrieval.activity_facts import read_activity_facts
    store = InMemoryDocumentStore()
    one = await store.insert_fact(NewFact('alpha', 'subject', 'uses', 'object', STAMP, source_episode_id=8))
    other = one.model_copy(update={'source_episode_id': 9})
    store.facts_for_graph = AsyncMock(side_effect=[[one], [other]])
    result = await read_activity_facts(store, 'alpha', {8, 9}, 3)
    assert result.facts == [] and result.truncated and result.status == 'failed'


async def test_mongo_bounded_query_never_uses_zero_as_an_unlimited_cursor():
    from scone_memory.backends.mongo import MongoDocumentStore
    calls = []
    class Cursor:
        def find(self, query):
            calls.append(('find', query))
            return self
        def sort(self, key, direction):
            calls.append(('sort', key, direction))
            return self
        def limit(self, value):
            calls.append(('limit', value))
            return self
        async def __aiter__(self):
            for row in []:
                yield row
    store = object.__new__(MongoDocumentStore)
    store.facts = Cursor()
    assert await store.facts_for_graph('alpha', 8, 0) == [] and calls == []
    assert await store.facts_for_graph('alpha', 8, 99999) == []
    assert calls == [('find', {'space':'alpha', 'source_episode_id':8}), ('sort', '_id', 1), ('limit', 2001)]


async def test_http_fact_budget_is_validated_and_partial_status_is_visible():
    from fastapi.testclient import TestClient
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api import create_app
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
        events=InMemoryEventLog()).open()
    try:
        for index in range(3):
            await engine.documents.insert_fact(NewFact('alpha', f's{index}', 'uses', 'object', STAMP))
        with TestClient(create_app(engine, {'test-key':'alpha'})) as client:
            headers = {'Authorization':'Bearer test-key'}
            data = client.get('/v1/graph?fact_limit=1', headers=headers).json()
            assert data['fact_read_status'] == 'bounded' and data['facts_truncated'] and data['truncated']
            assert len([node for node in data['nodes'] if node['kind'] == 'claim']) == 1
            for value in ('0', '2001', '1.5', 'no'):
                assert client.get('/v1/graph?fact_limit=' + value, headers=headers).status_code == 422
    finally:
        await engine.close()
