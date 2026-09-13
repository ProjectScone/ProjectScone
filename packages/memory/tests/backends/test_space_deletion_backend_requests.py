"""Offline adapter contracts: scoped inventory and durable catalog visibility."""
from unittest.mock import AsyncMock, Mock

import pytest

from scone_memory.core.errors import InvalidInput
from .test_space_deletion_catalog import intent


def adapter(backend, monkeypatch):
    if backend == 'postgres':
        pytest.importorskip('psycopg_pool')
        pytest.importorskip('pgvector.psycopg')
        from scone_memory.backends.postgres import PostgresDocumentStore
        store = PostgresDocumentStore('postgresql://unused')
        monkeypatch.setattr(store, '_row', AsyncMock())
        monkeypatch.setattr(store, '_rows', AsyncMock(return_value=[]))
        return store
    if backend == 'mongo':
        pytest.importorskip('motor.motor_asyncio')
        from scone_memory.backends.mongo import MongoDocumentStore
        store = object.__new__(MongoDocumentStore)
        store._space_deletions = AsyncMock()
        return store
    pytest.importorskip('elasticsearch')
    from scone_memory.backends.elastic import ElasticsearchDocumentStore
    return ElasticsearchDocumentStore(client=AsyncMock(), refresh=False)


@pytest.mark.parametrize('backend', ['postgres', 'mongo', 'elastic'])
@pytest.mark.parametrize('write', [False, True])
async def test_foreign_payload_refused_at_point_boundary(backend, write, monkeypatch):
    store = adapter(backend, monkeypatch)
    row = {'space': 'foreign', '_id': 'foreign', 'payload': intent('foreign').model_dump_json()}
    if backend == 'postgres':
        store._row.return_value = row
    elif backend == 'mongo':
        store._space_deletions.find_one.return_value = row
    else:
        monkeypatch.setattr(store, '_doc', AsyncMock(return_value=row))
    with pytest.raises(InvalidInput, match='identity'):
        if write:
            await store.record_space_deletion(intent())
        else:
            await store.space_deletion('alpha')


@pytest.mark.parametrize('backend', ['postgres', 'mongo', 'elastic'])
async def test_invalid_page_cursor_refuses_before_requests(backend, monkeypatch):
    store = adapter(backend, monkeypatch)
    for cursor in [True, 'alpha\n', ('alpha',), 1]:
        with pytest.raises(InvalidInput):
            await store.page_space_deletions(cursor, 1)


async def test_elastic_catalog_forces_refresh_even_when_bulk_refresh_disabled(monkeypatch):
    store = adapter('elastic', monkeypatch)
    row = {'space': 'alpha', 'payload': intent().model_dump_json()}
    monkeypatch.setattr(store, '_doc', AsyncMock(return_value=row))
    store.client.search.return_value = {'hits': {'hits': [{'_source': row}]}}
    assert await store.record_space_deletion(intent()) == intent()
    assert store.client.index.call_args.kwargs['refresh'] is True
    assert store.client.index.call_args.kwargs['op_type'] == 'create'
    assert await store.page_space_deletions('aardvark', 1) == [intent()]
    assert store.client.search.call_args.kwargs['search_after'] == ['aardvark']
    assert store.client.search.call_args.kwargs['size'] == 1
    await store.clear_space_deletion('alpha')
    assert store.client.delete.call_args.kwargs['refresh'] is True


async def test_postgres_first_value_conflict_and_parameterized_page(monkeypatch):
    store = adapter('postgres', monkeypatch)
    store._row.side_effect = [None, {'payload': intent().model_dump_json()}]
    assert await store.record_space_deletion(intent()) == intent()
    sql, values = store._row.call_args_list[0].args
    assert values == ('alpha', intent().model_dump_json()) and 'ON CONFLICT' in sql
    await store.page_space_deletions('alpha', 2)
    assert store._rows.call_args.args[1] == ('alpha', 2)
    assert 'LIMIT %s' in store._rows.call_args.args[0]


async def test_postgres_chunk_inventory_is_scoped_and_sorted(monkeypatch):
    store = adapter('postgres', monkeypatch)
    store._rows.return_value = [{'id': 1, 'episode_id': 4}, {'id': 2, 'episode_id': 4}]
    assert await store.chunk_index('alpha') == [(1, 4), (2, 4)]
    sql, values = store._rows.call_args.args
    assert values == ('alpha',) and 'WHERE space = %s ORDER BY id' in sql


async def test_mongo_chunk_inventory_is_scoped(monkeypatch):
    store = adapter('mongo', monkeypatch)
    cursor = AsyncMock()
    cursor.__aiter__.return_value = [{'_id': 1, 'episode_id': 4}]
    store.chunks = Mock()
    store.chunks.find.return_value.sort.return_value = cursor
    assert await store.chunk_index('alpha') == [(1, 4)]
    assert store.chunks.find.call_args.args == ({'space': 'alpha'}, {'_id': 1, 'episode_id': 1})


async def test_elastic_chunk_inventory_walks_past_first_page(monkeypatch):
    store = adapter('elastic', monkeypatch)
    monkeypatch.setattr(store, 'PAGE', 2)
    store.client.search.side_effect = [
        {'hits': {'hits': [{'_source': {'chunk_id': i, 'episode_id': 4}, 'sort': [i]} for i in [1, 2]]}},
        {'hits': {'hits': [{'_source': {'chunk_id': 3, 'episode_id': 5}, 'sort': [3]}]}},
    ]
    assert await store.chunk_index('alpha') == [(1, 4), (2, 4), (3, 5)]
    for call in store.client.search.call_args_list:
        assert call.kwargs['query'] == {'term': {'space': 'alpha'}}
        assert call.kwargs['size'] == 2
    assert store.client.search.call_args.kwargs['search_after'] == [2]
