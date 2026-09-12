"""Network-adapter recovery visibility and parameter boundaries, without services."""
from unittest.mock import AsyncMock

import pytest

from scone_memory.core.errors import InvalidInput
from .test_retirement_catalog import intent


async def test_elastic_bulk_refresh_setting_does_not_delay_retirement_visibility():
    pytest.importorskip("elasticsearch")
    from scone_memory.backends.elastic import ElasticsearchDocumentStore

    client = AsyncMock()
    record = intent()
    doc = {"space": record.space, "episode_id": record.episode_id, "payload": record.model_dump_json()}
    client.get.return_value = {"_source": doc}
    client.search.return_value = {"hits": {"hits": [{"_source": doc}]}}
    store = ElasticsearchDocumentStore(client=client, refresh=False)
    assert await store.record_retirement(record) == record
    assert client.index.call_args.kwargs["refresh"] is True
    assert client.index.call_args.kwargs["op_type"] == "create"
    assert await store.page_retirements(("alpha", 0 + 1), 1) == [record]
    assert client.search.call_args.kwargs["size"] == 1
    assert client.search.call_args.kwargs["search_after"] == ["alpha", 1]
    await store.clear_retirement("alpha", 1)
    assert client.delete.call_args.kwargs["refresh"] is True
    assert client.delete.call_args.kwargs["id"] == "alpha|1"


async def test_postgres_conflict_returns_original_and_keeps_values_parameterized(monkeypatch):
    pytest.importorskip("psycopg_pool")
    pytest.importorskip("pgvector.psycopg")
    from scone_memory.backends.postgres import PostgresDocumentStore

    store = PostgresDocumentStore("postgresql://unused")
    record = intent().model_copy(update={"content_hash": "literal ' quote"})
    row = {"space": "alpha", "episode_id": 1, "payload": record.model_dump_json()}
    fetch = AsyncMock(side_effect=[None, row])
    monkeypatch.setattr(store, "_row", fetch)
    assert await store.record_retirement(record) == record
    sql, values = fetch.call_args_list[0].args
    assert "literal ' quote" not in sql and "literal ' quote" in values[2]
    assert fetch.call_args_list[1].args[1] == ("alpha", 1)
    page = AsyncMock(return_value=[])
    monkeypatch.setattr(store, "_rows", page)
    assert await store.page_retirements(("alpha", 1), 2) == []
    assert page.call_args.args[1] == ("alpha", 1, 2)
    assert "LIMIT %s" in page.call_args.args[0]


@pytest.mark.parametrize("backend", ["postgres", "mongo", "elastic"])
async def test_invalid_cursor_is_refused_before_any_backend_call(backend):
    if backend == "postgres":
        pytest.importorskip("psycopg_pool")
        pytest.importorskip("pgvector.psycopg")
        from scone_memory.backends.postgres import PostgresDocumentStore
        store = PostgresDocumentStore("postgresql://unused")
    elif backend == "mongo":
        pytest.importorskip("motor.motor_asyncio")
        from scone_memory.backends.mongo import MongoDocumentStore
        store = object.__new__(MongoDocumentStore)
    else:
        pytest.importorskip("elasticsearch")
        from scone_memory.backends.elastic import ElasticsearchDocumentStore
        store = ElasticsearchDocumentStore(client=AsyncMock())
    for cursor in [("alpha", True), ("alpha\n", 1), ("alpha", 0), ("alpha", 1, 2)]:
        with pytest.raises(InvalidInput):
            await store.page_retirements(cursor, 1)


@pytest.mark.parametrize("backend", ["postgres", "mongo", "elastic"])
@pytest.mark.parametrize("write", [False, True])
async def test_point_operation_rejects_a_self_consistent_foreign_result(backend, write, monkeypatch):
    foreign = intent("foreign")
    row = {"space": "foreign", "episode_id": 1, "payload": foreign.model_dump_json()}
    if backend == "postgres":
        pytest.importorskip("psycopg_pool")
        pytest.importorskip("pgvector.psycopg")
        from scone_memory.backends.postgres import PostgresDocumentStore
        store = PostgresDocumentStore("postgresql://unused")
        monkeypatch.setattr(store, "_row", AsyncMock(return_value=row))
    elif backend == "mongo":
        pytest.importorskip("motor.motor_asyncio")
        from scone_memory.backends.mongo import MongoDocumentStore
        store = object.__new__(MongoDocumentStore)
        store._retirements = AsyncMock()
        store._retirements.find_one.return_value = row
    else:
        pytest.importorskip("elasticsearch")
        from scone_memory.backends.elastic import ElasticsearchDocumentStore
        store = ElasticsearchDocumentStore(client=AsyncMock())
        monkeypatch.setattr(store, "_doc", AsyncMock(return_value=row))
    with pytest.raises(InvalidInput, match="identity"):
        if write:
            await store.record_retirement(intent())
        else:
            await store.retirement("alpha", 1)
