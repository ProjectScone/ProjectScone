"""Mongo-specific adapter regressions; live tests use disposable databases."""
from __future__ import annotations

import inspect
import os
from uuid import uuid4

import pytest

from scone_memory.backends.mongo import MongoDocumentStore
from scone_memory.core.errors import SconeError
from scone_memory.core.ports import NewFactLink, NewTombstone


async def test_fact_links_remains_an_async_store_operation() -> None:
    pytest.importorskip("pymongo")
    store = MongoDocumentStore("mongodb://127.0.0.1:1/?connect=false")
    try:
        operation = store.fact_links("isolated", 1)
        assert inspect.iscoroutine(operation)
        operation.close()
    finally:
        await store.close()


@pytest.mark.parametrize("operation", ["counter", "revision", "tombstone", "link"])
async def test_missing_write_result_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    pytest.importorskip("pymongo")
    from pymongo.errors import DuplicateKeyError

    async def missing(*args: object, **kwargs: object) -> None:
        return None

    async def duplicate(*args: object, **kwargs: object) -> None:
        raise DuplicateKeyError("synthetic duplicate")

    store = MongoDocumentStore("mongodb://127.0.0.1:1/?connect=false")
    when = "2026-01-01T00:00:00Z"
    try:
        with pytest.raises(SconeError, match="MongoDB"):
            if operation == "counter":
                monkeypatch.setattr(store.counters, "find_one_and_update", missing)
                await store._next_id("facts")
            elif operation == "revision":
                monkeypatch.setattr(store.revisions, "find_one_and_update", missing)
                await store.bump_revision("alpha")
            elif operation == "tombstone":
                monkeypatch.setattr(store.tombstones, "insert_one", missing)
                monkeypatch.setattr(store.tombstones, "find_one", missing)
                await store.record_tombstone(NewTombstone("alpha", 1, "hash", when))
            else:
                async def next_id(name: str) -> int:
                    return 1
                monkeypatch.setattr(store, "_next_id", next_id)
                monkeypatch.setattr(store._fact_links, "find_one", missing)
                monkeypatch.setattr(store._fact_links, "insert_one", duplicate)
                await store.insert_fact_link(NewFactLink("alpha", 1, 2, "supports", when))
    finally:
        await store.close()


@pytest.mark.mongo
async def test_fact_links_read_both_directions_and_erase_only_their_space() -> None:
    url = os.environ.get("SCONE_TEST_MONGO_URL")
    if not url:
        pytest.skip("SCONE_TEST_MONGO_URL is not configured")
    store = await MongoDocumentStore(url, "scone_test_" + uuid4().hex).open()
    when = "2026-01-01T00:00:00Z"
    try:
        first = await store.insert_fact_link(NewFactLink("alpha", 1, 2, "supports", when))
        second = await store.insert_fact_link(NewFactLink("alpha", 3, 1, "contradicts", when))
        foreign = await store.insert_fact_link(NewFactLink("beta", 1, 2, "supports", when))
        duplicate = await store.insert_fact_link(NewFactLink("alpha", 1, 2, "supports", when))
        assert duplicate == first
        assert await store.fact_links("alpha", 1) == [first, second]
        assert await store.fact_links("alpha", 2) == [first]
        assert await store.fact_links("alpha", 99) == []
        assert await store.fact_links("beta", 1) == [foreign]
        erased = await store.delete_space("alpha", when)
        assert erased.links == 2
        assert await store.fact_links("alpha", 1) == []
        assert await store.fact_links("beta", 1) == [foreign]
    finally:
        await store.drop()
        await store.close()
