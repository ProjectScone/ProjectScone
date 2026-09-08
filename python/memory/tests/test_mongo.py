"""Mongo-specific adapter regressions; live tests use disposable databases."""
from __future__ import annotations

import inspect
import os
from uuid import uuid4

import pytest

from scone_memory.backends.mongo import MongoDocumentStore
from scone_memory.core.errors import SconeError
from scone_memory.core.ports import NewFact, NewFactLink, NewTombstone


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


@pytest.mark.mongo
async def test_bounded_graph_reads_keep_scope_order_and_hard_caps() -> None:
    url = os.environ.get("SCONE_TEST_MONGO_URL")
    if not url:
        pytest.skip("SCONE_TEST_MONGO_URL is not configured")
    store = await MongoDocumentStore(url, "scone_test_" + uuid4().hex).open()
    when = "2025-01-01T00:00:00Z"
    try:
        facts = [await store.insert_fact(NewFact("alpha", "Beacon", "owns", str(i), valid_from=when))
                 for i in range(135)]
        await store.insert_fact(NewFact("beta", "Beacon", "owns", "private", valid_from=when))
        links = [await store.insert_fact_link(NewFactLink("alpha", 1 if i % 2 else i + 2,
                 i + 2 if i % 2 else 1, "supports", when)) for i in range(135)]
        foreign = await store.insert_fact_link(NewFactLink("beta", 1, 2, "supports", when))
        assert await store.facts_by_subject("alpha", "Beacon", 2) == facts[:2]
        assert len(await store.facts_by_subject("alpha", "Beacon", 10_000)) == 129
        assert await store.fact_links_from("alpha", 1, 2) == links[:2]
        assert len(await store.fact_links_from("alpha", 1, 10_000)) == 129
        assert await store.fact_links_between("alpha", [1, 2, 3, 3], 49) == links[:2]
        assert await store.fact_links_between("beta", [1, 2], 49) == [foreign]
        assert await store.fact_links_between("alpha", [], 49) == []
        assert await store.fact_links_between("alpha", [1, 2, 3], 1) == links[:1]
        for limit in (0, -1):
            assert await store.facts_by_subject("alpha", "Beacon", limit) == []
            assert await store.fact_links_from("alpha", 1, limit) == []
            assert await store.fact_links_between("alpha", [1, 2], limit) == []
        assert await store.get_fact_link("alpha", links[0].link_id) == links[0]
        assert await store.get_fact_link("alpha", foreign.link_id) is None
        assert await store.facts_by_subject("alpha", "missing", 2) == []
        for target in range(2, 17):
            for kind in ("extends", "derived_from", "contradicts"):
                await store.insert_fact_link(NewFactLink("alpha", 1, target, kind, when))
        bounded = await store.fact_links_between("alpha", list(range(1, 140)), 10_000)
        assert len(bounded) == 49
        assert all(1 <= row.from_fact <= 16 and 1 <= row.to_fact <= 16 for row in bounded)
        indexes = await store.facts.index_information()
        assert [("space", 1), ("subject", 1), ("_id", 1)] in [v["key"] for v in indexes.values()]
    finally:
        await store.drop()
        await store.close()
