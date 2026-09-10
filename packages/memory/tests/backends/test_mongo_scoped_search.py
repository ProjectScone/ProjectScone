"""Real Mongo lexical filtering must precede candidate truncation."""

from dataclasses import replace
import asyncio
import os
from uuid import uuid4

import pytest

from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import MongoDocumentStore, QdrantVectorIndex
from scone_memory.core.ports import TextFilter
from scone_memory.retrieval.filters import parse_filter
from ..retrieval.test_scoped_recall import (
    test_episode_scope_finds_older_file_beyond_a_thousand_distractors as scope_distractors,
    test_scope_intersects_metadata_tags_space_and_inclusive_times as scope_intersection,
    test_empty_source_prefix_excludes_missing_source_before_limit as scope_empty_prefix,
)


@pytest.fixture
async def mongo_engine():
    uri = os.environ.get("SCONE_TEST_MONGO_URL")
    if not uri:
        pytest.skip("set SCONE_TEST_MONGO_URL to run real Mongo scope contracts")
    identity = "scone_scope_" + uuid4().hex
    documents = MongoDocumentStore(uri, identity)
    await documents.open()
    qdrant = os.environ.get("SCONE_TEST_QDRANT_URL")
    vectors = QdrantVectorIndex(qdrant, identity) if qdrant else InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), chunk_target=2000).open()
    try:
        yield engine
    finally:
        try:
            if qdrant:
                await vectors.drop()
        finally:
            try:
                await documents.drop()
            finally:
                await engine.close()


async def test_mongo_scope_recovers_all_four_targets_beyond_thousand_distractors(mongo_engine):
    await scope_distractors(mongo_engine)


async def test_mongo_scope_intersects_normalized_times_tags_metadata_and_conditions(mongo_engine):
    await scope_intersection(mongo_engine)


async def test_mongo_empty_prefix_requires_a_retained_nonnull_source(mongo_engine):
    await scope_empty_prefix(mongo_engine)


@pytest.mark.parametrize("prefix", ["/literal/[a].*+?^$(){}|\\/", "/Literal/%_/", "/Literal/\x00/"])
async def test_mongo_source_prefix_is_literal_and_case_sensitive(mongo_engine, prefix):
    await mongo_engine.remember("alpha", "quasar deployment", source="/literal/aaaaaaaa/")
    await mongo_engine.remember("alpha", "quasar deployment case variant", source=prefix.swapcase() + "other")
    wanted = await mongo_engine.remember("alpha", "quasar deployment " + "details " * 40, source=prefix + "wanted")
    result = await mongo_engine.documents.search_text("alpha", "quasar deployment", 1, TextFilter(source_prefix=prefix))
    assert [identifier for identifier, _ in result] == [(await mongo_engine.documents.chunks_of("alpha", wanted.episode_id))[0].chunk_id]


@pytest.mark.parametrize("condition", [
    {"field": "status", "is": "published"},
    {"all": [{"field": "priority", "at_least": 10}, {"field": "status", "is": "draft", "not": True}]},
    {"any": [{"field": "priority", "above": 19}, {"field": "owner", "has": "ops"}]},
])
async def test_mongo_conditions_are_exact_before_limit(mongo_engine, condition):
    for number in range(35):
        await mongo_engine.remember("alpha", f"quasar deployment {number}",
                                    metadata={"status": "draft", "priority": "9", "owner": "other"})
    await mongo_engine.remember("alpha", "quasar deployment missing metadata")
    wanted = await mongo_engine.remember("alpha", "quasar deployment " + "details " * 40,
        metadata={"status": "published", "priority": "20", "owner": "ops"})
    result = await mongo_engine.documents.search_text("alpha", "quasar deployment", 1,
                                                       TextFilter(conditions=parse_filter(condition)))
    assert [identifier for identifier, _ in result] == [(await mongo_engine.documents.chunks_of("alpha", wanted.episode_id))[0].chunk_id]


async def test_mongo_join_excludes_cross_space_and_deleted_sources(mongo_engine):
    source = await mongo_engine.remember("beta", "quasar deployment", kind="file", source="")
    chunk = (await mongo_engine.documents.chunks_of("beta", source.episode_id))[0]
    documents = mongo_engine.documents
    await documents.chunks.update_one({"_id": chunk.chunk_id}, {"$set": {"space": "alpha"}})
    assert await documents.search_text("alpha", "quasar deployment", 1, TextFilter(kind="file")) == []
    await documents.episodes.delete_one({"_id": source.episode_id})
    assert await documents.search_text("alpha", "quasar deployment", 1, TextFilter(source_prefix="")) == []
    # The unscoped lane retains its old contract; the engine independently
    # revalidates orphan candidates before exposing any episode content.
    assert [identifier for identifier, _ in await documents.search_text("alpha", "quasar deployment", 1, TextFilter())] == [chunk.chunk_id]


async def test_mongo_as_of_remains_a_chunk_time_filter(mongo_engine):
    source = await mongo_engine.remember("alpha", "quasar deployment", kind="observation", created_at="2020-01-01")
    chunk = (await mongo_engine.documents.chunks_of("alpha", source.episode_id))[0]
    await mongo_engine.documents.chunks.update_one({"_id": chunk.chunk_id}, {"$set": {"created_at": "2030-01-01T00:00:00.000Z"}})
    scope = TextFilter(kind="observation", since="2020-01-01T00:00:00.000Z", until="2020-01-01T00:00:00.000Z")
    assert await mongo_engine.documents.search_text("alpha", "quasar deployment", 1,
                                                    replace(scope, as_of="2025-01-01T00:00:00.000Z")) == []
    assert [identifier for identifier, _ in await mongo_engine.documents.search_text(
        "alpha", "quasar deployment", 1, replace(scope, as_of="2030-01-01T00:00:00.000Z"))] == [chunk.chunk_id]


@pytest.mark.parametrize("cancel", [False, True])
async def test_mongo_condition_cursor_closes_after_limit_or_cancellation(mongo_engine, monkeypatch, cancel):
    for number in range(3):
        await mongo_engine.remember("alpha", f"quasar deployment {number}", metadata={"status": "published"})
    original = mongo_engine.documents.chunks.aggregate
    entered, blocked = asyncio.Event(), asyncio.Event()
    closed = False

    async def aggregate(*args, **kwargs):
        cursor = await original(*args, **kwargs)

        class TrackedCursor:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if cancel:
                    entered.set()
                    await blocked.wait()
                return await cursor.__anext__()

            async def close(self):
                nonlocal closed
                await cursor.close()
                closed = True

        return TrackedCursor()

    monkeypatch.setattr(mongo_engine.documents.chunks, "aggregate", aggregate)
    operation = asyncio.create_task(mongo_engine.documents.search_text("alpha", "quasar deployment", 1,
        TextFilter(conditions=parse_filter({"field": "status", "is": "published"}))))
    if cancel:
        await asyncio.wait_for(entered.wait(), 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert len(await operation) == 1
    assert closed
