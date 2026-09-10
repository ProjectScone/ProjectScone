"""Episode scope must reach lexical retrieval before its candidate limit."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.ports import TextFilter


@pytest.fixture(params=["memory", "sqlite"])
async def scoped_engine(request, tmp_path):
    if request.param == "sqlite":
        path = tmp_path / "scope.db"
        documents, vectors = SqliteDocumentStore(path), SqliteVectorIndex(path)
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), chunk_target=2000).open()
    yield engine
    await engine.close()


async def test_episode_scope_finds_older_file_beyond_a_thousand_distractors(scoped_engine):
    engine = scoped_engine
    old = await engine.remember(
        "alpha", "quasar deployment " + "archived supporting details " * 30,
        kind="file", source="/archive/runbook.md", created_at="2020-01-01")
    future = await engine.remember(
        "alpha", "quasar deployment " + "later supporting details " * 30,
        kind="observation", source="/future/note", created_at="2030-01-01")
    for n in range(1000):
        await engine.remember("alpha", f"quasar deployment {n}", kind="conversation",
                              source=f"/chat/{n}", created_at="2025-01-01")

    cases = [
        ({"kind": "file"}, old.episode_id),
        ({"source_prefix": "/archive/"}, old.episode_id),
        ({"until": "2020-01-01"}, old.episode_id),
        ({"since": "2030-01-01"}, future.episode_id),
    ]
    outcomes = []
    for scope, expected in cases:
        started = perf_counter()
        result = await engine.recall("alpha", "quasar deployment", limit=1, **scope)
        elapsed = (perf_counter() - started) * 1000
        ids = [item.episode_id for item in result.items]
        print(f"{engine.documents.name} {scope}: ids={ids}, expected={expected}, {elapsed:.2f} ms")
        outcomes.append(ids == [expected])
    assert all(outcomes), f"scoped retrieval coverage: {sum(outcomes)}/{len(outcomes)}"


async def test_scope_intersects_metadata_tags_space_and_inclusive_times(scoped_engine):
    engine = scoped_engine
    attrs = dict(kind="file", source="/Archive/%_notes/guide.md", created_at="2020-01-01",
                 tags=["approved"], metadata={"owner": "ops", "status": "published"})
    wanted = await engine.remember("alpha", "quasar deployment approved", **attrs)
    variants = [
        {"source": "/archive/%_notes/guide.md"},
        {"source": "/Archive/xxnotes/guide.md"},
        {"source": None},
        {"kind": "conversation"},
        {"created_at": "2019-12-31"},
        {"created_at": "2020-01-02"},
        {"tags": []},
        {"metadata": {"owner": "other", "status": "published"}},
        {"metadata": {"owner": "ops", "status": "draft"}},
    ]
    for n, changes in enumerate(variants):
        await engine.remember("alpha", f"quasar deployment {n}", **{**attrs, **changes})
    await engine.remember("beta", "quasar deployment foreign", **attrs)

    result = await engine.recall(
        "alpha", "quasar deployment", limit=1, kind="file", source_prefix="/Archive/%_notes/",
        since="2019-12-31T18:00:00-06:00", until="2020-01-01T00:00:00Z", as_of="2020-01-01",
        tags=["approved"], where={"owner": "ops"}, conditions={"field": "status", "is": "published"})
    assert [item.episode_id for item in result.items] == [wanted.episode_id]


async def test_empty_source_prefix_excludes_missing_source_before_limit(scoped_engine):
    engine = scoped_engine
    await engine.remember("alpha", "quasar deployment", source=None)
    wanted = await engine.remember("alpha", "quasar deployment " + "detail " * 50, source="")
    result = await engine.documents.search_text("alpha", "quasar deployment", 1, TextFilter(source_prefix=""))
    chunks = await engine.documents.chunks_of("alpha", wanted.episode_id)
    assert [chunk_id for chunk_id, _ in result] == [chunks[0].chunk_id]


async def test_store_ignoring_episode_scope_still_has_engine_postfilter(scoped_engine):
    engine = scoped_engine

    class LegacyStore:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        async def search_text(self, space, query, limit, filter):
            return await self.inner.search_text(
                space, query, limit, replace(filter, kind=None, source_prefix=None, since=None, until=None))

    await engine.remember("alpha", "quasar deployment", kind="conversation", source="/chat/1")
    engine.documents = LegacyStore(engine.documents)
    result = await engine.recall("alpha", "quasar deployment", kind="file", source_prefix="/archive/")
    assert result.items == []
