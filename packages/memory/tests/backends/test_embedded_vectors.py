"""Chroma, LanceDB and Milvus Lite as vector indexes. The published contract
runs over all three (tests/test_contract.py); this file holds what is
specific to them: filters built from user-supplied strings, persistence
across reopen, and the environment wiring. All run embedded, no server."""

from __future__ import annotations

import importlib.util

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InvalidInput, MemoryEngine
from scone_memory.runtime.config import Settings, build_engine, build_vectors
from scone_memory.core.ports import VectorPoint

needs_chroma = pytest.mark.skipif(importlib.util.find_spec("chromadb") is None, reason="chromadb not installed")
needs_lancedb = pytest.mark.skipif(importlib.util.find_spec("lancedb") is None, reason="lancedb not installed")
needs_milvus = pytest.mark.skipif(importlib.util.find_spec("milvus_lite") is None, reason="milvus-lite not installed")


def make(kind: str, tmp_path):
    if kind == "chroma":
        from scone_memory.backends import ChromaVectorIndex

        return ChromaVectorIndex(path=str(tmp_path / "chroma"))
    if kind == "milvus":
        from scone_memory.backends import MilvusVectorIndex

        return MilvusVectorIndex(str(tmp_path / "milvus.db"))
    from scone_memory.backends import LanceDBVectorIndex

    return LanceDBVectorIndex(str(tmp_path / "lance"))


@pytest.fixture(params=[pytest.param("chroma", marks=needs_chroma), pytest.param("lancedb", marks=needs_lancedb), pytest.param("milvus", marks=needs_milvus)])
def kind(request):
    return request.param


async def test_quotes_in_tags_and_values_are_data_not_syntax(kind, tmp_path):
    engine = await MemoryEngine(InMemoryDocumentStore(), make(kind, tmp_path), HashEmbedder()).open()
    await engine.remember("default", "the o'reilly book on sql", tags=["o'reilly", 'say "hi"'], metadata={"owner": "mark's", "b": "back\\slash"})
    await engine.remember("default", "another book on sql", tags=["plain"], metadata={"owner": "ana"})
    for tags in (["o'reilly"], ['say "hi"']):
        assert [i.episode_id for i in (await engine.recall("default", "book on sql", tags=tags)).items] == [1], tags
    for where in ({"owner": "mark's"}, {"b": "back\\slash"}):
        assert [i.episode_id for i in (await engine.recall("default", "book on sql", where=where)).items] == [1], where
    for hostile_value in ("x' OR 1=1 --", 'x" or space == "default', 'x" or 1 == 1'):
        hostile = await engine.recall("default", "book on sql", where={"owner": hostile_value})
        assert hostile.items == [] and not hostile.degraded, "a would-be predicate matches nothing and breaks nothing"


async def test_a_limit_beyond_the_index_size_and_an_empty_index_are_fine(kind, tmp_path):
    engine = await MemoryEngine(InMemoryDocumentStore(), make(kind, tmp_path), HashEmbedder()).open()
    empty = await engine.recall("default", "anything")
    assert empty.items == [] and not empty.degraded
    await engine.remember("default", "one lonely note")
    full = await engine.recall("default", "lonely note", limit=50)
    assert len(full.items) == 1 and not full.degraded


async def test_data_and_width_survive_a_reopen(kind, tmp_path):
    first = await MemoryEngine(InMemoryDocumentStore(), make(kind, tmp_path), HashEmbedder(dim=64)).open()
    await first.remember("default", "kept on disk")
    await first.vectors.close()
    reopened = make(kind, tmp_path)
    await reopened.ensure(64)
    [vec] = await HashEmbedder(dim=64).embed(["kept on disk"])
    assert [cid for cid, _ in await reopened.search("default", vec, 5)] == [1]
    with pytest.raises(ValueError, match="64-d"):
        await make(kind, tmp_path).ensure(32)


async def test_upsert_replaces_a_point_in_place(kind, tmp_path):
    index = make(kind, tmp_path)
    await index.ensure(2)
    await index.upsert([VectorPoint(1, "default", 1, "2025-01-01T00:00:00.000Z", [1.0, 0.0], tags=("old",))])
    await index.upsert([VectorPoint(1, "default", 1, "2025-01-01T00:00:00.000Z", [0.0, 1.0], tags=("new",))])
    assert await index.search("default", [0.0, 1.0], 5, tags=("new",)) == [(1, pytest.approx(1.0))]
    assert await index.search("default", [0.0, 1.0], 5, tags=("old",)) == [], "the old tags went with the old point"
    await index.delete([1])
    assert await index.search("default", [0.0, 1.0], 5) == []


@needs_chroma
async def test_chroma_from_the_environment(tmp_path):
    engine = await build_engine(Settings.from_env({"SCONE_VECTORS": "chroma", "SCONE_CHROMA_PATH": str(tmp_path / "c")}))
    assert engine.vectors.name == "chroma"
    assert build_vectors(Settings.from_env({"SCONE_VECTORS": "chroma"})).name == "chroma", "no path: in-process ephemeral"


@needs_lancedb
async def test_lancedb_from_the_environment(tmp_path):
    engine = await build_engine(Settings.from_env({"SCONE_VECTORS": "lancedb", "SCONE_LANCEDB_PATH": str(tmp_path / "l")}))
    assert engine.vectors.name == "lancedb"
    with pytest.raises(InvalidInput, match="SCONE_LANCEDB_PATH"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "lancedb"}))


@needs_milvus
async def test_milvus_from_the_environment(tmp_path):
    engine = await build_engine(Settings.from_env({"SCONE_VECTORS": "milvus", "SCONE_MILVUS_URI": str(tmp_path / "m.db")}))
    assert engine.vectors.name == "milvus"
    with pytest.raises(InvalidInput, match="SCONE_MILVUS_URI"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "milvus"}))
