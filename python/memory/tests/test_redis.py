"""Redis with RediSearch as a vector index. The published contract runs
over it from tests/test_contract.py when SCONE_TEST_REDIS_URL points at a
redis-stack server; this file holds what is specific to Redis: query
escaping, the width recorded in the index, and the environment wiring."""

from __future__ import annotations

import os
import uuid

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InvalidInput, MemoryEngine
from scone_memory.config import Settings, build_engine, build_vectors
from scone_memory.ports import VectorPoint

URL = os.environ.get("SCONE_TEST_REDIS_URL")
pytestmark = [pytest.mark.redis, pytest.mark.skipif(not URL, reason="needs a live Redis with RediSearch")]


@pytest.fixture
async def index():
    from scone_memory.backends import RedisVectorIndex

    idx = RedisVectorIndex(URL, prefix=f"scone_test_{uuid.uuid4().hex[:8]}")
    yield idx
    await idx.drop()
    await idx.close()


async def test_query_syntax_in_tags_and_values_is_data(index):
    engine = await MemoryEngine(InMemoryDocumentStore(), index, HashEmbedder()).open()
    await engine.remember("default", "the o'reilly book on sql", tags=["o'reilly", "new york"], metadata={"owner": "mark's"})
    await engine.remember("default", "another book on sql", tags=["plain"], metadata={"owner": "ana"})
    assert [i.episode_id for i in (await engine.recall("default", "book on sql", tags=["o'reilly"])).items] == [1]
    assert [i.episode_id for i in (await engine.recall("default", "book on sql", tags=["new york"])).items] == [1]
    assert [i.episode_id for i in (await engine.recall("default", "book on sql", where={"owner": "mark's"})).items] == [1]
    hostile = await engine.recall("default", "book on sql", where={"owner": "x} | @space:{default"})
    assert hostile.items == [] and not hostile.degraded, "a value shaped like a query matches nothing and breaks nothing"


async def test_the_width_is_read_back_from_the_index(index):
    from scone_memory.backends import RedisVectorIndex

    await index.ensure(64)
    await index.upsert([VectorPoint(1, "default", 1, "2025-01-01T00:00:00.000Z", [1.0] + [0.0] * 63)])
    again = RedisVectorIndex(URL, prefix=index.prefix)
    await again.ensure(64)
    assert [cid for cid, _ in await again.search("default", [1.0] + [0.0] * 63, 5)] == [1]
    with pytest.raises(ValueError, match="64-d"):
        await RedisVectorIndex(URL, prefix=index.prefix).ensure(32)
    await again.close()


async def test_upsert_replaces_and_delete_removes(index):
    await index.ensure(2)
    await index.upsert([VectorPoint(1, "default", 1, "2025-01-01T00:00:00.000Z", [1.0, 0.0], tags=("old",))])
    await index.upsert([VectorPoint(1, "default", 1, "2025-01-01T00:00:00.000Z", [0.0, 1.0], tags=("new",))])
    assert await index.search("default", [0.0, 1.0], 5, tags=("new",)) == [(1, pytest.approx(1.0))]
    assert await index.search("default", [0.0, 1.0], 5, tags=("old",)) == []
    assert await index.search("default", [1.0, 0.0], 5) == [(1, pytest.approx(0.0, abs=1e-6))], "distance 1 reads as similarity 0"
    await index.delete([1])
    assert await index.search("default", [0.0, 1.0], 5) == []


async def test_redis_from_the_environment():
    prefix = f"scone_test_{uuid.uuid4().hex[:8]}"
    engine = await build_engine(Settings.from_env({"SCONE_VECTORS": "redis", "SCONE_REDIS_URL": URL, "SCONE_REDIS_PREFIX": prefix}))
    assert engine.vectors.name == "redis" and engine.vectors.prefix == prefix
    await engine.remember("default", "wired through the environment")
    assert (await engine.recall("default", "environment")).items[0].episode_id == 1
    await engine.vectors.drop()
    await engine.vectors.close()
    with pytest.raises(InvalidInput, match="SCONE_REDIS_URL"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "redis"}))
