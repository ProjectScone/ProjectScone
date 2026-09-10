"""PostgreSQL with pgvector as documents, vectors and evidence together.
The published contract and the events contract run over it from
tests/test_contract.py and tests/test_events.py when
SCONE_TEST_POSTGRES_URL points at a server with the vector extension;
this file holds what is specific to Postgres: the shared pool, the
schema stamp, persistence across reopen, and the environment wiring."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from scone_memory import HashEmbedder, InvalidInput, MemoryEngine
from scone_memory.runtime.config import Settings, build_engine, build_events, build_vectors

URL = os.environ.get("SCONE_TEST_POSTGRES_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not URL, reason="needs a live PostgreSQL with pgvector")]


@pytest.fixture
async def schema():
    from scone_memory.backends.postgres import Pool

    name = f"scone_test_{uuid.uuid4().hex[:8]}"
    yield name
    pool = Pool(URL)
    await pool.open()
    async with pool.connection() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    pool.users = 1
    await pool.release()


async def test_one_pool_serves_all_three_stores_and_closes_with_the_last(schema):
    from scone_memory.backends import PostgresDocumentStore

    documents = await PostgresDocumentStore(URL, schema).open()
    vectors, events = documents.vectors(), await documents.events().open()
    assert vectors.pool is documents.pool is events.pool and documents.pool.users == 3
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), events=events).open()
    await engine.remember("default", "one database for everything")
    await documents.close()
    assert documents.pool.opened, "two stores still use the pool"
    [vec] = await HashEmbedder().embed(["one database for everything"])
    assert [cid for cid, _ in await vectors.search("default", vec, 5)] == [1]
    await vectors.close()
    await events.close()
    assert not documents.pool.opened, "the last user closed it"


async def test_another_builds_schema_is_refused_not_migrated(schema):
    from scone_memory.backends.postgres import Pool, PostgresDocumentStore, SchemaMismatch

    pool = Pool(URL)
    await pool.open()
    async with pool.connection() as conn:
        await conn.execute(f"CREATE SCHEMA {schema}; CREATE TABLE {schema}.meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                           f" INSERT INTO {schema}.meta VALUES ('schema_version', '1')")
    pool.users = 1
    await pool.release()
    store = PostgresDocumentStore(URL, schema)
    with pytest.raises(SchemaMismatch, match="holds schema v1"):
        await store.open()
    await store.close()


async def test_data_and_width_survive_a_reopen(schema):
    from scone_memory.backends import PostgresDocumentStore

    first = await PostgresDocumentStore(URL, schema).open()
    engine = await MemoryEngine(first, first.vectors(), HashEmbedder(dim=64), events=None).open()
    await engine.remember("default", "kept in postgres", tags=["kept"])
    await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    await first.close()
    await engine.vectors.close()

    again = await PostgresDocumentStore(URL, schema).open()
    reopened = await MemoryEngine(again, again.vectors(), HashEmbedder(dim=64), events=None).open()
    result = await reopened.recall("default", "kept in postgres", tags=["kept"])
    assert [i.episode_id for i in result.items] == [1]
    assert [f.object for f in await reopened.facts("default")] == ["Lisbon"]
    with pytest.raises(ValueError, match="64-d"):
        await again.vectors().ensure(32)
    await again.close()
    await reopened.vectors.close()


async def test_concurrent_writers_share_the_pool(schema):
    from scone_memory.backends import PostgresDocumentStore

    documents = await PostgresDocumentStore(URL, schema).open()
    engine = await MemoryEngine(documents, documents.vectors(), HashEmbedder(), events=await documents.events().open()).open()
    await asyncio.gather(*(engine.remember("default", f"note number {i} about topic {i % 3}") for i in range(24)))
    status = await engine.status("default")
    assert status.episodes == 24 and status.revision == 24
    assert len(await engine.events.query("default", kind="remember", limit=100)) == 24
    for store in (documents, engine.vectors, engine.events):
        await store.close()


async def test_the_scope_filters_run_in_sql_on_both_lanes(schema):
    from scone_memory.backends import PostgresDocumentStore

    documents = await PostgresDocumentStore(URL, schema).open()
    engine = await MemoryEngine(documents, documents.vectors(), HashEmbedder(), events=None).open()
    for i in range(30):  # enough matching-but-out-of-scope chunks to crowd a lane's LIMIT
        await engine.remember("default", f"quarterly report {i} on revenue", metadata={"user_id": "ana"}, tags=["finance"])
    await engine.remember("default", "quarterly report on revenue for mark", metadata={"user_id": "mark"}, tags=["finance", "mine"],
                          created_at="2020-01-01")
    scoped = await engine.recall("default", "quarterly report revenue", where={"user_id": "mark"}, limit=3)
    assert [i.metadata["user_id"] for i in scoped.items] == ["mark"] and scoped.items[0].lanes == {"vector": 1, "text": 1}
    tagged = await engine.recall("default", "quarterly report revenue", tags=["mine"], limit=3)
    assert [i.episode_id for i in tagged.items] == [31]
    dated = await engine.recall("default", "quarterly report revenue", as_of="2021-01-01", limit=3)
    assert [i.episode_id for i in dated.items] == [31], "only the one that happened by then"
    await documents.close()
    await engine.vectors.close()


async def test_postgres_from_the_environment(schema):
    env = {"SCONE_DOCUMENTS": "postgres", "SCONE_VECTORS": "postgres", "SCONE_POSTGRES_URL": URL, "SCONE_POSTGRES_SCHEMA": schema}
    engine = await build_engine(Settings.from_env(env))
    assert (engine.documents.name, engine.vectors.name, engine.events.name) == ("postgres", "postgres", "postgres")
    assert engine.vectors.pool is engine.documents.pool is engine.events.pool, "one pool, not three"
    await engine.remember("default", "wired through the environment")
    assert (await engine.recall("default", "environment")).items[0].episode_id == 1
    for store in (engine.documents, engine.vectors, engine.events):
        await store.close()
    with pytest.raises(InvalidInput, match="SCONE_POSTGRES_URL"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "postgres"}))
    with pytest.raises(InvalidInput, match="SCONE_POSTGRES_URL"):
        build_events(Settings.from_env({"SCONE_EVENTS": "postgres"}))
