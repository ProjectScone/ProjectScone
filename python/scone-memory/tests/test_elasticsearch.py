"""Elasticsearch as documents, vectors and evidence together. The published
contract and the events contract run over it from tests/test_contract.py
and tests/test_events.py when SCONE_TEST_ELASTICSEARCH_URL points at an
8.x node; this file holds what is specific to Elasticsearch: the shared
client, the schema stamp, persistence across reopen, the scope copies on
chunks, and the environment wiring."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from scone_memory import HashEmbedder, InvalidInput, MemoryEngine
from scone_memory.config import Settings, build_engine, build_events, build_vectors

URL = os.environ.get("SCONE_TEST_ELASTICSEARCH_URL")
pytestmark = [pytest.mark.elasticsearch, pytest.mark.skipif(not URL, reason="needs a live Elasticsearch 8")]


@pytest.fixture
async def prefix():
    from scone_memory.backends import ElasticsearchDocumentStore

    name = f"scone_test_{uuid.uuid4().hex[:8]}"
    yield name
    store = ElasticsearchDocumentStore(URL, prefix=name)
    await store.drop()
    await store.close()


async def test_one_client_serves_all_three_stores_and_closes_with_the_last(prefix):
    from scone_memory.backends import ElasticsearchDocumentStore

    documents = await ElasticsearchDocumentStore(URL, prefix=prefix).open()
    vectors, events = documents.vectors(), await documents.events().open()
    assert vectors.client is documents.client is events.client and documents.shared.users == 3
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), events=events).open()
    await engine.remember("default", "one cluster for everything")
    await documents.close()
    [vec] = await HashEmbedder().embed(["one cluster for everything"])
    assert [cid for cid, _ in await vectors.search("default", vec, 5)] == [1], "two stores still use the client"
    await vectors.close()
    assert documents.shared.users == 1
    await events.close()
    assert documents.shared.users == 0


async def test_another_builds_indices_are_refused_not_migrated(prefix):
    from scone_memory.backends import ElasticsearchDocumentStore
    from scone_memory.backends.elastic import SchemaMismatch

    seed = ElasticsearchDocumentStore(URL, prefix=prefix)
    await seed.shared.ensure_index("meta", {"properties": {"value": {"type": "keyword"}}})
    await seed.client.index(index=seed.shared.index("meta"), id="schema_version", document={"value": "1"}, refresh=True)
    await seed.close()
    store = ElasticsearchDocumentStore(URL, prefix=prefix)
    with pytest.raises(SchemaMismatch, match="hold schema v1"):
        await store.open()
    await store.close()


async def test_data_and_width_survive_a_reopen(prefix):
    from scone_memory.backends import ElasticsearchDocumentStore

    first = await ElasticsearchDocumentStore(URL, prefix=prefix).open()
    engine = await MemoryEngine(first, first.vectors(), HashEmbedder(dim=64), events=None).open()
    await engine.remember("default", "kept in elasticsearch", tags=["kept"])
    await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    await first.close()
    await engine.vectors.close()

    again = await ElasticsearchDocumentStore(URL, prefix=prefix).open()
    reopened = await MemoryEngine(again, again.vectors(), HashEmbedder(dim=64), events=None).open()
    assert [i.episode_id for i in (await reopened.recall("default", "kept in elasticsearch", tags=["kept"])).items] == [1]
    assert [f.object for f in await reopened.facts("default")] == ["Lisbon"]
    with pytest.raises(ValueError, match="64-d"):
        await again.vectors().ensure(32)
    await again.close()
    await reopened.vectors.close()


async def test_the_scope_filters_run_in_the_engine_on_both_lanes(prefix):
    from scone_memory.backends import ElasticsearchDocumentStore

    documents = await ElasticsearchDocumentStore(URL, prefix=prefix).open()
    engine = await MemoryEngine(documents, documents.vectors(), HashEmbedder(), events=None).open()
    for i in range(30):  # enough matching-but-out-of-scope chunks to crowd a lane's size
        await engine.remember("default", f"quarterly report {i} on revenue", metadata={"user_id": "ana"}, tags=["finance"])
    await engine.remember("default", "quarterly report on revenue for mark", metadata={"user_id": "mark's"}, tags=["finance", "mine"],
                          created_at="2020-01-01")
    scoped = await engine.recall("default", "quarterly report revenue", where={"user_id": "mark's"}, limit=3)
    assert [i.metadata["user_id"] for i in scoped.items] == ["mark's"] and scoped.items[0].lanes == {"vector": 1, "text": 1}
    tagged = await engine.recall("default", "quarterly report revenue", tags=["mine"], limit=3)
    assert [i.episode_id for i in tagged.items] == [31]
    dated = await engine.recall("default", "quarterly report revenue", as_of="2021-01-01", limit=3)
    assert [i.episode_id for i in dated.items] == [31], "only the one that happened by then"
    hostile = await engine.recall("default", "quarterly report revenue", where={"user_id": "x\" OR 1=1"}, limit=3)
    assert hostile.items == [] and not hostile.degraded
    await documents.close()
    await engine.vectors.close()


async def test_concurrent_writers_share_the_client(prefix):
    from scone_memory.backends import ElasticsearchDocumentStore

    documents = await ElasticsearchDocumentStore(URL, prefix=prefix).open()
    engine = await MemoryEngine(documents, documents.vectors(), HashEmbedder(), events=await documents.events().open()).open()
    await asyncio.gather(*(engine.remember("default", f"note number {i} about topic {i % 3}") for i in range(12)))
    status = await engine.status("default")
    assert status.episodes == 12 and status.revision == 12
    assert len(await engine.events.query("default", kind="remember", limit=100)) == 12
    for store in (documents, engine.vectors, engine.events):
        await store.close()


async def test_elasticsearch_from_the_environment(prefix):
    env = {"SCONE_DOCUMENTS": "elasticsearch", "SCONE_VECTORS": "elasticsearch", "SCONE_ELASTICSEARCH_URL": URL, "SCONE_ELASTICSEARCH_PREFIX": prefix}
    engine = await build_engine(Settings.from_env(env))
    assert (engine.documents.name, engine.vectors.name, engine.events.name) == ("elasticsearch", "elasticsearch", "elasticsearch")
    assert engine.vectors.client is engine.documents.client is engine.events.client, "one client, not three"
    await engine.remember("default", "wired through the environment")
    assert (await engine.recall("default", "environment")).items[0].episode_id == 1
    for store in (engine.documents, engine.vectors, engine.events):
        await store.close()
    with pytest.raises(InvalidInput, match="SCONE_ELASTICSEARCH_URL"):
        build_vectors(Settings.from_env({"SCONE_VECTORS": "elasticsearch"}))
    with pytest.raises(InvalidInput, match="SCONE_ELASTICSEARCH_URL"):
        build_events(Settings.from_env({"SCONE_EVENTS": "elasticsearch"}))
