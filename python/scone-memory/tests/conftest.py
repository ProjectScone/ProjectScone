"""Fixtures. ``engine`` is parametrised over every backend pair that is
reachable, so the same behavioural tests run against dicts always and
against MongoDB and Qdrant whenever ``SCONE_TEST_MONGO_URL`` and
``SCONE_TEST_QDRANT_URL`` point at live servers."""

from __future__ import annotations

import importlib.util
import os
import uuid

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.events import InMemoryEventLog, SqliteEventLog
from scone_memory.testing import Clock

MONGO_URL = os.environ.get("SCONE_TEST_MONGO_URL")
QDRANT_URL = os.environ.get("SCONE_TEST_QDRANT_URL")
POSTGRES_URL = os.environ.get("SCONE_TEST_POSTGRES_URL")
REDIS_URL = os.environ.get("SCONE_TEST_REDIS_URL")
ELASTIC_URL = os.environ.get("SCONE_TEST_ELASTICSEARCH_URL")


def backends():
    yield pytest.param("memory", id="memory")
    yield pytest.param("sqlite", id="sqlite")
    yield pytest.param("qdrant-local", id="qdrant-local")
    yield pytest.param("chroma", id="chroma", marks=pytest.mark.skipif(importlib.util.find_spec("chromadb") is None, reason="chromadb not installed"))
    yield pytest.param("lancedb", id="lancedb", marks=pytest.mark.skipif(importlib.util.find_spec("lancedb") is None, reason="lancedb not installed"))
    yield pytest.param("milvus-lite", id="milvus-lite", marks=pytest.mark.skipif(importlib.util.find_spec("milvus_lite") is None, reason="milvus-lite not installed"))
    needs_langchain = pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None, reason="langchain-core not installed")
    yield pytest.param("langchain-filtered", id="langchain-filtered", marks=needs_langchain)
    yield pytest.param("langchain-postfilter", id="langchain-postfilter", marks=needs_langchain)
    if MONGO_URL:
        yield pytest.param("mongo", id="mongo", marks=pytest.mark.mongo)
        yield pytest.param("mongo+qdrant-local", id="mongo+qdrant-local", marks=pytest.mark.mongo)
    if QDRANT_URL:
        yield pytest.param("qdrant", id="qdrant", marks=pytest.mark.qdrant)
    if POSTGRES_URL:
        yield pytest.param("postgres", id="postgres", marks=pytest.mark.postgres)
    if REDIS_URL:
        yield pytest.param("redis", id="redis", marks=pytest.mark.redis)
    if ELASTIC_URL:
        yield pytest.param("elasticsearch", id="elasticsearch", marks=pytest.mark.elasticsearch)


@pytest.fixture(params=list(backends()))
async def engine(request, tmp_path):
    clock = Clock()
    events = InMemoryEventLog()
    if request.param == "memory":
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    elif request.param == "sqlite":
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex

        path = tmp_path / "memory.db"
        documents, vectors = SqliteDocumentStore(path), SqliteVectorIndex(path)
        events = SqliteEventLog(path, clock=clock)
    elif request.param.startswith("mongo"):
        from scone_memory.backends import MongoDocumentStore, QdrantVectorIndex

        documents = MongoDocumentStore(MONGO_URL, f"scone_test_{uuid.uuid4().hex[:8]}")
        await documents.open()
        vectors = QdrantVectorIndex(":memory:", "scone_test") if "qdrant" in request.param else InMemoryVectorIndex()
    elif request.param == "qdrant-local":
        from scone_memory.backends import QdrantVectorIndex

        documents = InMemoryDocumentStore()
        vectors = QdrantVectorIndex(":memory:", "scone_test")
    elif request.param == "postgres":
        from scone_memory.backends import PostgresDocumentStore

        documents = PostgresDocumentStore(POSTGRES_URL, schema=f"scone_test_{uuid.uuid4().hex[:8]}")
        await documents.open()
        vectors = documents.vectors()
        events = await documents.events(clock=clock).open()
    elif request.param == "elasticsearch":
        from scone_memory.backends import ElasticsearchDocumentStore

        documents = ElasticsearchDocumentStore(ELASTIC_URL, prefix=f"scone_test_{uuid.uuid4().hex[:8]}")
        await documents.open()
        vectors = documents.vectors()
        events = await documents.events(clock=clock).open()
    elif request.param == "redis":
        from scone_memory.backends import RedisVectorIndex

        documents = InMemoryDocumentStore()
        vectors = RedisVectorIndex(REDIS_URL, prefix=f"scone_test_{uuid.uuid4().hex[:8]}")
    elif request.param == "chroma":
        from scone_memory.backends import ChromaVectorIndex

        documents = InMemoryDocumentStore()
        vectors = ChromaVectorIndex(collection=f"scone_test_{uuid.uuid4().hex[:8]}")
    elif request.param == "lancedb":
        from scone_memory.backends import LanceDBVectorIndex

        documents = InMemoryDocumentStore()
        vectors = LanceDBVectorIndex(str(tmp_path / "lance"))
    elif request.param == "milvus-lite":
        from scone_memory.backends import MilvusVectorIndex

        documents = InMemoryDocumentStore()
        vectors = MilvusVectorIndex(str(tmp_path / "milvus.db"))
    elif request.param.startswith("langchain"):
        # The bridge over langchain_core's InMemoryVectorStore, once with a
        # filter_builder (the store filters) and once without (the bridge
        # over-fetches and filters on its own metadata).
        from langchain_core.vectorstores import InMemoryVectorStore

        from scone_memory.backends import LangChainVectorIndex
        from scone_memory.backends.langchain import LangChainVectorIndex as Bridge

        def builder(space, as_of_ts, tags, where):
            return lambda doc: Bridge._matches(doc.metadata, space, as_of_ts, tags, where)

        documents = InMemoryDocumentStore()
        vectors = LangChainVectorIndex(score="cosine_similarity", filter_builder=builder if request.param.endswith("filtered") else None)
        vectors.bind(InMemoryVectorStore(embedding=vectors.embeddings))
    else:
        from scone_memory.backends import QdrantVectorIndex

        documents = InMemoryDocumentStore()
        vectors = QdrantVectorIndex(QDRANT_URL, f"scone_test_{uuid.uuid4().hex[:8]}")
    e = await MemoryEngine(documents, vectors, HashEmbedder(), chunk_target=200, clock=clock, events=events).open()
    e.test_clock = clock  # type: ignore[attr-defined]
    yield e
    for store in (documents, vectors, events):
        if hasattr(store, "drop"):
            await store.drop()
        if hasattr(store, "close"):
            await store.close()
