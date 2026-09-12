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
from scone_memory.observability.events import InMemoryEventLog, SqliteEventLog
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
    yield pytest.param("langchain-faiss", id="langchain-faiss", marks=pytest.mark.skipif(
        importlib.util.find_spec("faiss") is None or importlib.util.find_spec("langchain_community") is None,
        reason="faiss-cpu and langchain-community not installed"))
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
    # Databases whose library keeps a background server alive past
    # `close()`; released by path in the teardown. See `_release_embedded`.
    embedded: list[str] = []
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
        embedded.append(str(tmp_path / "milvus.db"))
        vectors = MilvusVectorIndex(embedded[-1])
    elif request.param == "langchain-faiss":
        # The bridge over a real third-party store: FAISS with inner product
        # over unit vectors (cosine), and a callable filter over metadata.
        import warnings

        from scone_memory.backends import LangChainVectorIndex
        from scone_memory.backends.langchain import LangChainVectorIndex as Bridge

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_community.vectorstores import FAISS
            from langchain_community.vectorstores.utils import DistanceStrategy

        def faiss_builder(space, as_of_ts, tags, where):
            return lambda meta: Bridge._matches(meta, space, as_of_ts, tags, where)  # FAISS hands the metadata dict

        documents = InMemoryDocumentStore()
        vectors = LangChainVectorIndex(score="cosine_similarity", filter_builder=faiss_builder)
        vectors.bind(FAISS(embedding_function=vectors.embeddings, index=faiss.IndexFlatIP(256), docstore=InMemoryDocstore(),
                           index_to_docstore_id={}, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True))
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
    _release_embedded(embedded)


def _release_embedded(paths: list[str]) -> None:
    """Stop the embedded servers this fixture's own databases started.

    `milvus_lite` keeps one background gRPC server per distinct `.db`
    path in a module-level registry until `atexit`, and `MilvusClient.close`
    releases only the client connection -- so closing the index cannot
    stop the server. A fresh `tmp_path/milvus.db` per parametrisation
    therefore accumulated one live server per test for the life of the
    process: a full-suite run held 47 `grpc/_server.py` serving threads
    and 53 idle pool workers at once, and logged
    `GOAWAY ... ENHANCE_YOUR_CALM: too_many_pings`.

    Only this fixture's own paths, never `release_all()`: another test's
    client may still be using its own database, and nothing here should
    reach into a store it did not create. Guarded, because a
    `milvus_lite` without this call is a reason to leak a server rather
    than to fail a suite.
    """
    if not paths:
        return
    try:
        from milvus_lite.server_manager import server_manager_instance
    except Exception:  # pragma: no cover - milvus-lite absent or changed
        return
    for path in paths:
        try:
            server_manager_instance.release_server(path)
        except Exception:  # pragma: no cover - best effort teardown
            pass


def stores():
    """Every document store, and no vector variants: local ones always,
    remote ones when CI points the test at a live server. For contracts
    that live in the document store, such as the ledger and its pager."""
    yield "memory"
    yield "sqlite"
    if os.environ.get("SCONE_TEST_MONGO_URL"):
        yield pytest.param("mongo", marks=pytest.mark.mongo)
    if os.environ.get("SCONE_TEST_POSTGRES_URL"):
        yield pytest.param("postgres", marks=pytest.mark.postgres)
    if os.environ.get("SCONE_TEST_ELASTICSEARCH_URL"):
        yield pytest.param("elasticsearch", marks=pytest.mark.elasticsearch)


@pytest.fixture(params=list(stores()))
async def ledger_engine(request, tmp_path):
    name = f"scone_test_{uuid.uuid4().hex[:8]}"
    vectors = InMemoryVectorIndex()
    if request.param == "memory":
        documents = InMemoryDocumentStore()
    elif request.param == "sqlite":
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex

        documents, vectors = SqliteDocumentStore(tmp_path / "m.db"), SqliteVectorIndex(tmp_path / "m.db")
    elif request.param == "mongo":
        from scone_memory.backends import MongoDocumentStore

        documents = MongoDocumentStore(os.environ["SCONE_TEST_MONGO_URL"], name)
        await documents.open()
    elif request.param == "postgres":
        from scone_memory.backends import PostgresDocumentStore

        documents = PostgresDocumentStore(os.environ["SCONE_TEST_POSTGRES_URL"], schema=name)
        await documents.open()
        vectors = documents.vectors()
    else:
        from scone_memory.backends import ElasticsearchDocumentStore

        documents = ElasticsearchDocumentStore(os.environ["SCONE_TEST_ELASTICSEARCH_URL"], prefix=name)
        await documents.open()
        vectors = documents.vectors()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), clock=Clock()).open()
    yield engine
    for store in (documents, vectors):
        if hasattr(store, "drop"):
            await store.drop()
    await engine.close()
