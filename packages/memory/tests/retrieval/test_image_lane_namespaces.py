"""The image lane's index must not be the text vectors' storage under another handle.

Every first-party vector index defaults to one namespace (schema ``scone``,
prefix ``scone``, collection ``scone_chunks``...), so ``image_vectors=
PostgresVectorIndex(url)`` beside ``vectors=PostgresVectorIndex(url)`` would
write image vectors over caption vectors keyed by the same chunk ids. Each
index names where its rows live (``location``); the engine refuses two equal
ones. Nothing here connects to a server: the handles are only built.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.embedders import HashImageEmbedder


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def engine_with(vectors: object, images: object) -> MemoryEngine:
    return MemoryEngine(InMemoryDocumentStore(), vectors, HashEmbedder(),  # type: ignore[arg-type]
                        image_embedder=HashImageEmbedder(dim=64), image_vectors=images)  # type: ignore[arg-type]


def refused(vectors: object, images: object) -> bool:
    try:
        engine_with(vectors, images)
    except InvalidInput as error:
        assert "index of its own" in str(error)
        return True
    return False


class _Client:
    """A stand-in for an injected client: only its identity matters here."""


#: backend -> (text index, image index on the same storage, image index on its own storage).
Builder = Callable[..., tuple[object, object, object]]


def _postgres(tmp_path, monkeypatch):
    from scone_memory.backends.postgres import Pool, PostgresVectorIndex
    url = "postgresql://localhost/scone"
    shared = Pool(url)
    return [(PostgresVectorIndex(url), PostgresVectorIndex(url), PostgresVectorIndex(url, schema="scone_images")),
            (PostgresVectorIndex(url), PostgresVectorIndex(url), PostgresVectorIndex("postgresql://elsewhere/scone")),
            (PostgresVectorIndex(url, pool=shared), PostgresVectorIndex("ignored", pool=shared),
             PostgresVectorIndex(url, schema="images", pool=shared))]


def _elasticsearch(tmp_path, monkeypatch):
    from scone_memory.backends.elastic import ElasticsearchVectorIndex
    url, client = "http://localhost:9200", _Client()
    return [(ElasticsearchVectorIndex(url), ElasticsearchVectorIndex(url), ElasticsearchVectorIndex(url, prefix="images")),
            (ElasticsearchVectorIndex(url), ElasticsearchVectorIndex(url), ElasticsearchVectorIndex("http://elsewhere:9200")),
            (ElasticsearchVectorIndex(client=client), ElasticsearchVectorIndex(client=client),
             ElasticsearchVectorIndex(client=_Client()))]


def _opensearch(tmp_path, monkeypatch):
    import httpx

    from scone_memory.backends.opensearch import OpenSearchVectorIndex
    url, client = "http://localhost:9200", httpx.AsyncClient()
    return [(OpenSearchVectorIndex(url), OpenSearchVectorIndex(url + "/"), OpenSearchVectorIndex(url, index="images")),
            (OpenSearchVectorIndex(url), OpenSearchVectorIndex(url), OpenSearchVectorIndex("http://elsewhere:9200")),
            (OpenSearchVectorIndex(url, client=client), OpenSearchVectorIndex(url, client=client),
             OpenSearchVectorIndex(url, client=httpx.AsyncClient()))]


def _qdrant(tmp_path, monkeypatch):
    import qdrant_client

    from scone_memory.backends.qdrant import QdrantVectorIndex
    # A remote client asks the server its version when built; only identity matters here.
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", lambda *args, **kwargs: _Client())
    url, client = "http://localhost:6333", _Client()
    return [(QdrantVectorIndex(url), QdrantVectorIndex(url), QdrantVectorIndex(url, collection="images")),
            (QdrantVectorIndex(url), QdrantVectorIndex(url), QdrantVectorIndex("http://elsewhere:6333")),
            # An injected client chose the server, whatever url is also given.
            (QdrantVectorIndex(url, client=client), QdrantVectorIndex(url, client=client),
             QdrantVectorIndex(url, client=_Client())),
            # Each in-memory Qdrant client is a store of its own.
            (memory := QdrantVectorIndex(), QdrantVectorIndex(client=memory.client), QdrantVectorIndex())]


def _milvus(tmp_path, monkeypatch):
    import pymilvus

    from scone_memory.backends.milvus import MilvusVectorIndex
    monkeypatch.setattr(pymilvus, "MilvusClient", lambda *args, **kwargs: _Client())
    url, client = "http://localhost:19530", _Client()
    return [(MilvusVectorIndex(url), MilvusVectorIndex(url), MilvusVectorIndex(url, collection="images")),
            (MilvusVectorIndex(url), MilvusVectorIndex(url), MilvusVectorIndex("http://elsewhere:19530")),
            (MilvusVectorIndex(url, client=client), MilvusVectorIndex(url, client=client),
             MilvusVectorIndex(url, client=_Client()))]


def _redis(tmp_path, monkeypatch):
    from scone_memory.backends.redis import RedisVectorIndex
    url, client = "redis://localhost:6379", _Client()
    return [(RedisVectorIndex(url), RedisVectorIndex(url), RedisVectorIndex(url, prefix="images")),
            (RedisVectorIndex(url), RedisVectorIndex(url), RedisVectorIndex("redis://elsewhere:6379")),
            (RedisVectorIndex(client=client), RedisVectorIndex(client=client), RedisVectorIndex(client=_Client()))]


def _elasticache(tmp_path, monkeypatch):
    from scone_memory.backends.elasticache import ElastiCacheVectorIndex
    client = _Client()

    def build(url: str = "redis://localhost:6379", **kwargs):
        return ElastiCacheVectorIndex(url, cluster_mode=False, **kwargs)
    return [(build(), build("redis://localhost"), build(prefix="images")),
            (build(), build(), build("redis://localhost:6380")),
            (build(client=client), build(client=client), build(client=_Client()))]  # type: ignore[arg-type]


def _chroma(tmp_path, monkeypatch):
    from scone_memory.backends.chroma import ChromaVectorIndex
    import chromadb

    client = _Client()
    one, other = str(tmp_path / "chroma"), str(tmp_path / "chroma-images")
    (tmp_path / "sub").mkdir()
    # An HTTP client asks the server for its identity when built; only the configured url matters here.
    monkeypatch.setattr(chromadb, "HttpClient", lambda *args, **kwargs: _Client())
    url = "http://localhost:8000"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    home = str(tmp_path / "home" / "chroma")
    return [(ChromaVectorIndex(path=one), ChromaVectorIndex(path=f"{tmp_path}/sub/../chroma"),
             ChromaVectorIndex(path=one, collection="images")),
            (ChromaVectorIndex(path=one), ChromaVectorIndex(path=one), ChromaVectorIndex(path=other)),
            # Chroma's in-process clients share one store per process.
            (ChromaVectorIndex(), ChromaVectorIndex(), ChromaVectorIndex(collection="images")),
            (ChromaVectorIndex(url=url), ChromaVectorIndex(url=url), ChromaVectorIndex(url="http://elsewhere:8000")),
            # Chroma writes "~/chroma" to a directory named "~" where it runs, not under the home directory.
            (ChromaVectorIndex(path=home), ChromaVectorIndex(path=home), ChromaVectorIndex(path="~/chroma")),
            (ChromaVectorIndex(client=client), ChromaVectorIndex(client=client), ChromaVectorIndex(client=_Client()))]


def _lancedb(tmp_path, monkeypatch):
    from scone_memory.backends.lancedb import LanceDBVectorIndex
    connection = _Client()
    one = str(tmp_path / "lance")
    return [(LanceDBVectorIndex(one), LanceDBVectorIndex(one), LanceDBVectorIndex(one, table="images")),
            (LanceDBVectorIndex(one), LanceDBVectorIndex(one), LanceDBVectorIndex(str(tmp_path / "lance-images"))),
            (LanceDBVectorIndex(one, connection=connection), LanceDBVectorIndex(one, connection=connection),
             LanceDBVectorIndex(one, connection=_Client()))]


def _langchain(tmp_path, monkeypatch):
    from scone_memory.backends.langchain import LangChainVectorIndex
    store = _Client()
    return [(LangChainVectorIndex(store), LangChainVectorIndex(store), LangChainVectorIndex(_Client()))]


def _sqlite(tmp_path, monkeypatch):
    from scone_memory.backends import SqliteVectorIndex
    # SQLite opens "~" under the home directory, and "sub/.." is the directory above it.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "sub").mkdir()
    return [(SqliteVectorIndex(tmp_path / "m.db"), SqliteVectorIndex("~/sub/../m.db"),
             SqliteVectorIndex(tmp_path / "images.db"))]


BACKENDS = [
    pytest.param(_postgres, marks=pytest.mark.skipif(not _installed("psycopg_pool", "pgvector"), reason="postgres extra")),
    pytest.param(_elasticsearch, marks=pytest.mark.skipif(not _installed("elasticsearch"), reason="elasticsearch extra")),
    pytest.param(_opensearch, marks=pytest.mark.skipif(not _installed("httpx"), reason="opensearch extra")),
    pytest.param(_qdrant, marks=pytest.mark.skipif(not _installed("qdrant_client"), reason="qdrant extra")),
    pytest.param(_milvus, marks=pytest.mark.skipif(not _installed("pymilvus"), reason="milvus extra")),
    pytest.param(_redis, marks=pytest.mark.skipif(not _installed("redis"), reason="redis extra")),
    pytest.param(_elasticache, marks=pytest.mark.skipif(not _installed("redis"), reason="redis extra")),
    pytest.param(_chroma, marks=pytest.mark.skipif(not _installed("chromadb"), reason="chroma extra")),
    pytest.param(_lancedb, marks=pytest.mark.skipif(not _installed("lancedb"), reason="lancedb extra")),
    pytest.param(_langchain),
    pytest.param(_sqlite),
]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda build: build.__name__.strip("_"))
def test_an_image_index_on_the_text_indexs_storage_is_refused(backend, tmp_path, monkeypatch):
    cases = backend(tmp_path, monkeypatch)
    for n, (text, same, apart) in enumerate(cases):
        assert refused(text, same), f"case {n}: the same storage under another handle is refused"
        assert not refused(text, apart), f"case {n}: storage of its own is accepted"


def test_an_index_that_cannot_say_where_it_lives_is_recognised_only_as_itself():
    from scone_memory import InMemoryVectorIndex
    from scone_memory.backends import SqliteVectorIndex

    one = InMemoryVectorIndex()
    assert refused(one, one) and not refused(one, InMemoryVectorIndex())
    # Two in-memory SQLite handles are two databases, not one.
    assert not refused(SqliteVectorIndex(":memory:"), SqliteVectorIndex(":memory:"))
    from scone_memory.backends.langchain import LangChainVectorIndex
    # A bridge with no store yet holds nothing to share; bind() gives each its own later.
    assert not refused(LangChainVectorIndex(), LangChainVectorIndex())
