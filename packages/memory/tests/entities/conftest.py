"""Fixtures shared by the entity tests: an engine on every document store."""
from __future__ import annotations

import os
import uuid

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.testing import Clock


def stores():
    """Every document store with a ledger pager: local ones always, remote
    ones when CI points the test at a live server."""
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
