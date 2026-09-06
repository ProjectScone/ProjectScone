"""Fixtures. ``engine`` is parametrised over every backend pair that is
reachable, so the same behavioural tests run against dicts always and
against MongoDB and Qdrant whenever ``SCONE_TEST_MONGO_URL`` and
``SCONE_TEST_QDRANT_URL`` point at live servers."""

from __future__ import annotations

import os
import uuid

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

MONGO_URL = os.environ.get("SCONE_TEST_MONGO_URL")
QDRANT_URL = os.environ.get("SCONE_TEST_QDRANT_URL")


class Clock:
    """A clock the test moves by hand, so "now" is a value, not a race."""

    def __init__(self, start: str = "2025-01-01T00:00:00.000Z") -> None:
        self.now = start

    def __call__(self) -> str:
        return self.now


def backends():
    yield pytest.param("memory", id="memory")
    if MONGO_URL:
        yield pytest.param("mongo", id="mongo", marks=pytest.mark.mongo)
    if QDRANT_URL:
        yield pytest.param("qdrant", id="qdrant", marks=pytest.mark.qdrant)


@pytest.fixture(params=list(backends()))
async def engine(request):
    clock = Clock()
    if request.param == "memory":
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    elif request.param == "mongo":
        from scone_memory.backends import MongoDocumentStore

        documents = MongoDocumentStore(MONGO_URL, f"scone_test_{uuid.uuid4().hex[:8]}")
        await documents.open()
        vectors = InMemoryVectorIndex()
    else:
        from scone_memory.backends import QdrantVectorIndex

        documents = InMemoryDocumentStore()
        vectors = QdrantVectorIndex(QDRANT_URL, f"scone_test_{uuid.uuid4().hex[:8]}")
    e = await MemoryEngine(documents, vectors, HashEmbedder(), chunk_target=200, clock=clock).open()
    e.test_clock = clock  # type: ignore[attr-defined]
    yield e
    if hasattr(documents, "drop"):
        await documents.drop()
    if hasattr(vectors, "drop"):
        await vectors.drop()
