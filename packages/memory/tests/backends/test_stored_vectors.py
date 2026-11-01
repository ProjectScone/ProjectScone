"""Embedded indexes hand back the vectors they stored, by space.

Diversity compares the candidates with one another, so recall reads their
vectors from the index; an index that cannot hand them back makes every
diversified recall embed up to 200 passages again. Each embedded index is
held to the rules the in-memory and SQLite ones keep: only the space asked
about, a chunk without a vector absent, and the cosine between two vectors
the one they were written with.
"""

from __future__ import annotations

import importlib.util
import math
import uuid

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, MemoryEngine
from scone_memory.core.ports import VectorPoint

pytestmark = pytest.mark.asyncio

needs = {name: pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} not installed")
         for name, module in (("qdrant", "qdrant_client"), ("chroma", "chromadb"), ("lancedb", "lancedb"))}

WRITTEN = {1: [3.0, 4.0, 0.0], 2: [0.0, 2.0, 2.0], 3: [1.0, 0.0, 0.0]}


def make(kind: str, tmp_path):
    if kind == "qdrant":
        from scone_memory.backends.qdrant import QdrantVectorIndex

        return QdrantVectorIndex(":memory:", collection=f"scone_test_{uuid.uuid4().hex[:8]}")
    if kind == "chroma":
        from scone_memory.backends import ChromaVectorIndex

        return ChromaVectorIndex(path=str(tmp_path / "chroma"))
    from scone_memory.backends import LanceDBVectorIndex

    return LanceDBVectorIndex(str(tmp_path / "lance"))


@pytest.fixture(params=[pytest.param(kind, marks=needs[kind]) for kind in needs])
def kind(request):
    return request.param


def cosine(left, right) -> float:
    return sum(a * b for a, b in zip(left, right)) / (math.hypot(*left) * math.hypot(*right))


async def test_vectors_come_back_from_the_space_asked_about_and_keep_their_cosines(kind, tmp_path):
    index = make(kind, tmp_path)
    await index.ensure(3)
    await index.upsert([VectorPoint(chunk_id=chunk_id, space="alpha" if chunk_id < 3 else "beta", episode_id=chunk_id,
                                    created_at="2026-10-01T00:00:00Z", vector=vector)
                        for chunk_id, vector in WRITTEN.items()])
    try:
        alpha = await index.vectors_of("alpha", [1, 2, 3, 99])
        beta = await index.vectors_of("beta", [1, 2, 3])
        nothing = await index.vectors_of("alpha", [])
    finally:
        await index.close()
    assert set(alpha) == {1, 2} and set(beta) == {3} and nothing == {}
    assert cosine(alpha[1], alpha[2]) == pytest.approx(cosine(WRITTEN[1], WRITTEN[2]), abs=1e-6)
    assert all(cosine(alpha[key], WRITTEN[key]) == pytest.approx(1.0, abs=1e-6) for key in alpha)


async def test_a_diversified_recall_reads_the_index_and_embeds_only_the_question(kind, tmp_path):
    class Counting(HashEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            return await super().embed(texts)

    embedder = Counting()
    engine = await MemoryEngine(InMemoryDocumentStore(), make(kind, tmp_path), embedder).open()
    try:
        for note in ("The crane survey found rust on the jib.", "The crane survey found rust on the jib today.",
                     "The crane survey was booked for the third of May."):
            await engine.remember("default", note)
        embedder.calls = 0
        varied = await engine.recall("default", "crane survey rust jib", limit=2, diversity=0.7)
    finally:
        await engine.close()
    assert varied.diversity is not None and varied.diversity.vectors == "index"
    assert embedder.calls == 1
