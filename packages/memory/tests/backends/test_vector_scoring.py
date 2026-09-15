"""The built-in indexes score by cosine exactly as the plain formula does.

Both indexes rank their candidates by full-precision cosine and break ties
by chunk id, so a change in the last bit of a score can reorder a recall.
These tests hold the scores to the formula written out the slow way, on
vectors that are not unit length, so a faster route to the same number is
held to the same number and not merely a close one.
"""

from __future__ import annotations

from array import array
import math
import random

from scone_memory.backends import InMemoryVectorIndex, SqliteVectorIndex
from scone_memory.core.ports import VectorPoint

DIM = 48
WHEN = "2026-01-01T00:00:00Z"


def plain_cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def vectors(seed: int, count: int) -> list[list[float]]:
    chooser = random.Random(seed)
    # Mixed magnitudes, so every norm and every sum carries rounding.
    return [[chooser.uniform(-1, 1) * chooser.choice((1e-3, 1.0, 7.0, 1e3)) for _ in range(DIM)]
            for _ in range(count)]


def points(stored: list[list[float]], first: int = 1) -> list[VectorPoint]:
    return [VectorPoint(chunk_id=first + index, space="s", episode_id=first + index, created_at=WHEN, vector=vector)
            for index, vector in enumerate(stored)]


def expected(query: list[float], stored: dict[int, list[float]]) -> list[tuple[int, float]]:
    scored = [(chunk_id, plain_cosine(query, vector)) for chunk_id, vector in stored.items()]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


async def test_the_memory_index_scores_every_point_as_the_formula_does():
    index = InMemoryVectorIndex()
    await index.ensure(DIM)
    stored = vectors(7, 40) + [[0.0] * DIM]
    await index.upsert(points(stored))
    for query in vectors(8, 12) + [[0.0] * DIM]:
        found = await index.search("s", query, 100)
        assert found == expected(query, {index + 1: vector for index, vector in enumerate(stored)})


async def test_the_memory_index_scores_a_rewritten_point_by_its_new_vector():
    index = InMemoryVectorIndex()
    await index.ensure(DIM)
    first, second = vectors(11, 2)
    await index.upsert(points([first]))
    [query] = vectors(12, 1)
    assert await index.search("s", query, 5) == expected(query, {1: first})
    # The same chunk written again with a vector of another length and direction.
    rewritten = [value * 3.5 for value in second]
    await index.upsert(points([rewritten]))
    assert await index.search("s", query, 5) == expected(query, {1: rewritten})


async def test_the_memory_index_keeps_no_norm_for_a_point_it_no_longer_holds():
    index = InMemoryVectorIndex()
    await index.ensure(DIM)
    await index.upsert(points(vectors(13, 3)))
    await index.upsert([VectorPoint(chunk_id=9, space="other", episode_id=9, created_at=WHEN, vector=vectors(14, 1)[0])])
    await index.delete([2])
    assert set(index._norms) == set(index._points) == {1, 3, 9}
    await index.delete_space("s")
    assert set(index._norms) == set(index._points) == {9}


async def test_the_sqlite_index_scores_every_point_as_the_formula_does(tmp_path):
    index = SqliteVectorIndex(tmp_path / "vectors.db")
    await index.ensure(DIM)
    try:
        stored = vectors(21, 40) + [[0.0] * DIM]
        await index.upsert(points(stored))
        # The index keeps single precision; the formula sees what it kept.
        kept = {index + 1: list(array("f", vector)) for index, vector in enumerate(stored)}
        for query in vectors(22, 12) + [[0.0] * DIM]:
            found = await index.search("s", query, 100)
            assert found == expected(list(array("f", query)), kept)
    finally:
        await index.close()
