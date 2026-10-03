"""The in-memory index's fast pass must never change a search's answer.

Every test compares the index with an independent full scan of the same points, using the exact formula the
index has always used, so a result that differs in any chunk, order or last bit of a score fails."""
from __future__ import annotations

import asyncio
import math
import random
from operator import mul
from typing import Optional

import pytest

from scone_memory.backends import memory as memory_module
from scone_memory.backends.memory import InMemoryVectorIndex
from scone_memory.core.ports import VectorPoint
from scone_memory.core.timeutil import is_before_or_at

DIM = 12


def _point(chunk_id: int, vector: list[float], *, space: str = 's', day: int = 1, tag: str = 'a') -> VectorPoint:
    return VectorPoint(chunk_id=chunk_id, space=space, episode_id=chunk_id, created_at=f'2026-01-{day:02d}T00:00:00Z',
                       vector=vector, tags=(tag,), metadata={'kind': tag})


def _full_scan(points: list[VectorPoint], space: str, vector: list[float], limit: int,
               as_of: Optional[str] = None, tags: tuple[str, ...] = ()) -> list[tuple[int, float]]:
    query_norm = math.sqrt(sum(map(mul, vector, vector)))
    scored = []
    for point in points:
        if point.space != space or (as_of and not is_before_or_at(point.created_at, as_of)):
            continue
        if tags and not set(tags) <= set(point.tags):
            continue
        norm = math.sqrt(sum(map(mul, point.vector, point.vector)))
        scored.append((point.chunk_id, 0.0 if query_norm == 0 or norm == 0
                       else sum(map(mul, vector, point.vector)) / (query_norm * norm)))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:limit]


def _corpus(count: int, seed: int) -> list[VectorPoint]:
    rng = random.Random(seed)
    points = []
    for chunk_id in range(1, count + 1):
        vector = [rng.gauss(0, 1) for _ in range(DIM)]
        if chunk_id % 50 == 0:
            vector = [0.0] * DIM  # zero vectors score 0 and must still sort by chunk id
        points.append(_point(chunk_id, vector, space='s' if chunk_id % 7 else 't', day=1 + chunk_id % 28,
                             tag='a' if chunk_id % 3 else 'b'))
    return points


async def _index(points: list[VectorPoint]) -> InMemoryVectorIndex:
    index = InMemoryVectorIndex()
    await index.ensure(DIM)
    await index.upsert(points)
    return index


@pytest.mark.parametrize('path', ['numpy', 'sumprod', 'loop'])
def test_every_fast_path_returns_the_full_scan_to_the_last_bit(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if path in ('sumprod', 'loop'):
        monkeypatch.setattr(memory_module, '_numpy', None)
    if path == 'loop':
        monkeypatch.delattr(math, 'sumprod', raising=False)
    elif path == 'sumprod' and not hasattr(math, 'sumprod'):
        pytest.skip('math.sumprod needs Python 3.12')
    points = _corpus(6_000, seed=3)

    async def go() -> None:
        index = await _index(points)
        rng = random.Random(9)
        for trial in range(30):
            query = [rng.gauss(0, 1) for _ in range(DIM)]
            limit = (1, 5, 40, 300)[trial % 4]
            assert await index.search('s', query, limit) == _full_scan(points, 's', query, limit)
            as_of, tags = '2026-01-14T00:00:00Z', ('a',)
            assert (await index.search('s', query, limit, as_of=as_of, tags=tags)
                    == _full_scan(points, 's', query, limit, as_of=as_of, tags=tags))
    asyncio.run(go())


def test_a_crowd_of_exact_ties_at_the_cut_is_broken_by_chunk_id_as_in_a_full_scan() -> None:
    rng = random.Random(5)
    tied = [1.0] + [0.0] * (DIM - 1)
    points = [_point(chunk_id, [rng.gauss(0, 1) for _ in range(DIM)]) for chunk_id in range(1, 3_001)]
    # 600 identical vectors, more than the shortlist's spare places, scattered through the ids.
    for chunk_id in rng.sample(range(1, 3_001), 600):
        points[chunk_id - 1] = _point(chunk_id, list(tied))

    # Inserted out of id order, so a cut that kept insertion order among ties would keep the wrong ids.
    shuffled = list(points)
    rng.shuffle(shuffled)

    async def go() -> None:
        index = await _index(shuffled)
        for limit in (10, 400, 700):
            assert await index.search('s', tied, limit) == _full_scan(points, 's', tied, limit)
    asyncio.run(go())


def test_every_kind_of_write_is_seen_by_the_next_search() -> None:
    points = _corpus(3_000, seed=11)
    query = [1.0] * DIM

    async def go() -> None:
        index = await _index(points)
        await index.search('s', query, 5)  # builds the matrix
        best = _point(99_999, list(query))
        await index.upsert([best])
        assert (await index.search('s', query, 1))[0][0] == 99_999
        await index.delete([99_999])
        assert (await index.search('s', query, 5)) == _full_scan(points, 's', query, 5)
        await index.delete_space('s')
        assert await index.search('s', query, 5) == []
        assert await index.search('t', query, 5) == _full_scan(points, 't', query, 5)
    asyncio.run(go())


def test_near_ties_that_round_differently_are_still_scored_exactly() -> None:
    """Permutations of one vector have one exact cosine with an all-ones query: summed in the plain formula's
    order, the small components vanish beside 1e16 in every order alike. numpy's matrix product sums differently
    and splits them into several values, so a cut by fast score alone would drop some of the exact top (which a
    full scan breaks by chunk id). Only the margin keeps them."""
    rng = random.Random(1)
    base = [1e16, 1.0, 1e16, 3.0, 5.0, 7.0, 2.5e15, 2.5e15, 0.5, 9.0, 1.5, 11.0]
    seen: set[tuple[float, ...]] = set()
    points: list[VectorPoint] = []
    while len(points) < 1_500:
        order = base[:]
        rng.shuffle(order)
        if tuple(order) not in seen:
            seen.add(tuple(order))
            points.append(_point(len(points) + 1, order))
    query = [1.0] * DIM

    async def go() -> None:
        index = await _index(points)
        fast = index._fast_scores('s', points, query, math.sqrt(DIM))
        if fast is not None:  # the fixture must put fast-pass disagreement past the cut, or it proves nothing
            exact_top = [chunk_id for chunk_id, _ in _full_scan(points, 's', query, 10)]
            fast_rank = {p.chunk_id: r for r, p in enumerate(sorted(points, key=lambda p: -fast[p.chunk_id - 1]))}
            assert max(fast_rank[c] for c in exact_top) > 10 + memory_module.SHORTLIST_SPARE
        for limit in (1, 10, 50):
            assert await index.search('s', query, limit) == _full_scan(points, 's', query, limit)
    asyncio.run(go())
