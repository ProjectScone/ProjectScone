from __future__ import annotations

import asyncio
import random

from scone_memory.backends import InMemoryVectorIndex
from scone_memory.core.ports import VectorPoint

from .fast_index import FastExactVectorIndex


def _points(count: int, dim: int, seed: int) -> list[VectorPoint]:
    rng = random.Random(seed)
    points: list[VectorPoint] = []
    for chunk_id in range(1, count + 1):
        vector = [rng.gauss(0, 1) for _ in range(dim)]
        if chunk_id % 97 == 0:  # exact duplicates: ties broken by chunk id
            vector = list(points[-1].vector)
        points.append(VectorPoint(chunk_id=chunk_id, space='s', episode_id=chunk_id,
                                  created_at='2026-01-01T00:00:00Z', vector=vector))
    return points


async def _both(count: int, queries: int, limit: int) -> None:
    dim = 16
    slow, fast = InMemoryVectorIndex(), FastExactVectorIndex()
    for index in (slow, fast):
        await index.ensure(dim)
        await index.upsert(_points(count, dim, 7))
    rng = random.Random(11)
    for _ in range(queries):
        query = [rng.gauss(0, 1) for _ in range(dim)]
        assert await fast.search('s', query, limit) == await slow.search('s', query, limit)


def test_fast_search_returns_the_default_list_bit_for_bit() -> None:
    asyncio.run(_both(count=5_000, queries=40, limit=20))


def test_a_write_invalidates_the_snapshot() -> None:
    async def go() -> None:
        index = FastExactVectorIndex()
        await index.ensure(2)
        await index.upsert([VectorPoint(chunk_id=i, space='s', episode_id=i, created_at='2026-01-01T00:00:00Z',
                                        vector=[1.0, float(i) / 5000]) for i in range(1, 3001)])
        first = await index.search('s', [0.0, 1.0], 1)
        await index.upsert([VectorPoint(chunk_id=9999, space='s', episode_id=1, created_at='2026-01-01T00:00:00Z',
                                        vector=[0.0, 1.0])])
        assert first[0][0] != 9999 and (await index.search('s', [0.0, 1.0], 1))[0][0] == 9999
    asyncio.run(go())


def test_twohop_fusion_keeps_the_agreed_leader_and_lifts_what_hops_share() -> None:
    from .twohop import fuse

    first = ['a', 'b', 'c']
    hops = [['a', 'x', 'y'], ['b', 'x', 'z']]
    fused = fuse([first, *hops], 4)
    assert fused[0] == 'a' and fused.index('x') < fused.index('c') and len(fused) == 4
