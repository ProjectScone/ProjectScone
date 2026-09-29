"""Scone's in-memory vector index, searched in one matrix product instead of a Python loop.

A full HotpotQA dev corpus holds about 66,000 paragraphs. The default
``InMemoryVectorIndex.search`` scores every point with ``sum(map(mul, ...))``;
at that size one query takes seconds, and 7,405 queries take hours. This
subclass finds candidates with numpy and then scores those candidates with
the default's own formula, in the default's own order (score descending,
then chunk id). So the list it returns is the list the default returns, bit
for bit, provided the true top ``limit`` lies inside the numpy shortlist.
The shortlist is ``max(4 * limit, limit + 256)`` points; numpy's float64
dot products differ from the Python sums only in the last bits, so a point
outside it would need a score within about 1e-15 of hundreds of others.
``test_fast_index`` checks equality against the default on random data.

Filtered searches (``as_of``, tags, metadata, conditions) and small spaces
go to the default unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import mul
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scone_memory.backends import InMemoryVectorIndex

if TYPE_CHECKING:
    from scone_memory.core.ports import VectorPoint

SMALL = 2_000  # below this many points the default loop is fast enough


class FastExactVectorIndex(InMemoryVectorIndex):
    def __init__(self) -> None:
        super().__init__()
        self._snapshots: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        await super().upsert(points)
        self._snapshots.clear()

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        await super().delete(chunk_ids)
        self._snapshots.clear()

    async def delete_space(self, space: str) -> None:
        await super().delete_space(space)
        self._snapshots.clear()

    def _snapshot(self, space: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if space not in self._snapshots:
            points = [p for p in self._points.values() if p.space == space]
            ids = np.fromiter((p.chunk_id for p in points), dtype=np.int64, count=len(points))
            matrix = np.asarray([p.vector for p in points], dtype=np.float64)
            norms = np.fromiter((self._norms[p.chunk_id] for p in points), dtype=np.float64, count=len(points))
            self._snapshots[space] = (ids, matrix, norms)
        return self._snapshots[space]

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
        conditions: Any = None,
    ) -> list[tuple[int, float]]:
        if as_of or tags or where or conditions is not None:
            return await super().search(space, vector, limit, as_of, tags, where, conditions)
        ids, matrix, norms = self._snapshot(space)
        shortlist = max(4 * limit, limit + 256)
        if len(ids) <= max(SMALL, shortlist):
            return await super().search(space, vector, limit, as_of, tags, where, conditions)
        query = np.asarray(vector, dtype=np.float64)
        query_norm = float(np.sqrt(query @ query))
        if query_norm == 0:
            return await super().search(space, vector, limit, as_of, tags, where, conditions)
        with np.errstate(divide='ignore', invalid='ignore'):
            approx = np.where(norms == 0, 0.0, (matrix @ query) / (norms * query_norm))
        candidates = ids[np.argpartition(-approx, shortlist - 1)[:shortlist]]
        # The default's own score for each candidate, so the result is the default's to the last bit.
        exact_norm = super_norm(vector)
        scored = []
        for chunk_id in candidates.tolist():
            point = self._points[chunk_id]
            norm = self._norms[chunk_id]
            scored.append((chunk_id, 0.0 if exact_norm == 0 or norm == 0
                           else sum(map(mul, vector, point.vector)) / (exact_norm * norm)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]


def super_norm(vector: Sequence[float]) -> float:
    from scone_memory.backends.memory import _norm

    return float(_norm(vector))


def install() -> None:
    """Make the in-process engine builder use this index; everything else it assembles is unchanged."""
    import scone_memory.backends as backends

    backends.InMemoryVectorIndex = FastExactVectorIndex  # type: ignore[misc]
