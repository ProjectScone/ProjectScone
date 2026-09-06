"""Dict-backed stores: the reference implementation and the default.

Everything the engine needs in a few hundred lines with no server, so
the library works in a notebook, a test, or a sandbox with no network.
The Mongo and Qdrant adapters are checked against the same contract
tests as this one; when they disagree, this file is what "correct"
means.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import count
from typing import Mapping, Optional, Sequence

from ..lexical import Bm25
from ..models import Chunk, Episode, Fact
from ..ports import NewChunk, NewEpisode, NewFact, SpaceCounts, TextFilter, VectorPoint
from ..timeutil import is_before_or_at
from .validation import validate_vector


class InMemoryDocumentStore:
    name = "memory"

    def __init__(self) -> None:
        self._episodes: dict[int, Episode] = {}
        self._chunks: dict[int, Chunk] = {}
        self._facts: dict[int, Fact] = {}
        self._episode_ids = count(1)
        self._chunk_ids = count(1)
        self._fact_ids = count(1)
        self._bm25: dict[str, Bm25] = defaultdict(Bm25)
        self._revision: dict[str, int] = defaultdict(int)

    async def insert_episode(self, new: NewEpisode) -> Episode:
        episode = Episode(episode_id=next(self._episode_ids), **new.__dict__)
        self._episodes[episode.episode_id] = episode
        return episode

    async def episode_by_hash(self, space: str, content_hash: str) -> Optional[Episode]:
        for episode in self._episodes.values():
            if episode.space == space and episode.content_hash == content_hash:
                return episode
        return None

    async def get_episode(self, space: str, episode_id: int) -> Optional[Episode]:
        episode = self._episodes.get(episode_id)
        return episode if episode and episode.space == space else None

    async def delete_episode(self, space: str, episode_id: int) -> list[int]:
        episode = await self.get_episode(space, episode_id)
        if episode is None:
            return []
        del self._episodes[episode_id]
        removed = [c.chunk_id for c in self._chunks.values() if c.episode_id == episode_id]
        for chunk_id in removed:
            del self._chunks[chunk_id]
            self._bm25[space].remove(chunk_id)
        return removed

    async def insert_chunks(self, new: Sequence[NewChunk]) -> list[Chunk]:
        out = []
        for n in new:
            chunk = Chunk(chunk_id=next(self._chunk_ids), **n.__dict__)
            self._chunks[chunk.chunk_id] = chunk
            self._bm25[chunk.space].add(chunk.chunk_id, chunk.text)
            out.append(chunk)
        return out

    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]:
        found = []
        for chunk_id in chunk_ids:
            chunk = self._chunks.get(chunk_id)
            if chunk and chunk.space == space:
                found.append(chunk)
        return found

    async def search_text(
        self, space: str, query: str, limit: int, filter: TextFilter
    ) -> list[tuple[int, float]]:
        allowed = None
        if filter.as_of or filter.tags or filter.where:
            allowed = [
                c.chunk_id
                for c in self._chunks.values()
                if c.space == space and self._passes(c, filter)
            ]
        return self._bm25[space].search(query, limit, allowed)

    def _passes(self, chunk: Chunk, filter: TextFilter) -> bool:
        if filter.as_of and not is_before_or_at(chunk.created_at, filter.as_of):
            return False
        if filter.tags or filter.where:
            episode = self._episodes.get(chunk.episode_id)
            if episode is None:
                return False
            if filter.tags and not set(filter.tags) <= set(episode.tags):
                return False
            if any(episode.metadata.get(k) != v for k, v in filter.where.items()):
                return False
        return True

    async def recent_episodes(self, space: str, limit: int) -> list[Episode]:
        mine = [e for e in self._episodes.values() if e.space == space]
        mine.sort(key=lambda e: (e.created_at, e.episode_id), reverse=True)
        return mine[:limit]

    async def counts(self, space: str) -> SpaceCounts:
        counts = SpaceCounts()
        for episode in self._episodes.values():
            if episode.space != space:
                continue
            counts.episodes += 1
            counts.bytes += len(episode.content.encode())
            for tag in episode.tags:
                counts.tags[tag] = counts.tags.get(tag, 0) + 1
        counts.chunks = sum(1 for c in self._chunks.values() if c.space == space)
        return counts

    async def insert_fact(self, new: NewFact) -> Fact:
        fact = Fact(fact_id=next(self._fact_ids), **new.__dict__)
        self._facts[fact.fact_id] = fact
        return fact

    async def update_fact(self, fact: Fact) -> None:
        if fact.fact_id not in self._facts:
            raise KeyError(fact.fact_id)
        self._facts[fact.fact_id] = fact

    async def get_fact(self, space: str, fact_id: int) -> Optional[Fact]:
        fact = self._facts.get(fact_id)
        return fact if fact and fact.space == space else None

    async def list_facts(self, space: str, include_closed: bool) -> list[Fact]:
        return [
            f
            for f in self._facts.values()
            if f.space == space and (include_closed or f.status == "active")
        ]

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        return [
            f
            for f in self._facts.values()
            if f.space == space and f.subject == subject and f.predicate == predicate
        ]

    async def bump_revision(self, space: str) -> int:
        self._revision[space] += 1
        return self._revision[space]

    async def revision(self, space: str) -> int:
        return self._revision[space]


class InMemoryVectorIndex:
    name = "memory"

    def __init__(self) -> None:
        self._points: dict[int, VectorPoint] = {}
        self.dim: Optional[int] = None

    async def ensure(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim:
            raise ValueError(f"index holds {self.dim}-d vectors, embedder makes {dim}-d")
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        for point in points:
            validate_vector(point.vector, self.dim)
        for point in points:
            self._points[point.chunk_id] = point

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        validate_vector(vector, self.dim)
        scored = []
        for point in self._points.values():
            if point.space != space:
                continue
            if as_of and not is_before_or_at(point.created_at, as_of):
                continue
            if tags and not set(tags) <= set(point.tags):
                continue
            if where and any(point.metadata.get(k) != v for k, v in where.items()):
                continue
            scored.append((point.chunk_id, _cosine(vector, point.vector)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        for chunk_id in chunk_ids:
            self._points.pop(chunk_id, None)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
