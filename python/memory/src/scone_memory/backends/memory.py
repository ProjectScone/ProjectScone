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

from ..retrieval.lexical import Bm25
from ..core.models import IngestJob, Chunk, Episode, Fact, FactLink, Tombstone
from ..core.ports import DeletedSpace, NewJob, NewChunk, NewEpisode, NewFact, NewFactLink, NewTombstone, SpaceCounts, TextFilter, VectorPoint
from ..core.timeutil import is_before_or_at
from .validation import validate_vector


class InMemoryDocumentStore:
    name = "memory"

    def __init__(self) -> None:
        self._episodes: dict[int, Episode] = {}
        self._chunks: dict[int, Chunk] = {}
        self._facts: dict[int, Fact] = {}
        self._links: dict[int, FactLink] = {}
        self._tombstones: dict[tuple[str, int], Tombstone] = {}
        self._episode_ids = count(1)
        self._chunk_ids = count(1)
        self._fact_ids = count(1)
        self._link_ids = count(1)
        self._bm25: dict[str, Bm25] = defaultdict(Bm25)
        self._revision: dict[str, int] = defaultdict(int)
        self._inflight: set[tuple[str, str]] = set()
        self._deleted: dict[str, str] = {}
        self._jobs: dict[tuple[str, str], IngestJob] = {}
        self._job_seq: dict[tuple[str, str], int] = {}

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

    async def delete_space(self, space: str, deleted_at: str) -> DeletedSpace:
        episode_ids = [e.episode_id for e in self._episodes.values() if e.space == space]
        chunk_ids = tuple(sorted(c.chunk_id for c in self._chunks.values() if c.space == space))
        fact_ids = [i for i, f in self._facts.items() if f.space == space]
        link_ids = [i for i, l in self._links.items() if l.space == space]
        stones = [k for k in self._tombstones if k[0] == space]
        for episode_id in episode_ids:
            del self._episodes[episode_id]
        for chunk_id in chunk_ids:
            del self._chunks[chunk_id]
        self._bm25.pop(space, None)
        for fact_id in fact_ids:
            del self._facts[fact_id]
        for link_id in link_ids:
            del self._links[link_id]
        for key in stones:
            del self._tombstones[key]
        self._inflight = {mark for mark in self._inflight if mark[0] != space}
        for key in [k for k in self._jobs if k[0] == space]:
            del self._jobs[key]
            self._job_seq.pop(key, None)
        self._revision.pop(space, None)
        self._deleted[space] = deleted_at
        return DeletedSpace(chunk_ids=chunk_ids, episodes=len(episode_ids), facts=len(fact_ids),
                            links=len(link_ids), tombstones=len(stones))

    async def space_deleted(self, space: str) -> Optional[str]:
        return self._deleted.get(space)

    # -- ingest jobs ------------------------------------------------------

    async def create_job(self, new: NewJob) -> IngestJob:
        job = IngestJob(job_id=new.job_id, space=new.space, created_at=new.created_at,
                        request_id=new.request_id, items=list(new.items))
        self._jobs[(new.space, new.job_id)] = job
        self._job_seq[(new.space, new.job_id)] = len(self._job_seq)
        return job

    async def get_job(self, space: str, job_id: str) -> Optional[IngestJob]:
        return self._jobs.get((space, job_id))

    async def job_by_request(self, space: str, request_id: str) -> Optional[IngestJob]:
        return next((j for (s, _), j in self._jobs.items() if s == space and j.request_id == request_id), None)

    async def list_jobs(self, space: str, limit: int) -> list[IngestJob]:
        mine = [(key, job) for key, job in self._jobs.items() if key[0] == space]
        ordered = sorted(mine, key=lambda pair: (pair[1].created_at, self._job_seq[pair[0]]), reverse=True)
        return [job for _, job in ordered][:limit]

    async def update_job(self, job: IngestJob) -> None:
        self._jobs[(job.space, job.job_id)] = job

    async def mark_failed(self, space: str, episode_id: int, error: str, when: str) -> int:
        moved = 0
        for key, job in list(self._jobs.items()):
            if key[0] != space:
                continue
            items = []
            for item in job.items:
                if item.episode_id == episode_id and item.consolidated_at is None:
                    items.append(item.model_copy(update={
                        "state": "failed", "error": error, "attempts": item.attempts + 1}))
                    moved += 1
                else:
                    items.append(item)
            self._jobs[key] = job.model_copy(update={"items": items})
        return moved

    async def mark_consolidated(self, space: str, episode_ids: Sequence[int], when: str) -> int:
        wanted, moved = set(episode_ids), 0
        for key, job in list(self._jobs.items()):
            if key[0] != space:
                continue
            items = []
            for item in job.items:
                if item.episode_id in wanted and item.consolidated_at is None:
                    # A retry that works clears the error; the attempt count
                    # stays, because having had to retry is part of the story.
                    items.append(item.model_copy(update={
                        "consolidated_at": when, "state": "consolidated", "error": None}))
                    moved += 1
                else:
                    items.append(item)
            self._jobs[key] = job.model_copy(update={"items": items})
        return moved

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

    async def chunks_of(self, space: str, episode_id: int) -> list[Chunk]:
        found = [c for c in self._chunks.values() if c.space == space and c.episode_id == episode_id]
        return sorted(found, key=lambda c: c.ordinal)

    async def mark_inflight(self, space: str, content_hash: str) -> None:
        self._inflight.add((space, content_hash))

    async def clear_inflight(self, space: str, content_hash: str) -> None:
        self._inflight.discard((space, content_hash))

    async def inflight(self) -> list[tuple[str, str]]:
        return sorted(self._inflight)

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

    async def page_episodes(self, space, before, limit, kind):
        from heapq import nlargest
        return nlargest(limit, (e for e in self._episodes.values()
            if e.space == space and (before is None or e.episode_id < before)
            and (kind is None or e.kind == kind)), key=lambda e: e.episode_id)

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

    async def record_tombstone(self, new: NewTombstone) -> Tombstone:
        return self._tombstones.setdefault((new.space, new.episode_id), Tombstone(**new.__dict__))

    async def tombstone(self, space: str, episode_id: int) -> Optional[Tombstone]:
        return self._tombstones.get((space, episode_id))

    async def tombstone_by_hash(self, space: str, content_hash: str) -> Optional[Tombstone]:
        found = [t for (s, _), t in self._tombstones.items() if s == space and t.content_hash == content_hash]
        return max(found, key=lambda t: t.episode_id) if found else None

    async def list_tombstones(self, space: str) -> list[Tombstone]:
        return sorted((t for (s, _), t in self._tombstones.items() if s == space), key=lambda t: t.episode_id)

    async def chunk_index(self, space: str) -> list[tuple[int, int]]:
        """(chunk_id, episode_id) for every chunk of the space; for doctor."""
        return sorted((c.chunk_id, c.episode_id) for c in self._chunks.values() if c.space == space)

    async def insert_fact_link(self, new: NewFactLink) -> FactLink:
        for link in self._links.values():
            if (link.space, link.from_fact, link.to_fact, link.kind) == (new.space, new.from_fact, new.to_fact, new.kind):
                return link
        link = FactLink(link_id=next(self._link_ids), **new.__dict__)
        self._links[link.link_id] = link
        return link

    async def fact_links(self, space: str, fact_id: int) -> list[FactLink]:
        return [l for l in self._links.values() if l.space == space and fact_id in (l.from_fact, l.to_fact)]

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

    async def ids(self, space: str) -> list[int]:
        """Every chunk id with a vector in the space; for doctor."""
        return sorted(p.chunk_id for p in self._points.values() if p.space == space)

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

    async def delete_space(self, space: str) -> None:
        self._points = {chunk_id: p for chunk_id, p in self._points.items() if p.space != space}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
