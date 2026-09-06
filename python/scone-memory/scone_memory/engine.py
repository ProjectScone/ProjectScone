"""The engine: remember, recall, forget, and the fact ledger.

Built entirely against the protocols in ``ports``; nothing here knows
whether a dict, MongoDB, or Qdrant is underneath. A lane that fails
during recall is named in ``degraded`` and the other lane still
answers, because a thin answer that says it is thin beats a 500.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from . import fusion
from .chunker import DEFAULT_TARGET, chunk_spans
from .errors import InvalidInput, NotFound
from .lexical import tokenize
from .models import (
    MAX_CONTENT_BYTES,
    Added,
    Episode,
    EpisodeKind,
    Fact,
    RecallItem,
    RecallResult,
    Status,
)
from .ports import (
    DocumentStore,
    Embedder,
    NewChunk,
    NewEpisode,
    NewFact,
    TextFilter,
    VectorIndex,
    VectorPoint,
)
from .timeutil import format_rfc3339, now_rfc3339, parse_rfc3339

SPACE_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
KINDS = ("note", "chat", "file", "web", "connector")
MAX_QUERY = 1_000
MAX_LIMIT = 50
#: How many candidates each lane contributes before fusion.
LANE_DEPTH = 4


@dataclass
class Profile:
    static_facts: list[Fact] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)


class MemoryEngine:
    def __init__(
        self,
        documents: DocumentStore,
        vectors: VectorIndex,
        embedder: Embedder,
        chunk_target: int = DEFAULT_TARGET,
        clock: Callable[[], str] = now_rfc3339,
    ) -> None:
        self.documents = documents
        self.vectors = vectors
        self.embedder = embedder
        self.chunk_target = chunk_target
        self.clock = clock

    async def open(self) -> "MemoryEngine":
        await self.vectors.ensure(self.embedder.dim)
        return self

    # -- episodes ---------------------------------------------------------

    async def remember(
        self,
        space: str,
        content: str,
        kind: EpisodeKind = "note",
        source: Optional[str] = None,
        tags: Sequence[str] = (),
        created_at: Optional[str] = None,
    ) -> Added:
        check_space(space)
        if not content.strip():
            raise InvalidInput("content must not be empty")
        if len(content.encode()) > MAX_CONTENT_BYTES:
            raise InvalidInput(f"content exceeds {MAX_CONTENT_BYTES} bytes")
        if kind not in KINDS:
            raise InvalidInput(f"kind must be one of {KINDS}, got {kind!r}")
        clean_tags = normalise_tags(tags)
        when = self.clock()
        happened = normalise_time(created_at) if created_at else when
        digest = content_hash(space, content)
        existing = await self.documents.episode_by_hash(space, digest)
        if existing is not None:
            return Added(episode_id=existing.episode_id, deduplicated=True, chunks=0)

        episode = await self.documents.insert_episode(
            NewEpisode(
                space=space,
                kind=kind,
                content=content,
                content_hash=digest,
                created_at=happened,
                ingested_at=when,
                source=source,
                tags=clean_tags,
            )
        )
        spans = chunk_spans(content, self.chunk_target)
        chunks = await self.documents.insert_chunks(
            [
                NewChunk(
                    episode_id=episode.episode_id,
                    space=space,
                    ordinal=i,
                    start=s.start,
                    end=s.end,
                    text=content[s.start : s.end],
                    created_at=happened,
                )
                for i, s in enumerate(spans)
            ]
        )
        vectors = await self.embedder.embed([c.text for c in chunks])
        await self.vectors.upsert(
            [
                VectorPoint(
                    chunk_id=c.chunk_id,
                    space=space,
                    episode_id=episode.episode_id,
                    created_at=happened,
                    vector=v,
                    tags=clean_tags,
                )
                for c, v in zip(chunks, vectors)
            ]
        )
        await self.documents.bump_revision(space)
        return Added(episode_id=episode.episode_id, deduplicated=False, chunks=len(chunks))

    async def forget(self, space: str, episode_id: int) -> None:
        check_space(space)
        removed = await self.documents.delete_episode(space, episode_id)
        if not removed and await self.documents.get_episode(space, episode_id) is None:
            raise NotFound(f"episode {episode_id} not found in {space!r}")
        await self.vectors.delete(removed)
        await self.documents.bump_revision(space)

    async def episode(self, space: str, episode_id: int) -> Episode:
        check_space(space)
        found = await self.documents.get_episode(space, episode_id)
        if found is None:
            raise NotFound(f"episode {episode_id} not found in {space!r}")
        return found

    # -- recall -----------------------------------------------------------

    async def recall(
        self,
        space: str,
        query: str,
        limit: int = 5,
        as_of: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> RecallResult:
        check_space(space)
        query = query.strip()
        if not query or len(query) > MAX_QUERY:
            raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
        limit = max(1, min(limit, MAX_LIMIT))
        boundary = normalise_time(as_of) if as_of else None
        clean_tags = normalise_tags(tags)
        depth = limit * LANE_DEPTH
        degraded: list[str] = []

        vector_lane: list[tuple[int, float]] = []
        try:
            [qvec] = await self.embedder.embed([query])
            vector_lane = await self.vectors.search(space, qvec, depth, boundary, clean_tags)
        except Exception as e:  # noqa: BLE001 - the lane is reported, not hidden
            degraded.append(f"vectors: {type(e).__name__}: {e}")

        text_lane: list[tuple[int, float]] = []
        try:
            text_lane = await self.documents.search_text(
                space, query, depth, TextFilter(as_of=boundary, tags=clean_tags)
            )
        except Exception as e:  # noqa: BLE001
            degraded.append(f"text: {type(e).__name__}: {e}")

        if len(degraded) == 2:
            raise RuntimeError("both recall lanes failed: " + "; ".join(degraded))

        similarity = dict(vector_lane)
        fused = fusion.rrf([vector_lane, text_lane])
        chunks = {c.chunk_id: c for c in await self.documents.get_chunks(space, list(fused))}
        now = self.clock()
        items = [
            fusion.Fused(cid, score + fusion.recency_boost(chunks[cid].created_at, now), similarity.get(cid))
            for cid, score in fused.items()
            if cid in chunks
        ]
        items = fusion.order(items)
        items = fusion.cap_per_episode(items, {cid: c.episode_id for cid, c in chunks.items()})
        items = fusion.normalise(items[:limit])

        episodes: dict[int, Episode] = {}
        for item in items:
            eid = chunks[item.chunk_id].episode_id
            if eid not in episodes:
                found = await self.documents.get_episode(space, eid)
                if found is not None:
                    episodes[eid] = found

        result_items = []
        for item in items:
            chunk = chunks[item.chunk_id]
            episode = episodes.get(chunk.episode_id)
            result_items.append(
                RecallItem(
                    chunk_id=chunk.chunk_id,
                    episode_id=chunk.episode_id,
                    text=chunk.text,
                    score=round(item.score, 6),
                    similarity=None if item.similarity is None else round(item.similarity, 6),
                    created_at=chunk.created_at,
                    source=episode.source if episode else None,
                    tags=episode.tags if episode else (),
                )
            )
        facts = await self._facts_for_query(space, query, boundary or now)
        counts = await self.documents.counts(space)
        return RecallResult(
            items=result_items,
            facts=facts,
            degraded=degraded,
            returned_bytes=sum(len(i.text.encode()) for i in result_items),
            space_bytes=counts.bytes,
        )

    async def _facts_for_query(self, space: str, query: str, when: str, limit: int = 10) -> list[Fact]:
        terms = set(tokenize(query))
        if not terms:
            return []
        scored = []
        # Closed facts are included on purpose: asked about 2023, the
        # fact that held in 2023 is the answer even if it closed since.
        for fact in await self.documents.list_facts(space, include_closed=True):
            if not fact.holds_at(when):
                continue
            overlap = len(terms & set(tokenize(f"{fact.subject} {fact.predicate} {fact.object}")))
            if overlap:
                scored.append((-overlap, -fact.confidence, fact.fact_id, fact))
        scored.sort(key=lambda t: t[:3])
        return [t[3] for t in scored[:limit]]

    # -- facts ------------------------------------------------------------

    async def assert_fact(
        self,
        space: str,
        subject: str,
        predicate: str,
        object: str,
        valid_from: Optional[str] = None,
        confidence: float = 1.0,
        source_episode_id: Optional[int] = None,
    ) -> Fact:
        """Record that ``subject predicate object`` holds from ``valid_from``.

        An active fact with the same subject and predicate but a
        different object is closed at the new fact's start, with the
        reason naming what superseded it. A fact that arrives late (its
        start precedes an existing fact's start) is stored already
        closed against that existing fact rather than closing it: a
        stale record must not overwrite a fresher one.
        """
        check_space(space)
        subject = normalise_term(subject, "subject")
        predicate = normalise_term(predicate, "predicate")
        object = object.strip()
        if not object:
            raise InvalidInput("object must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise InvalidInput("confidence must be within 0..=1")
        start = normalise_time(valid_from) if valid_from else self.clock()

        rivals = [
            f for f in await self.documents.facts_for(space, subject, predicate) if f.status == "active"
        ]
        for rival in rivals:
            if rival.object == object:
                return rival
        newer = [r for r in rivals if parse_rfc3339(r.valid_from) > parse_rfc3339(start)]
        if newer:
            blocker = min(newer, key=lambda f: parse_rfc3339(f.valid_from))
            stale = await self.documents.insert_fact(
                NewFact(
                    space=space,
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    valid_from=start,
                    valid_until=blocker.valid_from,
                    confidence=confidence,
                    status="closed",
                    closed_reason=f"superseded by fact {blocker.fact_id}",
                    source_episode_id=source_episode_id,
                )
            )
            await self.documents.bump_revision(space)
            return stale

        fact = await self.documents.insert_fact(
            NewFact(
                space=space,
                subject=subject,
                predicate=predicate,
                object=object,
                valid_from=start,
                confidence=confidence,
                source_episode_id=source_episode_id,
            )
        )
        for rival in rivals:
            await self.documents.update_fact(
                rival.model_copy(
                    update={
                        "status": "closed",
                        "valid_until": start,
                        "closed_reason": f"superseded by fact {fact.fact_id}",
                    }
                )
            )
        await self.documents.bump_revision(space)
        return fact

    async def close_fact(self, space: str, fact_id: int, reason: str) -> Fact:
        check_space(space)
        reason = reason.strip()
        if not reason or len(reason) > 500:
            raise InvalidInput("reason must be 1..=500 chars")
        fact = await self.documents.get_fact(space, fact_id)
        if fact is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        if fact.status == "closed":
            return fact
        closed = fact.model_copy(
            update={"status": "closed", "valid_until": self.clock(), "closed_reason": reason}
        )
        await self.documents.update_fact(closed)
        await self.documents.bump_revision(space)
        return closed

    async def facts(
        self, space: str, include_closed: bool = False, as_of: Optional[str] = None
    ) -> list[Fact]:
        check_space(space)
        found = await self.documents.list_facts(space, include_closed=include_closed or as_of is not None)
        if as_of is not None:
            boundary = normalise_time(as_of)
            found = [f for f in found if f.holds_at(boundary)]
        return sorted(found, key=lambda f: f.fact_id)

    # -- overviews --------------------------------------------------------

    async def profile(self, space: str, limit: int = 10) -> Profile:
        check_space(space)
        limit = max(1, min(limit, 50))
        active = await self.documents.list_facts(space, include_closed=False)
        active.sort(key=lambda f: (-f.confidence, f.fact_id))
        recent = await self.documents.recent_episodes(space, limit)
        return Profile(
            static_facts=active[:limit],
            dynamic=[e.content[:200] for e in recent],
        )

    async def tags(self, space: str) -> dict[str, int]:
        check_space(space)
        return dict(sorted((await self.documents.counts(space)).tags.items()))

    async def status(self, space: str) -> Status:
        check_space(space)
        counts = await self.documents.counts(space)
        return Status(
            space=space,
            episodes=counts.episodes,
            chunks=counts.chunks,
            bytes=counts.bytes,
            revision=await self.documents.revision(space),
            embedder=self.embedder.id,
            document_store=self.documents.name,
            vector_index=self.vectors.name,
        )


# -- validation helpers -----------------------------------------------------


def check_space(space: str) -> None:
    if not SPACE_NAME.match(space or ""):
        raise InvalidInput(f"space name must be 1..=64 chars of [a-z0-9-_], got {space!r}")


def normalise_tags(tags: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for tag in tags:
        clean = tag.strip().casefold()
        if not clean:
            continue
        if len(clean) > 64:
            raise InvalidInput(f"tag too long: {tag!r}")
        if clean not in seen:
            seen.append(clean)
    return tuple(seen)


def normalise_term(value: str, what: str) -> str:
    clean = " ".join(value.strip().casefold().split())
    if not clean:
        raise InvalidInput(f"{what} must not be empty")
    return clean


def normalise_time(value: str) -> str:
    try:
        return format_rfc3339(parse_rfc3339(value))
    except ValueError as e:
        raise InvalidInput(f"not an RFC 3339 timestamp: {value!r}") from e


def content_hash(space: str, content: str) -> str:
    return hashlib.sha256(f"{space}\x00{content.strip()}".encode()).hexdigest()
