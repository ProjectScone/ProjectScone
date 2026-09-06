"""The engine: remember, recall, forget, and the fact ledger.

Built entirely against the protocols in ``ports``; nothing here knows
whether a dict, MongoDB, or Qdrant is underneath. A lane that fails
during recall is named in ``degraded`` and the other lane still
answers, because a thin answer that says it is thin beats a 500.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterable, Mapping, Optional, Sequence

from . import fusion
from .chunker import DEFAULT_TARGET, byte_spans, chunk_spans
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
    Event,
    EventLog,
    NewChunk,
    NewEpisode,
    NewEvent,
    NewFact,
    TextFilter,
    VectorIndex,
    VectorPoint,
)
from .timeutil import format_rfc3339, now_rfc3339, parse_rfc3339

SPACE_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
METADATA_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MAX_METADATA_KEYS = 16
MAX_METADATA_VALUE = 256
KINDS = ("note", "file", "conversation", "observation", "connector")
MAX_QUERY = 1_000
MAX_LIMIT = 50
#: How many candidates each lane contributes before fusion.
LANE_DEPTH = 4
#: Chunk texts per embedding call during batch ingest.
EMBED_BATCH = 64
#: Event kinds an outside process may append (long-running jobs
#: reporting progress). Engine kinds cannot be forged through this path.
EXTERNAL_EVENT_KINDS = ("job",)
ORIGINS = ("stated", "extracted", "inferred")
STATUSES = ("active", "closed", "proposed", "declined")
JOB_STATUSES = ("running", "completed", "failed")
MAX_EXTERNAL_PAYLOAD = 4096


@dataclass
class Profile:
    static_facts: list[Fact] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Record:
    """One thing to remember, for batch ingest and for import."""

    content: str
    kind: str = "note"
    source: Optional[str] = None
    tags: Sequence[str] = ()
    created_at: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Record":
        known = {k: data[k] for k in ("content", "kind", "source", "tags", "created_at", "metadata") if k in data}
        if "content" not in known:
            raise InvalidInput("a record needs content")
        return cls(**known)


@dataclass
class ImportSummary:
    episodes: int = 0
    #: Episodes already present in the target.
    deduplicated: int = 0
    facts: int = 0
    #: Facts already present in the target (same subject, predicate,
    #: object, interval and status).
    facts_skipped: int = 0


@dataclass(frozen=True)
class _Placement:
    covering: list
    restates: Optional[Fact]
    bound: Optional[str]
    bound_reason: Optional[str]


@dataclass(frozen=True)
class _Pending:
    slot: int
    new: NewEpisode
    #: Chunk texts, sliced in code points.
    texts: list[str]
    #: The same spans as UTF-8 byte offsets (spec rule 1.2).
    spans: list[tuple[int, int]]


@dataclass(frozen=True)
class _DupOf:
    """A record identical to an earlier one in the same batch; its id is
    known only after that one is written."""

    slot: int


class MemoryEngine:
    def __init__(
        self,
        documents: DocumentStore,
        vectors: VectorIndex,
        embedder: Embedder,
        chunk_target: int = DEFAULT_TARGET,
        clock: Callable[[], str] = now_rfc3339,
        events: Optional[EventLog] = None,
        record_queries: bool = False,
    ) -> None:
        self.documents = documents
        self.vectors = vectors
        self.embedder = embedder
        self.chunk_target = chunk_target
        self.clock = clock
        #: Evidence sink. None means no evidence is kept, and no metric
        #: can be computed; that absence is reported, never filled in.
        self.events = events
        #: False (the default) stores a sha256 prefix of each query instead
        #: of its text: a memory store is private, which is not consent to
        #: a second log of everything asked of it.
        self.record_queries = record_queries

    async def _emit(self, space: str, kind: str, payload: dict) -> Optional[Event]:
        if self.events is None:
            return None
        return await self.events.append(NewEvent(ts=self.clock(), space=space, kind=kind, payload=payload))

    def _query_for_evidence(self, query: str) -> dict:
        if self.record_queries:
            return {"query": query, "query_hashed": False}
        return {"query": hashlib.sha256(query.encode()).hexdigest()[:16], "query_hashed": True}

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
        metadata: Mapping[str, str] | None = None,
    ) -> Added:
        [added] = await self.remember_many(
            space,
            [Record(content, kind, source, tuple(tags), created_at, dict(metadata or {}))],
        )
        return added

    async def remember_many(self, space: str, records: Iterable[Record]) -> list[Added]:
        """Ingest a batch: one embedding call per EMBED_BATCH chunk texts
        instead of one per record, and one revision bump. Outcomes come
        back in input order; a record identical to an earlier one in the
        same batch is deduplicated against it.

        Nothing is written until every vector exists, so an embedder that
        fails leaves no orphan episodes. If a store fails after the first
        write, the episodes written so far are deleted again and the
        error is raised; a batch either lands whole or not at all.
        """
        check_space(space)
        started = time.perf_counter()
        records = list(records)
        try:
            resolved = await self._remember_many(space, records)
        except Exception as e:
            await self._emit(space, "remember", {
                "records": len(records), "error": f"{type(e).__name__}: {e}",
                "latency_ms": _ms(started),
            })
            raise
        fresh_count = sum(1 for a in resolved if not a.deduplicated)
        await self._emit(space, "remember", {
            "records": len(records),
            "fresh": fresh_count,
            "deduplicated": len(resolved) - fresh_count,
            "chunks": sum(a.chunks for a in resolved),
            "bytes": sum(len(r.content.encode()) for r, a in zip(records, resolved) if not a.deduplicated),
            "embedder": self.embedder.id,
            "latency_ms": _ms(started),
        })
        return resolved

    async def _remember_many(self, space: str, records: Sequence[Record]) -> list[Added]:
        when = self.clock()
        results: list[Optional[Added]] = []
        fresh: list[_Pending] = []
        seen: dict[str, int] = {}  # content hash -> index into results
        for record in records:
            content = record.content
            if not isinstance(content, str) or not content.strip():
                raise InvalidInput("content must not be empty")
            if len(content.encode()) > MAX_CONTENT_BYTES:
                raise InvalidInput(f"content exceeds {MAX_CONTENT_BYTES} bytes")
            kind = record.kind
            if kind not in KINDS:
                raise InvalidInput(f"kind must be one of {KINDS}, got {record.kind!r}")
            clean_tags = normalise_tags(record.tags)
            clean_meta = normalise_metadata(record.metadata or {})
            happened = normalise_time(record.created_at) if record.created_at else when
            digest = content_hash(space, content)
            if digest in seen:
                results.append(Added(episode_id=-1, deduplicated=True, chunks=0))
                fresh_or_dup = seen[digest]
                results[-1] = _DupOf(fresh_or_dup)  # resolved after inserts
                continue
            existing = await self.documents.episode_by_hash(space, digest)
            if existing is not None:
                seen[digest] = len(results)
                results.append(Added(episode_id=existing.episode_id, deduplicated=True, chunks=0))
                continue
            seen[digest] = len(results)
            results.append(None)
            spans = chunk_spans(content, self.chunk_target)
            fresh.append(
                _Pending(
                    slot=len(results) - 1,
                    texts=[content[sp.start : sp.end] for sp in spans],
                    new=NewEpisode(
                        space=space,
                        kind=kind,
                        content=content,
                        content_hash=digest,
                        created_at=happened,
                        ingested_at=when,
                        source=record.source,
                        tags=clean_tags,
                        metadata=clean_meta,
                    ),
                    spans=[(sp.start, sp.end) for sp in byte_spans(content, spans)],
                )
            )

        if fresh:
            texts = [t for p in fresh for t in p.texts]
            vectors: list[list[float]] = []
            for i in range(0, len(texts), EMBED_BATCH):
                vectors.extend(await self.embedder.embed(texts[i : i + EMBED_BATCH]))
            await self._write_batch(space, fresh, vectors, results)
            await self.documents.bump_revision(space)

        resolved: list[Added] = []
        for r in results:
            if isinstance(r, _DupOf):
                target = results[r.slot]
                resolved.append(Added(episode_id=target.episode_id, deduplicated=True, chunks=0))  # type: ignore[union-attr]
            else:
                resolved.append(r)  # type: ignore[arg-type]
        return resolved

    async def _write_batch(
        self, space: str, fresh: list["_Pending"], vectors: list[list[float]], results: list
    ) -> None:
        written: list[int] = []
        try:
            offset = 0
            for pending in fresh:
                episode = await self.documents.insert_episode(pending.new)
                written.append(episode.episode_id)
                chunks = await self.documents.insert_chunks(
                    [
                        NewChunk(
                            episode_id=episode.episode_id,
                            space=space,
                            ordinal=i,
                            start=a,
                            end=b,
                            text=text,
                            created_at=episode.created_at,
                        )
                        for i, ((a, b), text) in enumerate(zip(pending.spans, pending.texts))
                    ]
                )
                await self.vectors.upsert(
                    [
                        VectorPoint(
                            chunk_id=c.chunk_id,
                            space=space,
                            episode_id=episode.episode_id,
                            created_at=episode.created_at,
                            vector=v,
                            tags=pending.new.tags,
                            metadata=pending.new.metadata,
                        )
                        for c, v in zip(chunks, vectors[offset : offset + len(chunks)])
                    ]
                )
                offset += len(chunks)
                results[pending.slot] = Added(episode_id=episode.episode_id, deduplicated=False, chunks=len(chunks))
        except Exception:
            # Undo the partial batch so a retry starts clean.
            for episode_id in written:
                removed = await self.documents.delete_episode(space, episode_id)
                await self.vectors.delete(removed)
            raise

    async def forget(self, space: str, episode_id: int) -> None:
        check_space(space)
        started = time.perf_counter()
        removed = await self.documents.delete_episode(space, episode_id)
        if not removed and await self.documents.get_episode(space, episode_id) is None:
            await self._emit(space, "forget", {"episode_id": episode_id, "error": "NotFound", "latency_ms": _ms(started)})
            raise NotFound(f"episode {episode_id} not found in {space!r}")
        await self.vectors.delete(removed)
        await self.documents.bump_revision(space)
        await self._emit(space, "forget", {"episode_id": episode_id, "chunks_removed": len(removed), "latency_ms": _ms(started)})

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
        where: Mapping[str, str] | None = None,
    ) -> RecallResult:
        check_space(space)
        query = query.strip()
        if not query or len(query) > MAX_QUERY:
            raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
        limit = max(1, min(limit, MAX_LIMIT))
        boundary = normalise_time(as_of) if as_of else None
        clean_tags = normalise_tags(tags)
        clean_where = normalise_metadata(where or {})
        depth = limit * LANE_DEPTH
        degraded: list[str] = []
        started = time.perf_counter()
        latency: dict[str, float] = {}
        evidence = {
            **self._query_for_evidence(query),
            "limit": limit,
            "as_of": boundary,
            "tags": list(clean_tags),
            "where": clean_where,
            "embedder": self.embedder.id,
        }

        vector_lane: list[tuple[int, float]] = []
        try:
            t0 = time.perf_counter()
            [qvec] = await self.embedder.embed([query])
            latency["embed"] = _ms(t0)
            t0 = time.perf_counter()
            vector_lane = await self.vectors.search(space, qvec, depth, boundary, clean_tags, clean_where)
            latency["vector"] = _ms(t0)
        except Exception as e:  # noqa: BLE001 - the lane is reported, not hidden
            degraded.append(f"vectors: {type(e).__name__}: {e}")

        text_lane: list[tuple[int, float]] = []
        try:
            t0 = time.perf_counter()
            text_lane = await self.documents.search_text(
                space, query, depth, TextFilter(as_of=boundary, tags=clean_tags, where=clean_where)
            )
            latency["text"] = _ms(t0)
        except Exception as e:  # noqa: BLE001
            degraded.append(f"text: {type(e).__name__}: {e}")

        if len(degraded) == 2:
            latency["total"] = _ms(started)
            await self._emit(space, "recall", {**evidence, "degraded": degraded, "latency_ms": latency,
                                               "error": "both lanes failed"})
            raise RuntimeError("both recall lanes failed: " + "; ".join(degraded))

        similarity = dict(vector_lane)
        ranks = {
            "vector": {cid: i + 1 for i, (cid, _) in enumerate(vector_lane)},
            "text": {cid: i + 1 for i, (cid, _) in enumerate(text_lane)},
        }
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
                    lanes={lane: r[chunk.chunk_id] for lane, r in ranks.items() if chunk.chunk_id in r},
                    created_at=chunk.created_at,
                    source=episode.source if episode else None,
                    tags=episode.tags if episode else (),
                    metadata=dict(episode.metadata) if episode else {},
                )
            )
        facts = await self._facts_for_query(space, query, boundary or now)
        counts = await self.documents.counts(space)
        result = RecallResult(
            items=result_items,
            facts=facts,
            degraded=degraded,
            returned_bytes=sum(len(i.text.encode()) for i in result_items),
            space_bytes=counts.bytes,
        )
        latency["total"] = _ms(started)
        event = await self._emit(space, "recall", {
            **evidence,
            "latency_ms": latency,
            "degraded": degraded,
            # Only the returned items are recorded; candidates the lanes
            # saw but fusion dropped are not, so coverage is "returned".
            "items": [
                {"chunk_id": i.chunk_id, "episode_id": i.episode_id, "score": i.score,
                 "similarity": i.similarity, "lanes": i.lanes}
                for i in result_items
            ],
            "items_coverage": "returned",
            "lane_candidates": {"vector": len(vector_lane), "text": len(text_lane)},
            "facts": len(facts),
            "returned_bytes": result.returned_bytes,
            "space_bytes": result.space_bytes,
        })
        if event is not None:
            result.event_id = event.event_id
        return result

    async def record(self, space: str, kind: str, payload: Mapping[str, object]) -> Event:
        """Append an event from outside the engine: a job reporting its
        status. Validated so the evidence log cannot be polluted with
        forged engine events or unbounded payloads."""
        check_space(space)
        if self.events is None:
            raise InvalidInput("no event log is attached, so nothing can be recorded")
        if kind not in EXTERNAL_EVENT_KINDS:
            raise InvalidInput(f"kind must be one of {EXTERNAL_EVENT_KINDS}, got {kind!r}")
        import json

        if len(json.dumps(dict(payload))) > MAX_EXTERNAL_PAYLOAD:
            raise InvalidInput(f"payload exceeds {MAX_EXTERNAL_PAYLOAD} bytes")
        job_id, name, status = payload.get("job_id"), payload.get("name"), payload.get("status")
        if not isinstance(job_id, str) or not 1 <= len(job_id) <= 64:
            raise InvalidInput("job_id must be a string of 1..=64 chars")
        if not isinstance(name, str) or not 1 <= len(name) <= 120:
            raise InvalidInput("name must be a string of 1..=120 chars")
        if status not in JOB_STATUSES:
            raise InvalidInput(f"status must be one of {JOB_STATUSES}, got {status!r}")
        progress = payload.get("progress")
        if progress is not None:
            done, total = (progress.get("done"), progress.get("total")) if isinstance(progress, Mapping) else (None, None)
            if not (isinstance(done, int) and isinstance(total, int) and 0 <= done <= total):
                raise InvalidInput("progress must be {done, total} with 0 <= done <= total")
        for field_name in ("detail", "error"):
            value = payload.get(field_name)
            if value is not None and (not isinstance(value, str) or len(value) > 500):
                raise InvalidInput(f"{field_name} must be a string of at most 500 chars")
        return await self._emit(space, kind, dict(payload))  # type: ignore[return-value]

    async def feedback(
        self, space: str, recall_event_id: int, chunk_id: int, useful: bool, note: Optional[str] = None
    ) -> Event:
        """Record a person's judgement of one returned item. The only
        relevance evidence that does not come from a benchmark. Latest
        judgement per (recall, chunk) wins when metrics read these; the
        earlier ones stay as events."""
        check_space(space)
        if self.events is None:
            raise InvalidInput("no event log is attached, so feedback cannot be kept")
        recall = await self.events.get(space, recall_event_id)
        if recall is None or recall.kind != "recall":
            raise NotFound(f"recall event {recall_event_id} not found in {space!r}")
        returned = {int(i["chunk_id"]) for i in recall.payload.get("items", [])}
        if chunk_id not in returned:
            raise InvalidInput(f"chunk {chunk_id} was not returned by recall {recall_event_id}")
        if note is not None and len(note) > 500:
            raise InvalidInput("note must be at most 500 chars")
        return await self._emit(space, "feedback", {
            "recall_event_id": recall_event_id, "chunk_id": chunk_id, "useful": bool(useful), "note": note,
        })  # type: ignore[return-value]

    async def _facts_for_query(self, space: str, query: str, when: str, limit: int = 10) -> list[Fact]:
        terms = set(tokenize(query))
        if not terms:
            return []
        scored = []
        # Closed facts are included on purpose: asked about 2023, the
        # fact that held in 2023 is the answer even if it closed since.
        for fact in await self.documents.list_facts(space, include_closed=True):
            if fact.excluded or not fact.holds_at(when):
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
        origin: str = "stated",
        proposed: bool = False,
    ) -> Fact:
        """Record that ``subject predicate object`` holds from ``valid_from``.

        ``origin`` says who is speaking: a person or trusted program
        (stated), a model reading an episode (extracted), or a derivation
        (inferred). ``proposed`` parks the fact for a person to approve;
        until then it is outside the ledger and answers nothing.

        Every ledger fact with the same subject and predicate takes part,
        whatever its status, so the ledger stays a partition of time:

        - a fact with the same object whose interval covers the start is a
          restatement and is returned unchanged;
        - any fact whose interval covers the start (open or closed) ends at
          the start, with the reason naming what superseded it. A reason a
          person wrote is kept; only the bound moves;
        - the first fact that starts after the new one bounds it: the new
          fact is stored closed at that point. A stale record arriving late
          never overwrites a fresher one.
        """
        check_space(space)
        subject = normalise_term(subject, "subject")
        predicate = normalise_term(predicate, "predicate")
        object = object.strip()
        if not object:
            raise InvalidInput("object must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise InvalidInput("confidence must be within 0..=1")
        if origin not in ORIGINS:
            raise InvalidInput(f"origin must be one of {ORIGINS}, got {origin!r}")
        start = normalise_time(valid_from) if valid_from else self.clock()
        started = time.perf_counter()

        if proposed:
            fact = await self.documents.insert_fact(
                NewFact(
                    space=space, subject=subject, predicate=predicate, object=object, valid_from=start,
                    confidence=confidence, status="proposed", source_episode_id=source_episode_id, origin=origin,
                )
            )
            await self.documents.bump_revision(space)
            await self._emit(space, "fact_assert", {
                "fact_id": fact.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
                "outcome": "proposed", "superseded": [], "source_episode_id": source_episode_id,
                "latency_ms": _ms(started),
            })
            return fact

        placement = await self._place(space, subject, predicate, object, start)
        if placement.restates is not None:
            await self._emit(space, "fact_assert", {
                "fact_id": placement.restates.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
                "outcome": "restated", "superseded": [], "latency_ms": _ms(started),
            })
            return placement.restates
        fact = await self.documents.insert_fact(
            NewFact(
                space=space, subject=subject, predicate=predicate, object=object, valid_from=start,
                valid_until=placement.bound, confidence=confidence,
                status="closed" if placement.bound else "active",
                closed_reason=placement.bound_reason, source_episode_id=source_episode_id, origin=origin,
            )
        )
        await self._truncate(placement.covering, start, fact.fact_id)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_assert", {
            "fact_id": fact.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
            "outcome": "new_closed" if placement.bound else "new_active",
            "superseded": [r.fact_id for r in placement.covering],
            "source_episode_id": source_episode_id,
            "latency_ms": _ms(started),
        })
        return fact

    async def _place(self, space: str, subject: str, predicate: str, object: str, start: str, exclude_id: Optional[int] = None) -> "_Placement":
        """Where a fact starting at ``start`` sits among the ledger facts
        (active or closed) with the same subject and predicate."""
        start_dt = parse_rfc3339(start)
        rivals = [
            r for r in await self.documents.facts_for(space, subject, predicate)
            if r.in_ledger and r.fact_id != exclude_id
        ]
        covering = [r for r in rivals if _covers(r, start_dt)]
        restates = next((r for r in covering if r.object == object), None)
        later = [r for r in rivals if parse_rfc3339(r.valid_from) > start_dt]
        successor = min(later, key=lambda f: (parse_rfc3339(f.valid_from), f.fact_id)) if later else None
        return _Placement(
            covering=[] if restates else covering,
            restates=restates,
            bound=successor.valid_from if successor else None,
            bound_reason=f"superseded by fact {successor.fact_id}" if successor else None,
        )

    async def _truncate(self, covering: list[Fact], start: str, by_fact_id: int) -> None:
        for rival in covering:
            reason = rival.closed_reason
            if reason is None or reason.startswith("superseded by fact "):
                reason = f"superseded by fact {by_fact_id}"
            await self.documents.update_fact(
                rival.model_copy(update={"status": "closed", "valid_until": start, "closed_reason": reason})
            )

    async def approve(self, space: str, fact_id: int) -> Fact:
        """A person accepts a proposed fact: it enters the ledger exactly as
        an assertion at its own valid_from would, truncating what it
        covers and bounded by what starts later. A proposal that restates
        a fact already held is marked declined as a duplicate and the held
        fact is returned."""
        check_space(space)
        fact = await self._proposed(space, fact_id)
        placement = await self._place(space, fact.subject, fact.predicate, fact.object, fact.valid_from, exclude_id=fact.fact_id)
        if placement.restates is not None:
            await self.documents.update_fact(
                fact.model_copy(update={"status": "declined", "closed_reason": f"duplicate of fact {placement.restates.fact_id}"})
            )
            await self.documents.bump_revision(space)
            await self._emit(space, "fact_review", {"fact_id": fact_id, "decision": "duplicate", "of": placement.restates.fact_id})
            return placement.restates
        accepted = fact.model_copy(update={
            "status": "closed" if placement.bound else "active",
            "valid_until": placement.bound,
            "closed_reason": placement.bound_reason,
        })
        await self.documents.update_fact(accepted)
        await self._truncate(placement.covering, fact.valid_from, fact.fact_id)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_review", {
            "fact_id": fact_id, "decision": "approved", "superseded": [r.fact_id for r in placement.covering],
        })
        return accepted

    async def decline(self, space: str, fact_id: int, reason: str) -> Fact:
        """A person rejects a proposed fact. It never held; the reason is kept."""
        check_space(space)
        reason = _reason(reason)
        fact = await self._proposed(space, fact_id)
        declined = fact.model_copy(update={"status": "declined", "closed_reason": reason})
        await self.documents.update_fact(declined)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_review", {"fact_id": fact_id, "decision": "declined"})
        return declined

    async def _proposed(self, space: str, fact_id: int) -> Fact:
        fact = await self.documents.get_fact(space, fact_id)
        if fact is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        if fact.status != "proposed":
            raise InvalidInput(f"fact {fact_id} is {fact.status}, not proposed")
        return fact

    async def exclude(self, space: str, fact_id: int, reason: str) -> Fact:
        """Suppress a ledger fact from recall without touching its interval
        or its history. The third operation beside close (it stopped
        holding) and forget (the data is gone)."""
        check_space(space)
        reason = _reason(reason)
        fact = await self.documents.get_fact(space, fact_id)
        if fact is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        if not fact.in_ledger:
            raise InvalidInput(f"fact {fact_id} is {fact.status}; only ledger facts can be excluded")
        excluded = fact.model_copy(update={"excluded_reason": reason})
        await self.documents.update_fact(excluded)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_exclude", {"fact_id": fact_id, "action": "exclude"})
        return excluded

    async def include(self, space: str, fact_id: int) -> Fact:
        """Undo exclude."""
        check_space(space)
        fact = await self.documents.get_fact(space, fact_id)
        if fact is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        if not fact.excluded:
            return fact
        included = fact.model_copy(update={"excluded_reason": None})
        await self.documents.update_fact(included)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_exclude", {"fact_id": fact_id, "action": "include"})
        return included

    async def close_fact(self, space: str, fact_id: int, reason: str) -> Fact:
        check_space(space)
        reason = _reason(reason)
        fact = await self.documents.get_fact(space, fact_id)
        if fact is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        if fact.status == "closed":
            return fact
        if not fact.in_ledger:
            raise InvalidInput(f"fact {fact_id} is {fact.status}; decline a proposal instead of closing it")
        closed = fact.model_copy(
            update={"status": "closed", "valid_until": self.clock(), "closed_reason": reason}
        )
        await self.documents.update_fact(closed)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_close", {"fact_id": fact_id, "reason_kind": "manual"})
        return closed

    async def facts(
        self,
        space: str,
        include_closed: bool = False,
        as_of: Optional[str] = None,
        status: Optional[str] = None,
        include_excluded: bool = False,
    ) -> list[Fact]:
        """Ledger facts: active by default, closed too with include_closed,
        those holding at ``as_of`` when given. ``status`` selects one
        status instead (``proposed`` lists what awaits review). Excluded
        facts are left out unless asked for."""
        check_space(space)
        if status is not None and status not in STATUSES:
            raise InvalidInput(f"status must be one of {STATUSES}, got {status!r}")
        found = await self.documents.list_facts(space, include_closed=True)
        if status is not None:
            found = [f for f in found if f.status == status]
        elif as_of is not None:
            boundary = normalise_time(as_of)
            found = [f for f in found if f.holds_at(boundary)]
        else:
            allowed = ("active", "closed") if include_closed else ("active",)
            found = [f for f in found if f.status in allowed]
        if not include_excluded:
            found = [f for f in found if not f.excluded]
        return sorted(found, key=lambda f: f.fact_id)

    # -- overviews --------------------------------------------------------

    async def profile(self, space: str, limit: int = 10) -> Profile:
        check_space(space)
        limit = max(1, min(limit, 50))
        active = [f for f in await self.documents.list_facts(space, include_closed=False) if f.status == "active" and not f.excluded]
        active.sort(key=lambda f: (-f.confidence, f.fact_id))
        recent = await self.documents.recent_episodes(space, limit)
        return Profile(
            static_facts=active[:limit],
            dynamic=[e.content[:200] for e in recent],
        )

    async def tags(self, space: str) -> dict[str, int]:
        check_space(space)
        return dict(sorted((await self.documents.counts(space)).tags.items()))

    async def scopes(self, space: str) -> dict[str, dict[str, int]]:
        """Episode counts per metadata key and value: which users, agents
        and sessions have memory here. Walks the episodes, which is fine
        for an overview and avoids asking every store for a new query."""
        check_space(space)
        counts = await self.documents.counts(space)
        out: dict[str, dict[str, int]] = {}
        for episode in await self.documents.recent_episodes(space, max(counts.episodes, 1)):
            for key, value in episode.metadata.items():
                out.setdefault(key, {})
                out[key][value] = out[key].get(value, 0) + 1
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    async def status(self, space: str) -> Status:
        check_space(space)
        counts = await self.documents.counts(space)
        pending = sum(1 for f in await self.documents.list_facts(space, include_closed=True) if f.status == "proposed")
        return Status(
            space=space,
            episodes=counts.episodes,
            chunks=counts.chunks,
            bytes=counts.bytes,
            pending_review=pending,
            revision=await self.documents.revision(space),
            embedder=self.embedder.id,
            document_store=self.documents.name,
            vector_index=self.vectors.name,
        )

    # -- portability ------------------------------------------------------

    async def export(self, space: str) -> AsyncIterator[dict]:
        """Every episode and every fact in a space as plain dicts, the
        shape `import_records` accepts. Chunks and vectors are derived
        and are rebuilt on import, so a dump moves between stores and
        between embedders."""
        check_space(space)
        counts = await self.documents.counts(space)
        for episode in await self.documents.recent_episodes(space, max(counts.episodes, 1)):
            yield {
                "type": "episode",
                "episode_id": episode.episode_id,
                "kind": episode.kind,
                "content": episode.content,
                "source": episode.source,
                "tags": list(episode.tags),
                "metadata": dict(episode.metadata),
                "created_at": episode.created_at,
            }
        for fact in await self.documents.list_facts(space, include_closed=True):
            yield {"type": "fact", **fact.model_dump(exclude={"space"})}

    async def import_records(self, space: str, records: Iterable[Mapping]) -> ImportSummary:
        """Load an export. Episodes go through the normal ingest, so they
        are re-chunked, re-embedded and deduplicated; facts are stored as
        they were, closed ones included, because the ledger's history is
        part of what is being moved."""
        check_space(space)
        summary = ImportSummary()
        episodes: list[Record] = []
        source_ids: list[Optional[int]] = []
        facts: list[Mapping] = []
        for record in records:
            kind = record.get("type", "episode")
            if kind == "episode":
                episodes.append(Record.from_dict(record))
                source_ids.append(record.get("episode_id"))
            elif kind == "fact":
                facts.append(record)
            else:
                raise InvalidInput(f"unknown record type {kind!r}")
        # Ids are store-local (spec 3.1). Provenance in the dump names the
        # source store's episodes, so it is remapped through the ids this
        # import produced; a reference to an episode not in the dump is
        # dropped rather than pointed at an unrelated record.
        id_map: dict[int, int] = {}
        for old_id, added in zip(source_ids, await self.remember_many(space, episodes)):
            summary.episodes += 0 if added.deduplicated else 1
            summary.deduplicated += 1 if added.deduplicated else 0
            if old_id is not None:
                id_map[int(old_id)] = added.episode_id
        existing = {_fact_identity(f) for f in await self.documents.list_facts(space, include_closed=True)}
        for f in facts:
            source = f.get("source_episode_id")
            new = NewFact(
                space=space,
                subject=normalise_term(str(f["subject"]), "subject"),
                predicate=normalise_term(str(f["predicate"]), "predicate"),
                object=str(f["object"]),
                valid_from=normalise_time(str(f["valid_from"])),
                confidence=float(f.get("confidence", 1.0)),
                valid_until=normalise_time(str(f["valid_until"])) if f.get("valid_until") else None,
                status=str(f.get("status", "active")),
                closed_reason=f.get("closed_reason"),
                source_episode_id=id_map.get(int(source)) if source is not None else None,
                origin=str(f.get("origin", "stated")),
                excluded_reason=f.get("excluded_reason"),
            )
            if new.status not in STATUSES or new.origin not in ORIGINS:
                raise InvalidInput(f"fact record has status {new.status!r} and origin {new.origin!r}")
            identity = (
                new.subject, new.predicate, new.object, new.valid_from, new.valid_until,
                new.status, new.closed_reason, new.confidence,
            )
            if identity in existing:
                summary.facts_skipped += 1
                continue
            existing.add(identity)
            await self.documents.insert_fact(new)
            summary.facts += 1
        if summary.facts:
            await self.documents.bump_revision(space)
        return summary


# -- validation helpers -----------------------------------------------------


def _reason(reason: str) -> str:
    reason = reason.strip()
    if not reason or len(reason) > 500:
        raise InvalidInput("reason must be 1..=500 chars")
    return reason


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


def _fact_identity(fact: Fact) -> tuple:
    """Two facts are the same record only when every stored field agrees;
    a different reason or confidence is a different record."""
    return (
        fact.subject, fact.predicate, fact.object, fact.valid_from, fact.valid_until,
        fact.status, fact.closed_reason, fact.confidence,
    )


def _covers(fact: Fact, instant) -> bool:
    """True when ``instant`` lies in ``[valid_from, valid_until)``."""
    if parse_rfc3339(fact.valid_from) > instant:
        return False
    return fact.valid_until is None or parse_rfc3339(fact.valid_until) > instant


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


def normalise_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    if len(metadata) > MAX_METADATA_KEYS:
        raise InvalidInput(f"at most {MAX_METADATA_KEYS} metadata keys")
    clean: dict[str, str] = {}
    for key, value in metadata.items():
        if not METADATA_KEY.match(key or ""):
            raise InvalidInput(f"metadata key must match [a-z][a-z0-9_]{{0,31}}, got {key!r}")
        if not isinstance(value, str) or not value or len(value) > MAX_METADATA_VALUE:
            raise InvalidInput(f"metadata value for {key!r} must be 1..={MAX_METADATA_VALUE} chars")
        clean[key] = value
    return clean


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
