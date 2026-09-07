"""The engine: remember, recall, forget, and the fact ledger.

Built entirely against the protocols in ``ports``; nothing here knows
whether a dict, MongoDB, or Qdrant is underneath. A lane that fails
during recall is named in ``degraded`` and the other lane still
answers, because a thin answer that says it is thin beats a 500.
"""

from __future__ import annotations

import dataclasses

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterable, Mapping, Optional, Sequence

from ..retrieval import fusion
from ..ingestion.chunker import DEFAULT_TARGET, byte_spans, chunk_spans
from ..core.errors import Conflict, InvalidInput, NotFound
from ..retrieval.lexical import tokenize
from ..backends.blobs import BlobStore, InMemoryBlobStore
from ..core.models import (
    MAX_CONTENT_BYTES,
    Added,
    Attachment,
    BatchDecision,
    DecisionOutcome,
    Episode,
    EpisodeKind,
    Fact,
    FactLink,
    ForgetReceipt,
    DEPENDENCY_KINDS,
    LINK_KINDS,
    RecallItem,
    RecallResult,
    Status,
)
from ..capture.redact import SECRET_PATTERNS, redact_secrets  # noqa: F401 - re-exported for callers
from ..core.ports import (
    DocumentStore,
    DuplicateEvent,
    Embedder,
    Event,
    EventLog,
    NewChunk,
    NewEpisode,
    NewEvent,
    NewFact,
    NewFactLink,
    SourcePage,
    TextFilter,
    VectorIndex,
    VectorPoint,
)
from ..core.timeutil import format_rfc3339, now_rfc3339, parse_rfc3339

SPACE_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
METADATA_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MAX_METADATA_KEYS = 16
MAX_METADATA_VALUE = 256
KINDS = ("note", "file", "conversation", "observation", "connector")
MAX_QUERY = 1_000
MAX_LIMIT = 50
MAX_SOURCE = 1_000
#: How many candidates each lane contributes before fusion.
LANE_DEPTH = 4
#: Chunk texts per embedding call during batch ingest.
EMBED_BATCH = 64


def contextual_prefix(episode: "NewEpisode") -> str:
    """The context a chunk loses when cut from its episode: when it
    happened, where it came from, and whose it is. Prepended to the text
    that is embedded (research experiment 8, Anthropic's contextual
    retrieval without the LLM), never to the text that is stored, so
    invariant I1 holds and recall still returns exact excerpts. Only
    fields that exist are named; an empty prefix means no change."""
    parts = []
    if episode.created_at:
        parts.append(episode.created_at[:10])
    if episode.source:
        parts.append(episode.source)
    for key in ("user_id", "agent_id", "session_id"):
        value = episode.metadata.get(key) if episode.metadata else None
        if value:
            parts.append(f"{key.replace('_', ' ')} {value}")
    return " | ".join(parts)
#: Event kinds an outside process may append (long-running jobs
#: reporting progress). Engine kinds cannot be forged through this path.
EXTERNAL_EVENT_KINDS = ("job", "agent")
ORIGINS = ("stated", "extracted", "inferred")
STATUSES = ("active", "closed", "proposed", "declined")
JOB_STATUSES = ("running", "completed", "failed")
#: What an attachment may be. Anything not here is refused rather than
#: stored under a type the server would have to guess at on the way out.
ATTACHMENT_TYPES = (
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml",
    "application/pdf", "application/json", "text/plain", "text/markdown", "text/csv",
    "audio/mpeg", "audio/wav", "audio/webm", "video/mp4", "video/webm",
)
#: Bytes one attachment may carry. Evidence, not a file share.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

#: What one reviewed batch may do to the facts it names.
DECISIONS = ("approve", "decline", "exclude")
#: Ids one batch may carry. A review queue, not a migration.
MAX_DECISIONS = 500
MAX_EXTERNAL_PAYLOAD = 4096
AGENTS = ("claude-code", "codex", "other")
AGENT_EVENTS = ("session_start", "prompt", "response", "tool_use", "tool_result", "stop", "session_end")
MAX_AGENT_TEXT = 65_536


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
    #: Identity for deduplication. By default two records with the same
    #: text in a space are one episode. A transcript is different: the
    #: second "ok" in a conversation is a second turn, so a chat adapter
    #: keys each turn ("<session>#<n>") and identical text under different
    #: keys is stored twice, while a retried write of the same key is not.
    dedup_key: Optional[str] = None
    #: The identity a dump carries. Import passes it through so a moved
    #: store deduplicates exactly as its source did; callers leave it None.
    content_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping) -> "Record":
        known = {k: data[k] for k in ("content", "kind", "source", "tags", "created_at", "metadata", "dedup_key", "content_hash") if k in data}
        if "content" not in known:
            raise InvalidInput("a record needs content")
        return cls(**known)


@dataclass
class RecoveryReport:
    """What recover() found: episodes brought to a complete state, of
    which how many needed their chunks rebuilt, and marks with no
    episode behind them (the write never landed)."""

    completed: int = 0
    rechunked: int = 0
    forgotten: int = 0


@dataclass
class ImportSummary:
    episodes: int = 0
    #: Episodes already present in the target.
    deduplicated: int = 0
    facts: int = 0
    #: Relations stored, and those dropped or already present.
    links: int = 0
    links_skipped: int = 0
    #: Facts already present in the target (same subject, predicate,
    #: object, interval and status).
    facts_skipped: int = 0


@dataclass(frozen=True)
class _Placement:
    covering: list
    restates: Optional[Fact]
    bound: Optional[str]
    bound_reason: Optional[str]
    bound_by: Optional[int]


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
        contextual_embeddings: bool = False,
        similarity_floor: Optional[float] = None,
        demote_restated: bool = False,
        blobs: Optional[BlobStore] = None,
    ) -> None:
        if similarity_floor is not None and not -1.0 <= similarity_floor <= 1.0:
            raise InvalidInput("similarity_floor must be a cosine similarity in [-1, 1]")
        self.documents = documents
        self.vectors = vectors
        self.embedder = embedder
        #: Where an attachment's bytes live. In memory unless a store is
        #: given, so an engine with no configured blob directory keeps
        #: nothing across a restart rather than writing somewhere unasked.
        self.blobs = blobs if blobs is not None else InMemoryBlobStore()
        self._closed = False
        self.max_attachment_bytes = MAX_ATTACHMENT_BYTES
        self.chunk_target = chunk_target
        self.clock = clock
        #: Evidence sink. None means no evidence is kept, and no metric
        #: can be computed; that absence is reported, never filled in.
        self.events = events
        #: False (the default) stores a sha256 prefix of each query instead
        #: of its text: a memory store is private, which is not consent to
        #: a second log of everything asked of it.
        self.record_queries = record_queries
        #: Experiment 8: embed "<date> | <source> | <scopes>\n<chunk>" while
        #: storing the raw chunk. Off until the bench shows a gain; an
        #: engine's setting is recorded on every recall event so a number is
        #: never quoted without it.
        self.contextual_embeddings = contextual_embeddings
        #: Order a restated claim ahead of what it replaces (fusion.
        #: demote_restated). Off until the effect on ordinary retrieval is
        #: measured on E21's slice; see memory/EXPERIMENTS.md E34.
        self.demote_restated = demote_restated
        #: Experiment 9: with a floor, a recall whose best vector hit sits
        #: below it is flagged low_confidence so a reader can abstain
        #: instead of answering from weak evidence. None (the default) means
        #: no judgement; the floor is chosen from a measured sweep, never
        #: guessed here.
        self.similarity_floor = similarity_floor

    async def _emit(self, space: str, kind: str, payload: dict, dedup_key: Optional[str] = None) -> Optional[Event]:
        if self.events is None:
            return None
        return await self.events.append(NewEvent(ts=self.clock(), space=space, kind=kind, payload=payload, dedup_key=dedup_key))

    def _query_for_evidence(self, query: str) -> dict:
        if self.record_queries:
            return {"query": query, "query_hashed": False}
        return {"query": hashlib.sha256(query.encode()).hexdigest()[:16], "query_hashed": True}

    async def open(self) -> "MemoryEngine":
        await self.vectors.ensure(self.embedder.dim)
        await self.recover()
        return self

    async def close(self) -> None:
        """Release every store this engine holds.

        Only the engine knows which stores it holds, so closing is its
        job, not the caller's: the MCP server used to close two of the
        four by hand, and a fixture that tried ``await memory.close()``
        found nothing to call. A store with nothing to release has no
        close() and is skipped; requiring every backend to grow a no-op
        would widen the wrong side of the contract. Every store is asked
        even when one fails, and the first failure is raised afterwards,
        so a client that cannot disconnect does not leave a file open
        beside it. Closing twice is harmless.
        """
        if self._closed:
            return
        self._closed = True
        first: Optional[BaseException] = None
        for store in (self.documents, self.vectors, self.events, self.blobs):
            closer = getattr(store, "close", None)
            if store is None or not callable(closer):
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - every store must still be asked
                first = first or exc
        if first is not None:
            raise first

    async def recover(self) -> RecoveryReport:
        """Finish or forget what a crash interrupted. A remember marks its
        episode identity in the document store before writing and clears
        the mark once the rows and the vectors are all durable. Anything
        still marked here was cut off somewhere in between: an episode row
        without chunks, or chunks without vectors, or no row at all. Each
        is brought to the complete state (chunks and vectors rebuilt from
        the stored content, which never changed) or, if nothing landed,
        the mark is dropped. Recorded as one "recover" event per open when
        there was anything to do, so the evidence shows it happened."""
        report = RecoveryReport()
        marks = await self.documents.inflight()
        for space, digest in marks:
            episode = await self.documents.episode_by_hash(space, digest)
            if episode is None:
                report.forgotten += 1
            else:
                chunks = await self.documents.chunks_of(space, episode.episode_id)
                if not chunks:
                    spans = chunk_spans(episode.content, self.chunk_target)
                    chunks = await self.documents.insert_chunks(
                        [
                            NewChunk(
                                episode_id=episode.episode_id,
                                space=space,
                                ordinal=i,
                                start=bs.start,
                                end=bs.end,
                                text=episode.content[sp.start : sp.end],
                                created_at=episode.created_at,
                            )
                            for i, (sp, bs) in enumerate(zip(spans, byte_spans(episode.content, spans)))
                        ]
                    )
                    report.rechunked += 1
                as_new = NewEpisode(
                    space=space, kind=episode.kind, content=episode.content, content_hash=episode.content_hash,
                    created_at=episode.created_at, ingested_at=episode.ingested_at, source=episode.source,
                    tags=episode.tags, metadata=episode.metadata,
                )
                texts = [self._embed_text(as_new, c.text) for c in chunks]
                vectors: list[list[float]] = []
                for i in range(0, len(texts), EMBED_BATCH):
                    vectors.extend(await self.embedder.embed(texts[i : i + EMBED_BATCH]))
                await self.vectors.upsert(
                    [
                        VectorPoint(
                            chunk_id=c.chunk_id, space=space, episode_id=episode.episode_id, created_at=episode.created_at,
                            vector=v, tags=episode.tags, metadata=episode.metadata,
                        )
                        for c, v in zip(chunks, vectors)
                    ]
                )
                report.completed += 1
            await self.documents.clear_inflight(space, digest)
        if marks:
            for space in sorted({s for s, _ in marks}):
                await self._emit(space, "recover", {
                    "marks": sum(1 for s, _ in marks if s == space),
                    "completed": report.completed, "rechunked": report.rechunked, "forgotten": report.forgotten,
                })
        return report

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
        attachment_ids: Sequence[str] = (),
    ) -> Added:
        [added] = await self.remember_many(
            space,
            [Record(content, kind, source, tuple(tags), created_at, dict(metadata or {}))],
        )
        for attachment_id in dict.fromkeys(attachment_ids):
            await self.blobs.link(space, attachment_id, added.episode_id)
        return added

    # -- attachments ------------------------------------------------------

    async def attach(
        self, space: str, data: bytes, media_type: str, filename: Optional[str] = None
    ) -> Attachment:
        """Store bytes an episode will carry, addressed by their SHA-256.
        The same bytes stored twice are one attachment: the digest is the
        id, so a second store is a second reference."""
        check_space(space)
        if not data:
            raise InvalidInput("an attachment needs bytes")
        if len(data) > self.max_attachment_bytes:
            raise InvalidInput(
                f"an attachment takes at most {self.max_attachment_bytes} bytes, got {len(data)}"
            )
        if media_type not in ATTACHMENT_TYPES:
            raise InvalidInput(f"media_type must be one of {ATTACHMENT_TYPES}, got {media_type!r}")
        return await self.blobs.put(space, data, media_type, filename)

    async def attachment(self, space: str, attachment_id: str) -> tuple[Attachment, bytes]:
        """The stored bytes, for the space that stored them. A key for
        another space gets the same answer as an id that never existed."""
        check_space(space)
        return await self.blobs.get(space, attachment_id)

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
            digest = record.content_hash or content_hash(space, content, record.dedup_key)
            if not digest or len(digest) > 128:
                raise InvalidInput("content_hash must be 1..=128 chars")
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
            texts = [self._embed_text(p.new, t) for p in fresh for t in p.texts]
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

    def _embed_text(self, episode: NewEpisode, chunk_text: str) -> str:
        """What the embedder sees for a chunk. Stored text is never changed."""
        if not self.contextual_embeddings:
            return chunk_text
        prefix = contextual_prefix(episode)
        return f"{prefix}\n{chunk_text}" if prefix else chunk_text

    async def _write_batch(
        self, space: str, fresh: list["_Pending"], vectors: list[list[float]], results: list
    ) -> None:
        written: list[int] = []
        try:
            offset = 0
            for pending in fresh:
                # The mark outlives a crash; clear_inflight below is the
                # last thing this episode's write does (see recover()).
                await self.documents.mark_inflight(space, pending.new.content_hash)
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
                await self.documents.clear_inflight(space, pending.new.content_hash)
                results[pending.slot] = Added(episode_id=episode.episode_id, deduplicated=False, chunks=len(chunks))
        except Exception:
            # Undo the partial batch so a retry starts clean.
            for episode_id in written:
                removed = await self.documents.delete_episode(space, episode_id)
                await self.vectors.delete(removed)
            for pending in fresh:
                await self.documents.clear_inflight(space, pending.new.content_hash)
            raise

    async def impact(self, space: str, episode_id: int) -> ForgetReceipt:
        """What forgetting the episode would take with it and leave, with
        nothing removed. The claims and links that cite it are reported,
        not closed: a source being gone is a fact about the evidence."""
        check_space(space)
        if await self.documents.get_episode(space, episode_id) is None:
            raise NotFound(f"episode {episode_id} not found in {space!r}")
        carried = [a.attachment_id for a in await self.blobs.for_episode(space, episode_id)]
        released = await self.blobs.released_by(space, episode_id)
        facts = await self.documents.list_facts(space, include_closed=True)
        citing_links: dict[int, int] = {}
        for fact in facts:
            for link in await self.documents.fact_links(space, fact.fact_id):
                if link.source_episode_id == episode_id:
                    citing_links[link.link_id] = link.link_id
        return ForgetReceipt(
            episode_id=episode_id,
            chunks=len(await self.documents.chunks_of(space, episode_id)),
            attachments_released=released,
            attachments_kept=[a for a in carried if a not in released],
            facts_citing=sorted(f.fact_id for f in facts if f.source_episode_id == episode_id),
            links_citing=sorted(citing_links),
        )

    async def forget(self, space: str, episode_id: int) -> ForgetReceipt:
        """Remove the episode, its chunks and vectors, and release the
        attachments nothing else carries; return the receipt that
        ``impact`` would have shown. Claims and links stand."""
        check_space(space)
        started = time.perf_counter()
        try:
            receipt = await self.impact(space, episode_id)
        except NotFound:
            await self._emit(space, "forget", {"episode_id": episode_id, "error": "NotFound", "latency_ms": _ms(started)})
            raise
        removed = await self.documents.delete_episode(space, episode_id)
        await self.vectors.delete(removed)
        await self.blobs.unlink(space, episode_id)
        await self.documents.bump_revision(space)
        await self._emit(space, "forget", {
            "episode_id": episode_id, "chunks_removed": len(removed),
            "attachments_released": len(receipt.attachments_released),
            "facts_citing": len(receipt.facts_citing), "links_citing": len(receipt.links_citing),
            "latency_ms": _ms(started),
        })
        return receipt

    async def episode(self, space: str, episode_id: int) -> Episode:
        check_space(space)
        found = await self.documents.get_episode(space, episode_id)
        if found is None:
            raise NotFound(f"episode {episode_id} not found in {space!r}")
        carried = await self.blobs.for_episode(space, episode_id)
        return found.model_copy(update={"attachments": tuple(carried)}) if carried else found

    # -- recall -----------------------------------------------------------

    async def recall(
        self,
        space: str,
        query: str,
        limit: int = 5,
        as_of: Optional[str] = None,
        tags: Sequence[str] = (),
        where: Mapping[str, str] | None = None,
        history: bool = False,
        kind: Optional[str] = None,
        source_prefix: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> RecallResult:
        """``history`` (research experiment 3) also returns, for every
        subject and predicate among the matched facts, the closed facts that
        held before: what changed, when, and why. Off by default; the
        reader gets only what holds at ``as_of`` unless asked for the chain.

        ``kind``, ``source_prefix``, ``since`` and ``until`` narrow the
        candidates fusion produced by the episode's kind, the start of its
        source (literal text), and inclusive bounds on when it happened,
        the same way the Rust engine narrows: a filter that matches nothing
        in the candidate window yields nothing rather than reaching past
        it. ``tags`` and ``where`` are different: both lanes apply them
        before ranking. ``as_of`` stays what it was, fact validity and the
        upper bound the lanes already apply."""
        check_space(space)
        query = query.strip()
        if not query or len(query) > MAX_QUERY:
            raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
        limit = max(1, min(limit, MAX_LIMIT))
        boundary = normalise_time(as_of) if as_of else None
        clean_tags = normalise_tags(tags)
        clean_where = normalise_metadata(where or {})
        if kind is not None and kind not in KINDS:
            raise InvalidInput(f"kind must be one of {KINDS}, got {kind!r}")
        if source_prefix is not None and len(source_prefix) > MAX_SOURCE:
            raise InvalidInput(f"source_prefix must be at most {MAX_SOURCE} chars")
        since_at = normalise_time(since) if since else None
        until_at = normalise_time(until) if until else None
        narrowing = kind is not None or source_prefix is not None or since_at is not None or until_at is not None
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
            "contextual_embeddings": self.contextual_embeddings,
            "similarity_floor": self.similarity_floor,
            "narrow": {"kind": kind, "source_prefix": source_prefix, "since": since_at, "until": until_at},
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

        # An index that ranks without a cosine (a bridged store with an
        # unknown score) reports NaN: its order counts for fusion, but no
        # similarity is shown and no confidence is judged from it.
        similarity = {cid: (None if math.isnan(score) else score) for cid, score in vector_lane}
        # The best cosine the vector lane saw, before fusion, is the
        # confidence signal. A degraded vector lane cannot judge (None); a
        # lane that ran and found nothing is as weak as evidence gets.
        top_similarity = round(vector_lane[0][1], 6) if vector_lane and not math.isnan(vector_lane[0][1]) else None
        vector_ran = not any(d.startswith("vectors:") for d in degraded)
        low_confidence: Optional[bool] = None
        if self.similarity_floor is not None and vector_ran and (top_similarity is not None or not vector_lane):
            low_confidence = top_similarity is None or top_similarity < self.similarity_floor
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
        episodes: dict[int, Episode] = {}
        if narrowing:
            # Read every candidate's episode and keep the ones that fit,
            # before the limit is applied; the window is the lanes' depth.
            for item in items:
                eid = chunks[item.chunk_id].episode_id
                if eid not in episodes:
                    found = await self.documents.get_episode(space, eid)
                    if found is not None:
                        episodes[eid] = found
            items = [
                item for item in items
                if (episode := episodes.get(chunks[item.chunk_id].episode_id)) is not None
                and _fits(episode, kind, source_prefix, since_at, until_at)
            ]
        items = items[:limit]
        if self.demote_restated:
            items = fusion.demote_restated(
                items,
                {cid: c.text for cid, c in chunks.items()},
                {cid: c.created_at for cid, c in chunks.items()},
            )
        items = fusion.normalise(items)

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
        previous = await self._history_for(space, facts, boundary or now) if history else []
        counts = await self.documents.counts(space)
        result = RecallResult(
            items=result_items,
            facts=facts,
            history=previous,
            degraded=degraded,
            top_similarity=top_similarity,
            low_confidence=low_confidence,
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
            "top_similarity": top_similarity,
            "low_confidence": low_confidence,
            "facts": len(facts),
            "fact_ids": [f.fact_id for f in facts],
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
        if kind == "agent":
            return await self._record_agent(space, payload)
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

    async def _record_agent(self, space: str, payload: Mapping[str, object]) -> Event:
        p = dict(payload)
        if p.get("agent") not in AGENTS:
            raise InvalidInput(f"agent must be one of {AGENTS}")
        sid = p.get("session_id")
        if not isinstance(sid, str) or not 1 <= len(sid) <= 128:
            raise InvalidInput("session_id must be a string of 1..=128 chars")
        if p.get("event") not in AGENT_EVENTS:
            raise InvalidInput(f"event must be one of {AGENT_EVENTS}")
        for key, limit in (("project", 120), ("tool_name", 120), ("tool_use_id", 128), ("model", 120)):
            value = p.get(key)
            if value is not None and (not isinstance(value, str) or len(value) > limit):
                raise InvalidInput(f"{key} must be a string of at most {limit} chars")
        text = p.get("text")
        if text is not None:
            if not isinstance(text, str):
                raise InvalidInput("text must be a string")
            if len(text.encode()) > MAX_AGENT_TEXT:
                raise InvalidInput(f"text exceeds {MAX_AGENT_TEXT} bytes")
            p["text"] = redact_secrets(text)
        if p.get("episode_id") is not None and not isinstance(p["episode_id"], int):
            raise InvalidInput("episode_id must be an integer")
        if p.get("ok") is not None and not isinstance(p["ok"], bool):
            raise InvalidInput("ok must be a boolean")
        if p.get("duration_ms") is not None and not isinstance(p["duration_ms"], (int, float)):
            raise InvalidInput("duration_ms must be a number")
        source_event_id = p.get("source_event_id")
        if source_event_id is not None and (not isinstance(source_event_id, str) or not 1 <= len(source_event_id) <= 128):
            raise InvalidInput("source_event_id must be a string of 1..=128 chars")
        allowed = {"agent", "session_id", "project", "event", "text", "tool_name", "tool_use_id", "ok", "duration_ms",
                   "episode_id", "model", "source_event_id"}
        unknown = set(p) - allowed
        if unknown:
            raise InvalidInput(f"unknown agent fields: {sorted(unknown)}")
        if p.get("episode_id") is not None and await self.documents.get_episode(space, int(p["episode_id"])) is None:
            # A link is connector-reported provenance; it must at least point inside this space.
            raise InvalidInput(f"episode {p['episode_id']} is not in {space!r}")
        # Idempotent receipt is the sink's job: the key is unique per space,
        # a retry with the same payload returns the stored event, and the
        # same key with a different payload is a conflict, never a silent drop.
        key = f"agent:{p['agent']}:{sid}:{source_event_id}" if source_event_id is not None else None
        try:
            return await self._emit(space, "agent", p, dedup_key=key)  # type: ignore[return-value]
        except DuplicateEvent as e:
            raise InvalidInput(f"source_event_id {source_event_id!r} was already recorded with a different payload (event {e.existing.event_id})") from e

    async def graph(
        self,
        space: str,
        session_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        since: Optional[str] = None,
        limit: int = 400,
    ):
        """The recorded relations around a session or an episode (or the
        latest activity when neither is given). See ``graph.py``."""
        from ..retrieval import graph as G

        check_space(space)
        limit = max(1, min(limit, 2000))
        g = G.Graph()
        if self.events is None:
            return g
        focused = session_id is not None or episode_id is not None
        # A focused graph must reach the events that mention its subject even
        # when they are older than the window, and must leave out agent
        # events that merely happened nearby. A session focus keeps that
        # session's events; an episode focus keeps the agent events linked to
        # that episode. Unfocused: the newest window, as is.
        window = await self.events.query(space, since=since, limit=limit)
        g.truncated = len(window) >= limit
        if focused:
            scan = await self.events.query(space, kind="agent", since=since, limit=2000)
            g.truncated = g.truncated or len(scan) >= 2000
            if session_id is not None:
                agent_events = [e for e in scan if e.payload.get("session_id") == session_id]
            else:
                agent_events = [e for e in scan if e.payload.get("episode_id") == episode_id]
        else:
            agent_events = [e for e in window if e.kind == "agent"]
        recall_events = [e for e in window if e.kind == "recall" and "error" not in e.payload]
        feedback_events = [e for e in window if e.kind == "feedback"]

        episode_ids: set[int] = set()
        if episode_id is not None:
            episode_ids.add(episode_id)
        for e in agent_events:
            if e.payload.get("episode_id") is not None:
                episode_ids.add(int(e.payload["episode_id"]))
        chunk_ids: set[int] = set()
        for e in recall_events:
            for item in e.payload.get("items") or []:
                chunk_ids.add(int(item["chunk_id"]))
        chunks = await self.documents.get_chunks(space, sorted(chunk_ids)) if chunk_ids else []
        if focused:
            keep = {c.chunk_id for c in chunks if c.episode_id in episode_ids}
            recall_events = [e for e in recall_events if any(int(i["chunk_id"]) in keep for i in e.payload.get("items") or [])]
            chunks = [c for c in chunks if c.chunk_id in keep]
        for c in chunks:
            episode_ids.add(c.episode_id)

        for eid in sorted(episode_ids):
            ep = await self.documents.get_episode(space, eid)
            if ep is None:
                continue
            g.add(G.Node(f"episode:{eid}", "episode", ep.content[:80], ep.created_at,
                         {"kind": ep.kind, "source": ep.source, "tags": list(ep.tags), "metadata": dict(ep.metadata), "content": ep.content}))
        for c in chunks:
            if f"episode:{c.episode_id}" in g.nodes:
                g.add(G.Node(f"chunk:{c.chunk_id}", "chunk", c.text[:80], c.created_at, {"ordinal": c.ordinal, "start": c.start, "end": c.end, "text": c.text}))
                g.link(f"episode:{c.episode_id}", f"chunk:{c.chunk_id}", "chunked_into")
        facts = await self.documents.list_facts(space, include_closed=True)
        wanted = [f for f in facts if (f.source_episode_id in episode_ids) or not focused]
        for f in wanted:
            g.add(G.Node(f"claim:{f.fact_id}", "claim", f"{f.subject} {f.predicate.replace('_', ' ')} {f.object}", f.valid_from, {
                "status": f.status, "origin": f.origin, "confidence": f.confidence, "valid_from": f.valid_from,
                "valid_until": f.valid_until, "closed_reason": f.closed_reason, "excluded_reason": f.excluded_reason,
                "source_episode_id": f.source_episode_id, "superseded_by": f.superseded_by,
            }))
        # The typed relations among the claims drawn, each once.
        drawn = set()
        for f in wanted:
            for link in await self.documents.fact_links(space, f.fact_id):
                ends = (f"claim:{link.from_fact}", f"claim:{link.to_fact}")
                if link.link_id not in drawn and ends[0] in g.nodes and ends[1] in g.nodes:
                    drawn.add(link.link_id)
                    g.link(ends[0], ends[1], link.kind, source_episode_id=link.source_episode_id)
        for e in agent_events:
            G.add_agent_event(g, e)
        for e in recall_events:
            G.add_recall_event(g, e)
            for fid in e.payload.get("fact_ids") or []:
                g.link(f"recall:{e.event_id}", f"claim:{int(fid)}", "held")
        for e in feedback_events:
            if f"recall:{e.payload.get('recall_event_id')}" in g.nodes:
                G.add_feedback_event(g, e)
        G.add_claim_edges(g)
        return g

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

    async def _history_for(self, space: str, facts: Sequence[Fact], when: str, limit: int = 20) -> list[Fact]:
        """The closed ledger facts sharing a subject and predicate with any of
        ``facts`` and begun by ``when``, oldest first: the chain of what was
        believed before. Nothing that started after the reader's boundary
        leaks through; excluded facts stay out, as everywhere; proposals and
        declined candidates never held, so they are not history."""
        if not facts:
            return []
        keys = {(f.subject, f.predicate) for f in facts}
        shown = {f.fact_id for f in facts}
        chain = [
            f for f in await self.documents.list_facts(space, include_closed=True)
            if (f.subject, f.predicate) in keys and f.fact_id not in shown
            and f.status == "closed" and not f.excluded and f.valid_from <= when
        ]
        chain.sort(key=lambda f: (f.valid_from, f.fact_id))
        return chain[:limit]

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
        quote: Optional[str] = None,
        extends: Optional[int] = None,
        derived_from: Sequence[int] = (),
    ) -> Fact:
        """Record that ``subject predicate object`` holds from ``valid_from``;
        see ``_assert_placed`` for how it is placed among the ledger facts.

        ``extends`` names a fact this one adds detail to: both stay as they
        are and a link records the relation, so an extension may not share
        the extended fact's subject and predicate (that would supersede it;
        assert an update instead). ``derived_from`` names the ledger facts
        this one was inferred from; the claim is stored as ``inferred`` and
        a link records each premise. Both are checked before anything is
        written: an unknown target, one in another space, or one that is
        only proposed or declined refuses the whole assertion."""
        check_space(space)
        premises = [int(f) for f in derived_from]
        if premises:
            if origin == "stated":
                origin = "inferred"
            elif origin != "inferred":
                raise InvalidInput("a derived claim is inferred; it cannot claim another origin")
        targets = {}
        for target_id in ([extends] if extends is not None else []) + premises:
            target = await self.documents.get_fact(space, target_id)
            if target is None:
                raise NotFound(f"fact {target_id} not found in {space!r}")
            if not target.in_ledger:
                raise InvalidInput(f"fact {target_id} is {target.status}; only a held or closed fact can be built on")
            targets[target_id] = target
        if extends is not None and (targets[extends].subject, targets[extends].predicate) == (subject, predicate):
            raise InvalidInput("an extension cannot supersede what it extends; assert an update instead")
        fact = await self._assert_placed(space, subject, predicate, object, valid_from=valid_from, confidence=confidence,
                                         source_episode_id=source_episode_id, origin=origin, proposed=proposed, quote=quote)
        if extends is not None:
            await self.link_facts(space, fact.fact_id, extends, "extends")
        for premise in premises:
            await self.link_facts(space, fact.fact_id, premise, "derived_from")
        return fact

    async def link_facts(
        self, space: str, from_fact: int, to_fact: int, kind: str, *,
        source_episode_id: Optional[int] = None, quote: Optional[str] = None,
    ) -> FactLink:
        """Relate two facts of one space: ``from_fact`` extends / is derived
        from / contradicts / supports ``to_fact``. The same link twice is one
        link. A quote must sit in the source episode it names. A dependency
        (extends, derived_from) that would close a cycle is refused."""
        check_space(space)
        if kind not in LINK_KINDS:
            raise InvalidInput(f"link kind must be one of {LINK_KINDS}, got {kind!r}")
        if from_fact == to_fact:
            raise InvalidInput("a fact cannot be linked to itself")
        for fact_id in (from_fact, to_fact):
            if await self.documents.get_fact(space, fact_id) is None:
                raise NotFound(f"fact {fact_id} not found in {space!r}")
        if quote is not None:
            if not quote.strip() or len(quote) > 2000:
                raise InvalidInput("quote must be 1..2000 characters when given")
            if source_episode_id is None:
                raise InvalidInput("a quote needs a source_episode_id to be checked against")
            episode = await self.documents.get_episode(space, source_episode_id)
            if episode is None:
                raise NotFound(f"source episode {source_episode_id} not found in {space!r}")
            if quote not in episode.content:
                raise InvalidInput(f"quote is not a substring of episode {source_episode_id}; the link was not stored")
        for existing in await self.documents.fact_links(space, from_fact):
            if (existing.from_fact, existing.to_fact, existing.kind) == (from_fact, to_fact, kind):
                return existing
        if kind in DEPENDENCY_KINDS and await self._depends_on(space, to_fact, from_fact):
            raise InvalidInput(f"fact {from_fact} cannot {kind.replace('_', ' ')} fact {to_fact}: that would close a dependency cycle")
        link = await self.documents.insert_fact_link(NewFactLink(
            space=space, from_fact=from_fact, to_fact=to_fact, kind=kind, created_at=self.clock(),
            source_episode_id=source_episode_id, quote=quote,
        ))
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_link", {"link_id": link.link_id, "from_fact": from_fact, "to_fact": to_fact,
                                              "kind": kind, "source_episode_id": source_episode_id})
        return link

    async def fact(self, space: str, fact_id: int) -> Fact:
        """One fact of the space, whatever its status."""
        check_space(space)
        found = await self.documents.get_fact(space, fact_id)
        if found is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        return found

    async def fact_links(self, space: str, fact_id: int) -> list[FactLink]:
        """Every link naming the fact at either end, oldest first."""
        check_space(space)
        if await self.documents.get_fact(space, fact_id) is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        return await self.documents.fact_links(space, fact_id)

    async def _depends_on(self, space: str, start: int, target: int) -> bool:
        """Whether ``start`` reaches ``target`` along dependency links."""
        seen, frontier = {start}, [start]
        while frontier:
            fact_id = frontier.pop()
            for link in await self.documents.fact_links(space, fact_id):
                if link.from_fact != fact_id or link.kind not in DEPENDENCY_KINDS or link.to_fact in seen:
                    continue
                if link.to_fact == target:
                    return True
                seen.add(link.to_fact)
                frontier.append(link.to_fact)
            if len(seen) > 10_000:
                raise InvalidInput("dependency chain too long to check")
        return False

    async def _assert_placed(
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
        quote: Optional[str] = None,
    ) -> Fact:
        """Record that ``subject predicate object`` holds from ``valid_from``.

        ``quote`` is the exact substring of the source episode the claim
        rests on. When both a source and a quote are given, the quote must
        be non-empty and must occur in that episode's content, else the
        assertion is refused as ungrounded rather than stored. A source with
        no quote is stored and reads as ungrounded (``Fact.grounded`` is
        False); a quote with no source is refused, since there is nothing
        to check it against.

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
        if quote is not None:
            if not quote.strip():
                raise InvalidInput("quote must not be empty when given")
            if len(quote) > 2000:
                raise InvalidInput("quote must be at most 2000 characters")
            if source_episode_id is None:
                raise InvalidInput("a quote needs a source_episode_id to be checked against")
            episode = await self.documents.get_episode(space, source_episode_id)
            if episode is None:
                raise NotFound(f"source episode {source_episode_id} not found in {space!r}")
            if quote not in episode.content:
                raise InvalidInput(f"quote is not a substring of episode {source_episode_id}; the claim is ungrounded and was not stored")
        start = normalise_time(valid_from) if valid_from else self.clock()
        started = time.perf_counter()

        if proposed:
            fact = await self.documents.insert_fact(
                NewFact(
                    space=space, subject=subject, predicate=predicate, object=object, valid_from=start,
                    confidence=confidence, status="proposed", source_episode_id=source_episode_id, origin=origin, quote=quote,
                )
            )
            await self.documents.bump_revision(space)
            await self._emit(space, "fact_assert", {
                "fact_id": fact.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
                "outcome": "proposed", "superseded": [], "source_episode_id": source_episode_id,
                "grounded": fact.grounded, "latency_ms": _ms(started),
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
                closed_reason=placement.bound_reason, superseded_by=placement.bound_by,
                source_episode_id=source_episode_id, origin=origin, quote=quote,
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
            bound_by=successor.fact_id if successor else None,
        )

    async def _truncate(self, covering: list[Fact], start: str, by_fact_id: int) -> None:
        for rival in covering:
            reason = rival.closed_reason
            if reason is None or reason.startswith("superseded by fact "):
                reason = f"superseded by fact {by_fact_id}"
            await self.documents.update_fact(
                rival.model_copy(update={"status": "closed", "valid_until": start, "closed_reason": reason, "superseded_by": by_fact_id})
            )

    async def approve(self, space: str, fact_id: int, actor: Optional[str] = None) -> Fact:
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
            await self._emit(space, "fact_review", {"fact_id": fact_id, "decision": "duplicate", "of": placement.restates.fact_id, "actor": actor})
            return placement.restates
        accepted = fact.model_copy(update={
            "status": "closed" if placement.bound else "active",
            "valid_until": placement.bound,
            "closed_reason": placement.bound_reason,
            "superseded_by": placement.bound_by,
        })
        await self.documents.update_fact(accepted)
        await self._truncate(placement.covering, fact.valid_from, fact.fact_id)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_review", {
            "fact_id": fact_id, "decision": "approved", "superseded": [r.fact_id for r in placement.covering], "actor": actor,
        })
        return accepted

    async def decide(
        self,
        space: str,
        decision: str,
        fact_ids: Sequence[int],
        reason: Optional[str] = None,
        actor: Optional[str] = None,
        expect_revision: Optional[int] = None,
    ) -> BatchDecision:
        """One reviewed batch, settled together.

        Three things a loop of single decisions cannot do. The batch is
        applied by (valid_from, fact_id), not in the order the caller
        listed, because inside one subject and predicate each approval
        truncates what it covers and is bounded by what starts later, so
        the caller's draw order would otherwise decide the ledger. With
        ``expect_revision`` the space is checked once, before anything is
        applied, so a batch built from a stale reading is refused whole
        rather than half applied. And every id comes back with its own
        outcome, keyed by the id that was sent, so one refusal does not
        hide what the rest did.
        """
        check_space(space)
        if decision not in DECISIONS:
            raise InvalidInput(f"decision must be one of {DECISIONS}, got {decision!r}")
        if not fact_ids:
            raise InvalidInput("decide needs at least one fact id")
        if len(fact_ids) > MAX_DECISIONS:
            raise InvalidInput(f"decide takes at most {MAX_DECISIONS} ids, got {len(fact_ids)}")
        if decision in ("decline", "exclude"):
            reason = _reason(reason or "")
        current = await self.documents.revision(space)
        if expect_revision is not None and expect_revision != current:
            raise Conflict(
                f"space {space!r} is at revision {current}, the batch was built from {expect_revision}",
                revision=current,
            )
        known = {}
        outcomes: dict[int, DecisionOutcome] = {}
        for fact_id in fact_ids:
            if fact_id in outcomes or fact_id in known:
                continue
            found = await self.documents.get_fact(space, fact_id)
            if found is None:
                outcomes[fact_id] = DecisionOutcome(fact_id=fact_id, outcome="not_found")
            else:
                known[fact_id] = found
        for fact in sorted(known.values(), key=lambda f: (parse_rfc3339(f.valid_from), f.fact_id)):
            outcomes[fact.fact_id] = await self._decide_one(space, decision, fact.fact_id, reason, actor)
        applied = sum(1 for o in outcomes.values() if o.outcome not in ("not_found", "refused"))
        return BatchDecision(
            results=[outcomes[fact_id] for fact_id in dict.fromkeys(fact_ids)],
            applied=applied,
            revision=await self.documents.revision(space),
        )

    async def _decide_one(
        self, space: str, decision: str, fact_id: int, reason: Optional[str], actor: Optional[str]
    ) -> DecisionOutcome:
        try:
            if decision == "approve":
                landed = await self.approve(space, fact_id, actor=actor)
                if landed.fact_id != fact_id:
                    return DecisionOutcome(fact_id=fact_id, outcome="duplicate_of", held_fact_id=landed.fact_id)
                return DecisionOutcome(fact_id=fact_id, outcome="approved")
            if decision == "decline":
                await self.decline(space, fact_id, reason or "", actor=actor)
                return DecisionOutcome(fact_id=fact_id, outcome="declined")
            await self.exclude(space, fact_id, reason or "", actor=actor)
            return DecisionOutcome(fact_id=fact_id, outcome="excluded")
        except NotFound as e:
            return DecisionOutcome(fact_id=fact_id, outcome="not_found", error=str(e))
        except InvalidInput as e:
            return DecisionOutcome(fact_id=fact_id, outcome="refused", error=str(e))

    async def decline(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        """A person rejects a proposed fact. It never held; the reason is kept."""
        check_space(space)
        reason = _reason(reason)
        fact = await self._proposed(space, fact_id)
        declined = fact.model_copy(update={"status": "declined", "closed_reason": reason})
        await self.documents.update_fact(declined)
        await self.documents.bump_revision(space)
        await self._emit(space, "fact_review", {"fact_id": fact_id, "decision": "declined", "actor": actor})
        return declined

    async def _proposed(self, space: str, fact_id: int) -> Fact:
        fact = await self.documents.get_fact(space, fact_id)
        if fact is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
        if fact.status != "proposed":
            raise InvalidInput(f"fact {fact_id} is {fact.status}, not proposed")
        return fact

    async def exclude(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
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
        await self._emit(space, "fact_exclude", {"fact_id": fact_id, "action": "exclude", "actor": actor})
        return excluded

    async def include(self, space: str, fact_id: int, actor: Optional[str] = None) -> Fact:
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
        await self._emit(space, "fact_exclude", {"fact_id": fact_id, "action": "include", "actor": actor})
        return included

    async def close_fact(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
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
        await self._emit(space, "fact_close", {"fact_id": fact_id, "reason_kind": "manual", "actor": actor})
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

    async def pending_distillation(self, space: str) -> int:
        """Episodes no claim cites yet. The same definition the distiller
        and memory_pending use, so the three surfaces agree."""
        check_space(space)
        counts = await self.documents.counts(space)
        if counts.episodes == 0:
            return 0
        referenced = {f.source_episode_id for f in await self.documents.list_facts(space, include_closed=True)}
        return sum(1 for e in await self.documents.recent_episodes(space, counts.episodes) if e.episode_id not in referenced)

    async def source_page(self, space: str, *, before: Optional[int] = None,
                          limit: int = 25, kind: Optional[str] = None) -> SourcePage:
        """Browse retained sources by descending ID, not relevance or source date.

        Newer inserts are found by restarting the walk. Deleting the boundary
        record does not invalidate the next page. This is not a frozen snapshot.
        """
        check_space(space)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise InvalidInput("limit must be an integer from 1 through 100")
        if before is not None and (isinstance(before, bool) or not isinstance(before, int) or not 1 <= before <= 2**63-1):
            raise InvalidInput("before must be a positive signed 64-bit episode ID")
        if kind is not None and kind not in KINDS:
            raise InvalidInput(f"kind must be one of {KINDS}")
        page = getattr(self.documents, "page_episodes", None)
        if not callable(page):
            raise InvalidInput("this document store does not implement source inventory")
        rows = await page(space, before, limit + 1, kind)
        more = len(rows) > limit
        return SourcePage(rows[:limit], more, rows[limit-1].episode_id if more else None)

    async def episodes(self, space: str, where: Mapping[str, str], limit: Optional[int] = None) -> list[Episode]:
        """The episodes whose metadata matches every ``where`` pair, oldest
        first by (created_at, episode_id); with ``limit``, the newest N of
        them in that same order. This is a walk over the space's episodes,
        fine for a session's turns, not a query language."""
        check_space(space)
        clean = normalise_metadata(where)
        if not clean:
            raise InvalidInput("episodes() needs at least one where pair")
        counts = await self.documents.counts(space)
        found = [
            e for e in await self.documents.recent_episodes(space, max(counts.episodes, 1))
            if all(e.metadata.get(k) == v for k, v in clean.items())
        ]
        found.sort(key=lambda e: (e.created_at, e.episode_id))
        return found[-limit:] if limit else found

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

    async def revision(self, space: str) -> int:
        """The space's write counter. A page that renders a list can record
        the revision it read at and tell later whether the space moved."""
        check_space(space)
        return await self.documents.revision(space)

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
                "space": space,
                "episode_id": episode.episode_id,
                "kind": episode.kind,
                "content": episode.content,
                "content_hash": episode.content_hash,
                "source": episode.source,
                "tags": list(episode.tags),
                "metadata": dict(episode.metadata),
                "created_at": episode.created_at,
            }
        facts = await self.documents.list_facts(space, include_closed=True)
        for fact in facts:
            yield {"type": "fact", **fact.model_dump(exclude={"space"})}
        # A relation is part of the ledger's history, so it moves with it;
        # each link is read from both ends and written once.
        seen: set[int] = set()
        for fact in facts:
            for link in await self.documents.fact_links(space, fact.fact_id):
                if link.link_id not in seen:
                    seen.add(link.link_id)
                    yield {"type": "fact_link", **link.model_dump(exclude={"space"})}

    async def import_records(self, space: str, records: Iterable[Mapping]) -> ImportSummary:
        """Load an export. Episodes go through the normal ingest, so they
        are re-chunked, re-embedded and deduplicated; facts are stored as
        they were, closed ones included, because the ledger's history is
        part of what is being moved.

        Identity is bound to the space name (see content_hash). A line
        whose content_hash is the default derivation under the space it
        names is re-derived for this space, so a dump moved into a space
        of another name still deduplicates against what is remembered
        there next. A keyed identity (dedup_key) has no content to derive
        from and is passed through as the source made it; it keeps
        deduplicating a same-space move and not a renamed one."""
        check_space(space)
        summary = ImportSummary()
        episodes: list[Record] = []
        source_ids: list[Optional[int]] = []
        facts: list[Mapping] = []
        links: list[Mapping] = []
        for record in records:
            kind = record.get("type", "episode")
            if kind == "episode":
                episodes.append(_rederived(Record.from_dict(record), record.get("space"), space))
                source_ids.append(record.get("episode_id"))
            elif kind == "fact":
                facts.append(record)
            elif kind == "fact_link":
                links.append(record)
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
        existing = {_fact_identity(f): f.fact_id for f in await self.documents.list_facts(space, include_closed=True)}
        # Fact ids are store-local too: a link's ends are remapped through
        # the ids this import produced or found already present.
        fact_map: dict[int, int] = {}
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
                superseded_by=None,  # ids are store-local; the reason text keeps the history
                quote=f.get("quote"),
            )
            if new.status not in STATUSES or new.origin not in ORIGINS:
                raise InvalidInput(f"fact record has status {new.status!r} and origin {new.origin!r}")
            identity = (
                new.subject, new.predicate, new.object, new.valid_from, new.valid_until,
                new.status, new.closed_reason, new.confidence,
            )
            if identity in existing:
                summary.facts_skipped += 1
                if f.get("fact_id") is not None:
                    fact_map[int(f["fact_id"])] = existing[identity]
                continue
            stored = await self.documents.insert_fact(new)
            existing[identity] = stored.fact_id
            if f.get("fact_id") is not None:
                fact_map[int(f["fact_id"])] = stored.fact_id
            summary.facts += 1
        for record in links:
            kind = str(record.get("kind", ""))
            ends = (fact_map.get(int(record["from_fact"])), fact_map.get(int(record["to_fact"])))
            if kind not in LINK_KINDS or None in ends or ends[0] == ends[1]:
                # A link to a fact not in the dump is dropped, not pointed at
                # an unrelated record; a bad kind is a bad record.
                summary.links_skipped += 1
                continue
            source = record.get("source_episode_id")
            new_link = NewFactLink(
                space=space, from_fact=ends[0], to_fact=ends[1], kind=kind,
                created_at=normalise_time(str(record["created_at"])) if record.get("created_at") else self.clock(),
                source_episode_id=id_map.get(int(source)) if source is not None else None,
                quote=record.get("quote"),
            )
            already = any((l.to_fact, l.kind) == (ends[1], kind) for l in await self.documents.fact_links(space, ends[0]) if l.from_fact == ends[0])
            if already:
                summary.links_skipped += 1
                continue
            await self.documents.insert_fact_link(new_link)
            summary.links += 1
        if summary.facts or summary.links:
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


def _fits(episode: Episode, kind: Optional[str], source_prefix: Optional[str], since: Optional[str], until: Optional[str]) -> bool:
    """The narrowing rule, shared with the Rust engine: kind equal, source
    starting with the prefix as literal text (an episode without a source
    matches no prefix), created_at inside the inclusive bounds."""
    if kind is not None and episode.kind != kind:
        return False
    if source_prefix is not None and (episode.source is None or not episode.source.startswith(source_prefix)):
        return False
    if since is not None and episode.created_at < since:
        return False
    return not (until is not None and episode.created_at > until)


def _rederived(record: Record, source_space: Optional[str], space: str) -> Record:
    """The record with its content_hash dropped when it is the default
    derivation under the source space, so ingest derives it for the
    target space instead."""
    if record.content_hash is None or not source_space or source_space == space:
        return record
    if record.content_hash == content_hash(str(source_space), record.content):
        return dataclasses.replace(record, content_hash=None)
    return record


def content_hash(space: str, content: str, dedup_key: Optional[str] = None) -> str:
    if dedup_key is not None:
        if not dedup_key or len(dedup_key) > 256:
            raise InvalidInput("dedup_key must be 1..=256 chars")
        return hashlib.sha256(f"{space}\x00key\x00{dedup_key}".encode()).hexdigest()
    return hashlib.sha256(f"{space}\x00{content.strip()}".encode()).hexdigest()
