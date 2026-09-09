"""The engine: remember, recall, forget, and the fact ledger.

Built entirely against the protocols in ``ports``; nothing here knows
whether a dict, MongoDB, or Qdrant is underneath. A lane that fails
during recall is named in ``degraded`` and the other lane still
answers, because a thin answer that says it is thin beats a 500.
"""

from __future__ import annotations

import hashlib
import time
from uuid import uuid4
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterable, Mapping, Optional, Sequence, TypedDict, cast

from . import archive
from .archive import ImportSummary as ImportSummary, _fact_identity as _fact_identity, _rederived as _rederived
from ..ingestion import batch as ingestion_batch
from ..ingestion.batch import EMBED_BATCH as EMBED_BATCH
from ..ingestion.records import (
    Record as Record, RecoveryReport as RecoveryReport,
    _Pending as _Pending, _DupOf as _DupOf,
    content_hash as content_hash, contextual_prefix as contextual_prefix,
)
from ..retrieval import fact_recall
from ..retrieval.recall import (RecallRuntime, recall, LANE_DEPTH as LANE_DEPTH,
                                UNFILTERED_DEPTH as UNFILTERED_DEPTH)
from ..retrieval.episode_scope import episode_fits as _fits
from ..retrieval.fact_recall import FACT_SCOPE_CACHE_LIMIT as FACT_SCOPE_CACHE_LIMIT
from ..retrieval.overview import OverviewResult
from ..retrieval.reranking import Reranker, validate_candidate_limit, validate_rerank_options
from ..ingestion.chunker import DEFAULT_TARGET
from ..core.validation import (
    ORIGINS as ORIGINS,
    STATUSES as STATUSES,
    SPACE_NAME as SPACE_NAME,
    METADATA_KEY as METADATA_KEY,
    MAX_METADATA_KEYS as MAX_METADATA_KEYS,
    MAX_METADATA_VALUE as MAX_METADATA_VALUE,
    KINDS as KINDS,
    MAX_QUERY as MAX_QUERY,
    MAX_LIMIT as MAX_LIMIT,
    MAX_SOURCE as MAX_SOURCE,
    retention_policy as retention_policy,
    check_space as check_space,
    normalise_tags as normalise_tags,
    normalise_metadata as normalise_metadata,
    normalise_term as normalise_term,
    normalise_time as normalise_time,
)
from ..core.errors import Conflict, Gone, InvalidInput, NotFound
from ..backends.blobs import BlobStore, InMemoryBlobStore
from ..core.models import (
    Added,
    Attachment,
    BatchDecision,
    DecisionOutcome,
    Episode,
    EpisodeKind,
    Fact,
    FactLink,
    DoctorReport,
    ExpiryReport,
    ForgetReceipt,
    JobItem,
    IngestJob,
    SpaceReceipt,
    Tombstone,
    DEPENDENCY_KINDS,
    LINK_KINDS,
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
    NewEpisode,
    NewEvent,
    NewFact,
    NewFactLink,
    NewJob,
    NewTombstone,
    SourcePage,
    TextFilter,
    VectorIndex,
)
from ..core.timeutil import format_rfc3339, now_rfc3339, parse_rfc3339

#: Rows read at a time when walking the inventory for matches, and how
#: many such reads one page may cost. A filter matching nothing would
#: otherwise read a whole space to prove a negative.
SOURCE_WALK_PAGE = 200
SOURCE_WALK_READS = 25


#: Event kinds an outside process may append (long-running jobs
#: reporting progress). Engine kinds cannot be forged through this path.
EXTERNAL_EVENT_KINDS = ("job", "agent")
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


@dataclass(frozen=True)
class RecentActivity:
    """One ``dynamic`` excerpt with the episode it was cut from."""

    episode_id: int
    excerpt: str
    created_at: str


@dataclass
class Profile:
    static_facts: list[Fact] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)
    #: ``dynamic`` with its evidence: same order, same excerpts, newest first.
    recent: list[RecentActivity] = field(default_factory=list)


@dataclass
class Replaced:
    """What ``replace`` did: the record as stored (or found), the outcome,
    and the receipt for the episode that went, if one did."""

    added: Added
    outcome: str
    replaced: Optional[ForgetReceipt] = None


@dataclass(frozen=True)
class _Placement:
    covering: list
    restates: Optional[Fact]
    bound: Optional[str]
    bound_reason: Optional[str]
    bound_by: Optional[int]


class _RecallEventItem(TypedDict):
    """The identity field read from engine-authored recall event items."""

    chunk_id: int


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
        demote_restated: bool = True,
        blobs: Optional[BlobStore] = None,
        candidate_limit: int | None = None,
        reranker: Reranker | None = None,
        rerank_limit: int = 32,
        rerank_max_bytes: int = 64000,
        rerank_timeout: float = 1.0,
    ) -> None:
        if similarity_floor is not None and not -1.0 <= similarity_floor <= 1.0:
            raise InvalidInput("similarity_floor must be a cosine similarity in [-1, 1]")
        self.candidate_limit = validate_candidate_limit(candidate_limit)
        validate_rerank_options(rerank_limit, rerank_max_bytes, rerank_timeout)
        if reranker is not None and not callable(getattr(reranker, "rerank", None)):
            raise InvalidInput("reranker must provide an async rerank method")
        self.reranker = reranker
        self.rerank_limit = rerank_limit
        self.rerank_max_bytes = rerank_max_bytes
        self.rerank_timeout = rerank_timeout
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
        #: (space, premise ids) of every group a derivation pass has sent.
        self._derive_seen: set[tuple[str, frozenset[int]]] = set()
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
        #: Within one result, a statement and the statement that replaced
        #: it score almost the same, because they differ only at the end,
        #: so insertion order decided which came first. On MemoryAgentBench
        #: Conflict Resolution the superseded one led in 72 of 74 questions
        #: (E34); with this it leads in 1 (E35). On ordinary retrieval it
        #: changed nothing at all: every one of LongMemEval-S's stratified
        #: 60 came back identical, because the rule never found a pair to
        #: reorder (E32d). Measured cost nothing, measured benefit large,
        #: so it is on. Nothing is dropped either way, only ordered, and a
        #: caller who wants what was believed at the time turns it off.
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

        return await ingestion_batch.recover(self._ingestion_runtime())

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
        dedup_key: Optional[str] = None,
        replace: bool = False,
    ) -> Added:
        """One record. ``dedup_key`` names it across writes; ``replace``
        makes a changed record under a known key an update (see
        ``replace``) instead of a duplicate."""
        await self._living(space)
        record = Record(content, kind, source, tuple(tags), created_at, dict(metadata or {}), dedup_key=dedup_key)
        if replace:
            added = (await self.replace(space, record)).added
        else:
            [added] = await self.remember_many(space, [record])
        for attachment_id in dict.fromkeys(attachment_ids):
            await self.blobs.link(space, attachment_id, added.episode_id)
        return added

    async def replace(self, space: str, record: Record) -> "Replaced":
        """Store a keyed record as the current one under its key. A key
        nobody holds is accepted; the same content again is a duplicate
        and changes nothing; changed content is an update: the episode
        the key named is forgotten (its receipt returned; the claims that
        cited it stand) and the new one stored. The forget lands before
        the store, so a failure between them leaves the key empty rather
        than pointing at stale text; the error says so."""
        await self._living(space)
        check_space(space)
        if not record.dedup_key:
            raise InvalidInput("replace needs a dedup_key: it is the key that names what is being replaced")
        digest = content_hash(space, record.content, record.dedup_key)
        existing = await self.documents.episode_by_hash(space, digest)
        if existing is not None and existing.content == record.content:
            added = Added(episode_id=existing.episode_id, deduplicated=True, chunks=0, outcome="duplicate")
            return Replaced(added=added, outcome="duplicate", replaced=None)
        receipt = None
        if existing is not None:
            receipt = await self.forget(space, existing.episode_id)
        try:
            [added] = await self.remember_many(space, [record])
        except Exception as error:
            if receipt is not None:
                raise InvalidInput(
                    f"replace forgot episode {receipt.episode_id} and the new record then failed to land "
                    f"({type(error).__name__}); the key {record.dedup_key!r} names nothing now, so send it again"
                ) from error
            raise
        outcome = "updated" if receipt is not None else added.outcome
        added = added.model_copy(update={"outcome": outcome, "replaced": receipt})
        return Replaced(added=added, outcome=outcome, replaced=receipt)

    # -- attachments ------------------------------------------------------

    async def attach(
        self, space: str, data: bytes, media_type: str, filename: Optional[str] = None
    ) -> Attachment:
        """Store bytes an episode will carry, addressed by their SHA-256.
        The same bytes stored twice are one attachment: the digest is the
        id, so a second store is a second reference."""
        await self._living(space)
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
        await self._living(space)
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

    def _ingestion_runtime(self) -> ingestion_batch.IngestionRuntime:
        return ingestion_batch.IngestionRuntime(
            self.documents, self.vectors, self.embedder, self.clock, self.chunk_target,
            self._embed_text, self._emit,
        )

    async def _remember_many(self, space: str, records: Sequence[Record]) -> list[Added]:
        return await ingestion_batch.remember_many(self._ingestion_runtime(), space, records)

    def _embed_text(self, episode: NewEpisode, chunk_text: str) -> str:
        """What the embedder sees for a chunk. Stored text is never changed."""
        if not self.contextual_embeddings:
            return chunk_text
        prefix = contextual_prefix(episode)
        return f"{prefix}\n{chunk_text}" if prefix else chunk_text

    async def _write_batch(
        self, space: str, fresh: list["_Pending"], vectors: list[list[float]], results: list[Added | _DupOf | None]
    ) -> None:
        await ingestion_batch.write_batch(self._ingestion_runtime(), space, fresh, vectors, results)

    async def _episode_or_gone(self, space: str, episode_id: int) -> Episode:
        """The episode, or Gone when a tombstone says it was forgotten, or
        NotFound when the id never meant anything here."""
        found = await self.documents.get_episode(space, episode_id)
        if found is not None:
            return found
        stone = await self.documents.tombstone(space, episode_id)
        if stone is not None:
            raise Gone(f"episode {episode_id} was forgotten on {stone.forgotten_at}", stone.forgotten_at)
        raise NotFound(f"episode {episode_id} not found in {space!r}")

    async def tombstone(self, space: str, episode_id: int) -> Optional[Tombstone]:
        """The record that an episode was forgotten, or None."""
        check_space(space)
        return await self.documents.tombstone(space, episode_id)

    async def doctor(self, space: str) -> DoctorReport:
        """What references what across the space's stores, read only: chunks
        whose episode is gone, vectors whose chunk is gone, facts citing a
        forgotten or an unknown episode, links with a missing end, held
        attachments no episode carries. A store that cannot be walked is
        named in not_inspected rather than reported clean."""
        check_space(space)
        counts = await self.documents.counts(space)
        facts = await self.documents.list_facts(space, include_closed=True)
        report = DoctorReport(space=space, episodes=counts.episodes, chunks=counts.chunks, facts=len(facts))
        episode_ids = {e.episode_id for e in await self.documents.recent_episodes(space, max(counts.episodes, 1))}
        stones = {t.episode_id for t in await self.documents.list_tombstones(space)}
        report.tombstones = len(stones)
        for fact in facts:
            source = fact.source_episode_id
            if source is None or source in episode_ids:
                continue
            (report.facts_citing_forgotten if source in stones else report.facts_citing_unknown).append(fact.fact_id)
        fact_ids = {f.fact_id for f in facts}
        seen: set[int] = set()
        for fact in facts:
            for link in await self.documents.fact_links(space, fact.fact_id):
                if link.link_id in seen:
                    continue
                seen.add(link.link_id)
                if link.from_fact not in fact_ids or link.to_fact not in fact_ids:
                    report.links_with_missing_ends.append(link.link_id)
        report.links = len(seen)
        chunk_index = getattr(self.documents, "chunk_index", None)
        chunk_ids: Optional[set[int]] = None
        if callable(chunk_index):
            pairs = await chunk_index(space)
            chunk_ids = {chunk_id for chunk_id, _ in pairs}
            report.chunks_without_episode = sorted(chunk_id for chunk_id, episode_id in pairs if episode_id not in episode_ids)
        else:
            report.not_inspected.append("chunks")
        vector_ids = getattr(self.vectors, "ids", None)
        if callable(vector_ids) and chunk_ids is not None:
            report.vectors_without_chunk = sorted(v for v in await vector_ids(space) if v not in chunk_ids)
        else:
            report.not_inspected.append("vectors")
        held = getattr(self.blobs, "held", None)
        if callable(held):
            linked = await self.blobs.linked(space)
            report.attachments_unlinked = [a for a in await held(space) if a not in linked]
        else:
            report.not_inspected.append("attachments")
        report.healthy = not any([
            report.chunks_without_episode, report.vectors_without_chunk, report.facts_citing_forgotten,
            report.facts_citing_unknown, report.links_with_missing_ends, report.attachments_unlinked,
        ])
        return report

    async def expire(self, space: str, policy: Mapping[str, float], *, limit: int = 100,
                     dry_run: bool = False) -> ExpiryReport:
        """Forget the episodes a retention policy no longer keeps: for each
        kind in ``policy``, those whose own time is more than that many
        days before this engine's clock, oldest first, at most ``limit``
        in one pass. Facts never expire; the claims that cited a forgotten
        episode stand. ``dry_run`` reports and forgets nothing."""
        check_space(space)
        clean = retention_policy(policy)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise InvalidInput("limit must be an integer in 1..10000")
        now = parse_rfc3339(self.clock())
        counts = await self.documents.counts(space)
        due = []
        for episode in await self.documents.recent_episodes(space, max(counts.episodes, 1)):
            days = clean.get(episode.kind)
            if days is not None and (now - parse_rfc3339(episode.created_at)).total_seconds() > days * 86400:
                due.append(episode)
        due.sort(key=lambda e: (e.created_at, e.episode_id))
        report = ExpiryReport(space=space, policy=dict(clean), remaining=len(due), dry_run=dry_run)
        if dry_run:
            return report
        for episode in due[:limit]:
            report.receipts.append(await self.forget(space, episode.episode_id))
            report.forgotten.append(episode.episode_id)
        report.remaining = len(due) - len(report.forgotten)
        await self._emit(space, "expire", {"policy": dict(clean), "forgotten": len(report.forgotten),
                                           "remaining": report.remaining, "limit": limit})
        return report

    async def impact(self, space: str, episode_id: int) -> ForgetReceipt:
        """What forgetting the episode would take with it and leave, with
        nothing removed. The claims and links that cite it are reported,
        not closed: a source being gone is a fact about the evidence."""
        check_space(space)
        await self._episode_or_gone(space, episode_id)
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
        except NotFound as error:
            await self._emit(space, "forget", {"episode_id": episode_id, "error": type(error).__name__, "latency_ms": _ms(started)})
            raise
        episode = await self._episode_or_gone(space, episode_id)
        removed = await self.documents.delete_episode(space, episode_id)
        await self.vectors.delete(removed)
        await self.blobs.unlink(space, episode_id)
        # The tombstone outlives the episode: the id keeps meaning something
        # and the decision is not undone by a later import.
        stone = await self.documents.record_tombstone(NewTombstone(
            space=space, episode_id=episode_id, content_hash=episode.content_hash, forgotten_at=self.clock(),
        ))
        receipt = receipt.model_copy(update={"forgotten_at": stone.forgotten_at})
        await self.documents.bump_revision(space)
        await self._emit(space, "forget", {
            "episode_id": episode_id, "chunks_removed": len(removed),
            "attachments_released": len(receipt.attachments_released),
            "facts_citing": len(receipt.facts_citing), "links_citing": len(receipt.links_citing),
            "latency_ms": _ms(started),
        })
        return receipt

    # -- ingest jobs ---------------------------------------------------------

    def _able(self, *methods: str) -> bool:
        return all(callable(getattr(self.documents, name, None)) for name in methods)

    #: What a store must implement to record a batch, and to read one back.
    #: They are separate because a store may do one and not the other, and
    #: advertising the wrong one turns a refusal into a server error.
    RECORDS_JOBS = ("create_job", "update_job")
    READS_JOBS = ("get_job", "list_jobs")

    def _keeps_jobs(self) -> None:
        """Recording jobs is a store's choice: one that cannot keep them
        says so rather than pretending a batch was never tracked."""
        if not self._able(*self.RECORDS_JOBS):
            raise InvalidInput("this document store does not record ingest jobs")

    def _reads_jobs(self) -> None:
        """Reading a job back is a separate choice from recording one."""
        if not self._able(*self.READS_JOBS):
            raise InvalidInput("this document store does not read ingest jobs")

    async def ingest_batch(self, space: str, records: Iterable[Record], *,
                           request_id: Optional[str] = None) -> "IngestJob":
        """Ingest a batch and keep the receipt of what it became.

        Two receipts per record, never one: ``searchable_at`` is set here,
        because chunks and vectors land with the batch, and
        ``consolidated_at`` only when a model has read claims out of the
        record, which happens later and may never happen. A request id
        makes a retry the same job rather than a second one."""
        check_space(space)
        await self._living(space)
        self._keeps_jobs()
        if request_id:
            already = await self.documents.job_by_request(space, request_id)
            if already is not None:
                return already
        added = await self.remember_many(space, list(records))
        return await self.record_job(space, added, request_id=request_id)

    async def record_job(self, space: str, added: Sequence[Added], *,
                         request_id: Optional[str] = None) -> "IngestJob":
        """Keep the receipt for records that have just landed. Separate
        from ingesting them, because a caller may have written the batch
        its own way and still owes the person a receipt."""
        check_space(space)
        self._keeps_jobs()
        when = self.clock()
        items = tuple(
            JobItem(index=index, episode_id=one.episode_id, outcome=one.outcome,
                    state="searchable", searchable_at=when)
            for index, one in enumerate(added)
        )
        job = await self.documents.create_job(NewJob(
            job_id=uuid4().hex, space=space, created_at=when, request_id=request_id, items=items))
        await self._emit(space, "ingest_job", {
            "job_id": job.job_id, "records": len(items), "request_id": request_id,
        })
        return job

    async def job_for_request(self, space: str, request_id: str) -> Optional["IngestJob"]:
        """The job this request already made, if it made one."""
        check_space(space)
        if not self._able("job_by_request"):
            return None
        return await self.documents.job_by_request(space, request_id)

    async def job(self, space: str, job_id: str) -> "IngestJob":
        """One batch's receipt."""
        check_space(space)
        self._reads_jobs()
        found = await self.documents.get_job(space, job_id)
        if found is None:
            raise NotFound(f"job {job_id!r} in {space!r}")
        return found

    #: The most batches one page may carry.
    MAX_JOBS_PAGE = 100

    async def jobs(self, space: str, limit: int = 20, before: Optional[str] = None) -> list["IngestJob"]:
        """Recent batches, newest first. ``before`` continues from the last
        job of a previous page; a cursor naming no job is refused rather
        than quietly returning the newest page again."""
        check_space(space)
        self._reads_jobs()
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.MAX_JOBS_PAGE:
            raise InvalidInput(f"limit must be an integer from 1 through {self.MAX_JOBS_PAGE}")
        if before is not None and await self.documents.get_job(space, before) is None:
            raise NotFound(f"job {before!r} in {space!r}")
        return await self.documents.list_jobs(space, limit, before)

    async def cancel_job(self, space: str, job_id: str) -> "IngestJob":
        """Stop expecting more of this batch. What has already been read
        stays read and what was searchable stays searchable: cancelling a
        job abandons the work still to come, it does not delete memory."""
        job = await self.job(space, job_id)
        if job.cancelled_at:
            raise InvalidInput(f"job {job_id!r} was already cancelled at {job.cancelled_at}")
        when = self.clock()
        stopped = job.model_copy(update={
            "cancelled_at": when,
            "items": [item if item.consolidated_at else item.model_copy(update={"state": "cancelled"})
                      for item in job.items],
        })
        await self.documents.update_job(stopped)
        await self._emit(space, "ingest_job", {"job_id": job_id, "cancelled": len(stopped.items)})
        return stopped

    async def note_failed(self, space: str, episode_id: int, error: str) -> int:
        """Record that reading this record failed, against the record it
        failed on rather than against the batch. The attempt is counted,
        so a retry that works still shows it took two goes."""
        check_space(space)
        if not callable(getattr(self.documents, "mark_failed", None)):
            return 0
        return await self.documents.mark_failed(space, episode_id, error[:500], self.clock())

    async def note_consolidated(self, space: str, episode_ids: Sequence[int]) -> int:
        """Record that these episodes have been read into claims. Marking
        the same episode twice moves nothing, so a re-run of the extractor
        does not rewrite a receipt that already stands."""
        check_space(space)
        if not callable(getattr(self.documents, "mark_consolidated", None)):
            return 0
        return await self.documents.mark_consolidated(space, list(episode_ids), self.clock())

    # -- a whole space ------------------------------------------------------

    async def _living(self, space: str) -> None:
        """Refuse a space that was deleted: a key left in config must not
        re-create what was erased."""
        when = await self.space_deleted(space)
        if when is not None:
            raise NotFound(f"space {space!r} was deleted at {when}")

    async def space_deleted(self, space: str) -> Optional[str]:
        """When the space was deleted, or None while it lives. A store
        that cannot record a deletion has never deleted one."""
        check_space(space)
        deleted = getattr(self.documents, "space_deleted", None)
        return await deleted(space) if callable(deleted) else None

    async def _space_receipt(self, space: str) -> SpaceReceipt:
        counts = await self.documents.counts(space)
        facts = await self.documents.list_facts(space, include_closed=True)
        links: set[int] = set()
        for fact in facts:
            for link in await self.documents.fact_links(space, fact.fact_id):
                links.add(link.link_id)
        released, kept = await self.blobs.release_space(space, preview=True)
        return SpaceReceipt(
            space=space, episodes=counts.episodes, chunks=counts.chunks, facts=len(facts), links=len(links),
            tombstones=len(await self.documents.list_tombstones(space)),
            events=(await self.events.purge(space, preview=True)) if self.events is not None else 0,
            attachments_released=released, attachments_kept=kept,
        )

    async def space_impact(self, space: str) -> SpaceReceipt:
        """What deleting the space would take with it, with nothing removed."""
        check_space(space)
        await self._living(space)
        return await self._space_receipt(space)

    async def delete_space(self, space: str) -> SpaceReceipt:
        """Remove everything the space holds, in order: attachment holds
        (bytes only when no other space holds them), the records of the
        space with their vectors, then the event trail; mark the space
        deleted so no write re-creates it. Returns the receipt
        ``space_impact`` would have shown, with the counts of the deed."""
        check_space(space)
        await self._living(space)
        if not callable(getattr(self.documents, "delete_space", None)):
            raise InvalidInput("this document store does not implement delete-space")
        preview = await self._space_receipt(space)
        deleted_at = self.clock()
        released, kept = await self.blobs.release_space(space)
        gone = await self.documents.delete_space(space, deleted_at)
        sweep = getattr(self.vectors, "delete_space", None)
        if sweep is not None:
            await sweep(space)
        else:
            await self.vectors.delete(list(gone.chunk_ids))
        events = (await self.events.purge(space)) if self.events is not None else 0
        return preview.model_copy(update={
            "episodes": gone.episodes, "chunks": len(gone.chunk_ids), "facts": gone.facts, "links": gone.links,
            "tombstones": gone.tombstones, "events": events, "attachments_released": released,
            "attachments_kept": kept, "deleted_at": deleted_at,
        })

    async def episode(self, space: str, episode_id: int) -> Episode:
        check_space(space)
        found = await self._episode_or_gone(space, episode_id)
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
        conditions: Mapping[str, object] | None = None,
        candidate_limit: int | None = None,
        rerank: bool = True,
    ) -> RecallResult:
        """``history`` (research experiment 3) also returns, for every
        subject and predicate among the matched facts, the closed facts that
        held before: what changed, when, and why. Off by default; the
        reader gets only what holds at ``as_of`` unless asked for the chain.

        ``kind``, ``source_prefix``, ``since`` and ``until`` narrow by the
        episode's kind, literal source prefix and inclusive timestamps.
        Native SQLite and in-memory lexical lanes apply these before their
        candidate limit, so out-of-scope records cannot crowd out lexical
        matches. Vector candidates and stores that ignore these fields are
        still postfiltered within a bounded window; semantic-only matches
        and such fallback stores have no completeness guarantee. SQLite
        metadata conditions that require a Python recheck can also underfill
        the window. ``tags`` and ``where`` are applied in both lanes.
        ``as_of`` remains fact validity and the lanes' upper time bound.

        ``candidate_limit`` optionally sets each lane's candidate depth
        independently of returned ``limit``; None uses the configured depth
        or the legacy limit-based depth. A host-supplied reranker is optional.
        It sees only verified retained scoped passages within separate count,
        UTF-8 payload and cooperative async time budgets. Scores remain ranking
        signals, not confidence. A failed reranker retains baseline ordering;
        ``rerank=False`` explicitly disables the configured adapter."""
        runtime = RecallRuntime(
            documents=self.documents, vectors=self.vectors, embedder=self.embedder,
            clock=self.clock, emit=self._emit, query_for_evidence=self._query_for_evidence,
            candidate_limit=self.candidate_limit, reranker=self.reranker,
            rerank_limit=self.rerank_limit, rerank_max_bytes=self.rerank_max_bytes,
            rerank_timeout=self.rerank_timeout, contextual_embeddings=self.contextual_embeddings,
            demote_restated=self.demote_restated, similarity_floor=self.similarity_floor,
        )
        return await recall(runtime, space, query, limit, as_of, tags, where, history,
                            kind, source_prefix, since, until, conditions, candidate_limit, rerank)

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
        episode_id = p.get("episode_id")
        if episode_id is not None and not isinstance(episode_id, int):
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
        if episode_id is not None and await self.documents.get_episode(space, int(episode_id)) is None:
            # A link is connector-reported provenance; it must at least point inside this space.
            raise InvalidInput(f"episode {p['episode_id']} is not in {space!r}")
        # Idempotent receipt is the sink's job: the key is unique per space,
        # a retry with the same payload returns the stored event, and the
        # same key with a different payload is a conflict, never a silent drop.
        dedup_key = f"agent:{p['agent']}:{sid}:{source_event_id}" if source_event_id is not None else None
        try:
            return await self._emit(space, "agent", p, dedup_key=dedup_key)  # type: ignore[return-value]
        except DuplicateEvent as e:
            raise InvalidInput(f"source_event_id {source_event_id!r} was already recorded with a different payload (event {e.existing.event_id})") from e

    async def graph(
        self,
        space: str,
        session_id: Optional[str] = None,
        episode_id: Optional[int] = None,
        since: Optional[str] = None,
        limit: int = 400,
        *, fact_limit: int = 400,
    ):
        """The recorded relations around a session or an episode (or the
        latest activity when neither is given). See ``graph.py``."""
        from ..retrieval.activity_graph import build_activity_graph

        check_space(space)
        limit = max(1, min(limit, 2000))
        if type(fact_limit) is not int or not 1 <= fact_limit <= 2000:
            raise InvalidInput('fact_limit must be an integer in 1..2000')
        return await build_activity_graph(self.documents, self.events, space,
            session_id=session_id, episode_id=episode_id, since=since, limit=limit, fact_limit=fact_limit)

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
        returned = {int(i["chunk_id"]) for i in cast(Sequence[_RecallEventItem], recall.payload.get("items", []))}
        if chunk_id not in returned:
            raise InvalidInput(f"chunk {chunk_id} was not returned by recall {recall_event_id}")
        if note is not None and len(note) > 500:
            raise InvalidInput("note must be at most 500 chars")
        return await self._emit(space, "feedback", {
            "recall_event_id": recall_event_id, "chunk_id": chunk_id, "useful": bool(useful), "note": note,
        })  # type: ignore[return-value]

    async def _fact_fits_scope(self, space: str, fact: Fact, scope: TextFilter | None) -> bool:
        return await fact_recall.fact_fits_scope(self.documents, space, fact, scope)

    async def _facts_for_query(self, space: str, query: str, when: str, limit: int = 10,
                               scope: TextFilter | None = None, degraded: list[str] | None = None) -> list[Fact]:
        return await fact_recall.facts_for_query(self.documents, space, query, when, limit, scope, degraded)

    async def _scan_facts_for_query(self, space: str, query: str, when: str, limit: int = 10,
                                  scope: TextFilter | None = None) -> list[Fact]:
        return await fact_recall.scan_facts_for_query(self.documents, space, query, when, limit, scope)

    async def _history_for(self, space: str, facts: Sequence[Fact], when: str, limit: int = 20,
                           scope: TextFilter | None = None) -> list[Fact]:
        return await fact_recall.history_for(self.documents, space, facts, when, limit, scope)

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
        await self._living(space)
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
        await self._living(space)
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
        now = self.clock()
        active = [f for f in await self.documents.list_facts(space, include_closed=False)
                  if f.status == "active" and not f.excluded
                  and f.valid_from <= now and (f.valid_until is None or f.valid_until > now)]
        active.sort(key=lambda f: (-f.confidence, f.fact_id))
        recent = [RecentActivity(e.episode_id, e.content[:200], e.created_at)
                  for e in await self.documents.recent_episodes(space, limit)]
        return Profile(
            static_facts=active[:limit],
            dynamic=[r.excerpt for r in recent],
            recent=recent,
        )

    async def tags(self, space: str) -> dict[str, int]:
        check_space(space)
        return dict(sorted((await self.documents.counts(space)).tags.items()))

    async def pending_derivation(self, space: str) -> int:
        """Groups of active claims a derivation pass has not seen at their
        current membership. In memory only: a new process starts at zero
        seen, sends every group once, and restates rather than repeats."""
        check_space(space)
        return sum(1 for g in derivation_groups(await self.facts(space))
                   if (space, frozenset(f.fact_id for f in g)) not in self._derive_seen)

    async def pending_distillation(self, space: str) -> int:
        """Episodes no claim cites yet. The same definition the distiller
        and memory_pending use, so the three surfaces agree."""
        check_space(space)
        counts = await self.documents.counts(space)
        if counts.episodes == 0:
            return 0
        referenced = {f.source_episode_id for f in await self.documents.list_facts(space, include_closed=True)}
        return sum(1 for e in await self.documents.recent_episodes(space, counts.episodes) if e.episode_id not in referenced)

    async def cited_episode_ids(self, space: str) -> set[int]:
        """The episodes some claim cites, whatever the claim's status."""
        check_space(space)
        return {f.source_episode_id for f in await self.documents.list_facts(space, include_closed=True)
                if f.source_episode_id is not None}

    async def overview(
        self, space: str, *, limit: int = 20, before: int | None = None,
        where: Mapping[str, str] | None = None, kind: str | None = None,
        source_prefix: str | None = None, since: str | None = None, until: str | None = None,
        exclude_session_id: str | None = None, max_records: int = 200,
    ) -> OverviewResult:
        """Return bounded recent source evidence; coverage and continuation are explicit."""
        from ..retrieval.overview import overview

        return await overview(
            self.documents, space, limit=limit, before=before, where=where, kind=kind,
            source_prefix=source_prefix, since=since, until=until,
            exclude_session_id=exclude_session_id, max_records=max_records,
        )

    async def source_page(self, space: str, *, before: Optional[int] = None,
                          limit: int = 25, kind: Optional[str] = None,
                          conditions: Mapping[str, object] | None = None) -> SourcePage:
        """Browse retained sources by descending ID, not relevance or source date.

        Newer inserts are found by restarting the walk. Deleting the boundary
        record does not invalidate the next page. This is not a frozen snapshot.

        With ``conditions``, the walk keeps reading until the page is full,
        rather than filtering one page and handing back what survives. The
        latter turns a page of twenty-five into a page of one and makes the
        page size mean nothing. The walk is bounded, because a filter that
        matches nothing would otherwise read a whole space to prove it, and
        reaching that bound is reported as more to come rather than as the
        end.
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
        if conditions is None:
            rows = await page(space, before, limit + 1, kind)
            more = len(rows) > limit
            return SourcePage(rows[:limit], more, rows[limit-1].episode_id if more else None)

        from ..retrieval.filters import parse_filter

        narrow = parse_filter(conditions)
        kept: list[Episode] = []
        cursor, reads, ended = before, 0, False
        while len(kept) < limit and reads < SOURCE_WALK_READS:
            batch = await page(space, cursor, SOURCE_WALK_PAGE, kind)
            reads += 1
            filled = False
            for episode in batch:
                cursor = episode.episode_id
                if narrow.matches(episode.metadata):
                    kept.append(episode)
                    if len(kept) == limit:
                        filled = True
                        break
            if filled:
                # Stopped on a full page, not on the end of the store: the
                # rest of this batch has not been looked at yet.
                break
            if len(batch) < SOURCE_WALK_PAGE:
                # The store had nothing more to give, so this is the end of
                # the space rather than the end of what was read.
                ended = True
                break
        return SourcePage(kept, not ended, None if ended else cursor)

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
        async for record in archive.export_records(self.documents, space):
            yield record

    async def import_records(self, space: str, records: Iterable[Mapping], *, resurrect: bool = False) -> ImportSummary:
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
        await self._living(space)
        check_space(space)
        runtime = archive.ArchiveRuntime(self.documents, self.clock, self.remember_many)
        return await archive.import_records(runtime, space, records, resurrect=resurrect)


# -- validation helpers -----------------------------------------------------


def _reason(reason: str) -> str:
    reason = reason.strip()
    if not reason or len(reason) > 500:
        raise InvalidInput("reason must be 1..=500 chars")
    return reason


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


def _covers(fact: Fact, instant) -> bool:
    """True when ``instant`` lies in ``[valid_from, valid_until)``."""
    if parse_rfc3339(fact.valid_from) > instant:
        return False
    return fact.valid_until is None or parse_rfc3339(fact.valid_until) > instant


def derivation_groups(facts: Sequence[Fact]) -> list[list[Fact]]:
    """Claims grouped by subject, joined one hop through shared names: a
    claim whose object is another claim's subject puts both subjects in one
    group, so "mark works_at acme" and "acme based_in lisbon" meet. Pure;
    order is by the smallest fact id in each group."""
    parent: dict[str, str] = {}

    def key(name: str) -> str:
        return name.strip().casefold()

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    subjects = {key(f.subject) for f in facts}
    for f in facts:
        find(key(f.subject))
        if key(f.object) in subjects:
            union(key(f.subject), key(f.object))
    grouped: dict[str, list[Fact]] = {}
    for f in facts:
        grouped.setdefault(find(key(f.subject)), []).append(f)
    return sorted((sorted(g, key=lambda f: f.fact_id) for g in grouped.values()), key=lambda g: g[0].fact_id)
