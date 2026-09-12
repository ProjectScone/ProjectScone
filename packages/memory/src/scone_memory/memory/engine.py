"""The engine: remember, recall, forget, and the fact ledger.

Built entirely against the protocols in ``ports``; nothing here knows
whether a dict, MongoDB, or Qdrant is underneath. A lane that fails
during recall is named in ``degraded`` and the other lane still
answers, because a thin answer that says it is thin beats a 500.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Iterable, Mapping, Optional, Sequence, TypedDict, cast

from . import archive, catalog, fact_placement, fact_relationships, fact_review, retention, vector_identity
from .identity import join_match
from .catalog import (Profile as Profile, RecentActivity as RecentActivity,
                      SOURCE_WALK_PAGE as SOURCE_WALK_PAGE, SOURCE_WALK_READS as SOURCE_WALK_READS)
from .fact_review import DECISIONS as DECISIONS, MAX_DECISIONS as MAX_DECISIONS, _reason as _reason
from .fact_placement import Placement as _Placement, _covers as _covers
from .archive import ImportSummary as ImportSummary, _fact_identity as _fact_identity, _rederived as _rederived
from ..ingestion import batch as ingestion_batch, jobs as ingestion_jobs
from ..ingestion.batch import EMBED_BATCH as EMBED_BATCH
from ..ingestion.records import (
    Record as Record, RecoveryReport as RecoveryReport,
    _Pending as _Pending, _DupOf as _DupOf,
    content_hash as content_hash, contextual_prefix as contextual_prefix,
)
from ..retrieval import fact_recall
from ..retrieval.abstention import AbstentionPolicy
from ..retrieval.recall import (RecallRuntime, recall, LANE_DEPTH as LANE_DEPTH,
                                UNFILTERED_DEPTH as UNFILTERED_DEPTH)
from ..retrieval.episode_scope import episode_fits as _fits
from ..retrieval.fact_recall import FACT_SCOPE_CACHE_LIMIT as FACT_SCOPE_CACHE_LIMIT
from ..retrieval.overview import OverviewResult
from ..retrieval.reranking import Reranker, validate_candidate_limit, validate_rerank_options
from ..ingestion.chunker import DEFAULT_TARGET
from ..core.validation import (
    entity_key as entity_key,
    many_valued_predicates,
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
from ..core.errors import Conflict, InvalidInput, NotFound
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
    SourcePage,
    TextFilter,
    VectorIndex,
)
from ..core.timeutil import format_rfc3339, now_rfc3339, parse_rfc3339
from ..entities.service import EntityService

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

MAX_EXTERNAL_PAYLOAD = 4096
AGENTS = ("claude-code", "codex", "other")
AGENT_EVENTS = ("session_start", "prompt", "response", "tool_use", "tool_result", "stop", "session_end")
MAX_AGENT_TEXT = 65_536


@dataclass
class Replaced:
    """What ``replace`` did: the record as stored (or found), the outcome,
    and the receipt for the episode that went, if one did."""

    added: Added
    outcome: str
    replaced: Optional[ForgetReceipt] = None


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
        code_aware: bool = True,
        similarity_floor: Optional[float] = None,
        demote_restated: bool = True,
        blobs: Optional[BlobStore] = None,
        candidate_limit: int | None = None,
        reranker: Reranker | None = None,
        rerank_limit: int = 32,
        rerank_max_bytes: int = 64000,
        rerank_timeout: float = 1.0,
        many_valued: Iterable[str] = (),
        abstention: AbstentionPolicy | None = None,
        profile_policy: "catalog.ProfilePolicy | None" = None,
    ) -> None:
        if similarity_floor is not None and not -1.0 <= similarity_floor <= 1.0:
            raise InvalidInput("similarity_floor must be a cosine similarity in [-1, 1]")
        if abstention is not None and not abstention.fits(embedder.id, embedder.dim):
            raise InvalidInput(_other_scale(abstention, embedder.id, embedder.dim))
        #: Whether a source stored under a name that says it is code is cut
        #: at its declarations. Names say it, never the content: a note that
        #: quotes code is prose.
        self.code_aware = code_aware
        #: The measured floor this engine abstains by, when one was given.
        self.abstention = abstention
        #: Which claims a profile is made of; by default, all of them.
        self.profile_policy = profile_policy or catalog.ProfilePolicy()
        similarity_floor = abstention.floor if abstention is not None else similarity_floor
        self.candidate_limit = validate_candidate_limit(candidate_limit)
        validate_rerank_options(rerank_limit, rerank_max_bytes, rerank_timeout)
        if reranker is not None and not callable(getattr(reranker, "rerank", None)):
            raise InvalidInput("reranker must provide an async rerank method")
        self.reranker = reranker
        #: Predicates configured to hold many values at once; every other
        #: predicate holds one value at a time. Named, never inferred.
        self.many_valued = many_valued_predicates(many_valued)
        self.rerank_limit = rerank_limit
        self.rerank_max_bytes = rerank_max_bytes
        self.rerank_timeout = rerank_timeout
        self.documents = documents
        #: Every write goes through the index's writer check when it keeps one.
        self.vectors = vector_identity.guard(vectors, lambda: vector_identity.writer_of(self))
        self.embedder = embedder
        #: Where an attachment's bytes live. In memory unless a store is
        #: given, so an engine with no configured blob directory keeps
        #: nothing across a restart rather than writing somewhere unasked.
        self.blobs = blobs if blobs is not None else InMemoryBlobStore()
        self._closed = False
        #: Whether this engine's vectors can be compared with the stored
        #: ones; settled by open(). See memory.vector_identity.
        self.vector_identity: vector_identity.VectorIdentity | None = None
        self.max_attachment_bytes = MAX_ATTACHMENT_BYTES
        self.chunk_target = chunk_target
        #: Each space's entity projection, held between graph requests.
        self.entities = EntityService(self)
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
        self.vector_identity = await vector_identity.settle(self)
        await self.recover()
        return self

    @property
    def vector_block(self) -> str | None:
        """Why stored vectors must not be compared with this engine's, as of the
        last check, or None."""
        return None if self.vector_identity is None else self.vector_identity.blocked

    async def check_vectors(self) -> vector_identity.VectorIdentity:
        """Read who wrote the stored vectors now. Another process may have
        rebuilt or written them since this engine opened."""
        self.vector_identity = await vector_identity.observe(self)
        return self.vector_identity

    async def reembed_vectors(self) -> vector_identity.ReembedReport:
        """Re-embed every stored chunk with this engine's embedder and record it
        as the writer, which turns a disabled vector lane back on."""
        try:
            report = await vector_identity.rebuild(self)
        except BaseException:
            await self.check_vectors()
            raise
        self.vector_identity = vector_identity.VectorIdentity(
            "rebuilt", vector_identity.writer_of(self), vector_identity.writer_of(self))
        return report

    async def adopt_vector_identity(self) -> vector_identity.VectorIdentity:
        """Vouch that vectors stored before writers were recorded came from this
        engine's embedder and settings. The declaration is recorded as such."""
        self.vector_identity = await vector_identity.declare(self)
        return self.vector_identity

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
        await self.entities.aclose()
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
            self._embed_text, self._emit, code_aware=self.code_aware,
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

    def _retention_runtime(self) -> retention.RetentionRuntime:
        return retention.RetentionRuntime(
            self.documents, self.vectors, self.blobs, self.events, self.clock, self._emit,
            self._episode_or_gone, self.impact, self.forget, self._living,
            self.space_deleted, self._space_receipt,
        )

    async def _episode_or_gone(self, space: str, episode_id: int) -> Episode:
        """The episode, or Gone when a tombstone says it was forgotten, or
        NotFound when the id never meant anything here."""
        return await retention.episode_or_gone(self._retention_runtime(), space, episode_id)

    async def tombstone(self, space: str, episode_id: int) -> Optional[Tombstone]:
        """The record that an episode was forgotten, or None."""
        return await retention.tombstone(self._retention_runtime(), space, episode_id)

    async def doctor(self, space: str) -> DoctorReport:
        """What references what across the space's stores, read only: chunks
        whose episode is gone, vectors whose chunk is gone, facts citing a
        forgotten or an unknown episode, links with a missing end, held
        attachments no episode carries. A store that cannot be walked is
        named in not_inspected rather than reported clean."""
        return await retention.doctor(self._retention_runtime(), space)

    async def expire(self, space: str, policy: Mapping[str, float], *, limit: int = 100,
                     dry_run: bool = False) -> ExpiryReport:
        """Forget the episodes a retention policy no longer keeps: for each
        kind in ``policy``, those whose own time is more than that many
        days before this engine's clock, oldest first, at most ``limit``
        in one pass. Facts never expire; the claims that cited a forgotten
        episode stand. ``dry_run`` reports and forgets nothing."""
        return await retention.expire(self._retention_runtime(), space, policy, limit=limit, dry_run=dry_run)

    async def impact(self, space: str, episode_id: int) -> ForgetReceipt:
        """What forgetting the episode would take with it and leave, with
        nothing removed. The claims, links and kept restatements that cite
        it are reported, not closed: a source being gone is a fact about
        the evidence."""
        return await retention.impact(self._retention_runtime(), space, episode_id)

    async def forget(self, space: str, episode_id: int) -> ForgetReceipt:
        """Remove the episode, its chunks and vectors, and release the
        attachments nothing else carries; return the receipt that
        ``impact`` would have shown. Claims and links stand."""
        return await retention.forget(self._retention_runtime(), space, episode_id)

    # -- ingest jobs ---------------------------------------------------------

    def _jobs_runtime(self) -> ingestion_jobs.JobRuntime:
        return ingestion_jobs.JobRuntime(
            self.documents, self.clock, self._emit, self._living, self._able,
            self._keeps_jobs, self._reads_jobs, self.remember_many, self.record_job,
            self.job, self.MAX_JOBS_PAGE,
        )

    def _able(self, *methods: str) -> bool:
        return ingestion_jobs.able(self.documents, *methods)

    #: What a store must implement to record a batch, and to read one back.
    #: They are separate because a store may do one and not the other, and
    #: advertising the wrong one turns a refusal into a server error.
    RECORDS_JOBS = ingestion_jobs.RECORDS_JOBS
    READS_JOBS = ingestion_jobs.READS_JOBS

    def _keeps_jobs(self) -> None:
        """Recording jobs is a store's choice: one that cannot keep them
        says so rather than pretending a batch was never tracked."""
        ingestion_jobs.keeps_jobs(self._able, self.RECORDS_JOBS)

    def _reads_jobs(self) -> None:
        """Reading a job back is a separate choice from recording one."""
        ingestion_jobs.reads_jobs(self._able, self.READS_JOBS)

    async def ingest_batch(self, space: str, records: Iterable[Record], *,
                           request_id: Optional[str] = None) -> "IngestJob":
        """Ingest a batch and keep the receipt of what it became.

        Two receipts per record, never one: ``searchable_at`` is set here,
        because chunks and vectors land with the batch, and
        ``consolidated_at`` only when a model has read claims out of the
        record, which happens later and may never happen. A request id
        makes a retry the same job rather than a second one."""
        return await ingestion_jobs.ingest_batch(self._jobs_runtime(), space, records, request_id=request_id)

    async def record_job(self, space: str, added: Sequence[Added], *,
                         request_id: Optional[str] = None) -> "IngestJob":
        """Keep the receipt for records that have just landed. Separate
        from ingesting them, because a caller may have written the batch
        its own way and still owes the person a receipt."""
        return await ingestion_jobs.record_job(self._jobs_runtime(), space, added, request_id=request_id)

    async def job_for_request(self, space: str, request_id: str) -> Optional["IngestJob"]:
        """The job this request already made, if it made one."""
        return await ingestion_jobs.job_for_request(self._jobs_runtime(), space, request_id)

    async def job(self, space: str, job_id: str) -> "IngestJob":
        """One batch's receipt."""
        return await ingestion_jobs.job(self._jobs_runtime(), space, job_id)

    #: The most batches one page may carry.
    MAX_JOBS_PAGE = ingestion_jobs.MAX_JOBS_PAGE

    async def jobs(self, space: str, limit: int = 20, before: Optional[str] = None) -> list["IngestJob"]:
        """Recent batches, newest first. ``before`` continues from the last
        job of a previous page; a cursor naming no job is refused rather
        than quietly returning the newest page again."""
        return await ingestion_jobs.jobs(self._jobs_runtime(), space, limit=limit, before=before)

    async def cancel_job(self, space: str, job_id: str) -> "IngestJob":
        """Stop expecting more of this batch. What has already been read
        stays read and what was searchable stays searchable: cancelling a
        job abandons the work still to come, it does not delete memory."""
        return await ingestion_jobs.cancel_job(self._jobs_runtime(), space, job_id)

    async def note_failed(self, space: str, episode_id: int, error: str) -> int:
        """Record that reading this record failed, against the record it
        failed on rather than against the batch. The attempt is counted,
        so a retry that works still shows it took two goes."""
        return await ingestion_jobs.note_failed(self._jobs_runtime(), space, episode_id, error)

    async def note_consolidated(self, space: str, episode_ids: Sequence[int]) -> int:
        """Record that these episodes have been read into claims. Marking
        the same episode twice moves nothing, so a re-run of the extractor
        does not rewrite a receipt that already stands."""
        return await ingestion_jobs.note_consolidated(self._jobs_runtime(), space, episode_ids)

    # -- a whole space ------------------------------------------------------

    async def _living(self, space: str) -> None:
        """Refuse a space that was deleted: a key left in config must not
        re-create what was erased."""
        return await retention.living(self._retention_runtime(), space)

    async def space_deleted(self, space: str) -> Optional[str]:
        """When the space was deleted, or None while it lives. A store
        that cannot record a deletion has never deleted one."""
        return await retention.space_deleted(self._retention_runtime(), space)

    async def _space_receipt(self, space: str) -> SpaceReceipt:
        return await retention.space_receipt(self._retention_runtime(), space)

    async def space_impact(self, space: str) -> SpaceReceipt:
        """What deleting the space would take with it, with nothing removed."""
        return await retention.space_impact(self._retention_runtime(), space)

    async def delete_space(self, space: str) -> SpaceReceipt:
        """Remove everything the space holds, in order: attachment holds
        (bytes only when no other space holds them), the records of the
        space with their vectors, then the event trail; mark the space
        deleted so no write re-creates it. Returns the receipt
        ``space_impact`` would have shown, with the counts of the deed."""
        try:
            return await retention.delete_space(self._retention_runtime(), space)
        finally:
            self.entities.forget(space)

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
        graph_boost: bool = False,
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
        ``rerank=False`` explicitly disables the configured adapter.

        ``graph_boost`` adds the entity lane: passages naming the question's
        entities or their neighbours in the knowledge graph. The projection
        is built for it within the graph budget (and kept); if it is not
        ready in time the other lanes answer and ``degraded`` says so."""
        if self.vector_identity is not None:
            await self.check_vectors()
        projection = None
        unavailable = None
        notes: list[str] = []
        if graph_boost:
            from ..entities.read import load_projection
            from ..entities.service import ProjectionBuilding

            try:  # an invalid moment degrades the lane here, and recall then refuses it as always
                projection, read = await load_projection(self, space, mode="current", as_of=as_of)
                capped = read.get("reasons")
                if isinstance(capped, list) and capped:
                    notes.append("entity: graph_read_capped " + " ".join(map(str, capped)))
            except ProjectionBuilding:
                unavailable = "projection_building"
            except Exception as error:  # noqa: BLE001 - the optional lane degrades, recall answers
                unavailable = f"{type(error).__name__}: {error}"
        runtime = RecallRuntime(
            documents=self.documents, vectors=self.vectors, embedder=self.embedder,
            clock=self.clock, emit=self._emit, query_for_evidence=self._query_for_evidence,
            candidate_limit=self.candidate_limit, reranker=self.reranker,
            rerank_limit=self.rerank_limit, rerank_max_bytes=self.rerank_max_bytes,
            rerank_timeout=self.rerank_timeout, contextual_embeddings=self.contextual_embeddings,
            demote_restated=self.demote_restated, similarity_floor=self.similarity_floor,
            floor_dim=self.abstention.dim if self.abstention is not None else None,
            vector_block=self.vector_block,
        )
        return await recall(runtime, space, query, limit, as_of, tags, where, history,
                            kind, source_prefix, since, until, conditions, candidate_limit, rerank,
                            graph_boost=graph_boost, entity_projection=projection, entity_unavailable=unavailable,
                            entity_notes=notes)

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

    def _relationships_runtime(self) -> fact_relationships.FactRelationshipsRuntime:
        return fact_relationships.FactRelationshipsRuntime(self.documents, self.clock, self._emit,
            self._living, self._assert_placed, self.link_facts, self._depends_on)

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
        return await fact_relationships.assert_fact(self._relationships_runtime(), space, subject, predicate, object,
            valid_from=valid_from, confidence=confidence, source_episode_id=source_episode_id,
            origin=origin, proposed=proposed, quote=quote, extends=extends, derived_from=derived_from)

    async def link_facts(
        self, space: str, from_fact: int, to_fact: int, kind: str, *,
        source_episode_id: Optional[int] = None, quote: Optional[str] = None,
    ) -> FactLink:
        """Relate two facts of one space: ``from_fact`` extends / is derived
        from / contradicts / supports ``to_fact``. The same link twice is one
        link. A quote must sit in the source episode it names. A dependency
        (extends, derived_from) that would close a cycle is refused."""
        return await fact_relationships.link_facts(self._relationships_runtime(), space, from_fact, to_fact, kind,
            source_episode_id=source_episode_id, quote=quote)

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
        return await fact_relationships.depends_on(self.documents, space, start, target)

    def _placement_runtime(self) -> fact_placement.FactPlacementRuntime:
        return fact_placement.FactPlacementRuntime(self.documents, self.clock, self._emit,
            self._place, self._truncate, self.many_valued)

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
        links: Sequence[tuple[str, int]] = (),
    ) -> Fact:
        """Place a claim using the temporal ledger component."""
        return await fact_placement.assert_placed(self._placement_runtime(), space, subject, predicate, object,
            valid_from=valid_from, confidence=confidence, source_episode_id=source_episode_id,
            origin=origin, proposed=proposed, quote=quote, links=links)

    async def _place(self, space: str, subject: str, predicate: str, object: str, start: str, exclude_id: Optional[int] = None) -> "_Placement":
        return await fact_placement.place(self.documents, space, subject, predicate, object, start, exclude_id,
                                          many_valued=predicate in self.many_valued)

    async def _truncate(self, covering: list[Fact], start: str, by_fact_id: int) -> None:
        await fact_placement.truncate(self.documents, covering, start, by_fact_id)

    def _review_runtime(self) -> fact_review.FactReviewRuntime:
        return fact_review.FactReviewRuntime(
            self.documents, self.clock, self._emit, self._place, self._truncate,
            self._proposed, self.approve, self.decline, self.exclude, self._decide_one,
        )

    async def approve(self, space: str, fact_id: int, actor: Optional[str] = None) -> Fact:
        """A person accepts a proposed fact: it enters the ledger exactly as
        an assertion at its own valid_from would, truncating what it
        covers and bounded by what starts later. A proposal that restates
        a fact already held is marked declined as a duplicate and the held
        fact is returned."""
        return await fact_review.approve(self._review_runtime(), space, fact_id, actor=actor)

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
        return await fact_review.decide(self._review_runtime(), space, decision, fact_ids, reason=reason, actor=actor, expect_revision=expect_revision)

    async def _decide_one(
        self, space: str, decision: str, fact_id: int, reason: Optional[str], actor: Optional[str]
    ) -> DecisionOutcome:
        return await fact_review.decide_one(self._review_runtime(), space, decision, fact_id, reason, actor)

    async def decline(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        """A person rejects a proposed fact. It never held; the reason is kept."""
        return await fact_review.decline(self._review_runtime(), space, fact_id, reason, actor=actor)

    async def _proposed(self, space: str, fact_id: int) -> Fact:
        return await fact_review.proposed(self._review_runtime(), space, fact_id)

    async def exclude(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        """Suppress a ledger fact from recall without touching its interval
        or its history. The third operation beside close (it stopped
        holding) and forget (the data is gone)."""
        return await fact_review.exclude(self._review_runtime(), space, fact_id, reason, actor=actor)

    async def include(self, space: str, fact_id: int, actor: Optional[str] = None) -> Fact:
        """Undo exclude."""
        return await fact_review.include(self._review_runtime(), space, fact_id, actor=actor)

    async def close_fact(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        return await fact_review.close_fact(self._review_runtime(), space, fact_id, reason, actor=actor)

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
        return await catalog.facts(self.documents, space, include_closed, as_of, status, include_excluded)

    # -- overviews --------------------------------------------------------

    async def profile(self, space: str, limit: int = 10) -> Profile:
        return await catalog.profile(self, space, limit, policy=self.profile_policy)

    async def tags(self, space: str) -> dict[str, int]:
        return await catalog.tags(self.documents, space)

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
        return await catalog.pending_distillation(self.documents, space)

    async def cited_episode_ids(self, space: str) -> set[int]:
        """The episodes some claim cites, whatever the claim's status."""
        return await catalog.cited_episode_ids(self.documents, space)

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
        return await catalog.source_page(self.documents, space, before=before, limit=limit, kind=kind,
            conditions=conditions, walk_page=SOURCE_WALK_PAGE, walk_reads=SOURCE_WALK_READS)

    async def episodes(self, space: str, where: Mapping[str, str], limit: Optional[int] = None) -> list[Episode]:
        """The episodes whose metadata matches every ``where`` pair, oldest
        first by (created_at, episode_id); with ``limit``, the newest N of
        them in that same order. This is a walk over the space's episodes,
        fine for a session's turns, not a query language."""
        return await catalog.episodes(self.documents, space, where, limit)

    async def scopes(self, space: str) -> dict[str, dict[str, int]]:
        """Episode counts per metadata key and value: which users, agents
        and sessions have memory here. Walks the episodes, which is fine
        for an overview and avoids asking every store for a new query."""
        return await catalog.scopes(self.documents, space)

    async def revision(self, space: str) -> int:
        """The space's write counter. A page that renders a list can record
        the revision it read at and tell later whether the space moved."""
        check_space(space)
        return await self.documents.revision(space)

    async def status(self, space: str) -> Status:
        return await catalog.status(self.documents, space, identity=lambda: {
            "embedder": self.embedder.id, "document_store": self.documents.name, "vector_index": self.vectors.name,
            "abstention": self.abstention.record() if self.abstention is not None else None,
        })

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


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


def derivation_groups(facts: Sequence[Fact]) -> list[list[Fact]]:
    """Claims grouped by subject, joined one hop through shared names: a
    claim whose object names another claim's subject (``join_match``) puts
    both subjects in one group, so "mark works_at acme" and "acme based_in
    lisbon" meet. Pure; order is by the smallest fact id in each group."""
    parent: dict[str, str] = {}

    key = entity_key

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

    subjects: dict[str, set[str]] = {}
    for f in facts:
        subjects.setdefault(key(f.subject), set()).add(f.subject)
    for f in facts:
        find(key(f.subject))
        if any(join_match(f.object, named) for named in subjects.get(key(f.object), ())):
            union(key(f.subject), key(f.object))
    grouped: dict[str, list[Fact]] = {}
    for f in facts:
        grouped.setdefault(find(key(f.subject)), []).append(f)
    return sorted((sorted(g, key=lambda f: f.fact_id) for g in grouped.values()), key=lambda g: g[0].fact_id)


def _other_scale(policy: AbstentionPolicy, embedder_id: str, dim: int) -> str:
    """Why a floor cannot be carried from one embedder to another."""
    return (f"the abstention policy was measured with embedder {policy.embedder_id} "
            f"({policy.dim}-d); this engine embeds with {embedder_id} ({dim}-d), and one "
            f"embedder's similarities say nothing about another's")
