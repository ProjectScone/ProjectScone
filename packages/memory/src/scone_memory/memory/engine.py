"""The engine: remember, recall, forget, and the fact ledger.

Built entirely against the protocols in ``ports``; nothing here knows
whether a dict, MongoDB, or Qdrant is underneath. A lane that fails
during recall is named in ``degraded`` and the other lane still
answers, because a thin answer that says it is thin beats a 500.
"""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass
from functools import partial
from typing import (TYPE_CHECKING, AsyncIterator, Callable, Iterable, Literal, Mapping, Optional,
                    Sequence, TypedDict, cast)

from . import (archive, catalog, entity_merging, fact_placement, fact_relationships, fact_review, file_claims, retention,
               source_keys, vector_identity)
from .identity import join_match
from ..entities.merges import is_decision
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
from ..retrieval import fact_recall, fusion
from ..entities.meanings import RelationMeanings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..ingestion.code_graph import Resolve
    from ..ingestion.embedding_cache import EmbeddingCache
    # Type-only: the vocabulary store needs cryptography, and a base
    # install promises pydantic alone. Importing it here would make
    # `import scone_memory` fail wherever that extra is absent.
    from ..entities.vocabulary_store import VocabularyStore
    from ..core.models import Chunk
    from ..retrieval.feedback_prior import PriorTerms
    from ..retrieval.lessons import Lessons
    from ..ingestion.chunk_questions import QuestionLaneReport
    from ..providers.llm import ChatModel
    from ..retrieval.summary_traverse import Traversal
from ..retrieval.abstention import AbstentionPolicy
from ..retrieval.recall import (RecallRuntime, SummaryExpander, recall, LANE_DEPTH as LANE_DEPTH,
                                UNFILTERED_DEPTH as UNFILTERED_DEPTH)
from ..retrieval.episode_scope import episode_fits as _fits
from ..retrieval.image_lane import (ImageLane, records_writer, reembed_images as _reembed_images,
                                   remove_forgotten, writer_block)
from ..retrieval.fact_recall import FACT_SCOPE_CACHE_LIMIT as FACT_SCOPE_CACHE_LIMIT
from ..retrieval.overview import OverviewResult
from ..retrieval.synonyms import Synonyms

#: The vector lane's default voice when the embedder is a hash of tokens (see MemoryEngine).
HASHED_VECTOR_WEIGHT = 0.01
from ..retrieval.reranking import Reranker, validate_candidate_limit, validate_rerank_options
from ..ingestion.chunker import DEFAULT_TARGET
from ..core import extracted, forget_after
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
from ..core.errors import Conflict, Gone, InvalidInput, NotFound
from ..backends.blobs import BlobStore, InMemoryBlobStore
from ..core.models import (
    Added,
    BulkForgetReport,
    Attachment,
    BatchDecision,
    DecisionOutcome,
    Episode,
    EpisodeKind,
    Fact,
    FactLink,
    DoctorReport,
    ExpiryReport,
    ForgetDueReport,
    ForgetReceipt,
    ImageRebuildReport,
    ForgetStatus,
    IngestJob,
    SpaceReceipt,
    Tombstone,
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
    EmbeddingCheckpoint,
    ImageEmbedder,
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
    "application/octet-stream", "text/html", "application/xml", "text/xml", "application/x-ndjson",
    "text/tab-separated-values", "message/rfc822", "application/mbox", "application/rtf", "application/vnd.ms-outlook",
    "application/msword", "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # The macro-enabled, template and slideshow twins of the three above:
    # the same packages, read by the same readers.
    "application/vnd.ms-word.document.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    "application/vnd.ms-word.template.macroEnabled.12",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
    "application/vnd.ms-excel.template.macroEnabled.12",
    "application/vnd.ms-powerpoint.presentation.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.presentationml.template",
    "application/vnd.ms-powerpoint.template.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    "application/vnd.ms-powerpoint.slideshow.macroEnabled.12",
    "application/vnd.oasis.opendocument.text", "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation", "application/epub+zip",
    "application/hwp+zip",
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
    #: Extracted claims the replaced episode grounded that the new content
    #: no longer makes, closed. None when the store cannot read claims by
    #: episode, in which case nothing was closed.
    claims_closed: Optional[int] = 0
    #: The replaced episode grounded more claims than one read returns;
    #: some were not examined and may still stand.
    claims_unread: bool = False


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
        structure_aware: bool = False,
        semantic_aware: bool = False,
        heading_context: bool = False,
        embedding_budget: bool = False,
        code_graph: bool = False,
        similarity_floor: Optional[float] = None,
        #: How much newer memory is favoured in fusion: the recency term's
        #: size at age zero and the age at which it halves. The defaults
        #: break near-ties only; zero weight turns the term off.
        recency_weight: float = fusion.W_RECENCY,
        recency_half_life_days: float = fusion.RECENCY_HALF_LIFE_DAYS,
        #: How much recorded feedback moves a candidate in fusion (see
        #: retrieval/feedback_prior.py). Zero, the default, reads no feedback.
        feedback_weight: float = 0.0,
        demote_restated: bool = True,
        demote_superseded: bool = True,
        blobs: Optional[BlobStore] = None,
        candidate_limit: int | None = None,
        reranker: Reranker | None = None,
        rerank_limit: int = 32,
        rerank_max_bytes: int = 64000,
        rerank_timeout: float = 1.0,
        many_valued: Iterable[str] = (),
        relation_meanings: "RelationMeanings | None" = None,
        #: A store holding each space's own relation vocabulary. When one
        #: is attached, what a space holds beats what this process was
        #: configured with, so two processes read one space alike.
        vocabulary: "VocabularyStore | None" = None,
        vocabulary_spaces: "Sequence[str] | None" = None,
        abstention: AbstentionPolicy | None = None,
        profile_policy: "catalog.ProfilePolicy | None" = None,
        profile_bucket_rules: "catalog.BucketRules | None" = None,
        table_context_embeddings: bool = False,
        embedding_cache: "EmbeddingCache | None" = None,
        synonyms: "Synonyms | None" = None,
        context_lane: bool = False,
        lexical_stems: bool = True,
        lexical_exact_forms: bool = True,
        vector_weight: Optional[float] = None,
        chunk_tokens: int | None = None,
        chunk_overlap_tokens: int = 0,
        question_lane: bool = False,
        semantic_merge_threshold: float | None = None,
        #: The image lane: an embedder putting images and text queries into
        #: one space, and a vector index of its own for its vectors. Both or
        #: neither; without them a recall asking for the lane is told so.
        image_embedder: "ImageEmbedder | None" = None,
        image_vectors: VectorIndex | None = None,
    ) -> None:
        if image_embedder is not None and image_vectors is None:
            raise InvalidInput("the image lane keeps its vectors in an index of its own; image_vectors is missing")
        if image_vectors is not None and image_embedder is None:
            raise InvalidInput("the image lane embeds images and queries with one model; image_embedder is missing")
        if image_embedder is not None and not isinstance(image_embedder, ImageEmbedder):
            raise InvalidInput("image_embedder must have an id, a dim, embed_images and embed_texts")
        if _same_index(vectors, image_vectors):
            raise InvalidInput("image_vectors must be an index of its own, not the text vectors' index: "
                               "one index holds one embedder's vectors at one width")
        #: The image lane's embedder, or None when the engine has no image lane.
        self.image_embedder = image_embedder
        #: Its vectors, one per stored image, checked against its writer like the text vectors.
        self.image_vectors = (None if image_vectors is None else vector_identity.guard(
            image_vectors, lambda: cast(ImageEmbedder, image_embedder).id))
        #: How the lane keeps other image embedders' vectors out: ``recorded``
        #: when the index records its writer and refuses another (``image_block``);
        #: ``tagged`` when it cannot, so the lane searches only the vectors tagged
        #: with this embedder's id and ignores the rest; None without the lane.
        self.image_writer_check: Literal["recorded", "tagged"] | None = (
            None if self.image_vectors is None else "recorded" if records_writer(self.image_vectors) else "tagged")
        #: Why this engine's image embedder must not write to or compare with
        #: the image index, as the index recorded it when the engine opened; None otherwise.
        self.image_block: str | None = None
        #: Image vectors of forgotten images removed when the engine opened
        #: (left by forgets through an engine without the lane); None without
        #: the lane or when the image index cannot list what it holds.
        self.image_vectors_removed: int | None = None
        if vector_weight is None:
            # A hashed-token embedder ranks by word overlap, badly: a weak
            # echo of the text lane. Measured on LongMemEval-S (the frozen
            # 50 and 100 items outside them), every voice it had in the order
            # cost the fused ranking; at a hundredth it still runs -- the
            # confidence signal, passages the text lane did not find, and at
            # full voice when the text lane brings nothing (recall applies
            # that) -- and on both samples the fused ranking scored as
            # the text lane alone did (benchmarks/northstar-defaults-2026-09-14.results.md;
            # SCONE_VECTOR_WEIGHT=0.25 is the previous default). A real
            # embedder knows things the text lane does not and keeps its
            # full voice.
            vector_weight = HASHED_VECTOR_WEIGHT if embedder.id.startswith("hash-") else 1.0
        if isinstance(vector_weight, bool) or not isinstance(vector_weight, (int, float)) or not 0 < vector_weight <= 4:
            raise InvalidInput("vector_weight must be a number above 0 and at most 4")
        #: The vector lane's voice in rank fusion against the text lane's 1.0,
        #: resolved from the embedder when the caller did not say; on every
        #: recall event as fusion_weights.
        self.vector_weight = float(vector_weight)
        if type(lexical_stems) is not bool:
            raise InvalidInput("lexical_stems must be a boolean")
        #: Whether a query term's family (bills, billing, billed) is searched
        #: by stem prefix in the text lane; the index is untouched.
        self.lexical_stems = lexical_stems
        if type(lexical_exact_forms) is not bool:
            raise InvalidInput("lexical_exact_forms must be a boolean")
        #: With stem prefixes: whether a passage holding the query's own word
        #: in a family has the family weighed at that word's idf (still one
        #: term, counted once). Nothing without ``lexical_stems``. On by
        #: default: on LongMemEval-S it raised MRR on the frozen 50, the 100
        #: items outside them and 100 further items, and lost no R@5
        #: (benchmarks/exact-forms-2026-09-15.results.md).
        self.lexical_exact_forms = lexical_exact_forms
        if type(context_lane) is not bool:
            raise InvalidInput("context_lane must be a boolean")
        #: Whether what each chunk is under is indexed beside its text and
        #: searched as a lane of its own. Stored text never changes.
        self.context_lane = context_lane
        if type(question_lane) is not bool:
            raise InvalidInput("question_lane must be a boolean")
        #: Whether recall searches the questions ``build_chunk_questions``
        #: wrote into the context index, and whether that pass may run.
        self.question_lane = question_lane
        if type(table_context_embeddings) is not bool:
            raise InvalidInput('table_context_embeddings must be a boolean')
        self._table_context_embeddings = table_context_embeddings
        #: Vectors kept by the text they embed (ingestion/embedding_cache.py);
        #: None embeds every chunk of every stored record.
        self.embedding_cache = embedding_cache
        if synonyms is not None and not isinstance(synonyms, Synonyms):
            raise InvalidInput(f"synonyms must be a Synonyms list, not {type(synonyms).__name__}")
        #: The caller's synonym list; the text lane's query gains the other
        #: members of every group a query term is in. None leaves it alone.
        self.synonyms = synonyms
        if similarity_floor is not None and not -1.0 <= similarity_floor <= 1.0:
            raise InvalidInput("similarity_floor must be a cosine similarity in [-1, 1]")
        if abstention is not None and not abstention.fits(embedder.id, embedder.dim):
            raise InvalidInput(_other_scale(abstention, embedder.id, embedder.dim))
        #: What this space's predicates mean to each other: which are
        #: opposites, which read the same both ways, which carry through.
        #: None means the graph holds only what was said.
        self.relation_meanings = relation_meanings
        self.vocabulary = vocabulary
        #: Spaces whose vocabulary is read when this engine opens. A space
        #: not named here falls back to process configuration and says so,
        #: rather than reaching a thread-bound store from a request.
        self.vocabulary_spaces: tuple[str, ...] = tuple(vocabulary_spaces or ("default",))
        #: Whether a source stored under a name that says it is code is cut
        #: at its declarations. Names say it, never the content: a note that
        #: quotes code is prose.
        self.code_aware = code_aware
        self.structure_aware = structure_aware
        #: Whether prose is cut where its subject changes rather than
        #: where the length target lands. Off unless asked for: it decides
        #: what chunks exist, and stored offsets are part of the shared
        #: specification, so a space that already holds chunks cut another
        #: way must not silently start cutting differently.
        self.semantic_aware = semantic_aware
        if semantic_merge_threshold is not None:
            from ..ingestion.semantic_chunks import merge_refused

            refusal = merge_refused(semantic_merge_threshold)
            if refusal is not None:
                raise InvalidInput(refusal)
        #: The similarity at which a semantic cut's neighbouring chunks are
        #: joined again, within the target (ingestion/semantic_chunks.py);
        #: None keeps the first pass's cuts. Read by every semantic cut, the
        #: engine's rule or a record's own ``chunking="semantic"``.
        self.semantic_merge_threshold = semantic_merge_threshold
        #: Whether a chunk is embedded with the headings above it (for code,
        #: its file and declarations). Off unless asked for: it changes
        #: vectors, and a space's existing vectors were made without it.
        self.heading_context = heading_context
        #: Whether the context in front of a chunk is shortened to fit the
        #: embedder's declared window. Off unless asked for: it changes the
        #: vectors of chunks whose context would not fit.
        if embedding_budget:
            window = getattr(embedder, "max_input_tokens", None)
            if isinstance(window, bool) or not isinstance(window, int) or window < 1:
                raise InvalidInput("embedding_budget needs an embedder that declares its input window "
                                   "(max_input_tokens); this one declares none")
        if chunk_tokens is not None or chunk_overlap_tokens:
            from ..ingestion.token_chunks import refused

            reason = (refused(chunk_tokens, chunk_overlap_tokens) if chunk_tokens is not None
                      else "chunk_overlap_tokens needs chunk_tokens: an overlap in tokens is of a target in tokens")
            if reason is not None:
                raise InvalidInput(reason)
        counter = getattr(embedder, "count_tokens", None)
        if (embedding_budget or chunk_tokens is not None) and callable(counter):
            # Tried now: the vector writer's name and the chunk receipt say
            # the tokenizer counted, so it must be able to.
            setting = "embedding_budget" if embedding_budget else "chunk_tokens"
            try:
                counter("")
            except Exception as error:  # noqa: BLE001 - any failure to count refuses the setting, with its reason
                raise InvalidInput(f"{setting} counts with the embedder's tokenizer, which failed: "
                                   f"{type(error).__name__}: {error}") from error
        self.embedding_budget = embedding_budget
        #: The length chunker's target in tokens, packed from whole sentences
        #: (ingestion/token_chunks.py); None cuts at chunk_target characters.
        #: Code, structure, semantic and unit cuts keep the character target.
        self.chunk_tokens = chunk_tokens
        #: Tokens of the chunk before that a token-measured chunk starts with.
        self.chunk_overlap_tokens = chunk_overlap_tokens
        #: Whether remembering a source file also records what it says about
        #: itself — what it defines, imports and calls — as ordinary claims.
        #: Off unless asked for: it writes to the ledger, and a space's owner
        #: decides what goes in their ledger.
        self.code_graph = code_graph
        #: The measured floor this engine abstains by, when one was given.
        self.abstention = abstention
        #: Which claims a profile is made of; by default, all of them.
        self.profile_policy = profile_policy or catalog.ProfilePolicy()
        #: How a profile read in buckets places each claim as static or dynamic.
        self.profile_bucket_rules = profile_bucket_rules or catalog.BucketRules()
        similarity_floor = abstention.floor if abstention is not None else similarity_floor
        self.candidate_limit = validate_candidate_limit(candidate_limit)
        validate_rerank_options(rerank_limit, rerank_max_bytes, rerank_timeout)
        if reranker is not None and not callable(getattr(reranker, "rerank", None)):
            raise InvalidInput("reranker must provide an async rerank method")
        self.reranker = reranker
        #: Predicates configured to hold many values at once; every other
        #: predicate holds one value at a time. Named, never inferred.
        # What a person configures, plus what the framework extracts and
        # knows the shape of: a file's imports are many by nature.
        self.many_valued = many_valued_predicates(many_valued) | extracted.MANY_VALUED
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
        #: Whether recall moves a passage the ledger has retired below the
        #: passage that replaced it. On by default: it costs nothing when
        #: a query matched no fact, and one chain read per fact it did.
        self.demote_superseded = demote_superseded
        #: Experiment 9: with a floor, a recall whose best vector hit sits
        #: below it is flagged low_confidence so a reader can abstain
        #: instead of answering from weak evidence. None (the default) means
        #: no judgement; the floor is chosen from a measured sweep, never
        #: guessed here.
        self.similarity_floor = similarity_floor
        fusion.validate_recency(recency_weight, recency_half_life_days)
        self.recency_weight = float(recency_weight)
        self.recency_half_life_days = float(recency_half_life_days)
        from ..retrieval.feedback_prior import validate_feedback_weight

        validate_feedback_weight(feedback_weight)
        self.feedback_weight = float(feedback_weight)

    async def _emit(self, space: str, kind: str, payload: dict, dedup_key: Optional[str] = None) -> Optional[Event]:
        if self.events is None:
            return None
        return await self.events.append(NewEvent(ts=self.clock(), space=space, kind=kind, payload=payload, dedup_key=dedup_key))

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _query_for_evidence(self, query: str) -> dict:
        if self.record_queries:
            return {"query": query, "query_hashed": False}
        return {"query": self._query_hash(query), "query_hashed": True}

    async def open(self) -> "MemoryEngine":
        from . import space_cleanup

        # Identity settlement may rebuild vectors and call the embedder. Finish
        # accepted space erasures before it can inspect their source content.
        _, pending = await space_cleanup.recover(self._retention_runtime(), 100)
        if pending:
            raise InvalidInput("space cleanup remains; call recover() again before opening the engine")
        await self.vectors.ensure(self.embedder.dim)
        if self.image_vectors is not None:
            await self.image_vectors.ensure(cast(ImageEmbedder, self.image_embedder).dim)
        if self.vocabulary is not None:
            # Read on this thread, which owns the store's connection, and
            # held for the life of the engine. A vocabulary saved later
            # takes effect when the engine is reopened; that contract is
            # stated in entities/vocabulary.py rather than left to be
            # discovered.
            from ..entities.vocabulary import resolve_vocabulary

            for space in self.vocabulary_spaces:
                await resolve_vocabulary(self, space)
        self.vector_identity = await vector_identity.settle(self)
        report = await self.recover()
        if report.space_deletions_pending:
            raise InvalidInput("space cleanup remains; call recover() again before opening the engine")
        if report.retirements_pending:
            raise InvalidInput("source cleanup remains; call recover() again before opening the engine")
        if self.image_vectors is not None:
            self.image_block = await writer_block(self.image_vectors, cast(ImageEmbedder, self.image_embedder).id)
            self.image_vectors_removed = await remove_forgotten(self.documents, self.image_vectors)
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
        if self.embedding_cache is not None:
            # A model can change behind an id that did not; a rebuild is
            # the moment that is said, and nothing kept before it may be
            # served after it.
            self.embedding_cache.clear()
        try:
            report = await vector_identity.rebuild(self)
        except BaseException:
            await self.check_vectors()
            raise
        self.vector_identity = vector_identity.VectorIdentity(
            "rebuilt", vector_identity.writer_of(self), vector_identity.writer_of(self))
        return report

    async def reembed_images(self, space: str, *, limit: int = 100,
                             before: Optional[int] = None) -> ImageRebuildReport:
        """Embed the space's stored images again with this engine's image
        embedder: at most ``limit`` file episodes a pass, newest first, walking
        on from ``before``. On an index that records its writer, a rebuild
        under a new image model refuses the lane until a pass completes with
        no space holding another model's vectors, then records the new model.
        See ``retrieval.image_lane.reembed_images``."""
        try:
            return await _reembed_images(self, space, limit=limit, before=before)
        finally:
            # What the pass left the record as, whether it finished or not.
            if self.image_embedder is not None:
                self.image_block = await writer_block(cast(VectorIndex, self.image_vectors), self.image_embedder.id)

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
        for store in (self.documents, self.vectors, self.image_vectors, self.events, self.blobs, self.embedding_cache):
            closer = getattr(store, "close", None)
            if store is None or not callable(closer):
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - every store must still be asked
                first = first or exc
        if first is not None:
            raise first

    async def recover(self, *, retirement_limit: int = 100, space_deletion_limit: int = 100) -> RecoveryReport:
        """Finish interrupted deletions before repairing interrupted ingestion.

        Whole-space cleanup runs first, bounded by ``space_deletion_limit``.
        At most ``retirement_limit`` source cleanups follow. Both limits accept
        1..1000 and default to 100. A remaining backlog is disclosed and ingestion repair waits
        for a later call. A failing store leaves its intent and raises; callers
        serialize this operation with source writes. A remember marks its
        episode identity in the document store before writing and clears
        the mark once the rows and the vectors are all durable. Anything
        still marked here was cut off somewhere in between: an episode row
        without chunks, or chunks without vectors, or no row at all. Each
        is brought to the complete state (chunks and vectors rebuilt from
        the stored content, which never changed) or, if nothing landed,
        the mark is dropped. Recorded as one "recover" event per open when
        there was anything to do, so the evidence shows it happened."""

        from . import space_cleanup

        space_cleanup.cleanup_limit(space_deletion_limit)
        if type(retirement_limit) is not int or not 1 <= retirement_limit <= 1000:
            raise InvalidInput("retirement_limit must be 1..1000")
        spaces, spaces_pending = await space_cleanup.recover(self._retention_runtime(), space_deletion_limit)
        if spaces_pending:
            return RecoveryReport(spaces_deleted=spaces, space_deletions_pending=True)
        retired, pending = await retention.recover_forgets(self._retention_runtime(), retirement_limit)
        if pending:
            return RecoveryReport(retired=retired, retirements_pending=True, spaces_deleted=spaces)
        report = await ingestion_batch.recover(self._ingestion_runtime())
        report.retired = retired
        report.spaces_deleted = spaces
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
        dedup_key: Optional[str] = None,
        replace: bool = False,
        *, embedding_checkpoint: EmbeddingCheckpoint | None = None,
        chunking: Optional[str] = None,
        chunking_profile: Optional[str] = None,
        semantic_merge_threshold: Optional[float] = None,
        forget_after: Optional[str] = None,
    ) -> Added:
        """One record. ``dedup_key`` names it across writes; ``replace``
        makes a changed record under a known key an update (see
        ``replace``) instead of a duplicate. ``chunking`` names how this
        record is cut (length, code, structure, semantic); None keeps the
        engine's rule. ``chunking_profile`` names a genre (statute, paper,
        manual, qa, resume) whose boundaries structure chunking cuts at.
        ``semantic_merge_threshold`` joins this record's semantic chunks
        again at that similarity, over the engine's own threshold.

        ``forget_after`` schedules the memory to be forgotten: an RFC 3339
        time, a date, or a duration from this engine's clock such as ``30d``
        (``core.forget_after``). A time already past is refused. From that
        time recall does not return it, and ``forget_due`` forgets it."""
        await self._living(space)
        if replace and embedding_checkpoint is not None:
            raise InvalidInput('embedding checkpoints apply to append ingestion, not replacement')
        record = Record(content, kind, source, tuple(tags), created_at, dict(metadata or {}), dedup_key=dedup_key,
                        chunking=chunking, chunking_profile=chunking_profile,
                        semantic_merge_threshold=semantic_merge_threshold, forget_after=forget_after)
        if replace:
            added = (await self.replace(space, record)).added
        else:
            [added] = await self.remember_many(space, [record], embedding_checkpoint=embedding_checkpoint)
        for attachment_id in dict.fromkeys(attachment_ids):
            await self.blobs.link(space, attachment_id, added.episode_id)
        return added

    async def replace(self, space: str, record: Record, *, map_code: bool = True,
                      resolve: "Resolve | None" = None) -> "Replaced":
        """Store a keyed record as the current one under its key. A key
        nobody holds is accepted; the same content again is a duplicate
        and changes nothing; changed content is an update: the episode
        the key named is forgotten (its receipt returned; the claims that
        cited it stand) and the new one stored. Validation, chunking and
        embedding finish before forgetting, so preparation failure leaves
        the old source available. The subsequent forget/store is not an
        atomic swap: a storage failure between them can leave the key empty,
        and its error reports the removed source. Competing writers still
        need caller serialization across the commit phase.

        With the code graph on, the new content's claims are read and the
        replaced episode's claims it no longer makes are closed. A caller
        that reads claims itself, with a resolver this engine does not
        have, passes ``map_code=False`` and does both."""
        await self._living(space)
        check_space(space)
        if not record.dedup_key:
            raise InvalidInput("replace needs a dedup_key: it is the key that names what is being replaced")
        if record.content_hash is not None:
            raise InvalidInput("replace derives its identity from dedup_key; content_hash is only for import")
        started = time.perf_counter()
        runtime = self._ingestion_runtime()
        configuration = (runtime.embedder.id, runtime.embedder.dim, self.contextual_embeddings, self.chunk_target,
                         self.code_aware, self.code_graph, self.structure_aware, self.semantic_aware, self.heading_context,
                         self.embedding_budget, self.chunk_tokens, self.chunk_overlap_tokens,
                         self.semantic_merge_threshold)
        try:
            new = ingestion_batch.validated_record(space, record, self.clock())
            digest = new.content_hash
            prior_tombstone = await self.documents.tombstone_by_hash(space, digest)
            existing = await self.documents.episode_by_hash(space, digest)
            # An overdue memory is as good as gone: the same content again is
            # stored afresh, the overdue one forgotten below as any replaced one.
            if (existing is not None and existing.content == new.content
                    and not forget_after.is_due(existing.metadata, parse_rfc3339(self.clock()))):
                # The schedule the stored episode holds, which this write did not change.
                added = Added(episode_id=existing.episode_id, deduplicated=True, chunks=0, outcome="duplicate",
                              forget_after=existing.metadata.get(forget_after.KEY))
                return Replaced(added=added, outcome="duplicate", replaced=None)
            pending = await ingestion_batch.chunk_record(runtime, new)
            vectors, reused = await ingestion_batch.embed_pending_counted(runtime, space, [pending])
            # Embedding can yield for a long time. Never revoke a different source
            # or recreate one that the user forgot during preparation.
            await self._living(space)
            if (self.embedder is not runtime.embedder or self.documents is not runtime.documents
                    or self.vectors is not runtime.vectors
                    or (self.embedder.id, self.embedder.dim, self.contextual_embeddings, self.chunk_target,
                        self.code_aware, self.code_graph, self.structure_aware,
                        self.semantic_aware, self.heading_context, self.embedding_budget,
                        self.chunk_tokens, self.chunk_overlap_tokens,
                        self.semantic_merge_threshold) != configuration):
                raise InvalidInput("ingestion configuration changed while preparing replacement; retry with current settings")
            current = await self.documents.episode_by_hash(space, digest)
            current_tombstone = await self.documents.tombstone_by_hash(space, digest)
            if ((current.episode_id if current else None) != (existing.episode_id if existing else None)
                    or current_tombstone != prior_tombstone):
                raise InvalidInput("source changed while preparing replacement; inspect its current state and retry")
        except Exception as error:
            await self._emit(space, "remember", {
                "records": 1, "error": f"{type(error).__name__}: {error}", "latency_ms": _ms(started),
            })
            raise
        receipt = None
        if existing is not None:
            receipt = await self.forget(space, existing.episode_id)
        try:
            results: list[Added | _DupOf | None] = [None]
            await ingestion_batch.write_batch(runtime, space, [pending], vectors, results, reused=reused)
            await self.documents.bump_revision(space)
            added = cast(Added, results[0])
            retired = file_claims.Retired(closed=0)
            if self.code_graph and map_code:
                prepared = Record(new.content, kind=new.kind, source=new.source, created_at=new.created_at)
                mapped = await self._map_code(space, [prepared], [added], resolve=resolve)
                if receipt is not None:
                    # The old episode's claims stood through the forget, by
                    # its contract. What the new content no longer says is
                    # closed now, naming the file.
                    retired = await self.close_unstated(
                        space, receipt.episode_id,
                        kept=[(fact.subject, fact.predicate, fact.object) for fact in mapped],
                        reason=f"no longer stated by {new.source or record.dedup_key}", kind="source_changed")
        except Exception as error:
            await self._emit(space, "remember", {
                "records": 1, "error": f"{type(error).__name__}: {error}", "latency_ms": _ms(started),
            })
            if receipt is not None:
                raise InvalidInput(
                    f"replace forgot episode {receipt.episode_id}, then its storage or receipt stage failed "
                    f"({type(error).__name__}); inspect the current record under {record.dedup_key!r} before retrying"
                ) from error
            raise
        await self._emit(space, "remember", {
            "records": 1, "fresh": 1, "deduplicated": 0, "chunks": added.chunks,
            "bytes": len(new.content.encode()), "embedder": runtime.embedder.id, "latency_ms": _ms(started),
        })
        outcome = "updated" if receipt is not None else added.outcome
        added = added.model_copy(update={"outcome": outcome, "replaced": receipt})
        return Replaced(added=added, outcome=outcome, replaced=receipt,
                        claims_closed=retired.closed, claims_unread=retired.unread)

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

    async def remember_many(self, space: str, records: Iterable[Record], *,
                            embedding_checkpoint: EmbeddingCheckpoint | None = None,
                            partial: bool = False) -> list[Added]:
        """Ingest a batch: one embedding call per EMBED_BATCH chunk texts
        instead of one per record, and one revision bump. Outcomes come
        back in input order; a record identical to an earlier one in the
        same batch is deduplicated against it.

        Nothing is written until every vector exists, so an embedder that
        fails leaves no orphan episodes. If a store fails after the first
        write, the episodes written so far are deleted again and the
        error is raised; a batch either lands whole or not at all.

        ``partial`` is for a caller importing from somewhere messy: each
        record is judged on its own, one that cannot be stored comes back
        as ``failed`` with the reason, and the rest are stored. Without it
        a single bad record refuses the whole batch, which is the right
        default — a caller who sent one usually wants to fix it and send
        the lot again, and a half-stored batch nobody asked for is worse
        than a clear refusal.
        """
        await self._living(space)
        check_space(space)
        started = time.perf_counter()
        records = list(records)
        try:
            resolved = await self._remember_many(space, records, embedding_checkpoint=embedding_checkpoint, partial=partial)
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

    @property
    def table_context_embeddings(self) -> bool:
        """Source-table context policy; select it when constructing the engine."""
        return self._table_context_embeddings

    def _ingestion_runtime(self, embedding_checkpoint: EmbeddingCheckpoint | None = None) -> ingestion_batch.IngestionRuntime:
        from ..ingestion.document_outline import document_outline
        context_inputs = None
        if self.table_context_embeddings:
            from ..ingestion.table_context import embedding_inputs
            context_inputs = partial(embedding_inputs, blobs=self.blobs)
        return ingestion_batch.IngestionRuntime(
            self.documents, self.vectors, self.embedder, self.clock, self.chunk_target,
            self._embed_text, self._emit, embedding_checkpoint=embedding_checkpoint,
            embedding_cache=self.embedding_cache, code_aware=self.code_aware,
            structure_aware=self.structure_aware,
            semantic_aware=self.semantic_aware, semantic_merge_threshold=self.semantic_merge_threshold,
            heading_context=self.heading_context,
            embedding_budget=cast(int, getattr(self.embedder, "max_input_tokens")) if self.embedding_budget else None,
            count_tokens=(getattr(self.embedder, "count_tokens", None)
                          if self.embedding_budget or self.chunk_tokens is not None else None),
            chunk_tokens=self.chunk_tokens, chunk_overlap_tokens=self.chunk_overlap_tokens,
            context_inputs=context_inputs, verify_visual=self._verify_visual_record,
            context_lane=self.context_lane,
            document_outline=partial(document_outline, blobs=self.blobs),
            forget_overdue=self.forget,
        )

    async def _verify_visual_record(self, space: str, record: Record, episode_id: int | None) -> None:
        from ..ingestion.visual_document import verify_visual_record
        await verify_visual_record(self, space, record, episode_id)

    async def _remember_many(self, space: str, records: Sequence[Record], *,
                             embedding_checkpoint: EmbeddingCheckpoint | None = None,
                             partial: bool = False) -> list[Added]:
        added = await ingestion_batch.remember_many(self._ingestion_runtime(embedding_checkpoint), space, records,
                                                    partial=partial)
        if self.code_graph:
            await self._map_code(space, records, added)
        return added

    async def _map_code(self, space: str, records: Sequence[Record], added: Sequence[Added], *,
                        resolve: "Resolve | None" = None) -> list[Fact]:
        """What a source file says about itself, recorded as claims like any
        other: quoted from the line they were read on, cited to the episode
        they came from, and marked extracted rather than stated, because
        nobody said them — they were read.

        A file already in the space is not read again: its claims are
        already here, and asserting them again would say the same thing
        twice for no reason."""
        from ..ingestion.code_graph import record_claims
        from ..ingestion.code_resolution import file_resolver

        # A batch of files remembered together can follow relative imports
        # among themselves; a caller that walked a tree passes its own. One
        # file alone is no tree: a resolver over it would decline every
        # relative import, where the reader's own path arithmetic still
        # names a candidate, so a lone file keeps that.
        if resolve is None:
            # Only records the batch actually stored: a record refused by a
            # partial batch has no episode and may not even carry a path.
            sources = [record.source for record, outcome in zip(records, added)
                       if isinstance(record.source, str) and record.source and outcome.episode_id >= 0]
            resolve = file_resolver(sources) if len(sources) > 1 else None
        mapped: list[Fact] = []
        for record, outcome in zip(records, added):
            if outcome.deduplicated or outcome.episode_id < 0 or not record.source:
                continue
            await record_claims(self, space, episode_id=outcome.episode_id, content=record.content,
                                path=record.source, when=record.created_at or self.clock(), resolve=resolve,
                                _facts=mapped)
        return mapped

    async def close_unstated(self, space: str, episode_id: int, *, kept: Sequence[tuple[str, str, str]] = (),
                             reason: str, kind: str = "source_changed") -> "file_claims.Retired":
        """Close the extracted claims an episode grounded that are not in
        ``kept``: the file was replaced by content that no longer says
        them, or removed. What a person stated is never touched here."""
        return await file_claims.close_unstated(self._review_runtime(), space, episode_id, kept=kept,
                                                reason=reason, kind=kind)

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
            self.space_deleted, self._space_receipt, self.exclude, image_vectors=self.image_vectors,
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

    async def forget_matching(self, space: str, *, source_prefix: Optional[str] = None, tags: Sequence[str] = (),
                              conditions: Mapping[str, object] | None = None, kind: Optional[str] = None,
                              limit: int = 100, apply: bool = False, selection: Optional[str] = None,
                              with_claims: Literal["keep", "exclude"] = "keep") -> "BulkForgetReport":
        """Forget what a filter selects: a preview by default, the forgets only
        with the preview's selection digest. See ``memory.bulk_forget``."""
        from .bulk_forget import forget_matching

        return await forget_matching(self, space, source_prefix=source_prefix, tags=tags, conditions=conditions,
                                     kind=kind, limit=limit, apply=apply, selection=selection, with_claims=with_claims)

    async def forget_due(self, space: str, now: Optional[str] = None, *, limit: int = 100, dry_run: bool = False,
                         with_claims: Literal["keep", "exclude"] = "keep",
                         before: Optional[int] = None) -> "ForgetDueReport":
        """Forget the episodes whose ``forget_after`` has come, through the
        ordinary ``forget``: most overdue first, at most ``limit`` in a pass,
        from a walk of at most ``scheduled_forget.MAX_SCANNED`` episodes. The
        report says what went and why, and when either bound bit. ``now``
        may be earlier than this engine's clock, never later. See
        ``memory.scheduled_forget``."""
        from .scheduled_forget import forget_due

        return await forget_due(self, space, now=now, limit=limit, dry_run=dry_run, with_claims=with_claims,
                                before=before)

    async def forget_status(self, space: str, episode_id: int) -> ForgetStatus:
        """Observe retained, pending or completed source removal without writes."""
        return await retention.forget_status(self._retention_runtime(), space, episode_id)

    async def forget(self, space: str, episode_id: int, *,
                     with_claims: retention.ClaimPolicy = "keep") -> ForgetReceipt:
        """Remove the episode, its chunks and vectors, and release the
        attachments nothing else carries; return the receipt that
        ``impact`` would have shown. Claims and links stand, unless
        ``with_claims="exclude"`` takes the claims only this source
        supported out of recall, reversibly; see ``retention.forget``."""
        return await retention.forget(self._retention_runtime(), space, episode_id, with_claims)

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

    async def merge_space(self, space: str, *, into: str, confirm: Optional[str] = None,
                          preview: bool = False,
                          _authorize: Callable[[], None] | None = None) -> "archive.MergeReceipt":
        """Move episodes, claims and retained attachments, then close the source.

        Target tombstones take precedence. Known forgotten-source references
        are omitted and counted, preserving claims without inventing evidence.
        Preview reports verified attachment counts and bytes without writes.

        Quiesce source and destination writers throughout this operation: copy
        and validation use separate storage observations, not an atomic cutover.
        A copy failure leaves the source open for a retry; source deletion keeps
        the existing backend-specific cleanup failure semantics.
        """
        check_space(space)
        check_space(into)
        if space == into:
            raise InvalidInput(f"a space is not merged into itself ({space!r})")
        await self._living(space)
        await self._living(into)
        if not preview and confirm != space:
            raise InvalidInput(
                f"confirm must repeat the space being merged ({space!r}); a whole space does not move "
                f"by accident")
        from .space_transfer import merge
        return await merge(self, space, into, preview=preview, authorize=_authorize)

    async def space_impact(self, space: str) -> SpaceReceipt:
        """What deleting the space would take with it, with nothing removed."""
        return await retention.space_impact(self._retention_runtime(), space)

    async def delete_space(self, space: str) -> SpaceReceipt:
        """Remove everything the space holds, in order: attachment holds
        (bytes only when no other space holds them), the records of the
        space with their vectors, then the event trail; mark the space
        deleted so no write re-creates it. Returns the receipt
        ``space_impact`` would have shown, with the counts of the deed.

        Does **not** clear the space's relation vocabulary: that lives in a
        host-owned store this engine has no handle on, so a space recreated
        under this name would inherit it. The host clears that record."""
        try:
            return await retention.delete_space(self._retention_runtime(), space)
        finally:
            self.entities.forget(space)

    async def episode(self, space: str, episode_id: int) -> Episode:
        """The episode and its attachments. One past its ``forget_after``
        reads as Gone, swept or not; ``impact`` and ``forget`` still reach it."""
        check_space(space)
        found = self._not_overdue(await self._episode_or_gone(space, episode_id))
        carried = await self.blobs.for_episode(space, episode_id)
        return found.model_copy(update={"attachments": tuple(carried)}) if carried else found

    def _not_overdue(self, episode: Episode) -> Episode:
        """The episode, or Gone when its scheduled time has come."""
        if forget_after.is_due(episode.metadata, parse_rfc3339(self.clock())):
            due = episode.metadata[forget_after.KEY]
            raise Gone(f"episode {episode.episode_id} was due to be forgotten at {due}", due)
        return episode

    async def episode_by_key(self, space: str, dedup_key: str) -> Episode:
        """Read a keyed source and its attachment metadata. Unknown keys raise
        NotFound; a key with only a deletion record, or whose source is past
        its ``forget_after``, raises Gone. A changing
        key is retried at most three times before Conflict. The result does
        not reserve the key for a subsequent write or include prior versions."""
        return self._not_overdue(await source_keys.episode_by_key(self.documents, self.blobs, space, dedup_key))

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
        fusion: str = "rank",
        lanes: Sequence[str] = ("vector", "text"),
        require: Sequence[str] = (),
        exclude: Sequence[str] = (),
        diversity: Optional[float] = None,
        lessons: bool = False,
        expand_summaries: Optional[str] = None,
        expand_max_chunks: Optional[int] = None,
        image_lane: bool = False,
    ) -> RecallResult:
        """``image_lane`` adds the image lane: stored images nearest the query
        in the image embedder's space, fused by rank (``retrieval.image_lane``).
        Off unless asked for; an engine without an image embedder answers from
        the other lanes and ``degraded`` says so.

        ``lessons`` puts beside each returned passage what people said about it
        (``engine.lessons``), leaving the order as it was.

        ``expand_summaries`` (``follow`` or ``replace``) puts after each stored
        summary among the returned passages, or in its place, the chunks its
        citations rest on -- only those, never a forgotten document's -- each
        saying in ``via_summary`` which summary it came through; at most
        ``expand_max_chunks`` are added (``summary_expand``), so the answer can
        hold more than ``limit``, and ``expanded`` says what was added, refused
        and cut.

        ``history`` (research experiment 3) also returns, for every
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

        ``lanes`` names the lanes to run, "vector" and "text" by default. A
        lane not named is not run, and the result's ``lanes`` names the
        ones that answered.

        ``require`` and ``exclude`` are phrases a passage must all hold, or
        must hold none of, matched as whole words; they are checked across
        the fused candidates before the limit, and ``phrases`` says what
        they dropped and whether the answer came back short.

        ``diversity``, a weight from 0 to 1, fills the answer's places by
        maximal marginal relevance, so near-copies do not take several;
        ``diversity`` on the result says how.

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
            demote_restated=self.demote_restated, demote_superseded=self.demote_superseded,
            similarity_floor=self.similarity_floor,
            recency_weight=self.recency_weight, recency_half_life_days=self.recency_half_life_days,
            floor_dim=self.abstention.dim if self.abstention is not None else None,
            vector_block=self.vector_block,
            synonyms=self.synonyms,
            context_lane=self.context_lane,
            question_lane=self.question_lane,
            lexical_stems=self.lexical_stems,
            lexical_exact_forms=self.lexical_exact_forms,
            vector_weight=self.vector_weight,
            feedback_prior=self._feedback_prior if self.feedback_weight > 0 else None,
            image=None if self.image_vectors is None else ImageLane(cast(ImageEmbedder, self.image_embedder),
                                                                   self.image_vectors),
        )
        if type(lessons) is not bool:
            raise InvalidInput("lessons must be a boolean")
        summaries: Optional[SummaryExpander] = None
        if expand_summaries is not None:
            from ..retrieval.summary_expand import DEFAULT_MAX_CHUNKS, Expanded, check_expansion, expand_summaries as expand

            # Before the search, so a mistaken request spends no work and logs no recall.
            cap = DEFAULT_MAX_CHUNKS if expand_max_chunks is None else expand_max_chunks
            check_expansion(expand_summaries, cap)
            mode = cast("Literal['replace', 'follow']", expand_summaries)

            async def expand_found(items: Sequence[RecallItem], required: Sequence[str], excluded: Sequence[str],
                                   scope: TextFilter) -> Expanded:
                return await expand(self, space, items, mode=mode, max_chunks=cap, require=required, exclude=excluded,
                                    scope=scope)

            # Run inside recall, before its event is written and before lessons,
            # so the event lists the chunks a summary brought and they are read for lessons too.
            summaries = expand_found
        elif expand_max_chunks is not None:
            raise InvalidInput("expand_max_chunks is a cap on expand_summaries; ask for expand_summaries with it")
        result = await recall(runtime, space, query, limit, as_of, tags, where, history,
                            kind, source_prefix, since, until, conditions, candidate_limit, rerank,
                            graph_boost=graph_boost, fusion_mode=fusion, entity_projection=projection,
                            entity_unavailable=unavailable,
                            entity_notes=notes, lanes=lanes,
                            require=require, exclude=exclude, diversity=diversity, summaries=summaries,
                            image_lane=image_lane)
        if lessons:
            from ..retrieval.lessons import MAX_FEEDBACK_EVENTS, read_lessons, read_summary

            found = await read_lessons(self, space, chunk_ids=[item.chunk_id for item in result.items],
                                       max_events=MAX_FEEDBACK_EVENTS)
            result.lessons_read = read_summary(found)
            result.items = [item.model_copy(update={"lessons": found.lessons[item.chunk_id].record()})
                            if item.chunk_id in found.lessons else item for item in result.items]
        return result

    async def tree_recall(self, space: str, query: str, *, limit: Optional[int] = None, branching: Optional[int] = None,
                          max_depth: Optional[int] = None, text: bool = False, episode_ids: Optional[Sequence[int]] = None,
                          kind: Optional[str] = None, source_prefix: Optional[str] = None, tags: Sequence[str] = (),
                          since: Optional[str] = None, until: Optional[str] = None) -> "Traversal":
        """The chunks a descent of the stored summary trees in scope reaches:
        from each document's top summaries, the ``branching`` best at every
        step, down to chunks, each saying in ``via_tree`` the path that led
        there. No model is called. Unset bounds take the module's defaults;
        see ``summary_traverse``."""
        from ..retrieval.summary_traverse import DEFAULT_BRANCHING, DEFAULT_LIMIT, MAX_DEPTH, traverse_summaries

        return await traverse_summaries(self, space, query, limit=DEFAULT_LIMIT if limit is None else limit,
                                        branching=DEFAULT_BRANCHING if branching is None else branching,
                                        max_depth=MAX_DEPTH if max_depth is None else max_depth, text=text,
                                        episode_ids=episode_ids, kind=kind, source_prefix=source_prefix, tags=tags,
                                        since=since, until=until)

    async def record_turn(self, space: str, *, session_id: str, turn_id: str, mode: str,
                          latency_ms: Mapping[str, float]) -> Optional[Event]:
        """Append a ``conversation_turn`` event: how long one turn of a
        text or voice conversation took to prepare memory, to reach its
        first token, its first audio, and its end, in milliseconds from
        the moment the question was heard. Only the moments that came are
        given; nothing is invented. None with no event log attached."""
        check_space(space)
        if self.events is None:
            return None
        for name, value in (("session_id", session_id), ("turn_id", turn_id)):
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise InvalidInput(f"{name} must be a string of 1..=128 chars")
        if mode not in ("text", "voice"):
            raise InvalidInput("mode must be 'text' or 'voice'")
        timings: dict[str, float] = {}
        for moment, taken in latency_ms.items():
            if moment not in ("context", "first_token", "first_audio", "total") or isinstance(taken, bool) \
                    or not isinstance(taken, (int, float)) or not taken >= 0 or not math.isfinite(taken):
                raise InvalidInput("latency_ms names context, first_token, first_audio or total, each a finite non-negative number")
            timings[moment] = float(taken)
        return await self._emit(space, "conversation_turn", {"session_id": session_id, "turn_id": turn_id, "mode": mode,
                                                             "latency_ms": timings})

    async def record_idle(self, space: str, *, session_id: str, count: int, action: str, silent_ms: float,
                          turn_id: Optional[str] = None) -> Optional[Event]:
        """Append a ``conversation_idle`` event: a voice conversation waited
        ``silent_ms`` for a user who said nothing, the ``count``-th time in a
        row, and then prompted them (``turn_id`` is the prompt's turn), only
        noted it, or ended the conversation (``action``: prompt, noted or
        end). Checked whether or not an event log is attached; None when none is."""
        check_space(space)
        for name, value in (("session_id", session_id), ("turn_id", turn_id)):
            if (value is not None or name == "session_id") and (not isinstance(value, str) or not 1 <= len(value) <= 128):
                raise InvalidInput(f"{name} must be a string of 1..=128 chars")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise InvalidInput("count must be a positive integer")
        if action not in ("prompt", "noted", "end"):
            raise InvalidInput("action must be 'prompt', 'noted' or 'end'")
        if isinstance(silent_ms, bool) or not isinstance(silent_ms, (int, float)) or not math.isfinite(silent_ms) \
                or silent_ms < 0:
            raise InvalidInput("silent_ms must be a finite non-negative number")
        payload: dict[str, object] = {"session_id": session_id, "count": count, "action": action,
                                      "silent_ms": float(silent_ms)}
        if turn_id is not None:
            payload["turn_id"] = turn_id
        return await self._emit(space, "conversation_idle", payload)

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

    async def build_chunk_questions(self, space: str, chat: "ChatModel", *, per_chunk: int = 3,
                                    max_chunks: int = 2_000, after_chunk: Optional[int] = None,
                                    episode_ids: Optional[Sequence[int]] = None,
                                    model_name: str = "") -> "QuestionLaneReport":
        """Write the question lane for the space's chunks with ``chat``: up to
        ``per_chunk`` questions a chunk answers, kept only with a sentence
        quoted from it, in an index of their own (ingestion.chunk_questions).
        Refused while ``question_lane`` is off."""
        from ..ingestion.chunk_questions import build_chunk_questions

        return await build_chunk_questions(self, space, chat, per_chunk=per_chunk, max_chunks=max_chunks,
                                           after_chunk=after_chunk, episode_ids=episode_ids, model_name=model_name)

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

    async def lessons(self, space: str, *, window_days: int = 90, half_life_days: float = 30,
                      min_corroboration: int = 2, max_events: Optional[int] = None) -> "Lessons":
        """What people said about the passages of ``space`` over the last ``window_days``:
        per passage, the latest judgement of each recall weighed by age, counted, and
        stated as preferred, tentative, contested or dead end, with whether it can still be
        read. The read is bounded by ``max_events`` and says when that bit."""
        from ..retrieval.lessons import MAX_FEEDBACK_EVENTS, read_lessons

        check_space(space)
        return await read_lessons(self, space, window_days=window_days, half_life_days=half_life_days,
                                  min_corroboration=min_corroboration,
                                  max_events=MAX_FEEDBACK_EVENTS if max_events is None else max_events)

    async def _feedback_prior(self, space: str, candidates: Mapping[int, "Chunk"], now: str) -> "PriorTerms":
        from ..retrieval.feedback_prior import read_prior

        return await read_prior(self.events, self.documents, space, candidates, now=now, weight=self.feedback_weight)

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
        from ..retrieval.feedback_prior import fingerprint, question

        # What the passage said when it was judged, so a ranking prior can tell
        # when the id now names other text. A passage already gone gets none.
        # And which question it was judged for, so asking again is not corroboration.
        # A query kept in the clear is hashed as a hashed recall's is, so the same words are one question either way.
        asked = recall.payload.get("query")
        if recall.payload.get("query_hashed") is False:
            asked = self._query_hash(str(asked))
        marked: dict[str, object] = {"question": question(asked)}
        for chunk in await self.documents.get_chunks(space, [chunk_id]):
            episode = await self.documents.get_episode(space, chunk.episode_id)
            if episode is not None:
                marked["fingerprint"] = fingerprint(episode.content, chunk.text)
        return await self._emit(space, "feedback", {
            "recall_event_id": recall_event_id, "chunk_id": chunk_id, "useful": bool(useful), "note": note, **marked,
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

    async def reconsider(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        """Undo decline: a declined claim goes back to being a proposal.
        The decline stays in the event log with its reason."""
        return await fact_review.reconsider(self._review_runtime(), space, fact_id, reason, actor=actor)

    async def reopen(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        """Undo a close somebody made by hand: the claim holds again. A
        claim another claim superseded is refused, naming that claim."""
        return await fact_review.reopen(self._review_runtime(), space, fact_id, reason, actor=actor)

    async def close_fact(self, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
        return await fact_review.close_fact(self._review_runtime(), space, fact_id, reason, actor=actor)

    async def merge_entities(self, space: str, alias: str, into: str, *, reason: str,
                             actor: Optional[str] = None) -> Fact:
        """Record that ``alias`` names the entity ``into`` names, from now.
        Every view of the graph, and the walk retrieval takes through it,
        then treats the two as one entity; see ``memory.entity_merging``."""
        return await entity_merging.merge_entities(self, space, alias, into, reason=reason, actor=actor)

    async def unmerge_entities(self, space: str, alias: str, *, reason: str,
                               actor: Optional[str] = None) -> Fact:
        """Close the merge in force for ``alias``: the names part from now,
        and a view of an earlier moment still shows them joined."""
        return await entity_merging.unmerge_entities(self, space, alias, reason=reason, actor=actor)

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

    async def profile(self, space: str, limit: int = 10, *, buckets: "catalog.BucketBounds | None" = None) -> Profile:
        """The space's profile; with ``buckets``, its claims in static and
        dynamic buckets too, placed by ``profile_bucket_rules``."""
        return await catalog.profile(self, space, limit, policy=self.profile_policy, buckets=buckets,
                                     rules=self.profile_bucket_rules)

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
        """Return bounded recent source evidence; coverage and continuation are
        explicit. A source past its ``forget_after`` is left out and counted."""
        from ..retrieval.overview import overview

        return await overview(
            self.documents, space, limit=limit, before=before, where=where, kind=kind,
            source_prefix=source_prefix, since=since, until=until,
            exclude_session_id=exclude_session_id, max_records=max_records, now=self.clock(),
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

        A source past its ``forget_after`` is left out, swept or not, and
        counted in ``past_forget_after``; the page is still filled around it.
        """
        moment = self.clock()
        return await self._source_page(space, before=before, limit=limit, kind=kind, conditions=conditions, now=moment)

    async def _source_page(self, space: str, *, before: Optional[int], limit: int, kind: Optional[str],
                           conditions: Mapping[str, object] | None, now: Optional[str]) -> SourcePage:
        """``source_page``, and with ``now`` None the overdue sources too: what
        forgetting by filter selects from, since ``forget`` still reaches them."""
        return await catalog.source_page(self.documents, space, before=before, limit=limit, kind=kind,
            conditions=conditions, walk_page=SOURCE_WALK_PAGE, walk_reads=SOURCE_WALK_READS, now=now)

    async def episodes(self, space: str, where: Mapping[str, str], limit: Optional[int] = None) -> list[Episode]:
        """The episodes whose metadata matches every ``where`` pair, oldest
        first by (created_at, episode_id); with ``limit``, the newest N of
        them in that same order. This is a walk over the space's episodes,
        fine for a session's turns, not a query language. An episode past its
        ``forget_after`` is left out, swept or not."""
        return await catalog.episodes(self.documents, space, where, limit, now=self.clock())

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

    async def export(self, space: str, *, include_attachments: bool = False) -> AsyncIterator[dict]:
        """Every episode and every fact in a space as plain dicts, the
        shape `import_records` accepts. Chunks and vectors are derived
        and are rebuilt on import, so a dump moves between stores and
        between embedders. ``include_attachments=True`` selects archive/2:
        verified linked bytes and episode links travel with the ledger.
        Attachment transfer is bounded; neither profile is an atomic snapshot."""
        check_space(space)
        # What the space holds that an archive does not carry is counted
        # here, where the blob store is, and said in the header: a dump of
        # an illustrated space is not the whole of it, and nobody should
        # have to find that out by restoring one.
        if include_attachments:
            from .attachment_archive import export_records as export_attachments
            records = archive.export_records(self.documents, space, wrote_at=self.clock())
            for record in await export_attachments(records, self.blobs, self.documents, space,
                                                    self.max_attachment_bytes, ATTACHMENT_TYPES):
                yield record
            return
        left_behind = {"attachments": len(await self.blobs.linked(space))}
        async for record in archive.export_records(self.documents, space, wrote_at=self.clock(),
                                                   left_behind=left_behind):
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
        runtime = archive.ArchiveRuntime(self.documents, self.clock, self.remember_many,
                                         self.blobs, self.max_attachment_bytes, ATTACHMENT_TYPES)
        return await archive.import_records(runtime, space, records, resurrect=resurrect)


# -- validation helpers -----------------------------------------------------


def _same_index(vectors: object, image_vectors: object) -> bool:
    """Whether two indexes write the same rows: the same object, or two handles
    whose ``location`` is equal. An index that names no location (the in-memory
    one, a custom one) is recognised only as the same object."""
    if vectors is image_vectors:
        return True
    left = getattr(vectors, "location", None)
    return left is not None and left == getattr(image_vectors, "location", None)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


def derivation_groups(facts: Sequence[Fact]) -> list[list[Fact]]:
    """Claims grouped by subject, joined one hop through shared names: a
    claim whose object names another claim's subject (``join_match``) puts
    both subjects in one group, so "mark works_at acme" and "acme based_in
    lisbon" meet. Pure; order is by the smallest fact id in each group."""
    parent: dict[str, str] = {}
    # A merge decision says two names are one; it is not a claim to reason from.
    facts = [fact for fact in facts if not is_decision(fact)]

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
