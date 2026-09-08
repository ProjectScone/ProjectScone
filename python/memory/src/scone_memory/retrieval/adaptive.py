"""Host-owned, bounded evidence assessment and follow-up retrieval.

A ``sufficient`` status is the assessor's judgment, never proof that an answer
is correct or complete. Every disclosed passage is checked against retained
sources, independently of that judgment. The workflow is inspired by the local
RAGFlow sufficiency/query-generation reference, without its code or dependencies.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from itertools import islice, zip_longest
import time
import unicodedata
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.errors import InvalidInput
from ..core.models import Chunk, Episode, Fact, RecallItem, RecallResult
from ..core.ports import TextFilter
from ..memory.engine import MAX_QUERY, MemoryEngine, check_space
from .multihop import _source_matches
from .recall_scope import RecallScope
from .reranking import candidate_is_retained

EvidenceStatus = Literal["sufficient", "insufficient", "uncertain"]
EvidenceId = Annotated[str, Field(pattern=r"^(chunk|fact):[1-9][0-9]*$", max_length=64)]
FollowupQuery = Annotated[str, Field(min_length=1, max_length=MAX_QUERY)]


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: EvidenceId
    episode_id: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=128_000)
    subject: str | None = Field(default=None, max_length=128_000)
    predicate: str | None = Field(default=None, max_length=128_000)
    object: str | None = Field(default=None, max_length=128_000)


class EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: EvidenceStatus
    selected_ids: tuple[EvidenceId, ...] = Field(max_length=100)
    followup_queries: tuple[FollowupQuery, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def check_selection(self) -> EvidenceDecision:
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected_ids must be unique")
        if self.status == "sufficient" and not self.selected_ids:
            raise ValueError("sufficient requires selected evidence")
        if self.status == "sufficient" and self.followup_queries:
            raise ValueError("sufficient must not request follow-up retrieval")
        return self


class EvidenceAssessor(Protocol):
    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision: ...


class AdaptiveLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_rounds: int = Field(default=3, ge=1, le=4)
    max_queries: int = Field(default=6, ge=1, le=12)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    max_evidence_bytes: int = Field(default=16_000, ge=2, le=128_000)
    timeout_s: float = Field(default=30.0, ge=1, le=180, allow_inf_nan=False)


class AdaptiveRound(BaseModel):
    """Content-free diagnostic receipt; hashes do not include evidence."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    round_number: int
    query_hashes: tuple[str, ...]
    candidate_count: int
    evidence_bytes: int
    selected_count: int
    status: EvidenceStatus


class AdaptiveResult(BaseModel):
    """Selected retained evidence and an explicitly fallible model judgment."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    recall: RecallResult = Field(default_factory=RecallResult)
    status: EvidenceStatus = "uncertain"
    rounds: tuple[AdaptiveRound, ...] = ()
    queries_used: int = 0
    truncated: bool = False
    reasons: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def evidence_payload_bytes(candidates: tuple[EvidenceCandidate, ...]) -> int:
    """UTF-8 size of the entire candidate JSON array, without text clipping."""
    return len(json.dumps([candidate.model_dump(mode="json") for candidate in candidates],
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _query_key(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).split()).casefold()


@dataclass(frozen=True)
class _Evidence:
    candidate: EvidenceCandidate
    source: Episode
    item: RecallItem | None = None
    chunk: Chunk | None = None
    fact: Fact | None = None


class _Run:
    def __init__(self, memory: MemoryEngine, assessor: EvidenceAssessor, limits: AdaptiveLimits,
                 space: str, question: str, scope: RecallScope, excluded: str | None) -> None:
        self.memory, self.assessor, self.limits = memory, assessor, limits
        self.space, self.question, self.scope, self.excluded = space, question, scope, excluded
        self.filter = TextFilter(**scope.kwargs())
        self.boundary = memory.clock()
        self.deadline = time.monotonic() + limits.timeout_s
        self.rounds: list[AdaptiveRound] = []
        self.reasons: list[str] = []
        self.errors: list[str] = []
        self.queries: set[str] = set()
        self.truncated = False

    def check_deadline(self) -> None:
        # Also reject adapters which swallow cancellation and return late.
        if time.monotonic() >= self.deadline:
            raise asyncio.TimeoutError()

    def notice(self, reason: str, *, truncated: bool = False) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.truncated = self.truncated or truncated

    def source_valid(self, source: Episode, episode_id: int) -> bool:
        try:
            return (source.space == self.space and source.episode_id == episode_id
                    and _source_matches(source, self.filter, self.excluded))
        except ValueError:
            return False

    async def accept(self, item: RecallItem | Fact) -> _Evidence | None:
        """Resolve a recalled ID with point reads before trusting any content."""
        documents = self.memory.documents
        texts = (item.quote or "", item.subject, item.predicate, item.object) if isinstance(item, Fact) else (item.text,)
        if any(len(text.encode("utf-8")) > self.limits.max_evidence_bytes for text in texts):
            self.notice("max_evidence_bytes", truncated=True)
            return None
        if isinstance(item, Fact):
            retained = await documents.get_fact(self.space, item.fact_id)
            self.check_deadline()
            if retained is None or retained != item or retained.space != self.space:
                return None
            retained = retained.model_copy(deep=True)
            episode_id = retained.source_episode_id
            try:
                valid = (episode_id is not None and retained.quote is not None and bool(retained.quote)
                         and retained.status == "active" and not retained.excluded
                         and retained.holds_at(self.boundary))
            except ValueError:
                valid = False
            if not valid or episode_id is None:
                return None
            source = await documents.get_episode(self.space, episode_id)
            self.check_deadline()
            if source is None or not self.source_valid(source, episode_id):
                return None
            if retained.quote is None or retained.quote not in source.content:
                return None
            return _Evidence(EvidenceCandidate(id=f"fact:{retained.fact_id}", episode_id=episode_id,
                text=retained.quote, subject=retained.subject, predicate=retained.predicate,
                object=retained.object), source.model_copy(deep=True), fact=retained)
        chunks = await documents.get_chunks(self.space, [item.chunk_id])
        self.check_deadline()
        if len(chunks) != 1:
            return None
        chunk = chunks[0].model_copy(deep=True)
        if (chunk.chunk_id != item.chunk_id or chunk.episode_id != item.episode_id
                or chunk.text != item.text or chunk.created_at != item.created_at):
            return None
        source = await documents.get_episode(self.space, chunk.episode_id)
        self.check_deadline()
        if source is None or not self.source_valid(source, chunk.episode_id):
            return None
        # Timestamp comparisons are parsed above; omit the helper's lexical time
        # filters to support equivalent RFC3339 offsets correctly.
        span_scope = TextFilter(where=dict(self.scope.where), kind=self.scope.kind,
                                source_prefix=self.scope.source_prefix)
        if (not candidate_is_retained(chunk, source, self.space, span_scope)
                or chunk.end > len(source.content.encode("utf-8"))
                or not chunk.text):
            return None
        if item.source != source.source or item.metadata != source.metadata or item.tags != source.tags:
            return None
        return _Evidence(EvidenceCandidate(id=f"chunk:{chunk.chunk_id}", episode_id=chunk.episode_id,
            text=chunk.text), source.model_copy(deep=True), item=item.model_copy(deep=True), chunk=chunk)

    async def verify(self, pool: dict[str, _Evidence]) -> dict[str, _Evidence]:
        """Fresh bounded point reads plus a revision guard around the snapshot.

        Engine writes bump revisions. Direct store writers must provide their
        own transaction discipline, as with the existing multi-hop verifier.
        """
        if not pool:
            return {}
        revision = await self.memory.documents.revision(self.space)
        self.check_deadline()
        checked: dict[str, _Evidence] = {}
        for key, evidence in pool.items():
            record = evidence.fact if evidence.fact is not None else evidence.item
            if record is None:
                continue
            current = await self.accept(record)
            if current is not None and current == evidence:
                checked[key] = current
        final_revision = await self.memory.documents.revision(self.space)
        self.check_deadline()
        if final_revision != revision:
            checked.clear()
        if len(checked) != len(pool):
            self.notice("stale_evidence")
        return checked

    async def gather(self, query: str, pool: dict[str, _Evidence]) -> dict[str, _Evidence]:
        result = await self.memory.recall(self.space, query, limit=self.limits.candidate_limit,
            candidate_limit=self.limits.candidate_limit, rerank=False, **self.scope.kwargs())
        self.check_deadline()
        # A follow-up can yield while carried evidence is deleted or replaced.
        pool = await self.verify(pool)
        if result.degraded:
            self.notice("degraded_recall")
        if len(result.items) + len(result.facts) >= self.limits.candidate_limit:
            self.notice("candidate_window", truncated=True)
        # Preserve each lane's ranking and reserve positions for both kinds;
        # a full chunk window must not starve structured evidence entirely.
        records = (record for pair in zip_longest(result.items, result.facts)
                   for record in pair if record is not None)
        for record in islice(records, self.limits.candidate_limit):
            key = f"fact:{record.fact_id}" if isinstance(record, Fact) else f"chunk:{record.chunk_id}"
            if key in pool:
                continue
            if len(pool) >= self.limits.candidate_limit:
                self.notice("candidate_limit", truncated=True)
                break
            evidence = await self.accept(record)
            if evidence is None:
                self.notice("filtered_evidence")
                continue
            candidates = tuple(value.candidate for value in pool.values()) + (evidence.candidate,)
            if evidence_payload_bytes(candidates) > self.limits.max_evidence_bytes:
                self.notice("max_evidence_bytes", truncated=True)
                continue
            pool[key] = evidence
        return pool

    def result(self, pool: dict[str, _Evidence], status: EvidenceStatus) -> AdaptiveResult:
        items = [evidence.item for evidence in pool.values() if evidence.item is not None]
        facts = [evidence.fact for evidence in pool.values() if evidence.fact is not None]
        recall = RecallResult(items=items, facts=facts,
            returned_bytes=sum(len(item.text.encode("utf-8")) for item in items)
                + sum(len((fact.quote or "").encode("utf-8")) for fact in facts),
            degraded=[f"adaptive: {reason}" for reason in [*self.reasons, *self.errors]])
        return AdaptiveResult(recall=recall, status=status, rounds=tuple(self.rounds),
            queries_used=len(self.queries), truncated=self.truncated,
            reasons=tuple(self.reasons), errors=tuple(self.errors))

    async def execute(self) -> AdaptiveResult:
        pending = [self.question]
        pool: dict[str, _Evidence] = {}
        selected: dict[str, _Evidence] = {}
        status: EvidenceStatus = "uncertain"
        for round_number in range(1, self.limits.max_rounds + 1):
            hashes: list[str] = []
            for query in pending:
                key = _query_key(query)
                if key in self.queries:
                    self.notice("duplicate_queries")
                    continue
                if len(self.queries) >= self.limits.max_queries:
                    self.notice("max_queries", truncated=True)
                    break
                self.queries.add(key)
                hashes.append(hashlib.sha256(key.encode("utf-8")).hexdigest())
                pool = await self.gather(query, pool)
            pool = await self.verify(pool)
            if not pool:
                self.notice("no_evidence")
                selected = {}
                status = "insufficient"
                self.rounds.append(AdaptiveRound(round_number=round_number, query_hashes=tuple(hashes),
                    candidate_count=0, evidence_bytes=2, selected_count=0, status=status))
                break
            candidates = tuple(evidence.candidate for evidence in pool.values())
            self.rounds.append(AdaptiveRound(round_number=round_number, query_hashes=tuple(hashes),
                candidate_count=len(candidates), evidence_bytes=evidence_payload_bytes(candidates),
                selected_count=0, status="uncertain"))
            try:
                raw_decision = await self.assessor.assess(self.question, candidates)
                self.check_deadline()
                # Rebuild even model_construct/model_copy values: strict models
                # can otherwise bypass validation at an untrusted adapter edge.
                if not isinstance(raw_decision, EvidenceDecision):
                    raise ValueError("invalid decision type")
                decision = EvidenceDecision.model_validate(dict(vars(raw_decision)), strict=True)
                if not set(decision.selected_ids).issubset(pool):
                    raise ValueError("unknown evidence ID")
                if any(not query.strip() or len(query) > MAX_QUERY for query in decision.followup_queries):
                    raise ValueError("invalid follow-up query")
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                raise
            except Exception:
                # Exceptions may embed prompts, endpoints or secret values.
                self.errors.append("invalid_or_failed_assessment")
                return self.result({}, "uncertain")
            pool = await self.verify(pool)
            selected = {key: pool[key] for key in decision.selected_ids if key in pool}
            status = decision.status
            if len(selected) != len(decision.selected_ids):
                status = "uncertain"
            self.rounds[-1] = AdaptiveRound(round_number=round_number, query_hashes=tuple(hashes),
                candidate_count=len(candidates), evidence_bytes=evidence_payload_bytes(candidates),
                selected_count=len(selected), status=status)
            if status == "sufficient":
                break
            if not decision.followup_queries:
                self.notice("no_followup_queries")
                break
            pending = []
            seen = set(self.queries)
            for query in decision.followup_queries:
                key = _query_key(query)
                if key in seen:
                    self.notice("duplicate_queries")
                    continue
                seen.add(key)
                if len(self.queries) + len(pending) >= self.limits.max_queries:
                    self.notice("max_queries", truncated=True)
                    break
                pending.append(query.strip())
            if not pending:
                self.notice("no_new_queries")
                break
            if round_number == self.limits.max_rounds:
                self.notice("max_rounds", truncated=True)
                break
            # Only selected useful evidence is carried; rejected distractors
            # cannot crowd out the next bounded retrieval window.
            pool = selected.copy()
        verified = await self.verify(selected)
        if len(verified) != len(selected):
            status = "uncertain"
        self.check_deadline()
        return self.result(verified, status)


class AdaptiveRetriever:
    """Optional retrieval strategy; assessor lifecycle belongs to the caller.

    All queries use a private validated copy of the same caller scope. Session
    exclusions are applied to source records because MemoryEngine.recall has no
    session-exclusion parameter; a bounded candidate window can underfill.
    The cooperative deadline includes recall, assessment and source checks.
    Timeout or unexpected failures discard evidence which cannot be verified.
    """
    def __init__(self, memory: MemoryEngine, assessor: EvidenceAssessor, *,
                 limits: AdaptiveLimits | None = None) -> None:
        self._memory, self.assessor = memory, assessor
        self._limits = AdaptiveLimits.model_validate((limits or AdaptiveLimits()).model_dump(), strict=True)

    @property
    def memory(self) -> MemoryEngine:
        return self._memory

    @property
    def limits(self) -> AdaptiveLimits:
        return self._limits

    async def retrieve(self, space: str, query: str, *, scope: RecallScope,
                       exclude_session_id: str | None = None) -> AdaptiveResult:
        check_space(space)
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
            raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
        if exclude_session_id is not None and not isinstance(exclude_session_id, str):
            raise InvalidInput("exclude_session_id must be a string")
        fixed_scope = RecallScope.validated(**scope.kwargs())
        run = _Run(self.memory, self.assessor, self.limits, space, query.strip(), fixed_scope, exclude_session_id)
        try:
            return await asyncio.wait_for(run.execute(), timeout=self.limits.timeout_s)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            run.notice("timeout", truncated=True)
            run.errors.append("timeout")
            return run.result({}, "uncertain")
        except Exception:
            run.errors.append("retrieval_failed")
            return run.result({}, "uncertain")
