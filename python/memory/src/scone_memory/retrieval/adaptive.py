"""Host-owned, bounded evidence assessment and follow-up retrieval.

A ``sufficient`` status is the assessor's judgment, never proof that an answer
is correct or complete. Every disclosed passage is checked against retained
sources, independently of that judgment. The workflow is inspired by the local
RAGFlow sufficiency/query-generation reference, without its code or dependencies.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
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
from .multihop import MultiHopLimits, _source_matches, expand_multihop
from .recall_scope import RecallScope
from .reranking import candidate_is_retained

EvidenceStatus = Literal["sufficient", "insufficient", "uncertain"]
EvidenceBasis = Literal["assessed_selection", "verified_candidates", "unselected_candidates", "none"]
FallbackStatus = Literal["not_used", "retained", "empty", "verification_failed", "verification_timeout"]
FailurePolicy = Literal["empty", "retain_verified"]
EmptySelectionPolicy = Literal["empty", "retain_verified"]
EvidenceId = Annotated[str, Field(pattern=r"^(chunk|fact):[1-9][0-9]*$", max_length=64)]
EvidenceGroup = Annotated[tuple[EvidenceId, ...], Field(min_length=2, max_length=100)]
EvidenceGroups = Annotated[tuple[EvidenceGroup, ...], Field(max_length=100)]
FollowupQuery = Annotated[str, Field(min_length=1, max_length=MAX_QUERY)]
ASSESSMENT_FAILURE_REASONS = frozenset({
    "assessment_timeout", "assessment_provider_failed", "invalid_assessment",
})
GRAPH_REASONS = frozenset({
    "max_store_calls", "max_candidates", "max_nodes", "max_edges", "max_bytes", "max_hops",
    "candidate_window", "unsupported_bounded_links", "unsupported_bounded_subjects",
    "unsupported_link_revalidation", "stale_evidence", "degraded_recall", "no_seeds",
    "candidate_limit", "max_evidence_bytes", "atomic_group_omitted", "graph_grouping_limit",
    "filtered_evidence", "graph_failed", "timeout",
})


class EvidenceAssessmentError(ValueError):
    """Content-free assessment failure shared by model-neutral adapters."""

    def __init__(self, reason: str) -> None:
        super().__init__("evidence assessment failed")
        self.reason = reason if reason in ASSESSMENT_FAILURE_REASONS else "invalid_or_failed_assessment"


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
    selected_groups: EvidenceGroups = ()
    followup_queries: tuple[FollowupQuery, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def check_selection(self) -> EvidenceDecision:
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected_ids must be unique")
        if self.status == "sufficient" and not self.selected_ids:
            raise ValueError("sufficient requires selected evidence")
        if self.status == "sufficient" and self.followup_queries:
            raise ValueError("sufficient must not request follow-up retrieval")
        _validate_groups(self.selected_groups, set(self.selected_ids))
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


class AdaptiveGraphReceipt(BaseModel):
    """Per-expansion coverage, not evidence or answer completeness.

    ``store_calls`` counts traversal and the walker's own revalidation. Extra
    adaptive checks are candidate-bounded within the deadline. The round cap
    bounds the number of these separately budgeted expansions.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    candidate_count: int = Field(default=0, ge=0)
    added_count: int = Field(default=0, ge=0)
    omitted_count: int = Field(default=0, ge=0)
    store_calls: int = Field(default=0, ge=0)
    complete: bool = False
    truncated: bool = False
    reasons: tuple[str, ...] = ()


class AdaptiveResult(BaseModel):
    """Selected retained evidence and an explicitly fallible model judgment."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    recall: RecallResult = Field(default_factory=RecallResult)
    selected_groups: EvidenceGroups = ()
    status: EvidenceStatus = "uncertain"
    evidence_basis: EvidenceBasis = "none"
    fallback_status: FallbackStatus = "not_used"
    graph_expansions: tuple[AdaptiveGraphReceipt, ...] = ()
    rounds: tuple[AdaptiveRound, ...] = ()
    queries_used: int = 0
    truncated: bool = False
    reasons: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_groups(self) -> AdaptiveResult:
        delivered = {f"chunk:{item.chunk_id}" for item in self.recall.items}
        delivered.update(f"fact:{fact.fact_id}" for fact in self.recall.facts)
        _validate_groups(self.selected_groups, delivered)
        return self


def _validate_groups(groups: tuple[tuple[str, ...], ...], selected_ids: set[str]) -> None:
    members = [member for group in groups for member in group]
    if len(set(members)) != len(members):
        raise ValueError("group members must be unique within and across groups")
    if not set(members).issubset(selected_ids):
        raise ValueError("every group member must be selected")


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


class _AssessmentFailed(Exception):
    """Signal recovery outside the workflow timer; diagnostics stay on the run."""


class _Run:
    def __init__(self, memory: MemoryEngine, assessor: EvidenceAssessor, limits: AdaptiveLimits,
                 space: str, question: str, scope: RecallScope, excluded: str | None,
                 failure_policy: FailurePolicy, graph_limits: MultiHopLimits | None,
                 empty_selection_policy: EmptySelectionPolicy) -> None:
        self.memory, self.assessor, self.limits = memory, assessor, limits
        self.space, self.question, self.scope, self.excluded = space, question, scope, excluded
        self.boundary = memory.clock()
        self.filter = TextFilter(as_of=self.boundary if graph_limits is not None else None, **scope.kwargs())
        self.failure_policy = failure_policy
        self.empty_selection_policy = empty_selection_policy
        self.graph_limits = graph_limits
        self.graph_expansions: list[AdaptiveGraphReceipt] = []
        self.graph_snapshots: dict[str, _Evidence] = {}
        self.graph_stale_ids: set[str] = set()
        self.hard_deadline = time.monotonic() + limits.timeout_s
        reserve = min(1.0, limits.timeout_s / 4) if failure_policy == "retain_verified" else 0.0
        self.deadline = self.hard_deadline - reserve
        self.stage: Literal["retrieval", "assessment", "verification"] = "retrieval"
        self.fallback_pool: dict[str, _Evidence] = {}
        self.fallback_groups: tuple[tuple[str, ...], ...] = ()
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

    def retain_snapshot(self, evidence: _Evidence) -> _Evidence | None:
        if self.graph_limits is None:
            return evidence
        key = evidence.candidate.id
        previous = self.graph_snapshots.get(key)
        changed = previous is not None and (previous.candidate != evidence.candidate
            or previous.source != evidence.source or previous.chunk != evidence.chunk or previous.fact != evidence.fact)
        if key in self.graph_stale_ids or changed:
            self.graph_stale_ids.add(key)
            self.notice("stale_evidence")
            return None
        if previous is None:
            self.graph_snapshots[key] = deepcopy(evidence)
        return evidence

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
            return self.retain_snapshot(_Evidence(EvidenceCandidate(id=f"fact:{retained.fact_id}", episode_id=episode_id,
                text=retained.quote, subject=retained.subject, predicate=retained.predicate,
                object=retained.object), source.model_copy(deep=True), fact=retained))
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
        return self.retain_snapshot(_Evidence(EvidenceCandidate(id=f"chunk:{chunk.chunk_id}", episode_id=chunk.episode_id,
            text=chunk.text), source.model_copy(deep=True), item=item.model_copy(deep=True), chunk=chunk))

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
            elif self.graph_limits is not None:
                self.graph_stale_ids.add(key)
        final_revision = await self.memory.documents.revision(self.space)
        self.check_deadline()
        if final_revision != revision:
            checked.clear()
        if len(checked) != len(pool):
            self.notice("stale_evidence")
        return checked

    async def gather(self, query: str, pool: dict[str, _Evidence], groups: tuple[tuple[str, ...], ...]
                     ) -> tuple[dict[str, _Evidence], tuple[tuple[str, ...], ...]]:
        result = await self.memory.recall(self.space, query, limit=self.limits.candidate_limit,
            candidate_limit=self.limits.candidate_limit, rerank=False, **self.scope.kwargs())
        self.check_deadline()
        # A follow-up can yield while carried evidence is deleted or replaced.
        pool = await self.verify(pool)
        pool, groups = self.prune_groups(pool, groups)
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
        return pool, groups

    def prune_groups(self, pool: dict[str, _Evidence], groups: tuple[tuple[str, ...], ...]
                     ) -> tuple[dict[str, _Evidence], tuple[tuple[str, ...], ...]]:
        retained = tuple(group for group in groups if set(group).issubset(pool))
        if len(retained) == len(groups):
            return pool, groups
        self.notice("atomic_group_omitted")
        omitted = {member for group in groups if not set(group).issubset(pool) for member in group}
        return {key: evidence for key, evidence in pool.items() if key not in omitted}, retained

    async def expand_graph(self, pool: dict[str, _Evidence], groups: tuple[tuple[str, ...], ...]
                           ) -> tuple[dict[str, _Evidence], tuple[tuple[str, ...], ...]]:
        from .adaptive_graph import exact_fact_groups, merge_groups

        if self.graph_limits is None:
            return pool, groups
        facts = tuple(evidence.fact for evidence in pool.values() if evidence.fact is not None)
        index = len(self.graph_expansions)
        self.graph_expansions.append(AdaptiveGraphReceipt())
        if not facts:
            self.graph_expansions[index] = AdaptiveGraphReceipt(reasons=("no_seeds",))
            return pool, groups
        reasons: set[str] = set()
        store_calls = 0
        graph_count = 0
        try:
            revision = await self.memory.documents.revision(self.space)
            self.check_deadline()
            expansion = await expand_multihop(self.memory.documents, self.space,
                seeds=RecallResult(facts=list(facts)), limits=self.graph_limits,
                scope=TextFilter(as_of=self.boundary, **self.scope.kwargs()), exclude_session_id=self.excluded)
            self.check_deadline()
            store_calls = expansion.counts.store_calls
            reasons.update(reason if reason in GRAPH_REASONS else "graph_failed" for reason in expansion.coverage.reasons)
            graph_count = len(expansion.facts)
            original_ids = set(pool)
            original_facts = {fact.fact_id: fact for fact in facts}
            current = await self.verify(pool)
            # Membership includes whole discovered components before packing.
            # Existing IDs retain their original snapshots; even an updated
            # record with the same ID cannot replace one which failed checks.
            all_facts = dict(original_facts)
            for fact in expansion.facts:
                if fact.fact_id in original_facts:
                    if fact != original_facts[fact.fact_id]:
                        current.pop(f"fact:{fact.fact_id}", None)
                        reasons.add("stale_evidence")
                    continue
                all_facts[fact.fact_id] = fact
                if any(len(text.encode("utf-8")) > self.limits.max_evidence_bytes
                       for text in (fact.quote or "", fact.subject, fact.predicate, fact.object)):
                    reasons.add("max_evidence_bytes")
                evidence = await self.accept(fact)
                if evidence is None:
                    reasons.add("filtered_evidence")
                else:
                    current[evidence.candidate.id] = evidence
            current = await self.verify(current)
            final_revision = await self.memory.documents.revision(self.space)
            self.check_deadline()
            if final_revision != revision:
                reasons.add("stale_evidence")
                raise ValueError("graph snapshot changed")
            try:
                host_groups = merge_groups((*groups, *exact_fact_groups(tuple(all_facts.values()))))
            except ValueError:
                # A dense/oversized membership window is a visible budget
                # omission. Do not expose an arbitrary partial fact component.
                reasons.add("graph_grouping_limit")
                fact_ids = {f"fact:{fact_id}" for fact_id in all_facts}
                current = {key: value for key, value in current.items() if key not in fact_ids}
                current, host_groups = self.prune_groups(current, groups)
            members = {member for group in host_groups for member in group}
            standalone = [(key,) for key in current if key not in members]
            # Atomic components go first so unrelated recall chunks cannot
            # consume every slot needed by a newly discovered connecting fact.
            units = [*host_groups, *standalone]
            offered: dict[str, _Evidence] = {}
            for unit in units:
                if not set(unit).issubset(current):
                    reasons.add("atomic_group_omitted")
                    continue
                if len(offered) + len(unit) > self.limits.candidate_limit:
                    reasons.add("candidate_limit")
                    continue
                prospective = tuple(value.candidate for value in offered.values()) + tuple(current[key].candidate for key in unit)
                if evidence_payload_bytes(prospective) > self.limits.max_evidence_bytes:
                    reasons.add("max_evidence_bytes")
                    continue
                offered.update((key, current[key]) for key in unit)
            offered, retained_groups = self.prune_groups(offered, host_groups)
            if len(retained_groups) != len(host_groups):
                reasons.add("atomic_group_omitted")
            considered_ids = original_ids | {f"fact:{fact_id}" for fact_id in all_facts}
            omitted = len(considered_ids - set(offered))
            truncated = expansion.coverage.truncated or bool(reasons & {
                "candidate_limit", "max_evidence_bytes", "graph_grouping_limit"})
            self.graph_expansions[index] = AdaptiveGraphReceipt(candidate_count=graph_count,
                added_count=len(set(offered) - original_ids), omitted_count=omitted, store_calls=store_calls,
                complete=expansion.coverage.complete and not omitted and not reasons,
                truncated=truncated, reasons=tuple(sorted(reasons)))
            for reason in reasons:
                self.notice(reason, truncated=truncated)
            self.check_deadline()
            return offered, retained_groups
        except asyncio.CancelledError:
            self.graph_expansions[index] = AdaptiveGraphReceipt(candidate_count=graph_count, store_calls=store_calls,
                truncated=True, reasons=("timeout",))
            raise
        except Exception as error:
            reasons.add("timeout" if isinstance(error, asyncio.TimeoutError) else "graph_failed")
            self.graph_expansions[index] = AdaptiveGraphReceipt(candidate_count=graph_count, store_calls=store_calls,
                truncated=isinstance(error, asyncio.TimeoutError), reasons=tuple(sorted(reasons)))
            raise

    def result(self, pool: dict[str, _Evidence], status: EvidenceStatus,
               groups: tuple[tuple[str, ...], ...] = (), *, basis: EvidenceBasis | None = None,
               fallback_status: FallbackStatus = "not_used") -> AdaptiveResult:
        items = [evidence.item for evidence in pool.values() if evidence.item is not None]
        facts = [evidence.fact for evidence in pool.values() if evidence.fact is not None]
        recall = RecallResult(items=items, facts=facts,
            returned_bytes=sum(len(item.text.encode("utf-8")) for item in items)
                + sum(len((fact.quote or "").encode("utf-8")) for fact in facts),
            degraded=[f"adaptive: {reason}" for reason in [*self.reasons, *self.errors]])
        return AdaptiveResult(recall=recall, selected_groups=groups, status=status, rounds=tuple(self.rounds),
            evidence_basis=(basis or "assessed_selection") if pool else "none", fallback_status=fallback_status,
            graph_expansions=tuple(self.graph_expansions),
            queries_used=len(self.queries), truncated=self.truncated,
            reasons=tuple(self.reasons), errors=tuple(self.errors))

    async def recover_assessment(self) -> AdaptiveResult:
        """Recheck original candidates once, using only the remaining deadline."""
        if self.failure_policy == "empty":
            return self.result({}, "uncertain")
        self.deadline = self.hard_deadline
        self.stage = "verification"
        remaining = self.hard_deadline - time.monotonic()
        if remaining <= 0:
            self.notice("timeout", truncated=True)
            return self.result({}, "uncertain", fallback_status="verification_timeout")
        try:
            verified = await asyncio.wait_for(self.verify(self.fallback_pool), timeout=remaining)
            verified, groups = self.prune_groups(verified, self.fallback_groups)
            self.check_deadline()
            return self.result(verified, "uncertain", groups, basis="verified_candidates",
                               fallback_status="retained" if verified else "empty")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.notice("timeout", truncated=True)
            return self.result({}, "uncertain", fallback_status="verification_timeout")
        except Exception:
            return self.result({}, "uncertain", fallback_status="verification_failed")

    async def execute(self) -> AdaptiveResult:
        pending = [self.question]
        pool: dict[str, _Evidence] = {}
        selected: dict[str, _Evidence] = {}
        selected_groups: tuple[tuple[str, ...], ...] = ()
        status: EvidenceStatus = "uncertain"
        empty_selection = False
        for round_number in range(1, self.limits.max_rounds + 1):
            empty_selection = False
            hashes: list[str] = []
            carried_group_count = len(selected_groups)
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
                self.stage = "retrieval"
                pool, selected_groups = await self.gather(query, pool, selected_groups)
            self.stage = "verification"
            pool = await self.verify(pool)
            pool, selected_groups = self.prune_groups(pool, selected_groups)
            if self.graph_limits is not None:
                pool, selected_groups = await self.expand_graph(pool, selected_groups)
            if not pool:
                self.notice("no_evidence")
                selected = {}
                status = "uncertain" if len(selected_groups) < carried_group_count else "insufficient"
                self.rounds.append(AdaptiveRound(round_number=round_number, query_hashes=tuple(hashes),
                    candidate_count=0, evidence_bytes=2, selected_count=0, status=status))
                break
            candidates = tuple(evidence.candidate for evidence in pool.values())
            self.rounds.append(AdaptiveRound(round_number=round_number, query_hashes=tuple(hashes),
                candidate_count=len(candidates), evidence_bytes=evidence_payload_bytes(candidates),
                selected_count=0, status="uncertain"))
            # The adapter receives separate candidate objects. Recovery must not
            # inherit even deliberately bypassed mutations of frozen models.
            self.fallback_pool = deepcopy(pool)
            self.fallback_groups = selected_groups
            self.stage = "assessment"
            try:
                raw_decision = await self.assessor.assess(self.question,
                    tuple(candidate.model_copy(deep=True) for candidate in candidates))
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
            except EvidenceAssessmentError as error:
                # Recheck the code: caller-owned adapters can mutate exceptions.
                supplied_reason = getattr(error, "reason", None)
                reason = supplied_reason if type(supplied_reason) is str and supplied_reason in ASSESSMENT_FAILURE_REASONS else "invalid_or_failed_assessment"
                self.errors.append(reason)
                raise _AssessmentFailed() from None
            except Exception:
                # Exceptions may embed prompts, endpoints or secret values.
                self.errors.append("invalid_or_failed_assessment")
                raise _AssessmentFailed() from None
            self.stage = "verification"
            pool = await self.verify(pool)
            selected = {key: pool[key] for key in decision.selected_ids if key in pool}
            if self.graph_limits is not None:
                from .adaptive_graph import merge_groups
                selected_groups = merge_groups((*selected_groups, *decision.selected_groups))
            else:
                selected_groups = decision.selected_groups
            if decision.selected_ids:
                selected, selected_groups = self.prune_groups(selected, selected_groups)
            else:
                selected_groups = ()
            status = decision.status
            empty_selection = not decision.selected_ids
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
        basis: EvidenceBasis = "assessed_selection"
        if empty_selection and self.empty_selection_policy == "retain_verified":
            # A valid empty selection is not proof that every candidate is useless.
            # Recover only the last offered snapshot, never an earlier discarded pool.
            selected = self.fallback_pool
            selected_groups = self.fallback_groups
            basis = "unselected_candidates"
        verified = await self.verify(selected)
        verified, selected_groups = self.prune_groups(verified, selected_groups)
        if len(verified) != len(selected):
            status = "uncertain"
        self.check_deadline()
        if verified and basis == "unselected_candidates":
            self.notice("empty_selection_retained")
        return self.result(verified, status, selected_groups, basis=basis)


class AdaptiveRetriever:
    """Optional retrieval strategy; assessor lifecycle belongs to the caller.

    All queries use a private validated copy of the same caller scope. Session
    exclusions are applied to source records because MemoryEngine.recall has no
    session-exclusion parameter; a bounded candidate window can underfill.
    The cooperative deadline includes recall, assessment and source checks.
    By default a failed assessment may return freshly reverified candidates,
    explicitly unassessed and uncertain. A small verification reserve stays
    within the same deadline. ``failure_policy="empty"`` disables this fallback.
    Recovery preserves only atomic groups from prior valid decisions; a failed
    provider response establishes no new grouping or selection authority. With
    explicit ``graph_limits``, host-verified exact components establish atomic
    groups before assessment, including first-round fallback. Traversal uses
    the run's fact-time boundary and excludes later-created source episodes.
    A valid terminal empty selection retains the final verified candidate pool
    by default, labeled ``unselected_candidates`` with the original insufficient
    or uncertain judgment. It does not retain discarded pools across follow-ups.
    ``empty_selection_policy="empty"`` preserves strict model-only selection.
    """
    def __init__(self, memory: MemoryEngine, assessor: EvidenceAssessor, *,
                 limits: AdaptiveLimits | None = None, failure_policy: FailurePolicy = "retain_verified",
                 graph_limits: MultiHopLimits | None = None,
                 empty_selection_policy: EmptySelectionPolicy = "retain_verified") -> None:
        if type(failure_policy) is not str or failure_policy not in ("empty", "retain_verified"):
            raise ValueError("failure_policy must be empty or retain_verified")
        if type(empty_selection_policy) is not str or empty_selection_policy not in ("empty", "retain_verified"):
            raise ValueError("empty_selection_policy must be empty or retain_verified")
        self._memory, self.assessor = memory, assessor
        self._limits = AdaptiveLimits.model_validate((limits or AdaptiveLimits()).model_dump(), strict=True)
        self._failure_policy = failure_policy
        self._empty_selection_policy = empty_selection_policy
        self._graph_limits = (MultiHopLimits.model_validate(dict(vars(graph_limits)), strict=True)
                              if graph_limits is not None else None)

    @property
    def memory(self) -> MemoryEngine:
        return self._memory

    @property
    def limits(self) -> AdaptiveLimits:
        return self._limits

    @property
    def failure_policy(self) -> FailurePolicy:
        return self._failure_policy

    @property
    def empty_selection_policy(self) -> EmptySelectionPolicy:
        return self._empty_selection_policy

    @property
    def graph_limits(self) -> MultiHopLimits | None:
        return self._graph_limits

    async def retrieve(self, space: str, query: str, *, scope: RecallScope,
                       exclude_session_id: str | None = None) -> AdaptiveResult:
        check_space(space)
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
            raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
        if exclude_session_id is not None and not isinstance(exclude_session_id, str):
            raise InvalidInput("exclude_session_id must be a string")
        fixed_scope = RecallScope.validated(**scope.kwargs())
        run = _Run(self.memory, self.assessor, self.limits, space, query.strip(), fixed_scope, exclude_session_id,
                   self.failure_policy, self.graph_limits, self.empty_selection_policy)
        try:
            return await asyncio.wait_for(run.execute(), timeout=max(0.0, run.deadline - time.monotonic()))
        except asyncio.CancelledError:
            raise
        except _AssessmentFailed:
            return await run.recover_assessment()
        except asyncio.TimeoutError:
            run.notice("timeout", truncated=True)
            run.errors.append("timeout")
            if run.stage == "assessment":
                return await run.recover_assessment()
            return run.result({}, "uncertain")
        except Exception:
            run.errors.append("retrieval_failed")
            return run.result({}, "uncertain")
