"""Positive fact/path coverage for explicit, caller-authored requirements.

This is not natural-language intent detection, semantic entailment, source
verification or proof of global completeness. The adaptive retriever owns scope
and retention. Only exact recorded triples can witness these requirements.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..memory.engine import MAX_QUERY
from .adaptive import EvidenceCandidate, EvidenceDecision, EvidenceId, FollowupQuery, evidence_payload_bytes
from .adaptive_graph import merge_groups


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")
    kind: Literal["fact", "path", "reachable_fact"]
    subject: str | None = Field(default=None, min_length=1, max_length=128)
    predicate: str = Field(min_length=1, max_length=128)
    object: str | None = Field(default=None, min_length=1, max_length=128)
    max_hops: int = Field(default=6, ge=1, le=6)
    via: tuple[Annotated[str, Field(min_length=1, max_length=128)], ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def valid_requirement(self) -> Self:
        if any(value is not None and not value.strip() for value in (self.subject, self.predicate, self.object)):
            raise ValueError("requirement identities must be nonblank")
        if self.kind == "path" and (self.subject is None or self.object is None or self.object == self.subject):
            raise ValueError("a path requires distinct explicit endpoints")
        if self.kind == "fact" and self.subject is None and self.object is None:
            raise ValueError("a fact requires at least one explicit endpoint")
        if self.kind == "fact" and self.max_hops != 6:
            raise ValueError("max_hops applies only to paths")
        if self.kind == "reachable_fact":
            if self.subject is None or not self.via or self.max_hops < 2:
                raise ValueError("a reachable fact requires an anchor, traversal predicates and at least two hops")
            if (any(not predicate.strip() for predicate in self.via) or len(set(self.via)) != len(self.via)
                    or self.predicate in self.via):
                raise ValueError("traversal predicates must be unique, nonblank and distinct from the answer predicate")
        elif self.via:
            raise ValueError("via applies only to reachable facts")
        return self

    def search_query(self) -> str:
        base = " ".join(value for value in (self.subject, self.predicate, self.object) if value is not None)
        remaining = MAX_QUERY - len(base)
        hints: list[str] = []
        for predicate in self.via:
            if len(predicate) + 1 <= remaining:
                hints.append(predicate)
                remaining -= len(predicate) + 1
        return " ".join(value for value in (self.subject, *hints, self.predicate, self.object) if value is not None)


class RequirementCoverage(BaseModel):
    """Positive witnesses, partial bridges and alternatives for one requirement."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    requirement: EvidenceRequirement
    witnessed: bool
    witness_ids: tuple[EvidenceId, ...] = Field(max_length=100)
    bridge_ids: tuple[EvidenceId, ...] = Field(default=(), max_length=100)
    selected_ids: tuple[EvidenceId, ...] = Field(max_length=100)
    followup_query: FollowupQuery | None
    work_used: int = Field(ge=0, le=2048)
    work_exhausted: bool

    @model_validator(mode='after')
    def consistent_witnesses(self) -> Self:
        if (len(set(self.witness_ids)) != len(self.witness_ids)
                or len(set(self.bridge_ids)) != len(self.bridge_ids)
                or len(set(self.selected_ids)) != len(self.selected_ids)
                or not set((*self.witness_ids, *self.bridge_ids)).issubset(self.selected_ids)):
            raise ValueError('coverage IDs must be unique and witnesses retained')
        if (self.witnessed != bool(self.witness_ids)
                or self.witnessed != (self.followup_query is None)
                or (self.witnessed and self.bridge_ids)
                or (not self.witnessed and self.selected_ids and not self.bridge_ids)):
            raise ValueError('coverage verdict does not match its witnesses')
        return self


class StructuredEvidenceAssessment(BaseModel):
    """Per-call diagnostics; not source verification or a global completeness claim."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    decision: EvidenceDecision
    coverage: tuple[RequirementCoverage, ...] = Field(min_length=1, max_length=8)
    work_limit: int = Field(ge=1, le=2048)
    work_used: int = Field(ge=0, le=2048)

    @model_validator(mode='after')
    def consistent_budget(self) -> Self:
        if self.work_used != sum(row.work_used for row in self.coverage) or self.work_used > self.work_limit:
            raise ValueError('coverage work exceeds or disagrees with its budget')
        return self


def _facts(candidates: tuple[EvidenceCandidate, ...]) -> dict[tuple[str, str], list[EvidenceCandidate]]:
    index: dict[tuple[str, str], list[EvidenceCandidate]] = {}
    for candidate in candidates:
        if (candidate.id.startswith("fact:") and candidate.subject is not None and candidate.subject.strip()
                and candidate.predicate is not None and candidate.predicate.strip()
                and candidate.object is not None and candidate.object.strip()):
            index.setdefault((candidate.subject, candidate.predicate), []).append(candidate)
    return index


@dataclass
class _Bridge:
    path: tuple[EvidenceCandidate, ...] = ()

    def consider(self, path: tuple[EvidenceCandidate, ...], max_hops: int) -> None:
        target = path[-1].object
        # Leave a hop for the missing relation; never truncate an entity name.
        if (len(self.path) < len(path) < max_hops and target is not None
                and target.strip() and len(target) <= 128):
            self.path = path

    def query(self, requirement: EvidenceRequirement) -> str:
        anchor = self.path[-1].object
        if requirement.kind == 'reachable_fact' and len(self.path) + 1 == requirement.max_hops:
            return EvidenceRequirement(kind='fact', subject=anchor, predicate=requirement.predicate,
                                       object=requirement.object).search_query()
        return EvidenceRequirement.model_validate({**requirement.model_dump(), 'subject':anchor}).search_query()


def _path(requirement: EvidenceRequirement, index: dict[tuple[str, str], list[EvidenceCandidate]],
          remaining: int, bridge: _Bridge) -> tuple[tuple[EvidenceCandidate, ...], int, bool]:
    if requirement.subject is None:
        raise ValueError("a path requires an explicit subject")
    pending: deque[tuple[str, tuple[EvidenceCandidate, ...], frozenset[str]]] = deque([
        (requirement.subject, (), frozenset((requirement.subject,)))])
    while pending:
        subject, path, visited = pending.popleft()
        if len(path) >= requirement.max_hops:
            continue
        for candidate in index.get((subject, requirement.predicate), []):
            if remaining == 0:
                return (), remaining, True
            remaining -= 1
            target = candidate.object
            if target is None or target in visited:
                continue
            extended = (*path, candidate)
            if target == requirement.object:
                return extended, remaining, False
            bridge.consider(extended, requirement.max_hops)
            pending.append((target, extended, visited | {target}))
    return (), remaining, False


def _reachable_fact(requirement: EvidenceRequirement, index: dict[tuple[str, str], list[EvidenceCandidate]],
                    remaining: int, bridge: _Bridge) -> tuple[tuple[EvidenceCandidate, ...], int, bool]:
    if requirement.subject is None:
        raise ValueError("a reachable fact requires an explicit subject")
    pending: deque[tuple[str, tuple[EvidenceCandidate, ...]]] = deque([(requirement.subject, ())])
    visited = {requirement.subject}
    witnesses: dict[str, EvidenceCandidate] = {}
    while pending:
        subject, path = pending.popleft()
        if path:
            for candidate in index.get((subject, requirement.predicate), []):
                if remaining == 0:
                    return tuple(witnesses.values()), remaining, True
                remaining -= 1
                if requirement.object is None or candidate.object == requirement.object:
                    witnesses.update((row.id, row) for row in (*path, candidate))
        # Reserve one hop for the requested final fact. A matching attribute
        # does not stop traversal: another reachable subject may have a value.
        if len(path) + 1 >= requirement.max_hops:
            continue
        for predicate in requirement.via:
            for candidate in index.get((subject, predicate), []):
                if remaining == 0:
                    return tuple(witnesses.values()), remaining, True
                remaining -= 1
                target = candidate.object
                if target is not None and target not in visited:
                    # BFS gives each entity one shortest witness. Merged routes
                    # and cycles cannot multiply downstream expansion work.
                    visited.add(target)
                    extended = (*path, candidate)
                    bridge.consider(extended, requirement.max_hops)
                    pending.append((target, extended))
    return tuple(witnesses.values()), remaining, False


class StructuredEvidenceAssessor:
    """An AdaptiveRetriever assessor for a fixed question and explicit plan.

    Fact requirements request recorded values for an exact subject/predicate,
    or recorded subjects for an exact predicate/object. With both endpoints,
    one particular value must occur. Path requirements ask
    for a simple directed path of one exact predicate to a named endpoint.
    Reachable facts ask for a final predicate after one or more directed hops
    using the explicit ``via`` predicates; max_hops includes the final fact.
    The final subject/value can be unknown. Each reachable subject contributes
    one shortest witness, with competing values retained. It need not be a leaf.
    Every requirement must be witnessed for ``sufficient``. This means the plan
    has recorded witnesses, not that the records or caller's intent are true.

    The optional bridge follow-up strategy carries one deepest partial route
    and searches from its reached entity. Partial routes are never witnesses.
    Depth ties follow input/traversal order; other branches are not enumerated.

    At most 8 requirements, 100 candidates, 128000 candidate UTF-8 bytes and
    2048 path-edge examinations per call. Candidate indexing/direct lookup are
    separately bounded by those counts. Missing paths are relative to supplied
    evidence and max_hops; no global terminal/negative/completeness claims are supported.
    """

    def __init__(self, question: str, requirements: tuple[EvidenceRequirement, ...], *, max_work: int = 256,
                 followup_strategy: Literal['requirement', 'bridge'] = 'requirement') -> None:
        if type(question) is not str or not question.strip() or len(question.encode()) > 8000:
            raise ValueError("question requires 1..8000 UTF-8 bytes")
        if type(requirements) is not tuple or not 1 <= len(requirements) <= 8:
            raise ValueError("provide 1..8 explicit requirements")
        if any(not isinstance(row, EvidenceRequirement) for row in requirements):
            raise ValueError("invalid requirement")
        self._requirements = tuple(EvidenceRequirement.model_validate(dict(vars(row))) for row in requirements)
        if len(set(self._requirements)) != len(self._requirements):
            raise ValueError("requirements must be unique")
        if type(max_work) is not int or not 1 <= max_work <= 2048:
            raise ValueError("max_work must be an integer in 1..2048")
        if not isinstance(followup_strategy, str) or followup_strategy not in ('requirement', 'bridge'):
            raise ValueError('invalid followup strategy')
        self._followup_strategy = followup_strategy
        self._question, self._max_work = question, max_work

    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        return (await self.assess_with_coverage(question, candidates)).decision

    async def assess_with_coverage(self, question: str,
                                  candidates: tuple[EvidenceCandidate, ...]) -> StructuredEvidenceAssessment:
        """Return every requirement's witnesses without retaining per-call state."""
        if type(question) is not str or question != self._question:
            raise ValueError("question does not match the explicit plan")
        if (type(candidates) is not tuple or len(candidates) > 100
                or any(not isinstance(row, EvidenceCandidate) for row in candidates)):
            raise ValueError("provide at most 100 evidence candidates")
        frozen = tuple(EvidenceCandidate.model_validate(dict(vars(row))) for row in candidates)
        if len({row.id for row in frozen}) != len(frozen) or evidence_payload_bytes(frozen) > 128000:
            raise ValueError("candidate identities or payload exceed limits")
        index = _facts(frozen)
        selected: dict[str, None] = {}
        groups: list[tuple[str, ...]] = []
        missing: list[str] = []
        uncertain = False
        coverage: list[RequirementCoverage] = []
        remaining = self._max_work
        for requirement in self._requirements:
            before, exhausted = remaining, False
            bridge = _Bridge()
            if requirement.kind == "fact":
                observations = (index.get((requirement.subject, requirement.predicate), [])
                    if requirement.subject is not None else [row for (_, predicate), rows in index.items()
                        if predicate == requirement.predicate for row in rows if row.object == requirement.object])
                witnesses = tuple(row for row in observations
                                  if requirement.object is None or row.object == requirement.object)
            elif requirement.kind == "path":
                witnesses, remaining, exhausted = _path(requirement, index, remaining, bridge)
                uncertain = uncertain or exhausted
            else:
                witnesses, remaining, exhausted = _reachable_fact(requirement, index, remaining, bridge)
                uncertain = uncertain or exhausted
            retained_bridge = bridge.path if not witnesses and self._followup_strategy == 'bridge' else ()
            query = None
            if not witnesses:
                query = bridge.query(requirement) if retained_bridge else requirement.search_query()
                missing.append(query)
            # Keep competing values for every witnessed subject/predicate, not
            # just the value along the selected route. No winner is inferred.
            ids = tuple(dict.fromkeys(row.id for witness in (*witnesses, *retained_bridge)
                for row in index[(witness.subject or "", witness.predicate or "")]))
            selected.update((identifier, None) for identifier in ids)
            coverage.append(RequirementCoverage(requirement=requirement, witnessed=bool(witnesses),
                witness_ids=tuple(row.id for row in witnesses), bridge_ids=tuple(row.id for row in retained_bridge),
                selected_ids=ids, followup_query=query,
                work_used=before - remaining, work_exhausted=exhausted))
            if len(ids) > 1:
                groups.append(ids)
        decision = EvidenceDecision(status="uncertain" if uncertain else "insufficient" if missing else "sufficient",
            selected_ids=tuple(selected), selected_groups=merge_groups(tuple(groups)),
            followup_queries=tuple(dict.fromkeys(missing))[:3])
        return StructuredEvidenceAssessment(decision=decision, coverage=tuple(coverage),
            work_limit=self._max_work, work_used=self._max_work - remaining)
