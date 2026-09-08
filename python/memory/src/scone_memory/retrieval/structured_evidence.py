"""Positive fact/path coverage for explicit, caller-authored requirements.

This is not natural-language intent detection, semantic entailment, source
verification or proof of global completeness. The adaptive retriever owns scope
and retention. Only exact recorded triples can witness these requirements.
"""
from __future__ import annotations

from collections import deque
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adaptive import EvidenceCandidate, EvidenceDecision, evidence_payload_bytes
from .adaptive_graph import merge_groups


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")
    kind: Literal["fact", "path"]
    subject: str | None = Field(default=None, min_length=1, max_length=128)
    predicate: str = Field(min_length=1, max_length=128)
    object: str | None = Field(default=None, min_length=1, max_length=128)
    max_hops: int = Field(default=6, ge=1, le=6)

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
        return self

    def search_query(self) -> str:
        return " ".join(value for value in (self.subject, self.predicate, self.object) if value is not None)


def _facts(candidates: tuple[EvidenceCandidate, ...]) -> dict[tuple[str, str], list[EvidenceCandidate]]:
    index: dict[tuple[str, str], list[EvidenceCandidate]] = {}
    for candidate in candidates:
        if (candidate.id.startswith("fact:") and candidate.subject is not None and candidate.subject.strip()
                and candidate.predicate is not None and candidate.predicate.strip()
                and candidate.object is not None and candidate.object.strip()):
            index.setdefault((candidate.subject, candidate.predicate), []).append(candidate)
    return index


def _path(requirement: EvidenceRequirement, index: dict[tuple[str, str], list[EvidenceCandidate]],
          remaining: int) -> tuple[tuple[EvidenceCandidate, ...], int, bool]:
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
            pending.append((target, extended, visited | {target}))
    return (), remaining, False


class StructuredEvidenceAssessor:
    """An AdaptiveRetriever assessor for a fixed question and explicit plan.

    Fact requirements request recorded values for an exact subject/predicate,
    or recorded subjects for an exact predicate/object. With both endpoints,
    one particular value must occur. Path requirements ask
    for a simple directed path of one exact predicate to a named endpoint.
    Every requirement must be witnessed for ``sufficient``. This means the plan
    has recorded witnesses, not that the records or caller's intent are true.

    At most 8 requirements, 100 candidates, 128000 candidate UTF-8 bytes and
    2048 path-edge examinations per call. Candidate indexing/direct lookup are
    separately bounded by those counts. Missing paths are relative to supplied
    evidence and max_hops; no terminal/negative/completeness claims are supported.
    """

    def __init__(self, question: str, requirements: tuple[EvidenceRequirement, ...], *, max_work: int = 256) -> None:
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
        self._question, self._max_work = question, max_work

    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
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
        remaining = self._max_work
        for requirement in self._requirements:
            if requirement.kind == "fact":
                observations = (index.get((requirement.subject, requirement.predicate), [])
                    if requirement.subject is not None else [row for (_, predicate), rows in index.items()
                        if predicate == requirement.predicate for row in rows if row.object == requirement.object])
                witnesses = tuple(observations) if requirement.object is None or any(
                    row.object == requirement.object for row in observations) else ()
            else:
                witnesses, remaining, exhausted = _path(requirement, index, remaining)
                uncertain = uncertain or exhausted
            if not witnesses:
                missing.append(requirement.search_query())
                continue
            # Keep competing values for every witnessed subject/predicate, not
            # just the value along the selected route. No winner is inferred.
            ids = tuple(dict.fromkeys(row.id for witness in witnesses
                for row in index[(witness.subject or "", witness.predicate or "")]))
            selected.update((identifier, None) for identifier in ids)
            if len(ids) > 1:
                groups.append(ids)
        return EvidenceDecision(status="uncertain" if uncertain else "insufficient" if missing else "sufficient",
            selected_ids=tuple(selected), selected_groups=merge_groups(tuple(groups)),
            followup_queries=tuple(dict.fromkeys(missing))[:3])
