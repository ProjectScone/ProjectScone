"""Counterfactual evidence contracts, bounded to five provenance families.

These cache semantic judgments, not proofs of truth. A world is an exact subset
of a fixed evidence packet. No monotonicity, independence, or causal inference
is assumed. The host must supply complete snapshots and trusted origin groups.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Literal

Status = Literal['supported', 'refuted', 'conflict', 'insufficient', 'uncertain', 'recompile']
MAX_FAMILIES = 5
POLICY = 'counterfactual-contract-v1:yes=.8:no=.2'


def _text(value: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > limit:
        raise ValueError('invalid contract text or byte limit')


def _finite(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('expected finite number')


@dataclass(frozen=True)
class Evidence:
    source_id: str
    origin: str
    text: str

    def __post_init__(self) -> None:
        _text(self.source_id, 256)
        _text(self.origin, 256)
        _text(self.text, 8000)


@dataclass(frozen=True)
class ContractRequest:
    # Must include workspace, permissions, temporal context and policy revision.
    context_key: str
    question: str
    claim: str
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        _text(self.context_key, 2000)
        _text(self.question, 4000)
        _text(self.claim, 4000)
        if not isinstance(self.evidence, tuple) or len(self.evidence) > 20:
            raise ValueError('expected at most twenty immutable evidence records')
        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise ValueError('invalid evidence record')
        if len({item.source_id for item in self.evidence}) != len(self.evidence):
            raise ValueError('duplicate source identity')
        if sum(len(item.text.encode()) for item in self.evidence) > 16000:
            raise ValueError('evidence packet byte limit')


@dataclass(frozen=True)
class Judgment:
    support: float
    contradiction: float

    def __post_init__(self) -> None:
        for value in (self.support, self.contradiction):
            _finite(value)
            if not 0 <= value <= 1:
                raise ValueError('probability outside [0, 1]')

    @property
    def status(self) -> Status:
        if self.support >= .8 and self.contradiction >= .8:
            return 'conflict'
        if self.support >= .8 and self.contradiction <= .2:
            return 'supported'
        if self.contradiction >= .8 and self.support <= .2:
            return 'refuted'
        if self.support <= .2 and self.contradiction <= .2:
            return 'insufficient'
        return 'uncertain'


@dataclass(frozen=True)
class World:
    mask: int
    evidence: tuple[Evidence, ...]


def evidence_worlds(request: ContractRequest) -> tuple[World, ...]:
    origins = sorted({item.origin for item in request.evidence})
    if len(origins) > MAX_FAMILIES:
        raise ValueError('too many provenance families; never silently truncate evidence')
    positions = {origin: 1 << index for index, origin in enumerate(origins)}
    return tuple(World(mask, tuple(sorted(
        (item for item in request.evidence if mask & positions[item.origin]),
        key=lambda item: item.source_id,
    ))) for mask in range(1 << len(origins)))


@dataclass(frozen=True)
class Evaluation:
    status: Status
    reason: str
    judgment: Judgment | None = None


@dataclass(frozen=True)
class MemoryContract:
    request: ContractRequest
    judgments: tuple[Judgment, ...]
    model: str
    expires_at: float
    policy: str = POLICY

    def __post_init__(self) -> None:
        _text(self.model, 160)
        _finite(self.expires_at)
        if self.policy != POLICY or not isinstance(self.judgments, tuple):
            raise ValueError('unsupported contract policy')
        if len(self.judgments) != len(evidence_worlds(self.request)):
            raise ValueError('incomplete counterfactual table')
        if not all(isinstance(item, Judgment) for item in self.judgments):
            raise ValueError('invalid contract judgment')
        if self.judgments[0].status != 'insufficient':
            raise ValueError('an empty evidence world cannot establish a private claim')

    def evaluate(self, current: ContractRequest, *, now: float) -> Evaluation:
        _finite(now)
        if now >= self.expires_at:
            return Evaluation('recompile', 'contract expired')
        if (current.context_key, current.question, current.claim) != (
            self.request.context_key, self.request.question, self.request.claim,
        ):
            return Evaluation('recompile', 'question, claim or context changed')
        original = {item.source_id: item for item in self.request.evidence}
        if any(original.get(item.source_id) != item for item in current.evidence):
            return Evaluation('recompile', 'new or changed evidence')
        retained = frozenset(item.source_id for item in current.evidence)
        origins = sorted({item.origin for item in self.request.evidence})
        mask = 0
        for index, origin in enumerate(origins):
            members = {item.source_id for item in self.request.evidence if item.origin == origin}
            present = members & retained
            if present and present != members:
                return Evaluation('recompile', 'partially changed provenance family')
            if present:
                mask |= 1 << index
        judgment = self.judgments[mask]
        return Evaluation(judgment.status, 'exact previously evaluated evidence world', judgment)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> MemoryContract:
        """Load a trusted local artifact, validating bounds and table completeness.

        This is not an authenticity check. A serving integration must protect
        artifacts like other decision memory and never load client-supplied tables.
        """
        from pydantic import TypeAdapter
        if not isinstance(value, str) or len(value.encode()) > 128000:
            raise ValueError('contract artifact byte limit')
        return TypeAdapter(cls).validate_json(value, strict=True)

    def minimal_withdrawals(self) -> tuple[tuple[str, ...], ...]:
        """Smallest sets whose withdrawal loses support from the full packet.

        Describes tested worlds only. A superset of a cut need not remain unsafe:
        withdrawing contradictory evidence can restore support.
        """
        if self.judgments[-1].status != 'supported':
            return ()
        origins = sorted({item.origin for item in self.request.evidence})
        full = len(self.judgments) - 1
        cuts: list[int] = []
        for removed in sorted(range(1, full + 1), key=lambda mask: (mask.bit_count(), mask)):
            if self.judgments[full ^ removed].status == 'supported':
                continue
            if not any(cut & removed == cut for cut in cuts):
                cuts.append(removed)
        return tuple(tuple(origin for index, origin in enumerate(origins)
                           if cut & (1 << index)) for cut in cuts)


def compile_contract(request: ContractRequest, judgments: tuple[Judgment, ...], *,
                     model: str, expires_at: float) -> MemoryContract:
    return MemoryContract(request, judgments, model, expires_at)
