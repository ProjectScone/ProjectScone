"""Shared reporting contract. Scores describe a named metric, not generic quality."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: int
    title: str
    hypothesis: str
    baseline: str
    method: str
    metric: str
    baseline_score: float
    method_score: float
    higher_is_better: bool
    units: str
    cases: int
    evidence_kind: Literal['simulation', 'live_jev', 'mixed']
    details: list[dict[str, object]]
    limitations: tuple[str, ...]
    references: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.experiment_id) is not int or not 1 <= self.experiment_id <= 20:
            raise ValueError('experiment ID must be 1..20')
        if type(self.cases) is not int or self.cases < 1 or not self.details:
            raise ValueError('experiment must retain observations')
        for value in (self.baseline_score, self.method_score, *self.metrics.values()):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('experiment metrics must be finite numbers')
        if not self.limitations or not self.references:
            raise ValueError('experiment must state limits and related research')

    @property
    def delta(self) -> float:
        return self.method_score - self.baseline_score
