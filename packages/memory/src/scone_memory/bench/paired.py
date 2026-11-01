"""Paired comparisons of two runs over the same items.

A mean delta alone hides how many items moved. 0.02 on fifty items is a
single item changing its mind, and the same 0.02 can come from one large
win among many small losses. Reading a delta without its counts is how a
change gets called an improvement on the strength of noise.

So a comparison here reports, beside the mean delta, how many items the
candidate won, lost and tied, and the exact two-sided sign test over the
wins and losses. The sign test needs nothing but the counts: it asks how
surprising this split would be if each item were a fair coin. Ties carry
no direction and are left out of it, which is the test's own definition,
not a choice made here.

The sign test says whether the direction is credible; it says nothing
about size. ``bootstrap_mean_delta`` gives the size a range: a seeded
percentile bootstrap over the paired items.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class PairedOutcome:
    """How a candidate run did against a baseline, item by item."""

    n: int
    wins: int
    losses: int
    ties: int
    #: Mean of candidate minus baseline; ``None`` when there were no items.
    mean_delta: Optional[float]
    #: Exact two-sided sign test over wins and losses; 1.0 when none moved.
    sign_test_p: float


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test: the chance of a split at least this
    uneven if every item were a fair coin. Computed from integer binomial
    counts, so it is exact at any size rather than an approximation."""
    if type(wins) is not int or type(losses) is not int or wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative integers")
    moved = wins + losses
    if moved == 0:
        return 1.0
    tail = sum(math.comb(moved, count) for count in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** moved)


def _finite(values: Sequence[float], side: str) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{side} holds a non-finite score: {value!r}")


def paired_outcome(baseline: Sequence[float], candidate: Sequence[float], *,
                   tolerance: float = 0.0) -> PairedOutcome:
    """Compare two runs scored on the same items, in the same order.

    An item whose scores differ by no more than ``tolerance`` is a tie,
    so floating-point noise in a score does not count as a win or a loss.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"the runs score different numbers of items: {len(baseline)} and {len(candidate)}")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number")
    _finite(baseline, "baseline")
    _finite(candidate, "candidate")
    deltas = [after - before for before, after in zip(baseline, candidate)]
    wins = sum(delta > tolerance for delta in deltas)
    losses = sum(delta < -tolerance for delta in deltas)
    return PairedOutcome(
        n=len(deltas),
        wins=wins,
        losses=losses,
        ties=len(deltas) - wins - losses,
        mean_delta=math.fsum(deltas) / len(deltas) if deltas else None,
        sign_test_p=sign_test(wins, losses),
    )


MAX_RESAMPLES = 1_000_000


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval around a paired mean delta."""

    estimate: float
    low: float
    high: float
    confidence: float
    resamples: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        """Whether every plausible mean delta points the same way."""
        return self.low > 0 or self.high < 0


def bootstrap_mean_delta(baseline: Sequence[float], candidate: Sequence[float], *,
                         resamples: int = 10_000, confidence: float = 0.95, seed: int = 0) -> Optional[Interval]:
    """Percentile bootstrap interval for the mean of candidate minus baseline.

    The items are resampled as pairs, so an item hard for both runs stays
    hard for both in every resample; resampling each run on its own would
    throw that pairing away and widen the interval for no reason. The seed
    is part of the result, so a reported interval can be reproduced.
    ``None`` when there are no items to resample.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"the runs score different numbers of items: {len(baseline)} and {len(candidate)}")
    if type(resamples) is not int or not 1 <= resamples <= MAX_RESAMPLES:
        raise ValueError(f"resamples must be an integer from 1 to {MAX_RESAMPLES}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    _finite(baseline, "baseline")
    _finite(candidate, "candidate")
    deltas = [after - before for before, after in zip(baseline, candidate)]
    if not deltas:
        return None
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return Interval(
        estimate=math.fsum(deltas) / len(deltas),
        low=_nearest_rank(means, tail),
        high=_nearest_rank(means, 1 - tail),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def _nearest_rank(ordered: Sequence[float], q: float) -> float:
    return ordered[min(len(ordered), max(1, math.ceil(q * len(ordered)))) - 1]
