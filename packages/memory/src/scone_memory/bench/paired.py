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
"""

from __future__ import annotations

import math
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
