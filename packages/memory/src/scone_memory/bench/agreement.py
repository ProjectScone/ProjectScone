"""Whether two scorings put the same things in the same order.

Kendall's tau-b counts, over every pair of items, whether the two
scorings agree on which is higher. It is the usual measure for a query
performance predictor (does it rank queries the way their real scores
do?) and for whether two benchmarks rank systems alike. Ties are common
in both (many queries score 0 or 1), and tau-b is the variant that
corrects for them; tau-a would read a list full of ties as disagreement.

Items present in only one scoring are reported, never silently paired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, TypeVar

Key = TypeVar("Key", bound=Hashable)


@dataclass(frozen=True)
class Agreement:
    """How alike two scorings order the items they share."""

    #: Kendall's tau-b in [-1, 1]; ``None`` when either scoring is all ties.
    tau_b: Optional[float]
    items: int
    concordant: int
    discordant: int
    #: Pairs tied in the first scoring, the second, and both.
    tied_first: int
    tied_second: int
    tied_both: int
    #: Items scored by only one side.
    unmatched: int


def kendall_tau_b(first: Mapping[Key, float], second: Mapping[Key, float]) -> Agreement:
    """Kendall's tau-b over the items both scorings hold.

    Exact, over all pairs: suited to the hundreds of queries or tens of
    systems it is meant for, not to ranking a whole corpus.
    """
    shared = [key for key in first if key in second]
    for key in shared:
        if not (math.isfinite(first[key]) and math.isfinite(second[key])):
            raise ValueError(f"{key!r} has a non-finite score")
    unmatched = len(first) + len(second) - 2 * len(shared)
    concordant = discordant = tied_first = tied_second = tied_both = 0
    for index, left in enumerate(shared):
        for right in shared[index + 1:]:
            a = first[left] - first[right]
            b = second[left] - second[right]
            if a == 0 and b == 0:
                tied_both += 1
            elif a == 0:
                tied_first += 1
            elif b == 0:
                tied_second += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    untied_first = concordant + discordant + tied_second
    untied_second = concordant + discordant + tied_first
    denominator = math.sqrt(untied_first * untied_second)
    return Agreement(
        tau_b=(concordant - discordant) / denominator if denominator else None,
        items=len(shared), concordant=concordant, discordant=discordant,
        tied_first=tied_first, tied_second=tied_second, tied_both=tied_both, unmatched=unmatched,
    )
