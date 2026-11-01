"""Choosing passages for a token budget by what they cover, not one by one.

Filling a context window in relevance order spends the budget on the
best passage, then the next best, however much the second repeats the
first. What a reader needs is coverage: every relevant candidate should
have something in the window that stands for it. That is a facility
location objective, weighted by relevance (the demand-coverage form of
GeoRAG and the RCD objective, 2026):

    f(S) = sum over candidates i of  relevance(i) * max over j in S of like(i, j)
           + relevance_weight * sum over j in S of relevance(j)

``like`` is cosine clipped at zero. Both terms are monotone submodular,
so a greedy choice by gain per token, beside the best single passage
that fits, is within a constant factor of the best possible window
(Khuller, Moss and Naor, 1999; Leskovec et al., 2007). Gains are kept in
a lazy priority queue (CELF): a passage's gain can only fall as others
are chosen, so most are never recomputed.

The likeness of every pair is computed once, up front: with numpy as one
matrix product, without it as plain sums of products. Either way it is
the cost that grows with the square of the candidates.

Gain per token favours short passages, and a short passage of little
relevance can take the place of a long one that matters. Lin and Bilmes
(2010) rank by gain over ``cost ** cost_exponent`` instead: 1 is gain
per token, 0 ignores length until the budget runs out, and values
between trade the two. The default, 0.3, is the value they found best
for summaries and the best or nearly best here (below).

Measured gains from such packing depend on the reader: they appear for
multi-hop questions and weaker readers and fade with stronger ones
("What survives into context", 2026). It is offered, not imposed.

Measured (HotpotQA distractor dev, 1,000 questions, two embedders,
settings chosen on the other half of the questions): a window holding
both gold paragraphs, against filling in relevance order. With gain per
token the window lost at every budget, 0.022 to 0.081, because short
distractors were taken first; with the exponent chosen it was level
(+0.018 and +0.010 at 400 tokens, 0.000 and -0.001 at 250, -0.009 and
-0.013 at 150, only the last significant). The coverage term itself
added little: the chosen settings often made it negligible.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from operator import mul
from typing import Any, Generic, Hashable, Mapping, Sequence, TypeVar

try:  # numpy makes every likeness one matrix product; it is not required.
    import numpy as _numpy
except ImportError:  # pragma: no cover - numpy ships with most extras
    _numpy = None  # type: ignore[assignment]

Key = TypeVar("Key", bound=Hashable)


@dataclass(frozen=True)
class Packed(Generic[Key]):
    """The passages chosen for a budget."""

    #: In the order chosen: most gain per scaled cost first.
    chosen: tuple[Key, ...]
    cost: int
    budget: int
    objective: float
    #: True when the best single passage beat the greedy set and was used alone.
    single: bool
    #: Candidates costing more than the whole budget, never considered.
    too_large: int


def _units(vectors: Mapping[Key, Sequence[float]]) -> dict[Key, list[float]]:
    out = {}
    for key, vector in vectors.items():
        norm = math.sqrt(math.fsum(value * value for value in vector))
        out[key] = [value / norm for value in vector] if norm else [0.0] * len(vector)
    return out


def _likeness(keys: Sequence[Key], units: Mapping[Key, Sequence[float]]) -> dict[Key, dict[Key, float]]:
    """Cosine between every pair, clipped at zero; a zero vector is like nothing, itself included."""
    if _numpy is not None and keys:
        matrix: Any = _numpy.asarray([units[key] for key in keys], dtype=_numpy.float64)
        rows = _numpy.clip(matrix @ matrix.T, 0.0, None).tolist()
        return {left: dict(zip(keys, row)) for left, row in zip(keys, rows)}
    like: dict[Key, dict[Key, float]] = {key: {} for key in keys}
    for index, left in enumerate(keys):
        like[left][left] = 1.0 if any(units[left]) else 0.0
        for right in keys[index + 1:]:
            like[left][right] = like[right][left] = max(0.0, sum(map(mul, units[left], units[right])))
    return like


def pack(relevance: Mapping[Key, float], cost: Mapping[Key, int], vectors: Mapping[Key, Sequence[float]],
         budget: int, *, relevance_weight: float = 1.0, cost_exponent: float = 0.3) -> Packed[Key]:
    """The passages to put in a window of ``budget`` tokens.

    ``relevance`` weighs each candidate's demand for coverage and is
    clipped at zero; ``cost`` is its size in tokens and must be positive.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not math.isfinite(relevance_weight) or relevance_weight < 0:
        raise ValueError("relevance_weight must be a finite, non-negative number")
    if not math.isfinite(cost_exponent) or not 0 <= cost_exponent <= 1:
        raise ValueError("cost_exponent must be from 0 to 1")
    keys = list(relevance)
    for key in keys:
        if key not in cost or key not in vectors:
            raise ValueError(f"{key!r} has no cost or no vector")
        if type(cost[key]) is not int or cost[key] < 1:
            raise ValueError(f"{key!r} must cost a positive whole number of tokens")
        if not math.isfinite(relevance[key]):
            raise ValueError(f"{key!r} has a non-finite relevance")
    demand = {key: max(0.0, relevance[key]) for key in keys}
    like = _likeness(keys, _units({key: vectors[key] for key in keys}))
    fitting = [key for key in keys if cost[key] <= budget]

    def objective(chosen: Sequence[Key]) -> float:
        if not chosen:
            return 0.0
        covered = math.fsum(demand[i] * max(like[i][j] for j in chosen) for i in keys)
        return covered + relevance_weight * math.fsum(demand[j] for j in chosen)

    best_cover = {key: 0.0 for key in keys}

    def gain(candidate: Key) -> float:
        covered = math.fsum(demand[i] * max(0.0, like[i][candidate] - best_cover[i]) for i in keys)
        return covered + relevance_weight * demand[candidate]

    scale = {key: cost[key] ** cost_exponent for key in fitting}
    # CELF: a max-heap of (gain per scaled cost, gain, round computed, position, key)
    heap = [(-gain(key) / scale[key], gain(key), 0, position, key) for position, key in enumerate(fitting)]
    heapq.heapify(heap)
    chosen: list[Key] = []
    spent = 0
    round_ = 0
    while heap:
        ratio, value, computed, position, key = heapq.heappop(heap)
        if spent + cost[key] > budget:
            continue
        if computed != round_:
            fresh = gain(key)
            heapq.heappush(heap, (-fresh / scale[key], fresh, round_, position, key))
            continue
        if value <= 0:
            break
        chosen.append(key)
        spent += cost[key]
        round_ += 1
        for i in keys:
            best_cover[i] = max(best_cover[i], like[i][key])
    greedy_value = objective(chosen)
    single = max(fitting, key=lambda key: (objective([key]), -fitting.index(key)), default=None)
    if single is not None and objective([single]) > greedy_value:
        return Packed(chosen=(single,), cost=cost[single], budget=budget, objective=objective([single]),
                      single=True, too_large=len(keys) - len(fitting))
    return Packed(chosen=tuple(chosen), cost=spent, budget=budget, objective=greedy_value, single=False,
                  too_large=len(keys) - len(fitting))
