"""Pseudo-relevance feedback in embedding space, and when to hold it back.

Rocchio's rule moves the query towards what it found: the new query is
the old one plus ``beta`` times the mean of the top results. Over dense
vectors it needs no terms and no training (Li et al., 2022, "Average"
and "Rocchio" dense PRF), and a second search with the moved query
reaches passages the first one only nearly found.

It fails in a known way: when the top results are wrong, the query is
moved towards the wrong thing, and the second search confidently finds
more of it. So feedback can be gated on a predictor of the first list's
quality (``retrieval.qpp``): below ``gate`` the query is left as it was,
and the result says why.

Measured, and not used by recall. On DBpedia-Entity (381 queries, 999,000
entities, exact search), settings chosen on one half of the queries and
scored on the other (depth 3, 5 or 10; beta 0.25, 0.5 or 1):

- Ungated, nDCG@10 moved +0.0020 at 768-d (more queries gained than
  lost, 136/94, but the bootstrap interval of the mean holds zero) and
  -0.0045 at 3072-d (127/117). Neither is a gain to rely on.
- Gated on any of the six ``qpp`` signals, nothing was significant: the
  best was +0.0028 (nqc, 768-d), the worst -0.0051 (centred coherence,
  3072-d). Which side of the gate gets feedback was chosen from the
  data too, and it went against the reasoning above in 20 of 24 choices
  (six signals, two folds, two embedders): feedback was held back from
  the queries predicted to be going *best* and given to the rest.
- Choosing per query, with hindsight, between the nine settings and no
  feedback at all would have gained +0.0425 and +0.0340. The gain is
  there to have; a first list's shape does not say which queries hold it.

So this module is a measured mechanism to build on, not a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Feedback:
    """The query to search with, and whether feedback moved it."""

    vector: tuple[float, ...]
    expanded: bool
    #: Feedback vectors averaged into it.
    used: int
    why: str


def _unit(vector: Sequence[float]) -> Optional[list[float]]:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm):
        raise ValueError("a vector holds a non-finite value")
    return [value / norm for value in vector] if norm else None


def rocchio(query: Sequence[float], feedback: Sequence[Sequence[float]], *, beta: float = 0.5,
            depth: int = 5) -> Feedback:
    """The unit query plus ``beta`` times the mean of the first ``depth``
    unit feedback vectors, returned at unit length.

    Every vector is taken at unit length first, so a long passage vector
    does not pull harder than a short one; zero vectors are skipped.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be a finite, non-negative number")
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    unit_query = _unit(query)
    if unit_query is None:
        raise ValueError("the query vector is zero")
    units = []
    for vector in feedback[:depth]:
        if len(vector) != len(unit_query):
            raise ValueError(f"a feedback vector has {len(vector)} dimensions; the query has {len(unit_query)}")
        unit = _unit(vector)
        if unit is not None:
            units.append(unit)
    if not units or beta == 0:
        why = "no feedback vector had a direction" if not units else "beta is zero"
        return Feedback(vector=tuple(unit_query), expanded=False, used=0, why=why)
    moved = [q + beta * math.fsum(unit[index] for unit in units) / len(units)
             for index, q in enumerate(unit_query)]
    result = _unit(moved)
    if result is None:  # the feedback pointed exactly away from the query
        return Feedback(vector=tuple(unit_query), expanded=False, used=0,
                        why="the feedback cancelled the query out")
    return Feedback(vector=tuple(result), expanded=True, used=len(units),
                    why=f"moved towards the mean of {len(units)} results with beta {beta}")


def gated_rocchio(query: Sequence[float], feedback: Sequence[Sequence[float]], *, signal: Optional[float],
                  gate: float, beta: float = 0.5, depth: int = 5) -> Feedback:
    """``rocchio`` only when the first list's predicted quality ``signal``
    reaches ``gate``; otherwise the query unchanged, saying why. A missing
    signal is not evidence of quality, so it holds feedback back."""
    if not math.isfinite(gate):
        raise ValueError("gate must be finite")
    if signal is None or signal < gate:
        unit_query = _unit(query)
        if unit_query is None:
            raise ValueError("the query vector is zero")
        reason = ("no quality signal for the first list" if signal is None
                  else f"predicted quality {signal:.4g} is below the gate {gate:.4g}")
        return Feedback(vector=tuple(unit_query), expanded=False, used=0, why=reason)
    return rocchio(query, feedback, beta=beta, depth=depth)
