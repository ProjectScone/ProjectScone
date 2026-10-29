"""How many results to keep, chosen per query from where the scores fall.

A fixed ``k`` gives every question the same number of passages: too few
for a question with many good answers, and for a question with one, a
reader handed nine passages it should not trust. The scores often say
where the good ones end. Ranked by similarity, they tend to fall in
shelves: a steep drop after the passages that match, then a long flat
tail of passages that do not (Cluster-based Adaptive Retrieval, 2025;
Tail-Aware Adaptive-k, 2026).

``gap_cutoff`` cuts at the largest drop between ``min_keep`` and
``max_keep``, measured against the whole spread it looked at, less a
small charge for cutting late so that a drop at rank 20 must be larger
than one at rank 3 to win. When no drop stands out (``min_gap``) the
scores give no boundary, and it keeps ``max_keep`` rather than invent
one. The result names the rule that decided, so a reader can tell a
cut the scores chose from one a bound imposed.

Measured against a fixed ``k`` interpolated to the same mean number of
passages kept (two embedders, embeddinggemma and nomic-embed-text, the
cosines of each question to its candidates):

- **One answer passage** (SQuAD v1.1 dev, 1,000 questions, every
  paragraph of the article as candidates): the gold paragraph was kept
  0.052 and 0.042 more often at a mean of 1.45 and 1.59 passages (90/21
  and 117/37 questions; both p < 1e-10).
- **Two passages, the second unlike the question** (HotpotQA distractor
  dev, 1,000 questions): 0.084 to 0.107 less of the gold was kept. The
  bridge paragraph sits below the shelf the first one makes, so the cut
  falls between the two.

So it suits lookups of one fact, and a question that needs a second hop
should keep a fixed ``k``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Cutoff:
    """How many results to keep, and which rule decided it."""

    keep: int
    #: ``gap`` (the scores chose), ``no-boundary`` (no drop stood out, so
    #: ``max_keep``), ``short`` (fewer results than ``min_keep``).
    rule: str
    #: The chosen drop over the spread examined; ``None`` unless ``gap``.
    relative_gap: Optional[float]
    why: str


def gap_cutoff(scores: Sequence[float], *, min_keep: int = 1, max_keep: int = 20, min_gap: float = 0.15,
               late_charge: float = 0.05) -> Cutoff:
    """Where to cut a best-first score list.

    A cut after position ``i`` (1-based) is worth its drop
    ``scores[i-1] - scores[i]`` over the spread of the scores examined,
    less ``late_charge * i / max_keep``. The best cut is taken if its
    drop alone is at least ``min_gap`` of the spread.
    """
    if type(min_keep) is not int or type(max_keep) is not int or not 1 <= min_keep <= max_keep:
        raise ValueError("min_keep and max_keep must be integers with 1 <= min_keep <= max_keep")
    if not (0 <= min_gap <= 1 and math.isfinite(late_charge) and late_charge >= 0):
        raise ValueError("min_gap must lie in [0, 1] and late_charge must be finite and non-negative")
    for score in scores:
        if not math.isfinite(score):
            raise ValueError(f"a score is not finite: {score!r}")
    if any(later > earlier for earlier, later in zip(scores, scores[1:])):
        raise ValueError("scores must be best first")
    if len(scores) <= min_keep:
        return Cutoff(keep=len(scores), rule="short", relative_gap=None,
                      why=f"{len(scores)} results, no more than the {min_keep} always kept")
    examined = list(scores[:max_keep + 1])
    spread = examined[0] - examined[-1]
    last_cut = min(max_keep, len(scores) - 1)
    if spread <= 0:
        keep = min(max_keep, len(scores))
        return Cutoff(keep=keep, rule="no-boundary", relative_gap=None,
                      why=f"every score examined is equal, so the {keep} allowed were kept")
    best_position, best_value, best_gap = min_keep, -math.inf, 0.0
    for position in range(min_keep, last_cut + 1):
        gap = (examined[position - 1] - examined[position]) / spread
        value = gap - late_charge * position / max_keep
        if value > best_value:
            best_position, best_value, best_gap = position, value, gap
    if best_gap < min_gap:
        keep = min(max_keep, len(scores))
        return Cutoff(keep=keep, rule="no-boundary", relative_gap=None,
                      why=f"no drop reached {min_gap:.0%} of the spread, so the {keep} allowed were kept")
    return Cutoff(keep=best_position, rule="gap", relative_gap=best_gap,
                  why=f"cut after {best_position}: the scores dropped by {best_gap:.0%} of their spread there")
