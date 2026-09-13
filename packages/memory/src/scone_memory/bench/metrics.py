"""Rank-aware retrieval metrics, alongside the recall we already report.

Recall@k answers "was the evidence in the top k". It cannot answer
"where", so a change that moves every answer from tenth place to first
scores exactly the same, and a change that returns nine wasted passages
beside one good one scores perfectly. Both matter to somebody reading the
result, and neither is visible in the numbers this benchmark reported.

The reference RAG framework reports hit rate, MRR, precision, recall,
average precision and NDCG. Ours reported recall@k alone, so every
comparison drawn against that framework was weaker than it looked. All
five it had and we lacked are here now; each needs no model and no
labels beyond the ones the dataset already carries.

Relevance here is binary -- a returned source either is one of the
answer's sources or is not -- because that is what the datasets state.
Graded relevance would need graded labels, and inventing them would make
the numbers look richer without making them truer.
"""

from __future__ import annotations

from math import log2
from typing import Iterable, Sequence


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """One over the position of the first relevant source, or zero.

    About the **first** one reached, even when several are relevant:
    that is what makes it a measure of how soon a reader finds something
    useful rather than how much there was to find.
    """
    for position, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1.0 / position
    return 0.0


def mrr(pairs: Iterable[tuple[Sequence[str], set[str]]]) -> float:
    """The mean reciprocal rank over items; zero over no items."""
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in pairs]
    return sum(scores) / len(scores) if scores else 0.0


def precision_at(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """The share of the first ``k`` slots that were worth returning.

    A source returned twice fills two slots and answers one question, so
    it counts once above the line and twice below it. Divided by ``k``
    rather than by the number returned: a run that returns three things
    when asked for ten has spent the other seven on nothing, and hiding
    that would flatter it.
    """
    if k <= 0:
        return 0.0
    found = {source for source in retrieved[:k] if source in relevant}
    return len(found) / k


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Precision measured at each relevant hit, averaged over them all.

    Where the reciprocal rank stops at the first relevant source, this
    keeps counting, so it can tell `[a, b]` from `[a, x, b]` -- both find
    something first and only one found the rest early. That is the whole
    reason the reference reports both.

    Divided by how many were **findable**, not by how many were found: a
    run that returns one of three possible answers has missed two, and
    dividing by the one it found would call that perfect. A source
    returned twice fills two slots and answers one question, so it
    raises the denominator of the later hits without scoring again.
    """
    if not relevant:
        return 0.0
    seen: set[str] = set()
    total = 0.0
    for position, source in enumerate(retrieved, start=1):
        if source in relevant and source not in seen:
            seen.add(source)
            total += len(seen) / position
    return total / len(relevant)


def mean_average_precision(pairs: Iterable[tuple[Sequence[str], set[str]]]) -> float:
    """Average precision over items; zero over no items."""
    scores = [average_precision(retrieved, relevant) for retrieved, relevant in pairs]
    return sum(scores) / len(scores) if scores else 0.0


def hit_rate(pairs: Iterable[tuple[Sequence[str], set[str]]], k: int) -> float:
    """The share of items whose first ``k`` results held anything relevant.

    The coarsest of these and the one the reference leads with. It says
    nothing about where or how many, which is exactly why it belongs
    beside the others rather than instead of them: a run can lose half
    its precision without moving this at all.
    """
    items = list(pairs)
    if not items or k <= 0:
        return 0.0
    landed = sum(1 for retrieved, relevant in items
                 if any(source in relevant for source in retrieved[:k]))
    return landed / len(items)


def ndcg_at(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted gain against the best possible ordering, in [0, 1].

    Each relevant source is worth `1 / log2(position + 1)`, so the same
    answers ranked earlier score higher -- the property recall cannot
    see. The ideal is those same answers filling the first slots, so a
    run cannot be punished for evidence the dataset never had.
    """
    if k <= 0 or not relevant:
        return 0.0
    seen: set[str] = set()
    gained = 0.0
    for position, source in enumerate(retrieved[:k], start=1):
        if source in relevant and source not in seen:
            seen.add(source)
            gained += 1.0 / log2(position + 1)
    ideal = sum(1.0 / log2(position + 1)
                for position in range(1, min(k, len(relevant)) + 1))
    return gained / ideal if ideal else 0.0
