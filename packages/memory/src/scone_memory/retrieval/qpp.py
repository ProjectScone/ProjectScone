"""Predicting from a result list alone whether it is likely to be good.

A recall cannot see its own relevance, but it can see the shape of what
it found. Query performance prediction (QPP) reads that shape:

- ``dispersion`` -- the spread of the top ``k`` scores. NQC (Shtok et
  al., 2012) divides it by the corpus score; a dense index has no corpus
  score to give, so ``nqc`` here divides by the mean of the whole list
  that was read, and says so.
- ``wig`` -- how far the top ``k`` stand above that same reference
  (Zhou and Croft, 2007), without the query-length term, which belongs
  to term-matching scores.
- ``gap`` -- the top score less the ``k``-th.
- ``coherence`` -- the mean cosine between the top results' own vectors.
  Results that agree with one another are about one thing; results
  scattered over the space found nothing in particular. Raw cosines in
  an anisotropic space are high for any pair, so a ``Centre`` may be
  given to measure agreement from it.

Classical predictors transfer poorly to dense retrieval (Faggioli et al.,
2023), which is why none of these is trusted here: each is a number to
be checked against measured quality before anything acts on it.

Measured on DBpedia-Entity (381 judged queries, 999,000 entities, exact
search, the top 200 scores read, ``k`` 10, coherence over the top 5),
Kendall tau-b against each query's nDCG@10, for two embedders:

=====================  =====================  =============================
predictor              768-d embeddinggemma   3072-d text-embedding-3-large
=====================  =====================  =============================
dispersion             +0.320                 +0.376
nqc                    +0.300                 +0.359
wig                    +0.324                 +0.372
gap                    +0.309                 +0.374
coherence              +0.174                 +0.228
coherence, centred     +0.160                 +0.215
=====================  =====================  =============================

So the score-shaped signals rank queries by quality about equally well,
and better than coherence; centring did not help coherence on either
embedder. A tau-b of 0.3 orders pairs of queries rightly about two
times in three: enough to report beside an answer, not enough to act on
alone (see ``retrieval.feedback``, where gating on it did not help).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .geometry import Centre


@dataclass(frozen=True)
class Prediction:
    """Signals of one result list's quality; ``None`` where a signal had
    too little to read."""

    dispersion: Optional[float]
    nqc: Optional[float]
    wig: Optional[float]
    gap: Optional[float]
    coherence: Optional[float]
    k: int
    #: Scores read for the reference mean.
    depth: int
    #: Result vectors that the coherence was measured over.
    coherence_over: int


def _std(values: Sequence[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))


def _directions(vectors: Sequence[Sequence[float]], centre: Optional[Centre]) -> list[list[float]]:
    out = []
    for vector in vectors:
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not norm:
            continue
        unit = [value / norm for value in vector]
        if centre is not None:
            if len(centre.mean) != len(unit):
                raise ValueError("the centre and the vectors have different dimensions")
            unit = [value - mean for value, mean in zip(unit, centre.mean)]
            shifted = math.sqrt(math.fsum(value * value for value in unit))
            if shifted < 1e-12:
                continue
            unit = [value / shifted for value in unit]
        out.append(unit)
    return out


def coherence(vectors: Sequence[Sequence[float]], *, centre: Optional[Centre] = None) -> Optional[float]:
    """Mean pairwise cosine among ``vectors``; ``None`` with fewer than two
    that have a direction."""
    units = _directions(vectors, centre)
    if len(units) < 2:
        return None
    total = math.fsum(math.fsum(a * b for a, b in zip(units[i], units[j]))
                      for i in range(len(units)) for j in range(i + 1, len(units)))
    return total / (len(units) * (len(units) - 1) / 2)


def predict(scores: Sequence[float], *, k: int = 10, vectors: Sequence[Sequence[float]] = (),
            centre: Optional[Centre] = None) -> Prediction:
    """Every predictor for one list. ``scores`` are best first and as deep
    as was read (the reference mean is over all of them); ``vectors``, if
    given, are the top results' in the same order."""
    if type(k) is not int or k < 2:
        raise ValueError("k must be an integer of at least 2: one score has no spread")
    for score in scores:
        if not math.isfinite(score):
            raise ValueError(f"a score is not finite: {score!r}")
    if any(later > earlier for earlier, later in zip(scores, scores[1:])):
        raise ValueError("scores must be best first")
    top = list(scores[:k])
    dispersion = wig = nqc = gap = None
    if len(top) >= 2:
        reference = math.fsum(scores) / len(scores)
        dispersion = _std(top)
        wig = math.fsum(top) / len(top) - reference
        nqc = dispersion / abs(reference) if reference else None
        gap = top[0] - top[-1]
    agreed = coherence(vectors, centre=centre) if vectors else None
    return Prediction(dispersion=dispersion, nqc=nqc, wig=wig, gap=gap, coherence=agreed, k=k,
                      depth=len(scores), coherence_over=len(vectors))
