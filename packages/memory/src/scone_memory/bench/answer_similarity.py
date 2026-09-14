"""How close a generated answer is to its reference, by embedding.

Exact match and token F1 score a correct paraphrase as wrong. An embedding
similarity gives it credit, but its scale belongs to the embedder that made
it, so a pass needs a threshold measured on that embedder: a threshold
carries the embedder id and width it came from, and one from anywhere else
decides nothing. Measuring one needs no judge. Answers that match their
reference exactly are passes it must allow; each answer set against another
question's reference is a pass it must refuse; the threshold is the lowest
that lets through at most the target share of those. A pass is a score above
the threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

#: The share of mismatched pairs a threshold may pass, by default.
DEFAULT_FALSE_PASS = 0.05


class _Embeds(Protocol):
    id: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class SimilarityThreshold:
    value: float
    embedder_id: str
    dim: int
    #: What it was measured from: counts, rates and the target.
    measured: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerSimilarity:
    #: The best cosine to any reference; None when there was nothing to compare.
    score: Optional[float]
    #: Which reference it was.
    reference: Optional[int]
    #: None without a threshold measured on this embedder.
    passing: Optional[bool]
    threshold: Optional[float]
    embedder: str
    dim: int
    why: str = ""


def _cosine(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    if not left_norm or not right_norm:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


async def answer_similarity(embedder: _Embeds, *, answer: str, references: Sequence[str],
                            threshold: Optional[SimilarityThreshold] = None) -> AnswerSimilarity:
    """The answer's best cosine to any of ``references``, and whether it passes
    ``threshold`` when that was measured with this embedder at this width."""
    def unmeasured(why: str) -> AnswerSimilarity:
        return AnswerSimilarity(None, None, None, None, embedder.id, embedder.dim, why)

    kept = [reference for reference in references if reference.strip()]
    if not answer.strip():
        return unmeasured("the answer is empty")
    if not kept:
        return unmeasured("there is no reference to compare with")
    vectors = await embedder.embed([answer, *kept])
    scores = [_cosine(vectors[0], vector) for vector in vectors[1:]]
    known = [(score, index) for index, score in enumerate(scores) if score is not None]
    if not known:
        return unmeasured("the embedder gave a zero vector, which has no direction to compare")
    score, index = max(known, key=lambda pair: (pair[0], -pair[1]))
    reference = references.index(kept[index])
    if threshold is None:
        return AnswerSimilarity(score, reference, None, None, embedder.id, embedder.dim,
                                "no threshold was given, so nothing passes or fails")
    if (threshold.embedder_id, threshold.dim) != (embedder.id, embedder.dim):
        return AnswerSimilarity(score, reference, None, threshold.value, embedder.id, embedder.dim,
                                f"the threshold was measured with {threshold.embedder_id} ({threshold.dim}-d), "
                                f"not {embedder.id} ({embedder.dim}-d), so it decides nothing here")
    return AnswerSimilarity(score, reference, score > threshold.value, threshold.value, embedder.id, embedder.dim)


@dataclass(frozen=True)
class ThresholdMeasurement:
    threshold: Optional[SimilarityThreshold]
    #: Matched pairs above the threshold taken, as a share.
    pass_rate: Optional[float]
    #: Mismatched pairs above it, as a share.
    false_pass_rate: Optional[float]
    why: str = ""


def measure_threshold(matched: Sequence[float], mismatched: Sequence[float], *, embedder_id: str, dim: int,
                      target_false_pass: float = DEFAULT_FALSE_PASS) -> ThresholdMeasurement:
    """The lowest threshold that passes at most ``target_false_pass`` of the mismatched
    pairs' scores, with what it passes of the matched ones. None when a side is empty or
    when the only such threshold passes no matched answer, which would measure nothing."""
    if not 0.0 <= target_false_pass < 1.0:
        raise ValueError("target_false_pass is a share from 0 up to 1")
    if not matched or not mismatched:
        return ThresholdMeasurement(None, None, None, "a threshold needs matched and mismatched pairs to measure")
    allowed = math.floor(target_false_pass * len(mismatched))
    for candidate in sorted({-1.0, *mismatched}):
        false_passes = sum(score > candidate for score in mismatched)
        if false_passes > allowed:
            continue
        passes = sum(score > candidate for score in matched)
        if not passes:
            return ThresholdMeasurement(None, 0.0, false_passes / len(mismatched),
                                        "the lowest threshold under the target passes no matched answer")
        measured: dict[str, object] = {"matched": len(matched), "mismatched": len(mismatched), "target_false_pass": target_false_pass,
                    "pass_rate": passes / len(matched), "false_pass_rate": false_passes / len(mismatched)}
        return ThresholdMeasurement(SimilarityThreshold(candidate, embedder_id, dim, measured), passes / len(matched),
                                    false_passes / len(mismatched))
    raise AssertionError("the largest mismatched score is always under the target")
