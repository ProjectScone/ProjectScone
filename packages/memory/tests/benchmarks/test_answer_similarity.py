"""How close a generated answer is to its reference, by embedding, with a threshold measured per embedder.

Exact match and token F1 give nothing to a correct paraphrase: a one-word
reference restated in a sentence scores near zero. The reference framework
embeds both answers and passes at a fixed 0.8, a number that means
something different for every embedder. Here the similarity is the best
cosine to any reference, and a pass needs a threshold that names the
embedder and width it was measured with; used with another embedder it
decides nothing, and says why. A threshold is measured without a judge:
answers that match their reference exactly are the passes it must allow,
and each answer set against another question's reference is a pass it must
refuse, so the value taken is the lowest that lets through at most the
target share of those. A pass is a score above the threshold.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder
from scone_memory.bench.answer_similarity import SimilarityThreshold, answer_similarity, measure_threshold


class Fixed:
    """An embedder whose vectors are set by hand, so the cosines are known."""

    id = "fixed-test"
    dim = 2

    def __init__(self, vectors):
        self.vectors = vectors

    async def embed(self, texts):
        return [self.vectors[text] for text in texts]


VECTORS = {"Paris": [1.0, 0.0], "The capital is Paris.": [0.8, 0.6], "Lyon": [0.0, 1.0], "": [0.0, 0.0]}


async def test_the_score_is_the_best_cosine_to_any_reference():
    found = await answer_similarity(Fixed(VECTORS), answer="The capital is Paris.", references=["Lyon", "Paris"])
    assert found.score == pytest.approx(0.8) and found.reference == 1
    blank_first = await answer_similarity(Fixed(VECTORS), answer="Paris", references=["  ", "Lyon", "Paris"])
    assert blank_first.reference == 2, "the index is into the references given, blanks included"
    assert found.passing is None and "no threshold" in found.why
    assert found.embedder == "fixed-test" and found.dim == 2


async def test_a_threshold_measured_on_this_embedder_decides():
    threshold = SimilarityThreshold(value=0.75, embedder_id="fixed-test", dim=2)
    passed = await answer_similarity(Fixed(VECTORS), answer="The capital is Paris.", references=["Paris"], threshold=threshold)
    failed = await answer_similarity(Fixed(VECTORS), answer="Lyon", references=["Paris"], threshold=threshold)
    assert passed.passing is True and failed.passing is False
    assert passed.threshold == 0.75


@pytest.mark.parametrize("borrowed", [SimilarityThreshold(value=0.75, embedder_id="another", dim=2),
                                      SimilarityThreshold(value=0.75, embedder_id="fixed-test", dim=768)],
                         ids=["other-embedder", "other-width"])
async def test_a_threshold_from_another_embedder_decides_nothing(borrowed):
    found = await answer_similarity(Fixed(VECTORS), answer="The capital is Paris.", references=["Paris"], threshold=borrowed)
    assert found.score == pytest.approx(0.8) and found.passing is None
    assert "measured with" in found.why


async def test_nothing_to_compare_is_unmeasured_not_zero():
    no_answer = await answer_similarity(Fixed(VECTORS), answer="  ", references=["Paris"])
    no_reference = await answer_similarity(Fixed(VECTORS), answer="Paris", references=[])
    zero_vector = await answer_similarity(Fixed({**VECTORS, "blank": [0.0, 0.0]}), answer="blank", references=["Paris"])
    assert no_answer.score is None and no_reference.score is None and zero_vector.score is None
    assert all(item.passing is None and item.why for item in (no_answer, no_reference, zero_vector))
    assert "reference" in no_reference.why and "zero vector" in zero_vector.why and "empty" in no_answer.why


async def test_a_real_embedder_gives_a_score_in_range():
    found = await answer_similarity(HashEmbedder(), answer="The crane was surveyed in May.",
                                    references=["the crane survey happened in May"])
    assert found.score is not None and -1.0 <= found.score <= 1.0 and found.embedder == HashEmbedder().id


def test_the_threshold_taken_lets_through_at_most_the_target_share_of_mismatched_pairs():
    matched = [0.95, 0.9, 0.88, 0.85, 0.6]
    mismatched = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.62, 0.65, 0.7, 0.86]
    measured = measure_threshold(matched, mismatched, embedder_id="fixed-test", dim=2, target_false_pass=0.1)
    assert measured.threshold is not None and measured.threshold.value == pytest.approx(0.7, abs=0.011)
    assert measured.threshold.embedder_id == "fixed-test" and measured.threshold.dim == 2
    assert measured.false_pass_rate <= 0.1 and measured.pass_rate == pytest.approx(0.8)
    assert measured.threshold.measured["matched"] == 5 and measured.threshold.measured["mismatched"] == 10
    # 15% of ten mismatched pairs is one and a half: at most one may pass, never two.
    assert measure_threshold(matched, mismatched, embedder_id="fixed-test", dim=2, target_false_pass=0.15).false_pass_rate == 0.1


def test_no_threshold_is_taken_when_the_only_one_under_the_target_passes_no_matched_answer():
    measured = measure_threshold([0.5], [0.99, 0.99], embedder_id="fixed-test", dim=2, target_false_pass=0.0)
    assert measured.threshold is None and "no matched answer" in measured.why


def test_a_measurement_needs_pairs_on_both_sides():
    measured = measure_threshold([], [0.3], embedder_id="fixed-test", dim=2)
    assert measured.threshold is None and "needs matched and mismatched" in measured.why
    assert measure_threshold([0.3], [], embedder_id="fixed-test", dim=2).threshold is None


async def test_a_vector_that_is_not_finite_or_not_the_embedders_width_measures_nothing():
    """A NaN cosine compares false with everything, so it could win a maximum and read as a fail."""
    nan, inf = float("nan"), float("inf")
    for broken in ([nan, nan], [inf, 1.0], [1.0, 0.0, 0.0]):
        vectors = {**VECTORS, "bad": broken}
        threshold = SimilarityThreshold(value=0.75, embedder_id="fixed-test", dim=2)
        found = await answer_similarity(Fixed(vectors), answer="The capital is Paris.", references=["bad", "Paris"],
                                        threshold=threshold)
        assert found.score is None and found.passing is None, broken
        assert "finite" in found.why or "width" in found.why


def test_a_score_that_is_not_finite_is_refused_by_the_measurement():
    with pytest.raises(ValueError, match="finite"):
        measure_threshold([0.9, float("nan")], [0.1], embedder_id="fixed-test", dim=2)
