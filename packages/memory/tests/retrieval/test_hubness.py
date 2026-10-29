"""Hubs are near everything; CSLS makes them pay for it.

The fixture has a hub at the centre of a ring of passages. Every query
finds it close, and a plain cosine ranks it above the passage that
actually answers.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.retrieval.hubness import (
    hub_penalised, hubness, k_occurrence, neighbourhood_density, pool_density,
)


def test_k_occurrence_counts_each_result_once():
    counts = k_occurrence([["hub", "a"], ["hub", "b", "hub"], ["c"]])
    assert counts == {"hub": 2, "a": 1, "b": 1, "c": 1}


def test_a_hub_gives_a_large_positive_skew_and_even_use_none():
    hub_results = [["hub", f"p{index}"] for index in range(20)]
    skewed = hubness(hub_results, population=40)
    assert skewed.max_occurrence == 20
    assert skewed.skewness is not None and skewed.skewness > 3

    even = hubness([[f"p{index}", f"p{index + 1}"] for index in range(0, 40, 2)], population=40)
    assert even.skewness is None
    assert even.max_occurrence == 1


def test_passages_never_found_count_as_zero_occurrences():
    """The same results over a larger collection are more uneven: more
    passages were never found at all."""
    results = [["a", "b"], ["a", "c"], ["a", "d"]]
    assert hubness(results, population=100).skewness > hubness(results, population=4).skewness


def test_hubness_refuses_a_population_smaller_than_what_appeared():
    with pytest.raises(ValueError):
        hubness([["a", "b", "c"]], population=2)


def test_density_is_the_mean_of_the_k_nearest():
    assert neighbourhood_density([0.1, 0.9, 0.5, 0.7], 2) == pytest.approx(0.8)
    assert neighbourhood_density([0.4], 3) == pytest.approx(0.4)
    assert neighbourhood_density([], 3) is None


#: Six passages on orthogonal axes, and a hub on their diagonal: it is
#: as near every passage as any passage is near anything.
SPOKES = {f"p{index}": [1.0 if axis == index else 0.0 for axis in range(6)] for index in range(6)}
SPOKES["hub"] = [1.0] * 6


def test_the_hub_is_densest_among_the_candidates():
    density = pool_density(SPOKES, k=3)
    assert density["hub"] == pytest.approx(1 / math.sqrt(6))
    assert density["p0"] == pytest.approx(1 / math.sqrt(6) / 3)


def test_csls_puts_the_answer_above_the_hub_that_cosine_preferred():
    query = [1.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    similarity = {key: _cosine(query, vector) for key, vector in SPOKES.items()}
    assert max(similarity, key=similarity.__getitem__) == "hub"
    ordered = hub_penalised(similarity, pool_density(SPOKES, k=3), weight=0.5)
    assert ordered[0][0] == "p0"


def test_weight_zero_keeps_the_cosine_order():
    similarity = {"a": 0.9, "b": 0.8, "c": 0.7}
    ordered = hub_penalised(similarity, {"a": 0.9, "b": 0.1, "c": 0.0}, weight=0.0)
    assert [key for key, _ in ordered] == ["a", "b", "c"]


def test_a_candidate_without_density_takes_the_typical_penalty():
    ordered = dict(hub_penalised({"a": 0.5, "b": 0.5, "c": 0.5}, {"a": 0.2, "b": 0.4}, weight=1.0))
    assert ordered["c"] == pytest.approx(0.5 - 0.3)


def test_zero_vectors_have_no_density():
    assert pool_density({"blank": [0.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]}, k=1)["blank"] is None


def test_hub_penalised_refuses_a_negative_weight():
    with pytest.raises(ValueError):
        hub_penalised({"a": 1.0}, {"a": 0.5}, weight=-0.1)


def _cosine(left, right):
    return sum(a * b for a, b in zip(left, right)) / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))
