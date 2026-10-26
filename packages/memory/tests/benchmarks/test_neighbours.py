"""Exact neighbours are the ground truth an approximate index is judged by.

They must apply the index's own filter, never rank a zero vector, and
come out in the same order on every run.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.bench.neighbours import exact_top_k

VECTORS = {
    "east": [1.0, 0.0],
    "north-east": [1.0, 1.0],
    "north": [0.0, 2.0],
    "west": [-1.0, 0.0],
}


def test_ranks_by_cosine_not_by_length():
    """``north`` is the longest vector but points away from the query; a
    dot product would rank it above ``north-east``."""
    found = exact_top_k([1.0, 0.2], VECTORS, 3)
    assert [document for document, _ in found.ranked] == ["east", "north-east", "north"]
    assert found.ranked[0][1] == pytest.approx(1 / math.sqrt(1.04))
    assert found.considered == 4


def test_ties_are_broken_by_id_so_the_truth_is_stable():
    tied = {"b": [2.0, 0.0], "a": [1.0, 0.0], "c": [3.0, 0.0]}
    assert [document for document, _ in exact_top_k([1.0, 0.0], tied, 2).ranked] == ["a", "b"]


def test_applies_the_same_filter_the_index_was_given():
    found = exact_top_k([1.0, 0.0], VECTORS, 2, allowed=lambda document: document != "east")
    assert [document for document, _ in found.ranked] == ["north-east", "north"]
    assert found.filtered_out == 1
    assert found.considered == 3


def test_zero_vectors_are_skipped_and_counted_not_scored_as_zero():
    """A zero similarity would rank ``blank`` above ``west``, whose
    similarity is negative; it has no direction, so it has no rank."""
    found = exact_top_k([1.0, 0.0], {"west": [-1.0, 0.0], "blank": [0.0, 0.0]}, 2)
    assert [document for document, _ in found.ranked] == ["west"]
    assert found.skipped_zero == 1


def test_returns_fewer_than_k_when_fewer_exist():
    assert len(exact_top_k([1.0, 0.0], VECTORS, 10).ranked) == 4


@pytest.mark.parametrize("query, k", [([0.0, 0.0], 1), ([math.nan, 1.0], 1), ([1.0, 0.0], 0), ([1.0, 0.0], 1.5)])
def test_refuses_queries_it_cannot_rank_against(query, k):
    with pytest.raises(ValueError):
        exact_top_k(query, VECTORS, k)


def test_refuses_a_vector_of_another_dimension():
    with pytest.raises(ValueError, match="3 dimensions"):
        exact_top_k([1.0, 0.0], {"odd": [1.0, 0.0, 0.0]}, 1)
