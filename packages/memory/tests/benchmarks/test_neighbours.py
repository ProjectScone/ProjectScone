"""Exact neighbours are the ground truth an approximate index is judged by.

They must apply the index's own filter, never rank a zero vector, and
come out in the same order on every run.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.bench.neighbours import exact_top_k, neighbour_recall

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


def test_recall_counts_the_true_neighbours_returned():
    result = neighbour_recall(["east", "west"], [1.0, 0.0], VECTORS, 2)
    assert (result.found, result.expected, result.recall) == (1, 2, 0.5)


def test_a_document_tied_at_the_boundary_is_a_true_neighbour():
    """``a``, ``b`` and ``c`` point the same way; the exact top two are
    ``a`` and ``b`` by id, but an index returning ``c`` has returned a true
    second neighbour and must not be marked wrong for it."""
    tied = {"a": [1.0, 0.0], "b": [2.0, 0.0], "c": [3.0, 0.0], "d": [0.0, 1.0]}
    assert neighbour_recall(["a", "c"], [1.0, 0.0], tied, 2).recall == 1.0
    assert neighbour_recall(["a", "d"], [1.0, 0.0], tied, 2).recall == 0.5


def test_a_filtered_document_is_a_leak_not_a_hit():
    """``east`` is the best match, but the caller may not see it."""
    result = neighbour_recall(["east", "north-east"], [1.0, 0.0], VECTORS, 2,
                              allowed=lambda document: document != "east")
    assert result.outside_filter == 1
    assert result.found == 1


def test_duplicates_and_unknown_ids_are_reported_and_never_counted():
    result = neighbour_recall(["east", "east", "ghost"], [1.0, 0.0], VECTORS, 3)
    assert (result.found, result.duplicates, result.unknown) == (1, 1, 1)


def test_only_the_first_k_returned_are_read():
    assert neighbour_recall(["west", "north", "east"], [1.0, 0.0], VECTORS, 2).found == 0


def test_recall_with_nothing_to_find_is_none_not_zero():
    result = neighbour_recall(["east"], [1.0, 0.0], VECTORS, 2, allowed=lambda document: False)
    assert result.recall is None and result.expected == 0
