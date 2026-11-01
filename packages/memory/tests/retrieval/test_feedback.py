"""Dense Rocchio moves the query towards what it found, unless told not to."""

from __future__ import annotations

import math

import pytest

from scone_memory.retrieval.feedback import gated_rocchio, rocchio


def _cosine(left, right):
    return sum(a * b for a, b in zip(left, right)) / (math.hypot(*left) * math.hypot(*right))


def test_the_query_moves_towards_the_mean_of_its_results():
    moved = rocchio([1.0, 0.0], [[0.0, 1.0], [0.0, 3.0]], beta=1.0)
    assert moved.expanded and moved.used == 2
    assert moved.vector == pytest.approx((1 / math.sqrt(2), 1 / math.sqrt(2)))


def test_every_vector_counts_at_unit_length():
    """The long feedback vector must not pull harder than the short one."""
    moved = rocchio([1.0, 0.0, 0.0], [[0.0, 10.0, 0.0], [0.0, 0.0, 0.1]], beta=1.0)
    assert moved.vector[1] == pytest.approx(moved.vector[2])


def test_only_the_first_depth_results_are_read():
    moved = rocchio([1.0, 0.0], [[0.0, 1.0], [0.0, -1.0]], beta=1.0, depth=1)
    assert moved.used == 1 and moved.vector[1] > 0


def test_the_result_is_unit_length():
    moved = rocchio([3.0, 4.0], [[1.0, 0.0], [0.5, 0.5]], beta=0.7)
    assert math.hypot(*moved.vector) == pytest.approx(1.0)


def test_zero_beta_or_no_directed_feedback_leaves_the_query_alone():
    assert not rocchio([2.0, 0.0], [[0.0, 1.0]], beta=0.0).expanded
    unchanged = rocchio([2.0, 0.0], [[0.0, 0.0]], beta=0.5)
    assert not unchanged.expanded and unchanged.vector == (1.0, 0.0)


def test_feedback_that_cancels_the_query_is_refused():
    cancelled = rocchio([1.0, 0.0], [[-1.0, 0.0]], beta=1.0)
    assert not cancelled.expanded
    assert "cancelled" in cancelled.why


def test_the_gate_holds_feedback_back_below_its_threshold():
    held = gated_rocchio([1.0, 0.0], [[0.0, 1.0]], signal=0.1, gate=0.3, beta=1.0)
    assert not held.expanded and "below the gate" in held.why
    passed = gated_rocchio([1.0, 0.0], [[0.0, 1.0]], signal=0.4, gate=0.3, beta=1.0)
    assert passed.expanded
    assert _cosine(passed.vector, [0.0, 1.0]) > _cosine(held.vector, [0.0, 1.0])


def test_a_missing_signal_is_not_evidence_of_quality():
    assert not gated_rocchio([1.0, 0.0], [[0.0, 1.0]], signal=None, gate=0.0).expanded


@pytest.mark.parametrize("options", [{"beta": -0.1}, {"depth": 0}, {"beta": math.inf}])
def test_rocchio_refuses_settings_it_cannot_honour(options):
    with pytest.raises(ValueError):
        rocchio([1.0, 0.0], [[0.0, 1.0]], **options)


def test_rocchio_refuses_a_zero_query_and_mixed_dimensions():
    with pytest.raises(ValueError):
        rocchio([0.0, 0.0], [[0.0, 1.0]])
    with pytest.raises(ValueError):
        rocchio([1.0, 0.0], [[0.0, 1.0, 0.0]])
