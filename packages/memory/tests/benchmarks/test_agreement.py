"""Kendall's tau-b, held to hand-counted pairs and to its tie correction."""

from __future__ import annotations

import math

import pytest

from scone_memory.bench.agreement import kendall_tau_b


def test_identical_and_reversed_orders():
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    assert kendall_tau_b(scores, scores).tau_b == pytest.approx(1.0)
    assert kendall_tau_b(scores, {key: -value for key, value in scores.items()}).tau_b == pytest.approx(-1.0)


def test_one_swapped_pair_out_of_six():
    first = {"a": 1, "b": 2, "c": 3, "d": 4}
    second = {"a": 2, "b": 1, "c": 3, "d": 4}
    result = kendall_tau_b(first, second)
    assert (result.concordant, result.discordant) == (5, 1)
    assert result.tau_b == pytest.approx(4 / 6)


def test_ties_are_corrected_for_rather_than_read_as_disagreement():
    """Five pairs, one tied in the second scoring only, the rest agreeing:
    tau-b = 5 / sqrt(6 * 5); tau-a would give 5 / 6."""
    first = {"a": 1, "b": 2, "c": 3, "d": 4}
    second = {"a": 0, "b": 0, "c": 1, "d": 2}
    result = kendall_tau_b(first, second)
    assert (result.concordant, result.discordant, result.tied_second) == (5, 0, 1)
    assert result.tau_b == pytest.approx(5 / math.sqrt(6 * 5))


def test_a_scoring_of_all_ties_has_no_tau():
    result = kendall_tau_b({"a": 1, "b": 2, "c": 3}, {"a": 0, "b": 0, "c": 0})
    assert result.tau_b is None
    assert result.tied_second == 3


def test_items_held_by_one_side_are_counted_not_paired():
    result = kendall_tau_b({"a": 1, "b": 2, "x": 9}, {"a": 1, "b": 2, "y": 0, "z": 1})
    assert result.items == 2
    assert result.unmatched == 3


def test_refuses_non_finite_scores():
    with pytest.raises(ValueError):
        kendall_tau_b({"a": math.nan, "b": 1.0}, {"a": 1.0, "b": 2.0})
