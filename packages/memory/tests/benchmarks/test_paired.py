"""Paired comparisons report counts beside the delta.

A delta of 0.02 on fifty items is one item. These tests hold the counts
and the sign test to their textbook values, so a comparison can say how
many items moved and how surprising that split is.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.bench.paired import paired_outcome, sign_test


def test_counts_wins_losses_and_ties_item_by_item():
    outcome = paired_outcome([0.5, 0.5, 0.5, 0.5], [0.7, 0.4, 0.5, 0.9])
    assert (outcome.n, outcome.wins, outcome.losses, outcome.ties) == (4, 2, 1, 1)
    assert outcome.mean_delta == pytest.approx((0.2 - 0.1 + 0.0 + 0.4) / 4)


def test_sign_test_matches_the_binomial_tail():
    # 8 wins and 2 losses: 2 * (C(10,0) + C(10,1) + C(10,2)) / 2**10.
    assert sign_test(8, 2) == pytest.approx(2 * 56 / 1024)
    assert sign_test(10, 0) == pytest.approx(2 / 1024)


def test_an_even_split_is_not_evidence_of_anything():
    assert sign_test(5, 5) == 1.0
    assert sign_test(0, 0) == 1.0


def test_ties_are_left_out_of_the_sign_test():
    """Forty ties beside eight wins and two losses is still an 8-2 split:
    an item that did not move says nothing about direction."""
    moved = paired_outcome([0.0] * 10, [1.0] * 8 + [-1.0] * 2)
    padded = paired_outcome([0.0] * 50, [1.0] * 8 + [-1.0] * 2 + [0.0] * 40)
    assert padded.ties == 40
    assert padded.sign_test_p == moved.sign_test_p


def test_swapping_the_runs_swaps_wins_and_losses_and_keeps_the_p_value():
    baseline, candidate = [0.1, 0.4, 0.3, 0.9, 0.2], [0.2, 0.3, 0.6, 0.95, 0.2]
    forward = paired_outcome(baseline, candidate)
    backward = paired_outcome(candidate, baseline)
    assert (forward.wins, forward.losses) == (backward.losses, backward.wins)
    assert forward.sign_test_p == backward.sign_test_p
    assert forward.mean_delta == pytest.approx(-backward.mean_delta)


def test_tolerance_turns_floating_point_noise_into_ties():
    noisy = paired_outcome([0.3, 0.6], [0.1 + 0.2, 0.6 + 1e-12])
    assert noisy.wins == 2
    assert paired_outcome([0.3, 0.6], [0.1 + 0.2, 0.6 + 1e-12], tolerance=1e-9).ties == 2


def test_an_empty_comparison_has_no_delta_rather_than_a_zero_one():
    outcome = paired_outcome([], [])
    assert outcome.n == 0
    assert outcome.mean_delta is None
    assert outcome.sign_test_p == 1.0


def test_large_splits_stay_exact_and_tiny():
    p = sign_test(900, 100)
    assert 0.0 < p < 1e-100


@pytest.mark.parametrize("baseline, candidate", [([0.1], [0.1, 0.2]), ([math.nan], [0.1]), ([0.1], [math.inf])])
def test_refuses_runs_that_cannot_be_paired(baseline, candidate):
    with pytest.raises(ValueError):
        paired_outcome(baseline, candidate)


@pytest.mark.parametrize("wins, losses", [(-1, 2), (1.5, 2), (2, True)])
def test_sign_test_refuses_counts_that_are_not_counts(wins, losses):
    with pytest.raises(ValueError):
        sign_test(wins, losses)
