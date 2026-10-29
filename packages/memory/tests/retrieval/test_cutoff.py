"""The cut falls where the scores drop, and a bound never passes for one."""

from __future__ import annotations

import math

import pytest

from scone_memory.retrieval.cutoff import gap_cutoff


def test_cuts_after_the_shelf_of_matching_passages():
    cut = gap_cutoff([0.9, 0.88, 0.86, 0.5, 0.49, 0.48, 0.47], max_keep=6)
    assert (cut.keep, cut.rule) == (3, "gap")
    assert cut.relative_gap == pytest.approx(0.36 / (0.9 - 0.47))


def test_a_flat_list_has_no_boundary_and_keeps_the_maximum():
    cut = gap_cutoff([0.5 - index * 0.01 for index in range(30)], max_keep=10)
    assert (cut.keep, cut.rule, cut.relative_gap) == (10, "no-boundary", None)


def test_never_keeps_fewer_than_min_keep():
    """The largest drop is after the first result, but two are always kept."""
    cut = gap_cutoff([0.9, 0.3, 0.29, 0.1, 0.09], min_keep=2, max_keep=4)
    assert cut.keep >= 2


def test_a_late_drop_must_be_larger_than_an_early_one_to_win():
    scores = [1.0, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.42, 0.41]
    assert gap_cutoff(scores, max_keep=11, late_charge=0.0).keep == 10
    assert gap_cutoff(scores, max_keep=11, late_charge=0.5).keep == 1


def test_only_the_first_max_keep_plus_one_scores_set_the_spread():
    """A far tail would make every early drop look small."""
    head = [0.9, 0.85, 0.6, 0.58]
    assert gap_cutoff(head + [-5.0], max_keep=3).keep == gap_cutoff(head, max_keep=3).keep == 2


def test_fewer_results_than_min_keep_are_all_kept():
    cut = gap_cutoff([0.7, 0.2], min_keep=3)
    assert (cut.keep, cut.rule) == (2, "short")


def test_equal_scores_keep_what_is_allowed():
    assert gap_cutoff([0.4] * 8, max_keep=5).keep == 5


@pytest.mark.parametrize("scores, options", [
    ([0.1, 0.9], {}), ([math.nan, 0.1], {}), ([0.9, 0.1], {"min_keep": 0}),
    ([0.9, 0.1], {"min_keep": 3, "max_keep": 2}), ([0.9, 0.1], {"min_gap": 1.5}),
])
def test_refuses_what_it_cannot_cut(scores, options):
    with pytest.raises(ValueError):
        gap_cutoff(scores, **options)
