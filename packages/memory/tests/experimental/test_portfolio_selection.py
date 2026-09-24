"""Behavior checks for bounded evidence-selection research diagnostics."""
from __future__ import annotations

import math

import pytest

from research20.selection import (
    Passage,
    Probe,
    choose_information_gain,
    choose_joint,
    choose_missing,
    choose_reserved,
    information_gain,
    run,
    stop_when_sufficient,
)


def test_missing_premise_search_ignores_redundant_high_relevance() -> None:
    pool = (
        Passage("copy", 0.99, frozenset({"a"})),
        Passage("bridge", 0.4, frozenset({"b"})),
        Passage("end", 0.3, frozenset({"c"})),
    )
    chosen = choose_missing(pool, frozenset({"a", "b", "c"}), frozenset({"a"}), 2)
    assert [item.key for item in chosen] == ["bridge", "end"]


def test_joint_selection_finds_pair_with_no_singleton_reward() -> None:
    pool = (
        Passage("distractor", 0.99, frozenset({"d"}), utility=0.9),
        Passage("left", 0.4, frozenset({"a"}), utility=0.4),
        Passage("right", 0.3, frozenset({"b"}), utility=0.3),
    )
    groups = ((frozenset({"a", "b"}), 1.0), (frozenset({"d"}), 0.2))
    assert {item.key for item in choose_joint(pool, groups, 2)} == {"left", "right"}


def test_reserved_search_spends_one_slot_on_refutation() -> None:
    pool = (
        Passage("yes1", 0.9, frozenset(), stance=1),
        Passage("yes2", 0.8, frozenset(), stance=1),
        Passage("no", 0.1, frozenset(), stance=-1),
    )
    assert [item.key for item in choose_reserved(pool, 2)] == ["yes1", "no"]
    assert choose_reserved(pool[:2], 2) == pool[:2]


def test_information_gain_values_discrimination_over_relevance() -> None:
    decoy = Probe("decoy", 0.99, (0.5, 0.5))
    separator = Probe("separator", 0.3, (0.0, 1.0))
    assert information_gain(decoy, 0.5) == pytest.approx(0.0)
    assert information_gain(separator, 0.5) == pytest.approx(1.0)
    assert choose_information_gain((decoy, separator), 0.5) == separator


def test_sufficiency_stopping_obeys_budget_and_requires_all_premises() -> None:
    pool = (
        Passage("a", 0.9, frozenset({"a"})),
        Passage("b", 0.8, frozenset({"b"})),
        Passage("extra", 0.7, frozenset({"c"})),
    )
    assert len(stop_when_sufficient(pool, frozenset({"a", "b"}), 3)) == 2
    assert len(stop_when_sufficient(pool, frozenset({"a", "b"}), 1)) == 1
    assert stop_when_sufficient(pool, frozenset(), 3) == ()


def test_portfolio_retains_adverse_cases_and_recomputable_measurements() -> None:
    results = run()
    assert results == run()
    assert [result.experiment_id for result in results] == [6, 7, 8, 9, 10]
    for result in results:
        assert result.cases == len(result.details) == 8
        assert {row["case_kind"] for row in result.details} == {
            "target", "control", "adverse"
        }
        baseline = [float(str(row["baseline_value"])) for row in result.details]
        method = [float(str(row["method_value"])) for row in result.details]
        assert result.baseline_score == pytest.approx(sum(baseline) / 8)
        assert result.method_score == pytest.approx(sum(method) / 8)
        assert all(math.isfinite(value) for value in baseline + method)
        for row in result.details:
            assert int(str(row["method_used"])) <= int(str(row["budget"]))
            assert int(str(row["baseline_used"])) <= int(str(row["budget"]))
    stopping = results[-1]
    assert stopping.metrics["method_complete_rate"] < stopping.metrics["baseline_complete_rate"]
