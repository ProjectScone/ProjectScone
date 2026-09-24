"""Behavioral checks for bounded validity diagnostics; no model or service calls."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from research20.validity import (
    Alternative, ConditionalDecision, Evidence, TemporalFact, activate,
    affected_conclusions, bitemporal_value, independent_support, latest_ingestion,
    origin_support, preserve_alternatives, run, winner,
)


def test_dependency_closure_crosses_multiple_hops_and_cycles() -> None:
    dependencies = {'a': frozenset({'source', 'b'}), 'b': frozenset({'a'}),
                    'other': frozenset({'untouched'})}
    assert affected_conclusions(dependencies, frozenset({'source'})) == {'a', 'b'}
    assert affected_conclusions(dependencies, frozenset({'unknown'})) == set()
    assert affected_conclusions(dependencies, frozenset({'other'})) == {'other'}


def test_copied_support_does_not_increase_origin_confidence() -> None:
    evidence = (Evidence('root', .4), Evidence('root', .4), Evidence('other', .5))
    assert independent_support(evidence) == pytest.approx(.82)
    assert origin_support(evidence) == pytest.approx(.7)
    assert origin_support(()) == 0


@pytest.mark.parametrize('probability', [-.1, 1.1, float('nan')])
def test_support_rejects_invalid_probabilities(probability: float) -> None:
    with pytest.raises(ValueError):
        independent_support((Evidence('a', probability),))


def test_bitemporal_lookup_respects_both_time_axes_and_corrections() -> None:
    facts = (TemporalFact(1, 1, 'old'), TemporalFact(4, 4, 'new'),
             TemporalFact(1, 6, 'corrected old'))
    assert bitemporal_value(facts, event_at=5, known_at=7) == 'new'
    assert latest_ingestion(facts, known_at=7) == 'corrected old'
    assert bitemporal_value(facts, event_at=2, known_at=5) == 'old'
    assert bitemporal_value(facts, event_at=2, known_at=7) == 'corrected old'
    assert bitemporal_value(facts, event_at=0, known_at=7) is None


def test_conditional_activation_requires_known_satisfied_premises() -> None:
    decision = ConditionalDecision('deploy', (('region', 'us'), ('approved', 'yes')))
    assert activate(decision, {'region': 'us', 'approved': 'yes'}) == 'deploy'
    assert activate(decision, {'region': 'us'}) is None
    assert activate(decision, {'region': 'eu', 'approved': 'yes'}) is None
    assert activate(ConditionalDecision('always', ()), {}) == 'always'


def test_conflict_preservation_deduplicates_without_discarding_low_score_claim() -> None:
    alternatives = (Alternative('a', .9), Alternative('a', .8), Alternative('b', .2))
    assert preserve_alternatives(alternatives) == frozenset({'a', 'b'})
    assert winner(alternatives) == frozenset({'a'})
    assert winner(()) == preserve_alternatives(()) == frozenset()


def _number(value: object) -> float:
    assert isinstance(value, (int, float))
    return float(value)


def test_diagnostics_retain_controls_and_counterexamples_with_finite_scores() -> None:
    results = run()
    assert [result.experiment_id for result in results] == [1, 2, 3, 4, 5]
    for result in results:
        assert 6 <= result.cases == len(result.details) <= 12
        assert any(row['control'] for row in result.details)
        assert any(row['counterexample'] for row in result.details)
        assert result.evidence_kind == 'simulation'
        json.dumps(asdict(result), allow_nan=False)
    assert any(_number(row['method_error']) > _number(row['baseline_error']) for row in results[0].details)
    assert any(_number(row['method_loss']) > _number(row['baseline_loss']) for row in results[1].details)
    assert any(row['baseline_correct'] and not row['method_correct'] for row in results[2].details)
    assert any(not row['method_correct'] for row in results[3].details)
    assert any(_number(row['baseline_f1']) > _number(row['method_f1']) for row in results[4].details)
