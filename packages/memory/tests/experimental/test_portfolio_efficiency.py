"""Behavioral tests for bounded simulated efficiency policies."""
from research20.efficiency import (
    AuditItem,
    PremiseKey,
    RefreshItem,
    choose_precompute,
    compile_workload,
    reuse_premises,
    run,
    schedule_refresh,
    simulate_audits,
)


def test_unobserved_world_abstains_without_assuming_monotonicity() -> None:
    seen: list[int] = []

    def judge(mask: int) -> bool:
        seen.append(mask)
        return mask == 1  # Adding evidence reverses this judgment.

    result = compile_workload(4, (1,), (1, 3, 0), judge, fallback=False)
    assert result.answers == (True, None, None)
    assert seen == [1]
    assert result.total_calls == 1
    assert result.abstentions == 2


def test_fallback_memoizes_exact_world_and_counts_initial_unused_work() -> None:
    result = compile_workload(4, (0, 1), (3, 3), lambda mask: mask == 1)
    assert result.answers == (False, False)
    assert result.initial_calls == 2
    assert result.fallback_calls == 1
    assert result.total_calls == 3
    assert result.unused_initial == 2


def test_workload_estimate_does_not_peek_at_actual_requests() -> None:
    assert choose_precompute((0.05, 0.8, 0.1, 0.05), 2) == (1, 2)
    precomputed = compile_workload(4, (1, 2), (0, 0), lambda mask: mask == 0)
    lazy = compile_workload(4, (), (0, 0), lambda mask: mask == 0)
    assert precomputed.total_calls == 3
    assert lazy.total_calls == 1
    assert precomputed.fallback_calls == lazy.fallback_calls == 1


def test_refresh_estimates_can_miss_true_change() -> None:
    items = (
        RefreshItem('new', 1, 0.9, 2.0),
        RefreshItem('old', 5, 0.01, 5.0),
    )
    assert schedule_refresh(items, 1, risk_weighted=True) == ('new',)
    assert schedule_refresh(items, 1, risk_weighted=False) == ('old',)
    assert schedule_refresh(items, 0, risk_weighted=True) == ()


def test_reuse_rechecks_all_semantic_identity_dimensions() -> None:
    base = PremiseKey('claim', 'source-v1', 'workspace-a', 'model-a', 'policy-a')
    changed = (
        PremiseKey('other', 'source-v1', 'workspace-a', 'model-a', 'policy-a'),
        PremiseKey('claim', 'source-v2', 'workspace-a', 'model-a', 'policy-a'),
        PremiseKey('claim', 'source-v1', 'workspace-b', 'model-a', 'policy-a'),
        PremiseKey('claim', 'source-v1', 'workspace-a', 'model-b', 'policy-a'),
        PremiseKey('claim', 'source-v1', 'workspace-a', 'model-a', 'policy-b'),
    )
    result = reuse_premises((base, base, *changed), lambda key: key == base)
    assert result.calls == 6
    assert result.answers == (True, True, False, False, False, False, False)


def test_audits_spend_equal_budget_and_surprise_can_hurt() -> None:
    items = tuple(AuditItem(str(i), 0.9 if i >= 4 else 0.01, 1.0) for i in range(8))
    changes = (frozenset({'0', '1'}),)
    adaptive = simulate_audits(items, changes, 2, adaptive=True)
    uniform = simulate_audits(items, changes, 2, adaptive=False)
    assert adaptive.audits == uniform.audits == 2
    assert adaptive.residual_loss > uniform.residual_loss
    assert simulate_audits(items, changes, 0, adaptive=True).audits == 0


def test_five_reports_retain_adverse_cases_and_separate_simulated_units() -> None:
    results = run()
    assert [result.experiment_id for result in results] == list(range(11, 16))
    assert all(6 <= result.cases <= 12 for result in results)
    assert all(result.evidence_kind == 'simulation' for result in results)
    assert results[1].metrics['lazy_total_calls'] <= results[1].method_score
    assert results[4].metrics['adverse_cases'] >= 1
