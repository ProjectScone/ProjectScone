"""Deterministic count simulations. Policies never receive evaluator truth."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import math
from research20.common import ExperimentResult

SEMANTIC = 'https://arxiv.org/abs/2608.20845'
ROLLBACK = 'https://arxiv.org/abs/2608.10502'
RAGONITE = 'https://arxiv.org/abs/2412.10571'


def _probability(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('expected finite probability')


@dataclass(frozen=True)
class CompilationResult:
    answers: tuple[bool | None, ...]
    initial_calls: int
    fallback_calls: int
    unused_initial: int
    abstentions: int

    @property
    def total_calls(self) -> int:
        return self.initial_calls + self.fallback_calls


def compile_workload(state_count: int, initial_states: tuple[int, ...],
                     requests: tuple[int, ...], judge: Callable[[int], bool], *,
                     fallback: bool = True) -> CompilationResult:
    """Exact-world cache for one immutable evidence/context/model snapshot."""
    if state_count < 1 or any(not 0 <= x < state_count for x in (*initial_states, *requests)):
        raise ValueError('world outside state space')
    initial = tuple(dict.fromkeys(initial_states))
    cache = {state: judge(state) for state in initial}
    answers: list[bool | None] = []
    misses = 0
    for state in requests:
        if state not in cache and fallback:
            cache[state] = judge(state)
            misses += 1
        answers.append(cache.get(state))
    return CompilationResult(tuple(answers), len(initial), misses,
                             len(set(initial) - set(requests)), sum(x is None for x in answers))


def choose_precompute(estimated_mass: tuple[float, ...], budget: int) -> tuple[int, ...]:
    if budget < 0 or not estimated_mass:
        raise ValueError('nonempty estimates and nonnegative budget required')
    for mass in estimated_mass:
        _probability(mass)
    return tuple(sorted(range(len(estimated_mass)), key=lambda i: (-estimated_mass[i], i))[:budget])


@dataclass(frozen=True)
class RefreshItem:
    key: str
    age: int
    estimated_drift: float
    impact: float

    def __post_init__(self) -> None:
        _probability(self.estimated_drift)
        if self.age < 0 or not math.isfinite(self.impact) or self.impact < 0:
            raise ValueError('invalid refresh metadata')


def schedule_refresh(items: tuple[RefreshItem, ...], budget: int, *,
                     risk_weighted: bool) -> tuple[str, ...]:
    if budget < 0 or len({item.key for item in items}) != len(items):
        raise ValueError('nonnegative budget and unique keys required')
    ranked = sorted(items, key=lambda item: (
        -(item.estimated_drift * item.impact if risk_weighted else float(item.age)),
        -item.age, item.key))
    return tuple(item.key for item in ranked[:budget])


@dataclass(frozen=True)
class PremiseKey:
    claim: str
    evidence_version: str
    context: str
    model: str
    policy: str


@dataclass(frozen=True)
class ReuseResult:
    answers: tuple[bool, ...]
    calls: int


def reuse_premises(premises: tuple[PremiseKey, ...],
                  judge: Callable[[PremiseKey], bool]) -> ReuseResult:
    cache: dict[PremiseKey, bool] = {}
    answers: list[bool] = []
    for premise in premises:
        if premise not in cache:
            cache[premise] = judge(premise)
        answers.append(cache[premise])
    return ReuseResult(tuple(answers), len(cache))


@dataclass(frozen=True)
class AuditItem:
    key: str
    estimated_drift: float
    impact: float

    def __post_init__(self) -> None:
        _probability(self.estimated_drift)
        if not math.isfinite(self.impact) or self.impact < 0:
            raise ValueError('invalid impact')


@dataclass(frozen=True)
class AuditResult:
    audits: int
    discoveries: int
    residual_loss: float
    selections: tuple[tuple[str, ...], ...]


def _audit_selection(keys: tuple[str, ...], estimates: dict[str, float],
                     ages: dict[str, int], impacts: dict[str, float], budget: int,
                     round_index: int, adaptive: bool) -> tuple[str, ...]:
    if not keys or budget == 0:
        return ()
    count = min(budget, len(keys))
    offset = (round_index * count) % len(keys)
    rotating = keys[offset:] + keys[:offset]
    if not adaptive:
        return rotating[:count]
    exploration = rotating[:count // 3]
    ranked = sorted((key for key in keys if key not in exploration), key=lambda key: (
        -estimates[key] * (ages[key] + 1) * impacts[key], key))
    return (*exploration, *ranked[:count - len(exploration)])


def simulate_audits(items: tuple[AuditItem, ...], change_events: tuple[frozenset[str], ...],
                    budget: int, *, adaptive: bool) -> AuditResult:
    """Changes make records stale until audited; labels only enter observed updates."""
    keys = tuple(item.key for item in items)
    if budget < 0 or len(set(keys)) != len(keys):
        raise ValueError('nonnegative budget and unique keys required')
    if any(not event <= set(keys) for event in change_events):
        raise ValueError('unknown change identity')
    estimates = {item.key: item.estimated_drift for item in items}
    impacts = {item.key: item.impact for item in items}
    ages = dict.fromkeys(keys, 0)
    stale: set[str] = set()
    selections: list[tuple[str, ...]] = []
    discoveries = 0
    loss = 0.0
    for round_index, event in enumerate(change_events):
        stale.update(event)
        selected = _audit_selection(keys, estimates, ages, impacts, budget, round_index, adaptive)
        selections.append(selected)
        for key in keys:
            ages[key] += 1
        for key in selected:
            observed = key in stale
            discoveries += int(observed)
            estimates[key] = .75 * estimates[key] + .25 * int(observed)
            ages[key] = 0
            stale.discard(key)
        loss += sum(impacts[key] for key in sorted(stale))
    return AuditResult(sum(map(len, selections)), discoveries, loss, tuple(selections))


def _truth(mask: int) -> bool:
    # Nonmonotonic synthetic semantics: evidence bit 2 defeats support bit 1.
    return bool(mask & 1) and (not mask & 2 or mask & 12 == 12)


def _sparse() -> ExperimentResult:
    details: list[dict[str, object]] = []
    baseline_calls = method_calls = abstentions = requests_total = 0
    for families, pattern in ((2, 'repeat'), (2, 'full'), (3, 'drop-one'), (3, 'interior'),
                              (4, 'repeat'), (4, 'full'), (4, 'interior'), (5, 'repeat'),
                              (5, 'drop-one'), (5, 'full')):
        count = 1 << families
        initial = (0, count - 1, *(count - 1 ^ (1 << bit) for bit in range(families)))
        requests = {'repeat': (count - 1,) * 12, 'full': tuple(range(count)),
                    'drop-one': tuple(count - 1 ^ (1 << bit) for bit in range(families)) * 3,
                    'interior': (1, 3, 5, 1, 3, 5) if count > 5 else (1, 2)}[pattern]
        exhaustive = compile_workload(count, tuple(range(count)), requests, _truth)
        sparse = compile_workload(count, initial, requests, _truth)
        abstaining = compile_workload(count, initial, requests, _truth, fallback=False)
        baseline_calls += exhaustive.total_calls
        method_calls += sparse.total_calls
        abstentions += abstaining.abstentions
        requests_total += len(requests)
        details.append({'case': f'{families}-families/{pattern}', 'states': count,
                        'requests': list(requests), 'compiled_worlds': list(initial),
                        'baseline_calls': exhaustive.total_calls, 'method_calls': sparse.total_calls,
                        'initial_calls': sparse.initial_calls, 'fallback_calls': sparse.fallback_calls,
                        'unused_initial': sparse.unused_initial,
                        'answers_equal': exhaustive.answers == sparse.answers,
                        'no_fallback_abstentions': abstaining.abstentions})
    return ExperimentResult(11, 'Sparse counterfactual contracts',
        'Exact sparse tables plus paid fallback save compilation on concentrated deletion workloads.',
        'Compile all subsets', 'Compile empty/full/one-deletion worlds and memoize misses',
        'total judgment calls including compilation and fallback', float(baseline_calls), float(method_calls),
        False, 'simulated judgment calls', len(details), 'simulation', details,
        ('Synthetic deterministic judgments do not measure semantic accuracy or latency.',
         'Missing worlds abstain or incur paid fallback; no monotonic inference.',
         'Fixed packet/context/model; full coverage eliminates savings.'), (SEMANTIC,),
        {'no_fallback_abstentions': float(abstentions), 'no_fallback_coverage': 1 - abstentions / requests_total})


def _workload() -> ExperimentResult:
    details: list[dict[str, object]] = []
    all_calls = directed_calls = lazy_calls = directed_misses = lazy_misses = unused = 0
    scenarios = (
        ('accurate-hot', (0, 1) * 10, (.6, .3, .05, .05), 2),
        ('wrong-hot', (2, 3) * 10, (.6, .3, .05, .05), 2),
        ('uniform', tuple(range(8)) * 2, (.125,) * 8, 2),
        ('one-request', (7,), (.5, .3, .1, .1, 0., 0., 0., 0.), 3),
        ('no-requests', (), (.5, .5, 0., 0.), 2),
        ('no-precompute', (0, 1) * 5, (.5, .5, 0., 0.), 0),
        ('compile-all', (0,) * 10, (.7, .1, .1, .1), 4),
        ('demand-shift', (0,) * 6 + (7,) * 6, (.8, .1, .1, 0., 0., 0., 0., 0.), 2),
        ('large-concentrated', (15,) * 40, (0.,) * 15 + (1.,), 1),
        ('large-unseen-tail', (14, 15) * 10, (1.,) + (0.,) * 15, 1))
    for name, requests, estimates, budget in scenarios:
        initial = choose_precompute(estimates, budget)
        exhaustive = compile_workload(len(estimates), tuple(range(len(estimates))), requests, _truth)
        directed = compile_workload(len(estimates), initial, requests, _truth)
        lazy = compile_workload(len(estimates), (), requests, _truth)
        all_calls += exhaustive.total_calls
        directed_calls += directed.total_calls
        lazy_calls += lazy.total_calls
        directed_misses += directed.fallback_calls
        lazy_misses += lazy.fallback_calls
        unused += directed.unused_initial
        details.append({'case': name, 'estimated_mass': list(estimates), 'requests': list(requests),
                        'selected': list(initial), 'baseline_calls': exhaustive.total_calls,
                        'method_calls': directed.total_calls, 'initial_calls': directed.initial_calls,
                        'fallback_calls': directed.fallback_calls, 'unused_initial': directed.unused_initial,
                        'lazy_calls': lazy.total_calls, 'lazy_fallback_calls': lazy.fallback_calls,
                        'answers_equal': exhaustive.answers == directed.answers == lazy.answers})
    return ExperimentResult(12, 'Workload-directed precomputation',
        'Demand estimates reduce compile-all waste, trading total work against online misses versus lazy memoization.',
        'Compile all states', 'Precompute highest estimated demand states; memoize misses',
        'total calls including unused compilation', float(all_calls), float(directed_calls), False,
        'simulated judgment calls', len(details), 'simulation', details,
        ('Supplied estimates are deliberately wrong in adverse cases.',
         'Lazy memoization cannot lose on total unit-cost calls; precomputation only moves work earlier.',
         'Online call counts are not measured latency or provider performance.'), (SEMANTIC,),
        {'lazy_total_calls': float(lazy_calls), 'method_online_calls': float(directed_misses),
         'lazy_online_calls': float(lazy_misses), 'method_unused_initial_calls': float(unused)})


def _refresh() -> ExperimentResult:
    details: list[dict[str, object]] = []
    baseline_loss = method_loss = 0.0
    adverse = 0
    names = ('accurate', 'reversed-estimates', 'flat-risk', 'impact-dominant', 'no-drift',
             'all-drift', 'zero-budget', 'full-budget', 'wrong-impact-priority', 'oldest-changes')
    for index, name in enumerate(names):
        impacts: tuple[float, ...] = (1., 1., 2., 2., 5., 8.)
        risks: tuple[float, ...] = (.05, .1, .1, .2, .8, .9)
        changed = frozenset({'4', '5'})
        budget = 2
        if index == 1:
            risks = tuple(reversed(risks))
        elif index == 2:
            risks, impacts = (.2,) * 6, (1.,) * 6
        elif index == 3:
            risks = (.3,) * 6
        elif index == 4:
            changed = frozenset()
        elif index == 5:
            changed = frozenset(map(str, range(6)))
        elif index == 6:
            budget = 0
        elif index == 7:
            budget = 6
        elif index == 8:
            changed = frozenset({'0', '1'})
        elif index == 9:
            changed, impacts = frozenset({'0', '1', '2'}), (3.,) * 6
        items = tuple(RefreshItem(str(i), 6 - i, risks[i], impacts[i]) for i in range(6))
        fifo = schedule_refresh(items, budget, risk_weighted=False)
        weighted = schedule_refresh(items, budget, risk_weighted=True)
        fifo_loss = sum(item.impact for item in items if item.key in changed and item.key not in fifo)
        weighted_loss = sum(item.impact for item in items if item.key in changed and item.key not in weighted)
        baseline_loss += fifo_loss
        method_loss += weighted_loss
        adverse += weighted_loss > fifo_loss
        details.append({'case': name, 'budget': budget, 'actual_changed': sorted(changed),
                        'estimates': list(risks), 'impacts': list(impacts), 'baseline_selected': list(fifo),
                        'method_selected': list(weighted), 'baseline_loss': fifo_loss, 'method_loss': weighted_loss,
                        'baseline_calls': len(fifo), 'method_calls': len(weighted)})
    return ExperimentResult(13, 'Risk-weighted refresh scheduling',
        'Estimated drift times impact can reduce stale-answer loss under equal refresh budgets.',
        'FIFO oldest first', 'Highest estimated drift probability times impact',
        'unrepaired weighted stale loss', baseline_loss, method_loss, False, 'simulated impact units',
        len(details), 'simulation', details,
        ('Evaluator-only truth; supplied risk estimates can be wrong.',
         'Impact is supplied decision metadata; all refreshes have unit cost.',
         'One epoch with perfect repair assumed; no empirical calibration claim.'),
        (ROLLBACK, RAGONITE), {'adverse_cases': float(adverse)})


def _reuse() -> ExperimentResult:
    details: list[dict[str, object]] = []
    baseline_calls = method_calls = unsafe_errors = 0
    names = ('high-sharing', 'no-sharing', 'source-invalidation', 'context-isolation', 'model-upgrade',
             'policy-change', 'claim-change', 'repeat-after-invalidation', 'permission-change', 'time-change')
    for index, name in enumerate(names):
        base = PremiseKey('p', 'v1', 'workspace-a/time-1/access-a', 'm1', 'policy1')
        other = {
            2: PremiseKey('p', 'v2', base.context, 'm1', 'policy1'),
            3: PremiseKey('p', 'v1', 'workspace-b/time-1/access-a', 'm1', 'policy1'),
            4: PremiseKey('p', 'v1', base.context, 'm2', 'policy1'),
            5: PremiseKey('p', 'v1', base.context, 'm1', 'policy2'),
            6: PremiseKey('q', 'v1', base.context, 'm1', 'policy1'),
            7: PremiseKey('p', 'v2', base.context, 'm1', 'policy1'),
            8: PremiseKey('p', 'v1', 'workspace-a/time-1/access-b', 'm1', 'policy1'),
            9: PremiseKey('p', 'v1', 'workspace-a/time-2/access-a', 'm1', 'policy1')}.get(index, base)
        if index == 1:
            premises = tuple(PremiseKey(str(i), 'v1', base.context, 'm1', 'policy1') for i in range(12))
        elif index == 7:
            premises = (base, other) * 6
        else:
            premises = (base,) * 6 + (other,) * 6
        truth = {premise: premise == base for premise in premises}
        baseline = tuple(truth[premise] for premise in premises)
        reused = reuse_premises(premises, truth.__getitem__)
        unsafe_cache: dict[str, bool] = {}
        unsafe_answers: list[bool] = []
        for premise in premises:
            if premise.claim not in unsafe_cache:
                unsafe_cache[premise.claim] = truth[premise]
            unsafe_answers.append(unsafe_cache[premise.claim])
        errors = sum(a != b for a, b in zip(unsafe_answers, baseline, strict=True))
        baseline_calls += len(premises)
        method_calls += reused.calls
        unsafe_errors += errors
        details.append({'case': name, 'baseline_calls': len(premises), 'method_calls': reused.calls,
                        'cache_hits': len(premises) - reused.calls, 'answers_equal': reused.answers == baseline,
                        'claim_only_cache_errors': errors,
                        'identity_fields': ['claim', 'evidence_version', 'context', 'model', 'policy']})
    return ExperimentResult(14, 'Shared-premise judgment reuse',
        'Exact premise identities avoid repeated judgments while invalidation prevents stale reuse.',
        'Rejudge every premise occurrence', 'Memoize complete semantic identities', 'total premise calls',
        float(baseline_calls), float(method_calls), False, 'simulated judgment calls', len(details),
        'simulation', details,
        ('Caller must supply complete evidence, access, time, context, model and policy identities.',
         'Synthetic deterministic judges; caching preserves judgments, not objective truth.',
         'Downstream decision composition is outside scope; exact caching is established prior art.'),
        (SEMANTIC, ROLLBACK), {'unsafe_claim_only_cache_errors': float(unsafe_errors)})


def _audits() -> ExperimentResult:
    details: list[dict[str, object]] = []
    baseline_loss = method_loss = 0.0
    baseline_audits = method_audits = adverse = 0
    names = ('predicted-tail', 'surprise-head', 'recurring-tail', 'no-drift', 'all-drift',
             'uniform-estimates', 'low-budget-surprise', 'full-budget', 'moving-drift', 'reversed-estimates')
    for index, name in enumerate(names):
        risks: tuple[float, ...] = (.02,) * 6 + (.8, .9, .95)
        events: tuple[frozenset[str], ...] = (frozenset({'7', '8'}), frozenset(), frozenset(), frozenset())
        budget = 3
        if index == 1:
            events = (frozenset({'0', '1', '2'}), frozenset(), frozenset(), frozenset())
        elif index == 2:
            events = (frozenset({'7', '8'}),) * 4
        elif index == 3:
            events = (frozenset(),) * 4
        elif index == 4:
            events = (frozenset(map(str, range(9))),) * 4
        elif index == 5:
            risks = (.2,) * 9
        elif index == 6:
            budget = 1
            events = (frozenset({'0'}), frozenset(), frozenset(), frozenset())
        elif index == 7:
            budget = 9
        elif index == 8:
            events = tuple(frozenset({str(i), str(8 - i)}) for i in range(4))
        elif index == 9:
            risks = tuple(reversed(risks))
        items = tuple(AuditItem(str(i), risks[i], 1.) for i in range(9))
        uniform = simulate_audits(items, events, budget, adaptive=False)
        adaptive = simulate_audits(items, events, budget, adaptive=True)
        baseline_loss += uniform.residual_loss
        method_loss += adaptive.residual_loss
        baseline_audits += uniform.audits
        method_audits += adaptive.audits
        adverse += adaptive.residual_loss > uniform.residual_loss
        details.append({'case': name, 'budget_per_round': budget, 'estimated_drift': list(risks),
                        'actual_changes_by_round': [sorted(event) for event in events],
                        'baseline_selected': [list(keys) for keys in uniform.selections],
                        'method_selected': [list(keys) for keys in adaptive.selections],
                        'baseline_calls': uniform.audits, 'method_calls': adaptive.audits,
                        'baseline_loss': uniform.residual_loss, 'method_loss': adaptive.residual_loss,
                        'baseline_discoveries': uniform.discoveries, 'method_discoveries': adaptive.discoveries})
    return ExperimentResult(15, 'Adaptive drift audits',
        'Risk estimates updated only from audits can find stale judgments sooner than uniform rotation.',
        'Uniform cyclic audits', 'Risk-times-age audits with one exploration slot per three calls',
        'cumulative unrepaired stale loss', baseline_loss, method_loss, False, 'simulated impact-round units',
        len(details), 'simulation', details,
        ('Synthetic streams and supplied priors are not empirical drift rates.',
         'Equal per-round calls; audits assumed to detect and repair perfectly.',
         'Surprise low-prior drift can be missed; budgets below three reserve no exploration.',
         'Only observed audited labels update estimates; unaudited labels never enter selection.'),
        (RAGONITE, ROLLBACK), {'baseline_audit_calls': float(baseline_audits),
                             'method_audit_calls': float(method_audits), 'adverse_cases': float(adverse)})


def run() -> list[ExperimentResult]:
    return [_sparse(), _workload(), _refresh(), _reuse(), _audits()]
