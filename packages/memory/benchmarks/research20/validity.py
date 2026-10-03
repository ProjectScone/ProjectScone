"""Pure memory-validity mechanisms and transparent, hand-authored diagnostics.

Ground truth and provenance are fixture metadata, never inferred by a model.
These diagnostics establish consequences of representations, not RAG accuracy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

from research20.common import ExperimentResult

DEPENDENCY_REFERENCE = 'https://arxiv.org/abs/2608.10502'
BELIEF_REFERENCE = 'https://arxiv.org/abs/2605.05583'
TEMPORAL_REFERENCE = 'https://arxiv.org/abs/2609.16073'
TANGLE_REFERENCE = 'https://arxiv.org/abs/2608.13921'


def affected_conclusions(
    dependencies: Mapping[str, frozenset[str]], changed: frozenset[str],
) -> frozenset[str]:
    """Return transitive dependents, including changed stored conclusions.

    Dependencies mean necessary premises, not alternative sufficient proofs.
    Unknown source nodes are allowed; cycles terminate via the visited set.
    """
    reverse: dict[str, set[str]] = {}
    for conclusion, premises in dependencies.items():
        for premise in premises:
            reverse.setdefault(premise, set()).add(conclusion)
    visited = set(changed)
    pending = list(changed)
    while pending:
        for dependent in reverse.get(pending.pop(), set()):
            if dependent not in visited:
                visited.add(dependent)
                pending.append(dependent)
    return frozenset(visited.intersection(dependencies))


@dataclass(frozen=True)
class Evidence:
    origin: str
    probability: float


def independent_support(evidence: Sequence[Evidence]) -> float:
    """Noisy-OR under independent sufficient causes, not generic Bayesian truth."""
    complement = 1.0
    for item in evidence:
        if not math.isfinite(item.probability) or not 0 <= item.probability <= 1:
            raise ValueError('support probability must be finite and in [0, 1]')
        complement *= 1 - item.probability
    return 1 - complement


def origin_support(evidence: Sequence[Evidence]) -> float:
    """Treat same-origin copies as perfectly correlated; independent across origins."""
    independent_support(evidence)  # Validate even values hidden by a group maximum.
    groups: dict[str, float] = {}
    for item in evidence:
        groups[item.origin] = max(groups.get(item.origin, 0.0), item.probability)
    return independent_support(tuple(Evidence(key, value) for key, value in groups.items()))


@dataclass(frozen=True)
class TemporalFact:
    event_at: int
    observed_at: int
    value: str


def latest_ingestion(facts: Sequence[TemporalFact], *, known_at: int) -> str | None:
    visible = [fact for fact in facts if fact.observed_at <= known_at]
    return max(visible, key=lambda fact: fact.observed_at).value if visible else None


def bitemporal_value(
    facts: Sequence[TemporalFact], *, event_at: int, known_at: int,
) -> str | None:
    """Latest event effective by query time, as known by the observation cutoff.

    Events denote state transitions that persist until the next event. A later
    observation at the same event time corrects that event. Exact ties preserve
    input order; genuine simultaneous conflicts require separate handling.
    """
    eligible = [fact for fact in facts
                if fact.event_at <= event_at and fact.observed_at <= known_at]
    return max(eligible, key=lambda fact: (fact.event_at, fact.observed_at)).value if eligible else None


@dataclass(frozen=True)
class ConditionalDecision:
    value: str
    requirements: tuple[tuple[str, str], ...]


def activate(decision: ConditionalDecision, state: Mapping[str, str]) -> str | None:
    """Missing and contradicted conditions both withhold the stored decision."""
    if all(state.get(key) == expected for key, expected in decision.requirements):
        return decision.value
    return None


@dataclass(frozen=True)
class Alternative:
    value: str
    score: float


def preserve_alternatives(alternatives: Sequence[Alternative]) -> frozenset[str]:
    """Retain all supplied assertions, without adjudicating source quality."""
    return frozenset(item.value for item in alternatives)


def winner(alternatives: Sequence[Alternative]) -> frozenset[str]:
    if not alternatives:
        return frozenset()
    return frozenset({max(alternatives, key=lambda item: item.score).value})


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _invalidation() -> ExperimentResult:
    graph = {'a': frozenset({'s1'}), 'b': frozenset({'a'}),
             'c': frozenset({'s2'}), 'd': frozenset({'s3'})}
    # The target is authored independently of the graph traversal.
    cases: tuple[tuple[str, frozenset[str], frozenset[str], bool, bool], ...] = (
        ('one chain', frozenset({'s1'}), frozenset({'a', 'b'}), False, False),
        ('one isolated premise', frozenset({'s2'}), frozenset({'c'}), False, False),
        ('two premises', frozenset({'s2', 's3'}), frozenset({'c', 'd'}), False, False),
        ('all sources control', frozenset({'s1', 's2', 's3'}), frozenset(graph), True, False),
        ('irrelevant source', frozenset({'unrelated'}), frozenset(), True, False),
        ('changed conclusion', frozenset({'a'}), frozenset({'a', 'b'}), False, False),
        ('no changes control', frozenset(), frozenset(), True, False),
        ('unrecorded common premise', frozenset({'hidden'}), frozenset(graph), False, True),
    )
    details: list[dict[str, object]] = []
    baseline_errors: list[float] = []
    method_errors: list[float] = []
    baseline_work: list[float] = []
    method_work: list[float] = []
    missed: list[float] = []
    for name, changed, expected, control, counterexample in cases:
        baseline = frozenset(graph) if changed else frozenset()
        method = affected_conclusions(graph, changed)
        base_error = len(baseline.symmetric_difference(expected))
        method_error = len(method.symmetric_difference(expected))
        baseline_errors.append(float(base_error))
        method_errors.append(float(method_error))
        baseline_work.append(float(len(baseline)))
        method_work.append(float(len(method)))
        missed.append(float(len(expected - method)))
        details.append({'name': name, 'changed': sorted(changed),
                        'dependencies': {key: sorted(value) for key, value in graph.items()},
                        'expected_affected': sorted(expected), 'baseline': sorted(baseline),
                        'method': sorted(method), 'baseline_error': base_error,
                        'method_error': method_error, 'method_missed': len(expected - method),
                        'control': control, 'counterexample': counterexample})
    return ExperimentResult(
        1, 'Dependency-aware invalidation',
        'Complete necessary-premise lineage reduces needless invalidation after updates.',
        'Invalidate all conclusions after any source change.',
        'Invalidate only transitive dependents in the supplied graph.',
        'mean symmetric-difference invalidation error', _average(baseline_errors),
        _average(method_errors), False, 'conclusions per update', len(cases), 'simulation', details,
        ('Lineage is supplied metadata, not extracted or verified.',
         'The hidden common-premise counterexample leaves every conclusion stale.',
         'Errors weight needless and missed invalidations equally; costs are application dependent.',
         'Necessary-premise graphs do not represent alternative sufficient derivations.'),
        (DEPENDENCY_REFERENCE,),
        {'baseline_mean_invalidations': _average(baseline_work),
         'method_mean_invalidations': _average(method_work), 'method_missed_total': sum(missed)},
    )


def _correlated() -> ExperimentResult:
    cases: tuple[tuple[str, tuple[Evidence, ...], int, bool, bool], ...] = (
        ('false claim copied four times', (Evidence('a', .6),) * 4, 0, False, False),
        ('false claim copied twice', (Evidence('a', .4),) * 2, 0, False, False),
        ('true claim copied four times', (Evidence('a', .6),) * 4, 1, False, True),
        ('independent true control', (Evidence('a', .6), Evidence('b', .6)), 1, True, False),
        ('independent false control', (Evidence('a', .2), Evidence('b', .2)), 0, True, False),
        ('single source control', (Evidence('a', .7),), 1, True, False),
        ('copies plus independent source', (Evidence('a', .5), Evidence('a', .5), Evidence('b', .3)), 0, False, False),
        ('independent reporters mislabeled same origin', (Evidence('publisher', .7),) * 2, 1, False, True),
    )
    details: list[dict[str, object]] = []
    baseline_losses: list[float] = []
    method_losses: list[float] = []
    for name, evidence, label, control, counterexample in cases:
        baseline, method = independent_support(evidence), origin_support(evidence)
        base_loss, method_loss = (baseline - label) ** 2, (method - label) ** 2
        baseline_losses.append(base_loss)
        method_losses.append(method_loss)
        details.append({'name': name, 'evidence': [asdict(item) for item in evidence],
                        'truth_label': label, 'baseline_probability': baseline,
                        'method_probability': method, 'baseline_loss': base_loss,
                        'method_loss': method_loss, 'control': control, 'counterexample': counterexample})
    return ExperimentResult(
        2, 'Origin-correlated support',
        'Collapsing copied support can reduce overconfidence when provenance is accurate.',
        'Independent noisy-OR across every evidence item.',
        'Maximum support per supplied origin, followed by noisy-OR across origins.',
        'mean Brier loss', _average(baseline_losses), _average(method_losses), False,
        'squared probability error', len(cases), 'simulation', details,
        ('Origins, confidence values and binary labels are authored fixture metadata.',
         'Noisy-OR assumes independent sufficient causes, not arbitrary evidence likelihoods.',
         'Same-origin evidence is treated as perfectly correlated; different origins may still correlate.',
         'Lower confidence worsens loss on some true claims; these fixtures do not establish calibration.'),
        (BELIEF_REFERENCE,),
    )


@dataclass(frozen=True)
class TemporalCase:
    name: str
    facts: tuple[TemporalFact, ...]
    event_at: int
    known_at: int
    expected: str | None
    control: bool = False
    counterexample: bool = False


def _temporal() -> ExperimentResult:
    old, new = TemporalFact(1, 1, 'old'), TemporalFact(4, 4, 'new')
    cases = (
        TemporalCase('ordered control', (old, new), 5, 5, 'new', True),
        TemporalCase('late old event', (new, TemporalFact(1, 6, 'old')), 7, 7, 'new'),
        TemporalCase('historical event query', (old, new), 2, 5, 'old'),
        TemporalCase('future-effective record', (old, TemporalFact(10, 3, 'future')), 5, 5, 'old'),
        TemporalCase('observation cutoff control', (old, TemporalFact(2, 8, 'late')), 3, 5, 'old', True),
        TemporalCase('correction to old event', (old, new, TemporalFact(1, 6, 'fixed old')), 2, 7, 'fixed old'),
        TemporalCase('empty history control', (), 5, 5, None, True),
        TemporalCase('incorrect event timestamp', (old, TemporalFact(0, 6, 'new')), 7, 7, 'new', False, True),
    )
    details: list[dict[str, object]] = []
    baseline_scores: list[float] = []
    method_scores: list[float] = []
    for case in cases:
        baseline = latest_ingestion(case.facts, known_at=case.known_at)
        method = bitemporal_value(case.facts, event_at=case.event_at, known_at=case.known_at)
        baseline_scores.append(float(baseline == case.expected))
        method_scores.append(float(method == case.expected))
        details.append({**asdict(case), 'baseline': baseline, 'method': method,
                        'baseline_correct': baseline == case.expected,
                        'method_correct': method == case.expected})
    return ExperimentResult(
        3, 'Bitemporal event and observation memory',
        'Separating event time from knowledge time prevents late ingestion from replacing newer state.',
        'Most recently observed record visible at the observation cutoff.',
        'Latest effective event visible at both time cutoffs; latest observation breaks event ties.',
        'exact state accuracy', _average(baseline_scores), _average(method_scores), True,
        'fraction of queries', len(cases), 'simulation', details,
        ('Event timestamps and expected states are supplied, with one deliberately incorrect timestamp.',
         'State persists until the next event; interval expiry, retractions and simultaneous conflicts are unmodeled.',
         'Answers are as-known states, not unknowable retrospective truth.'),
        (TEMPORAL_REFERENCE, TANGLE_REFERENCE),
    )


def _conditional() -> ExperimentResult:
    decision = ConditionalDecision('deploy', (('region', 'us'), ('approved', 'yes')))
    cases: tuple[tuple[str, dict[str, str], str | None, bool, bool], ...] = (
        ('matching control', {'region': 'us', 'approved': 'yes'}, 'deploy', True, False),
        ('region changed', {'region': 'eu', 'approved': 'yes'}, None, False, False),
        ('approval revoked', {'region': 'us', 'approved': 'no'}, None, False, False),
        ('approval unknown', {'region': 'us'}, None, False, False),
        ('region unknown', {'approved': 'yes'}, None, False, False),
        ('irrelevant state control', {'region': 'us', 'approved': 'yes', 'color': 'blue'}, 'deploy', True, False),
        ('both changed', {'region': 'eu', 'approved': 'no'}, None, False, False),
        ('unrecorded safety precondition', {'region': 'us', 'approved': 'yes', 'safe': 'no'}, None, False, True),
    )
    details: list[dict[str, object]] = []
    baseline_scores: list[float] = []
    method_scores: list[float] = []
    for name, state, expected, control, counterexample in cases:
        baseline, method = decision.value, activate(decision, state)
        baseline_scores.append(float(baseline == expected))
        method_scores.append(float(method == expected))
        details.append({'name': name, 'decision': asdict(decision), 'state': state,
                        'expected': expected, 'baseline': baseline, 'method': method,
                        'baseline_correct': baseline == expected, 'method_correct': method == expected,
                        'control': control, 'counterexample': counterexample})
    return ExperimentResult(
        4, 'Conditional decision activation',
        'Retaining preconditions prevents decisions from carrying into incompatible contexts.',
        'Always return the stored decision value.',
        'Return the decision only if every stored equality precondition is known and satisfied.',
        'exact activation accuracy', _average(baseline_scores), _average(method_scores), True,
        'fraction of contexts', len(cases), 'simulation', details,
        ('Preconditions and context are supplied structured metadata; extraction is not evaluated.',
         'Unknown conditions deliberately withhold action and are scored as withholding in this fixture.',
         'An omitted safety condition makes both methods activate incorrectly.',
         'Equality conjunctions cannot express arbitrary policy, conflicting rules or partial observability.'),
        (DEPENDENCY_REFERENCE, TANGLE_REFERENCE),
    )


def _set_f1(predicted: frozenset[str], expected: frozenset[str]) -> float:
    denominator = len(predicted) + len(expected)
    return 2 * len(predicted & expected) / denominator if denominator else 1.0


def _conflicts() -> ExperimentResult:
    cases: tuple[tuple[str, tuple[Alternative, ...], frozenset[str], bool, bool], ...] = (
        ('unresolved date conflict', (Alternative('Monday', .9), Alternative('Tuesday', .8)), frozenset({'Monday', 'Tuesday'}), False, False),
        ('three unresolved suppliers', (Alternative('a', .9), Alternative('b', .7), Alternative('c', .6)), frozenset({'a', 'b', 'c'}), False, False),
        ('single claim control', (Alternative('a', .8),), frozenset({'a'}), True, False),
        ('duplicate agreement control', (Alternative('a', .8), Alternative('a', .7)), frozenset({'a'}), True, False),
        ('empty evidence control', (), frozenset(), True, False),
        ('low-score valid alternative', (Alternative('a', .95), Alternative('b', .1)), frozenset({'a', 'b'}), False, False),
        ('resolved stale alternative', (Alternative('a', .9), Alternative('b', .5)), frozenset({'a'}), False, True),
        ('three spurious alternatives', (Alternative('a', .9), Alternative('b', .4), Alternative('c', .3), Alternative('d', .2)), frozenset({'a'}), False, True),
    )
    details: list[dict[str, object]] = []
    baseline_scores: list[float] = []
    method_scores: list[float] = []
    baseline_sizes: list[float] = []
    method_sizes: list[float] = []
    for name, alternatives, expected, control, counterexample in cases:
        baseline, method = winner(alternatives), preserve_alternatives(alternatives)
        baseline_f1, method_f1 = _set_f1(baseline, expected), _set_f1(method, expected)
        baseline_scores.append(baseline_f1)
        method_scores.append(method_f1)
        baseline_sizes.append(float(len(baseline)))
        method_sizes.append(float(len(method)))
        details.append({'name': name, 'alternatives': [asdict(item) for item in alternatives],
                        'expected_plausible': sorted(expected), 'baseline': sorted(baseline),
                        'method': sorted(method), 'baseline_f1': baseline_f1,
                        'method_f1': method_f1, 'control': control, 'counterexample': counterexample})
    return ExperimentResult(
        5, 'Conflict-preserving alternatives',
        'Keeping unresolved alternatives retains plausible answers discarded by winner selection.',
        'Retain the highest-score assertion, with stable input-order tie breaking.',
        'Retain every distinct supplied alternative.',
        'mean plausible-alternative set F1', _average(baseline_scores), _average(method_scores), True,
        'set F1', len(cases), 'simulation', details,
        ('Plausibility labels and mutually exclusive alternatives are manually supplied.',
         'Retaining alternatives is not resolving truth or answering a single-value question.',
         'Resolved or spurious conflicts penalize indiscriminate retention.',
         'Set F1 rewards recall but hides application-specific cognitive and token costs; retained counts are also reported.'),
        (BELIEF_REFERENCE, TANGLE_REFERENCE),
        {'baseline_mean_retained': _average(baseline_sizes), 'method_mean_retained': _average(method_sizes)},
    )


def run() -> list[ExperimentResult]:
    """Execute five deterministic simulations without services, APIs or model calls."""
    return [_invalidation(), _correlated(), _temporal(), _conditional(), _conflicts()]
