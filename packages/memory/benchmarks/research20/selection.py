"""Synthetic selection diagnostics; policy metadata are supplied, not inferred."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
import random

from .common import ExperimentResult


@dataclass(frozen=True)
class Passage:
    key: str
    relevance: float
    premise_hints: frozenset[str]
    utility: float = 0.0
    stance: int = 1


@dataclass(frozen=True)
class Probe:
    key: str
    relevance: float
    probability_one: tuple[float, float]


Groups = tuple[tuple[frozenset[str], float], ...]
Row = dict[str, object]
KINDS = ("target",) * 4 + ("control",) * 2 + ("adverse",) * 2
FLARE = "https://arxiv.org/abs/2305.06983"
IG = "https://arxiv.org/abs/2601.17532"
SUFFICIENCY = "https://arxiv.org/abs/2411.06037"


def _rank(pool: tuple[Passage, ...], budget: int) -> tuple[Passage, ...]:
    return tuple(sorted(pool, key=lambda p: (-p.relevance, p.key))[:max(0, budget)])


def _hints(pool: tuple[Passage, ...]) -> frozenset[str]:
    return frozenset(value for item in pool for value in item.premise_hints)


def choose_missing(pool: tuple[Passage, ...], required: frozenset[str],
                   known: frozenset[str], budget: int) -> tuple[Passage, ...]:
    selected: tuple[Passage, ...] = ()
    while len(selected) < max(0, budget):
        remaining = tuple(p for p in pool if p not in selected)
        if not remaining:
            break
        missing = required - known - _hints(selected)
        selected += (min(remaining, key=lambda p: (
            -len(p.premise_hints & missing), -p.relevance, p.key)),)
    return selected


def _reward(premises: frozenset[str], groups: Groups) -> float:
    return sum(weight for needed, weight in groups if needed <= premises)


def choose_joint(pool: tuple[Passage, ...], groups: Groups, budget: int) -> tuple[Passage, ...]:
    subsets = combinations(sorted(pool, key=lambda p: p.key), min(max(0, budget), len(pool)))
    return max(subsets, key=lambda subset: (
        _reward(_hints(subset), groups), sum(p.utility for p in subset)), default=())


def choose_reserved(pool: tuple[Passage, ...], budget: int) -> tuple[Passage, ...]:
    if budget <= 0:
        return ()
    opposing = _rank(tuple(p for p in pool if p.stance < 0), 1)
    supporting = _rank(tuple(p for p in pool if p.stance >= 0), budget - len(opposing))
    selected = supporting + opposing
    return selected + _rank(tuple(p for p in pool if p not in selected), budget - len(selected))


def _entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def information_gain(probe: Probe, prior_one: float) -> float:
    p0, p1 = probe.probability_one
    marginal = (1 - prior_one) * p0 + prior_one * p1
    return _entropy(marginal) - (1 - prior_one) * _entropy(p0) - prior_one * _entropy(p1)


def choose_information_gain(pool: tuple[Probe, ...], prior_one: float) -> Probe:
    return min(pool, key=lambda p: (-information_gain(p, prior_one), -p.relevance, p.key))


def _posterior(probe: Probe, observed: int, prior_one: float) -> float:
    p0, p1 = probe.probability_one
    if observed == 0:
        p0, p1 = 1 - p0, 1 - p1
    denominator = (1 - prior_one) * p0 + prior_one * p1
    return prior_one if denominator == 0 else prior_one * p1 / denominator


def stop_when_sufficient(pool: tuple[Passage, ...], required: frozenset[str],
                         budget: int) -> tuple[Passage, ...]:
    selected: tuple[Passage, ...] = ()
    for item in _rank(pool, budget):
        if required <= _hints(selected):
            break
        selected += (item,)
    return selected


def _seeded(pool: tuple[Passage, ...], seed: int) -> tuple[Passage, ...]:
    rng = random.Random(seed)
    items = [replace(p, relevance=p.relevance + rng.uniform(-0.01, 0.01)) for p in pool]
    rng.shuffle(items)
    return tuple(items)


def _actual(selected: tuple[Passage, ...], truth: dict[str, frozenset[str]]) -> frozenset[str]:
    return frozenset(value for p in selected for value in truth[p.key])


def _row(index: int, kind: str, pool: tuple[Passage, ...], budget: int,
         baseline: tuple[Passage, ...], method: tuple[Passage, ...],
         baseline_value: float, method_value: float) -> Row:
    return {"case": index, "case_kind": kind,
            "candidate_pool": [{"id": p.key, "relevance": p.relevance,
                                "supplied_premise_hints": sorted(p.premise_hints),
                                "supplied_utility": p.utility, "supplied_stance": p.stance} for p in pool],
            "budget": budget, "baseline_selected": [p.key for p in baseline],
            "method_selected": [p.key for p in method],
            "baseline_used": len(baseline), "method_used": len(method),
            "baseline_value": baseline_value, "method_value": method_value}


def _average(rows: list[Row], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row[key]
        if not isinstance(value, (float, int)):
            raise TypeError(f"non-numeric observation: {key}")
        total += value
    return total / len(rows)


def _result(identifier: int, title: str, hypothesis: str, baseline: str, method: str,
            metric: str, rows: list[Row], limits: tuple[str, ...], reference: str,
            *, higher: bool = True, units: str = "fraction",
            metrics: dict[str, float] | None = None) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=identifier, title=title, hypothesis=hypothesis, baseline=baseline,
        method=method, metric=metric, baseline_score=_average(rows, "baseline_value"),
        method_score=_average(rows, "method_value"), higher_is_better=higher, units=units,
        cases=len(rows), evidence_kind="simulation", details=rows,
        limitations=limits + (
            "Eight constructed cases (four target, two control, two adverse); not held-out retrieval evaluation.",
            "Metadata are supplied oracle estimates, not discovered by a retriever or model.",
        ), references=(reference,), metrics=metrics or {})


def _missing_experiment() -> ExperimentResult:
    rows: list[Row] = []
    required, known = frozenset({"a", "b", "c"}), frozenset({"a"})
    for index, kind in enumerate(KINDS):
        pool: tuple[Passage, ...] = (Passage("repeat1", 0.95, frozenset({"a"})),
                Passage("repeat2", 0.9, frozenset({"a"})),
                Passage("bridge", 0.6, frozenset({"b"})),
                Passage("conclusion", 0.5, frozenset({"c"})))
        truth = {p.key: p.premise_hints for p in pool}
        if kind == "control":
            pool = tuple(replace(p, relevance=1.0 if p.key == "bridge" else
                                 0.98 if p.key == "conclusion" else p.relevance) for p in pool)
        if kind == "adverse":
            truth = {"repeat1": frozenset({"b"}), "repeat2": frozenset({"c"}),
                     "bridge": frozenset({"a"}), "conclusion": frozenset({"a"})}
        pool = _seeded(pool, 600 + index)
        baseline = _rank(pool, 2)
        method = choose_missing(pool, required, known, 2)
        row = _row(index, kind, pool, 2, baseline, method,
                   float(required <= known | _actual(baseline, truth)),
                   float(required <= known | _actual(method, truth)))
        row.update({"seed": 600 + index, "required": sorted(required), "known": sorted(known),
                    "scoring_truth_map": {k: sorted(v) for k, v in truth.items()}})
        rows.append(row)
    return _result(6, "Missing-premise retrieval", "Coverage selection can recover missing links under duplicate relevance.",
                   "Two highest-relevance passages", "Greedy uncovered-premise gain with relevance tie-break",
                   "Complete premise coverage", rows,
                   ("Premise dependencies are hand supplied; incorrect hints reverse the result.",), FLARE)


def _joint_experiment() -> ExperimentResult:
    rows: list[Row] = []
    groups: Groups = ((frozenset({"a", "b"}), 1.0), (frozenset({"c"}), 0.3), (frozenset({"d"}), 0.3))
    for index, kind in enumerate(KINDS):
        pool: tuple[Passage, ...] = (Passage("left", 0.6, frozenset({"a"}), 0.6),
                Passage("right", 0.5, frozenset({"b"}), 0.5),
                Passage("alone1", 0.9, frozenset({"c"}), 0.9),
                Passage("alone2", 0.8, frozenset({"d"}), 0.8))
        if kind == "control":
            pool = tuple(replace(p, utility=1.0 if p.key in {"left", "right"} else 0.2) for p in pool)
        pool = _seeded(pool, 700 + index)
        truth = {p.key: p.premise_hints for p in pool}
        if kind == "adverse":
            truth["right"] = frozenset({"irrelevant"})
        baseline = tuple(sorted(pool, key=lambda p: (-p.utility, p.key))[:2])
        method = choose_joint(pool, groups, 2)
        row = _row(index, kind, pool, 2, baseline, method,
                   _reward(_actual(baseline, truth), groups), _reward(_actual(method, truth), groups))
        row.update({"seed": 700 + index, "supplied_groups": [
            {"premises": sorted(needed), "reward": reward} for needed, reward in groups],
            "scoring_truth_map": {k: sorted(v) for k, v in truth.items()},
            "method_subsets_evaluated": math.comb(len(pool), 2), "baseline_candidates_scored": len(pool)})
        rows.append(row)
    return _result(7, "Evidence complementarity", "Joint utility can favor complete pairs over attractive individual passages.",
                   "Two highest supplied independent passage utilities", "Enumerate two-passage subsets and maximize completed-group reward",
                   "Realized complete-group reward", rows,
                   ("Both independent utilities and joint reward groups are supplied, not learned.",
                    "Equal retrieval budgets do not equal compute: six subsets evaluated; exact search scales combinatorially."),
                   SUFFICIENCY, units="reward")


def _counter_experiment() -> ExperimentResult:
    rows: list[Row] = []
    for index, kind in enumerate(KINDS):
        pool = _seeded((Passage("support1", 0.95, frozenset()),
                        Passage("support2", 0.9, frozenset()),
                        Passage("support3", 0.85, frozenset()),
                        Passage("challenge", 0.5, frozenset(), stance=-1)), 800 + index)
        contributions = {"support1": 1.0, "support2": 1.0, "support3": 1.0,
                         "challenge": -0.5 if kind == "control" else -3.0 - 0.1 * (index % 2)}
        truth = kind != "target"
        baseline = _rank(tuple(p for p in pool if p.stance >= 0), 3)
        method = choose_reserved(pool, 3)
        bp = sum(contributions[p.key] for p in baseline) > 0
        mp = sum(contributions[p.key] for p in method) > 0
        row = _row(index, kind, pool, 3, baseline, method, float(bp == truth), float(mp == truth))
        row.update({"seed": 800 + index, "observed_evidence_contributions": contributions,
                    "scoring_claim_true": truth, "baseline_prediction": bp, "method_prediction": mp})
        rows.append(row)
    return _result(8, "Counterevidence search", "A reserved challenge slot can expose false claims under confirmatory retrieval.",
                   "Three highest-relevance supportive passages", "Two supportive passages and one opposing passage",
                   "Claim decision accuracy", rows,
                   ("Stances and evidence strengths are supplied; misleading counterevidence harms true claims.",
                    "The baseline is deliberately confirmation-only, not a strong unbiased retriever.",
                    "Candidate generation and extra query costs are excluded; both see the same four-item pool."), FLARE)


def _log_loss(posterior: float, truth: int) -> float:
    return -math.log2(max(1e-12, posterior if truth == 1 else 1 - posterior))


def _information_experiment() -> ExperimentResult:
    rows: list[Row] = []
    for index, kind in enumerate(KINDS):
        rng = random.Random(900 + index)
        error = 0.08 + 0.01 * (index % 4)
        items = [Probe("topical", 0.95, (0.5, 0.5)),
                 Probe("separator", 0.4, (error, 1 - error)), Probe("weak", 0.6, (0.4, 0.6))]
        if kind == "control":
            items[0] = replace(items[0], probability_one=(error, 1 - error))
        rng.shuffle(items)
        pool = tuple(items)
        truth = index % 2
        observed = {"topical": truth, "separator": 1 - truth if kind == "adverse" else truth, "weak": truth}
        baseline = min(pool, key=lambda p: (-p.relevance, p.key))
        method = choose_information_gain(pool, 0.5)
        bp, mp = _posterior(baseline, observed[baseline.key], 0.5), _posterior(method, observed[method.key], 0.5)
        rows.append({"case": index, "case_kind": kind, "seed": 900 + index,
                     "budget": 1, "baseline_used": 1, "method_used": 1,
                     "candidate_pool": [{"id": p.key, "relevance": p.relevance,
                                         "supplied_probability_one_by_hypothesis": list(p.probability_one),
                                         "estimated_information_gain_bits": information_gain(p, 0.5)} for p in pool],
                     "prior_one": 0.5, "observed_outcomes": observed, "scoring_hypothesis": truth,
                     "baseline_selected": [baseline.key], "method_selected": [method.key],
                     "baseline_posterior_one": bp, "method_posterior_one": mp,
                     "baseline_value": _log_loss(bp, truth), "method_value": _log_loss(mp, truth)})
    return _result(9, "Decision information gain", "Expected discrimination helps decisions when likelihoods are reliable.",
                   "One highest-relevance probe", "One probe maximizing supplied binary mutual information",
                   "Realized hypothesis log loss", rows,
                   ("Likelihoods are supplied oracle estimates; outcomes are constructed, not sampled deployment data.",
                    "Uncertainty reduction is not truth: confidently wrong likelihoods can make the method worse.",
                    "The cited pruning work is related research, not reproduced by this probe-selection diagnostic."),
                   IG, higher=False, units="bits")


def _stopping_experiment() -> ExperimentResult:
    rows: list[Row] = []
    required = frozenset({"a", "b", "c"})
    for index, kind in enumerate(KINDS):
        first = frozenset({"a"}) if kind == "control" else required
        pool = _seeded((Passage("first", 0.95, first), Passage("second", 0.85, frozenset({"b"})),
                        Passage("third", 0.75, frozenset({"c"})), Passage("extra", 0.65, frozenset({"d"}))), 1000 + index)
        truth = {p.key: p.premise_hints for p in pool}
        if kind == "adverse":
            truth["first"] = frozenset({"a"})
        baseline, method = _rank(pool, 4), stop_when_sufficient(pool, required, 4)
        row = _row(index, kind, pool, 4, baseline, method, float(len(baseline)), float(len(method)))
        row.update({"seed": 1000 + index, "required": sorted(required),
                    "scoring_truth_map": {k: sorted(v) for k, v in truth.items()},
                    "baseline_complete": float(required <= _actual(baseline, truth)),
                    "method_complete": float(required <= _actual(method, truth))})
        rows.append(row)
    return _result(10, "Sufficiency stopping", "A support stop can save reads when sufficiency estimates are reliable.",
                   "Read four ranked passages", "Stop on supplied complete-premise coverage with four-read maximum",
                   "Evidence passages consumed", rows,
                   ("Savings trade off against completeness when hints falsely imply sufficiency.",
                    "Only evidence reads counted; indexing, annotation and sufficiency-estimator costs are excluded.",
                    "Both use the same ranking and four-read maximum; the method may consume fewer reads."),
                   SUFFICIENCY, higher=False, units="passages",
                   metrics={"baseline_complete_rate": _average(rows, "baseline_complete"),
                            "method_complete_rate": _average(rows, "method_complete")})


def run() -> list[ExperimentResult]:
    return [_missing_experiment(), _joint_experiment(), _counter_experiment(),
            _information_experiment(), _stopping_experiment()]
