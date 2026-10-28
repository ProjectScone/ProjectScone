"""A budgeted window chosen for coverage.

Two near-copies of the best passage and one passage about the other half
of the question: relevance order spends the budget on the copies;
coverage spends it on one copy and the other half.
"""

from __future__ import annotations

import itertools

import pytest

from scone_memory.retrieval.packing import pack

RELEVANCE = {"copy-1": 0.9, "copy-2": 0.88, "other-half": 0.6, "noise": 0.05}
COST = {"copy-1": 100, "copy-2": 100, "other-half": 100, "noise": 100}
VECTORS = {
    "copy-1": [1.0, 0.0, 0.0],
    "copy-2": [0.99, 0.01, 0.0],
    "other-half": [0.0, 1.0, 0.0],
    "noise": [0.0, 0.0, 1.0],
}


def test_coverage_takes_the_other_half_over_a_second_copy():
    packed = pack(RELEVANCE, COST, VECTORS, budget=200)
    assert set(packed.chosen) == {"copy-1", "other-half"}
    assert packed.cost == 200 and not packed.single


def test_the_greedy_window_is_close_to_the_best_possible_one():
    """Brute force over every affordable window: greedy should reach the
    optimum on a case this small."""
    for budget in (100, 200, 300):
        packed = pack(RELEVANCE, COST, VECTORS, budget=budget)
        best = max(_objective(subset) for size in range(1, 5) for subset in itertools.combinations(RELEVANCE, size)
                   if sum(COST[key] for key in subset) <= budget)
        assert packed.objective == pytest.approx(best)


def test_cost_counts_so_a_cheap_passage_can_beat_an_expensive_one():
    packed = pack({"long": 1.0, "short": 0.9}, {"long": 300, "short": 50},
                  {"long": [1.0, 0.0], "short": [0.98, 0.2]}, budget=400)
    assert packed.chosen[0] == "short"


def test_the_cost_exponent_stops_a_short_weak_passage_jumping_the_queue():
    """Per token, the short passage wins; by gain alone, or by the default
    scaled cost, the relevant one does."""
    relevance = {"short-weak": 0.2, "long-strong": 1.0}
    cost = {"short-weak": 10, "long-strong": 100}
    vectors = {"short-weak": [0.0, 1.0], "long-strong": [1.0, 0.0]}
    assert pack(relevance, cost, vectors, budget=200, cost_exponent=1.0).chosen[0] == "short-weak"
    assert pack(relevance, cost, vectors, budget=200, cost_exponent=0.0).chosen[0] == "long-strong"
    assert pack(relevance, cost, vectors, budget=200).chosen[0] == "long-strong"


def test_the_best_single_passage_is_used_when_it_beats_the_greedy_set():
    """Greedy by gain per token takes the cheap, weak passage first and then
    cannot afford the one that matters."""
    packed = pack({"weak": 0.1, "strong": 1.0}, {"weak": 1, "strong": 100},
                  {"weak": [0.0, 1.0], "strong": [1.0, 0.0]}, budget=100, cost_exponent=1.0)
    assert packed.chosen == ("strong",)
    assert packed.single


def test_passages_larger_than_the_budget_are_counted_and_never_chosen():
    packed = pack({"huge": 1.0, "small": 0.2}, {"huge": 5000, "small": 10},
                  {"huge": [1.0, 0.0], "small": [0.0, 1.0]}, budget=100)
    assert packed.chosen == ("small",)
    assert packed.too_large == 1


def test_never_spends_more_than_the_budget():
    relevance = {f"p{index}": 1 / (index + 1) for index in range(12)}
    cost = {f"p{index}": 37 + 11 * index for index in range(12)}
    vectors = {f"p{index}": [1.0, index / 12, (index % 3) / 3] for index in range(12)}
    for budget in (40, 120, 333, 600):
        assert pack(relevance, cost, vectors, budget).cost <= budget


@pytest.mark.parametrize("relevance, cost, budget", [
    ({"a": 1.0}, {"a": 0}, 10), ({"a": 1.0}, {"a": 1.5}, 10), ({"a": 1.0}, {}, 10), ({"a": 1.0}, {"a": 1}, 0),
    ({"a": float("nan")}, {"a": 1}, 10),
])
def test_refuses_inputs_it_cannot_pack(relevance, cost, budget):
    with pytest.raises(ValueError):
        pack(relevance, cost, {"a": [1.0]}, budget)


@pytest.mark.parametrize("cost_exponent", [-0.1, 1.5, float("nan")])
def test_refuses_a_cost_exponent_outside_zero_to_one(cost_exponent):
    with pytest.raises(ValueError, match="cost_exponent"):
        pack({"a": 1.0}, {"a": 1}, {"a": [1.0]}, 10, cost_exponent=cost_exponent)


def _objective(subset):
    import math
    def like(a, b):
        u, v = VECTORS[a], VECTORS[b]
        return max(0.0, sum(x * y for x, y in zip(u, v)) / (math.hypot(*u) * math.hypot(*v)))
    covered = sum(RELEVANCE[i] * max(like(i, j) for j in subset) for i in RELEVANCE)
    return covered + sum(RELEVANCE[j] for j in subset)


def test_the_window_is_the_same_with_and_without_numpy(monkeypatch):
    pytest.importorskip("numpy")
    import random

    import scone_memory.retrieval.packing as packing

    rng = random.Random(7)
    relevance = {f"p{index}": rng.random() for index in range(40)}
    cost = {key: rng.randint(20, 90) for key in relevance}
    vectors = {key: [rng.gauss(0, 1) for _ in range(16)] for key in relevance}
    vectors["p3"] = [0.0] * 16
    with_numpy = pack(relevance, cost, vectors, budget=400)
    monkeypatch.setattr(packing, "_numpy", None)
    without = pack(relevance, cost, vectors, budget=400)
    assert with_numpy.chosen == without.chosen
    assert with_numpy.objective == pytest.approx(without.objective, rel=1e-12)
