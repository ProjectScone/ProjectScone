from scone_memory.retrieval.fusion import Fused, cap_per_episode, normalise, order, rrf


def test_tied_scores_order_by_chunk_id():
    items = [Fused(9, 0.5, None), Fused(3, 0.5, None), Fused(7, 0.9, None)]
    assert [f.chunk_id for f in order(items)] == [7, 3, 9]


def test_top_item_is_normalised_to_one():
    items = normalise(order([Fused(1, 0.02, None), Fused(2, 0.01, None)]))
    assert items[0].score == 1.0
    assert items[1].score == 0.5


def test_rrf_rewards_appearing_in_both_lanes():
    scores = rrf([[(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)]])
    assert scores[2] > scores[1] > 0
    assert scores[2] > scores[3]


def test_per_episode_cap_keeps_two():
    items = [Fused(i, 1.0 - i / 100, None) for i in range(1, 6)]
    kept = cap_per_episode(items, {1: 10, 2: 10, 3: 10, 4: 20, 5: 10}, cap=2)
    assert [f.chunk_id for f in kept] == [1, 2, 4]


def test_the_recency_term_is_a_weight_that_halves_every_half_life_and_zero_favours_nothing():
    import math

    import pytest

    from scone_memory.core.errors import InvalidInput
    from scone_memory.retrieval.fusion import MAX_RECENCY_WEIGHT, recency_boost, validate_recency

    now = "2024-03-31T00:00:00Z"
    fresh, old = "2024-03-31T00:00:00Z", "2024-03-01T00:00:00Z"
    assert recency_boost(fresh, now, weight=0.2, half_life_days=30.0) == pytest.approx(0.2)
    assert recency_boost(old, now, weight=0.2, half_life_days=30.0) == pytest.approx(0.1), "a half-life halves"
    assert recency_boost("2024-03-31T00:00:00Z", "2024-05-30T00:00:00Z", weight=0.2, half_life_days=30.0) == pytest.approx(0.05), "and halves again"
    assert recency_boost(old, now, weight=0.2, half_life_days=3.0) < recency_boost(old, now, weight=0.2, half_life_days=300.0), \
        "a shorter half-life forgets faster"
    assert recency_boost(old, now, weight=0.0, half_life_days=30.0) == 0.0 == recency_boost(fresh, now, weight=0.0)
    assert recency_boost(old, now) == recency_boost(old, now, weight=0.005, half_life_days=30.0), "the defaults are the constants"
    from scone_memory.retrieval.fusion import MAX_RECENCY_HALF_LIFE_DAYS

    for weight, half_life in ((-0.1, 30.0), (MAX_RECENCY_WEIGHT + 1, 30.0), (float("nan"), 30.0), (0.1, 0.0), (0.1, -1.0), (0.1, float("inf")),
                              (0.1, MAX_RECENCY_HALF_LIFE_DAYS + 1), (True, 30.0)):
        with pytest.raises(InvalidInput, match="recency_"):
            validate_recency(weight, half_life)
