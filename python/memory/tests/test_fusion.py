from scone_memory.fusion import Fused, cap_per_episode, normalise, order, rrf


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
