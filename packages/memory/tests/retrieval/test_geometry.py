"""Centred cosine removes the direction every vector shares.

The fixture is a cone: every vector leans along the first axis, so raw
cosines sit close together. Taken from the centre, the directions that
tell passages apart are what is left to compare.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.retrieval.geometry import (
    Centre, anisotropy, centre_of, centred_cosine, merge_centres, recentred,
)

CONE = {
    "a": [5.0, 1.0, 0.0],
    "b": [5.0, 0.0, 1.0],
    "c": [5.0, -1.0, 0.0],
    "d": [5.0, 0.0, -1.0],
}


def test_the_centre_is_the_mean_of_unit_vectors_not_of_raw_ones():
    """A long vector must not pull the centre towards itself."""
    centre = centre_of([[10.0, 0.0], [0.0, 1.0]])
    assert centre is not None
    assert centre.mean == pytest.approx((0.5, 0.5))
    assert centre.count == 2


def test_zero_vectors_are_counted_and_left_out_of_the_centre():
    centre = centre_of([[0.0, 0.0], [1.0, 0.0]])
    assert centre is not None
    assert (centre.count, centre.skipped_zero) == (1, 1)
    assert centre_of([[0.0, 0.0]]) is None


def test_centres_merge_to_the_centre_of_the_union():
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, -1.0], [0.5, 2.0]]
    whole = centre_of(vectors)
    first, second = centre_of(vectors[:2]), centre_of(vectors[2:])
    assert whole is not None and first is not None and second is not None
    merged = merge_centres(first, second)
    assert merged.mean == pytest.approx(whole.mean)
    assert merged.count == whole.count


def test_centring_spreads_scores_that_the_shared_direction_squeezed_together():
    centre = centre_of(CONE.values())
    assert centre is not None
    query = [5.0, 1.0, 0.2]
    raw = {key: _cosine(query, vector) for key, vector in CONE.items()}
    centred = {key: centred_cosine(query, vector, centre) for key, vector in CONE.items()}
    assert max(raw.values()) - min(raw.values()) < 0.1
    assert max(centred.values()) - min(centred.values()) > 1.5
    assert centred["a"] == max(centred.values())
    assert centred["c"] == min(centred.values())


def test_recentred_orders_candidates_and_puts_the_undirected_last():
    centre = Centre(mean=(1.0, 0.0), count=1)
    ordered = recentred([0.0, 1.0], {"at-centre": [1.0, 0.0], "up": [1.0, 1.0], "down": [1.0, -1.0]}, centre)
    assert [key for key, _ in ordered] == ["up", "down", "at-centre"]
    assert ordered[-1][1] == -math.inf


def test_anisotropy_is_near_one_for_a_cone_and_near_zero_for_an_even_spread():
    """``n`` directions spread evenly round a circle average ``-1/(n-1)``
    over distinct pairs, which is close to zero for many directions."""
    assert anisotropy(list(CONE.values()), pairs=200).mean_cosine > 0.9
    circle = [[math.cos(math.radians(degree)), math.sin(math.radians(degree))] for degree in range(360)]
    spread = anisotropy(circle, pairs=20_000)
    assert spread.mean_cosine is not None and abs(spread.mean_cosine) < 0.02


def test_anisotropy_is_reproducible_and_empty_without_two_directions():
    vectors = [[1.0, 0.2], [0.3, 1.0], [0.9, 0.9], [1.0, -0.4]]
    assert anisotropy(vectors, seed=3) == anisotropy(vectors, seed=3)
    assert anisotropy([[1.0, 0.0], [0.0, 0.0]]).mean_cosine is None


@pytest.mark.parametrize("vectors", [[[1.0, 0.0], [1.0, 0.0, 0.0]], [[math.inf, 0.0]]])
def test_centre_refuses_mixed_dimensions_and_non_finite_values(vectors):
    with pytest.raises(ValueError):
        centre_of(vectors)


def _cosine(left, right):
    return sum(a * b for a, b in zip(left, right)) / (math.hypot(*left) * math.hypot(*right))
