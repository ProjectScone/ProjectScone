"""Query performance predictors read the shape of a result list."""

from __future__ import annotations

import math

import pytest

from scone_memory.retrieval.geometry import Centre
from scone_memory.retrieval.qpp import coherence, predict


def test_a_clear_winner_spreads_the_top_and_stands_above_the_rest():
    clear = predict([0.9, 0.5, 0.45, 0.44, 0.43, 0.42], k=3)
    flat = predict([0.5, 0.49, 0.48, 0.47, 0.46, 0.45], k=3)
    assert clear.dispersion > flat.dispersion
    assert clear.wig > flat.wig
    assert clear.gap == pytest.approx(0.9 - 0.45)


def test_nqc_divides_the_spread_by_the_list_mean():
    prediction = predict([0.8, 0.4, 0.3, 0.1], k=2)
    assert prediction.dispersion == pytest.approx(0.2)
    assert prediction.nqc == pytest.approx(0.2 / 0.4)
    assert prediction.depth == 4


def test_a_list_too_short_to_spread_predicts_nothing():
    prediction = predict([0.7], k=5)
    assert (prediction.dispersion, prediction.nqc, prediction.wig, prediction.gap) == (None, None, None, None)


def test_coherence_is_high_for_agreeing_results_and_low_for_scattered_ones():
    agreeing = [[1.0, 0.1, 0.0], [1.0, 0.0, 0.1], [1.0, 0.05, 0.05]]
    scattered = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert coherence(agreeing) > 0.95
    assert coherence(scattered) == pytest.approx(0.0)


def test_centring_removes_agreement_that_only_the_shared_direction_gave():
    """Every vector leans along the first axis; from the centre, the three
    point in different directions."""
    leaning = [[5.0, 1.0, 0.0], [5.0, -0.5, 0.87], [5.0, -0.5, -0.87]]
    centre = Centre(mean=(1.0, 0.0, 0.0), count=3)
    assert coherence(leaning) > 0.9
    assert coherence(leaning, centre=centre) < 0.0


def test_coherence_needs_two_directions():
    assert coherence([[1.0, 0.0]]) is None
    assert coherence([[0.0, 0.0], [1.0, 0.0]]) is None
    assert predict([0.9, 0.8], k=2).coherence is None


def test_vectors_feed_the_coherence_signal():
    prediction = predict([0.9, 0.8, 0.7], k=2, vectors=[[1.0, 0.0], [0.0, 1.0]])
    assert prediction.coherence == pytest.approx(0.0)
    assert prediction.coherence_over == 2


@pytest.mark.parametrize("scores, k", [([0.1, 0.9], 2), ([math.nan, 0.1], 2), ([0.9, 0.1], 1)])
def test_refuses_lists_it_cannot_read(scores, k):
    with pytest.raises(ValueError):
        predict(scores, k=k)
