"""What a lane earns the right to say on this query, from the shape of
its own scores.

One weight for every query is a bet that a lane is equally worth
listening to whatever it was asked. It is not: the same vector lane that
places the answer first for a question written in the corpus's own words
ranks twenty near-identical neighbours for a question whose words it has
never seen, and on that query its order is close to arbitrary. The
lane's own scores say which of the two happened, before anything is
fused, without a model.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.retrieval.lane_trust import MIN_JUDGED, lane_voice


def lane(*scores: float) -> list[tuple[int, float]]:
    return [(i + 1, score) for i, score in enumerate(scores)]


def test_a_lane_that_ranks_one_thing_far_above_the_rest_speaks_at_full_voice():
    voice, judged = lane_voice(lane(0.91, 0.12, 0.11, 0.10, 0.10), floor=0.25)
    assert judged["separation"] == pytest.approx(1.0, abs=0.02)
    assert voice == pytest.approx(1.0, abs=0.02)
    assert judged["voice"] == voice


def test_a_lane_whose_candidates_are_indistinguishable_keeps_the_voice_it_was_given():
    voice, judged = lane_voice(lane(0.40, 0.40, 0.40, 0.40), floor=0.25)
    assert judged["separation"] == 0.0 and voice == 0.25
    assert judged["reason"] == "no spread"


def test_the_voice_rises_with_the_separation_and_never_leaves_the_band():
    voices = [lane_voice(lane(1.0, *[gap] * 6), floor=0.25)[0] for gap in (0.95, 0.7, 0.4, 0.0)]
    assert voices == sorted(voices), "a sharper lane is never given a quieter voice"
    assert all(0.25 <= voice <= 1.0 for voice in voices)


def test_a_lane_too_small_to_have_a_shape_is_not_called_confident():
    """One hit is the top and the bottom at once: its separation would
    read as perfect, which is the opposite of what one hit shows."""
    for scores in ([0.9], [0.9, 0.1]):
        voice, judged = lane_voice(lane(*scores), floor=0.25)
        assert (voice, judged["reason"]) == (0.25, f"fewer than {MIN_JUDGED} candidates to judge")
        assert "separation" not in judged
    assert len([0.9, 0.1]) < MIN_JUDGED <= 3


def test_a_lane_that_reports_no_score_is_judged_on_nothing_and_keeps_its_voice():
    """A store that ranks without a score hands back NaN; its order may
    be perfect, but nothing in it says so."""
    voice, judged = lane_voice(lane(float("nan"), float("nan"), float("nan")), floor=0.25)
    assert (voice, judged["reason"]) == (0.25, "no scores to judge")
    voice, judged = lane_voice(lane(0.9, float("nan"), 0.1), floor=0.25)
    assert (voice, judged["reason"]) == (0.25, "no scores to judge")


def test_an_empty_lane_earns_nothing_and_is_not_judged():
    voice, judged = lane_voice([], floor=0.25)
    assert (voice, judged["reason"]) == (0.25, "no candidates")


@pytest.mark.parametrize("floor", [1.0, 2.5])
def test_a_lane_already_trusted_as_much_as_the_other_is_left_alone(floor):
    """The band runs from the voice the operator set up to the voice the
    lane it argues with has. Set at or above that, there is no band, and
    the rule has nothing to say rather than something quieter."""
    voice, judged = lane_voice(lane(0.91, 0.12, 0.11, 0.10), floor=floor)
    assert (voice, judged["reason"]) == (floor, "no band to move in")


def test_the_ceiling_can_be_named_and_the_voice_stays_under_it():
    voice, _ = lane_voice(lane(0.91, 0.12, 0.11, 0.10), floor=0.2, ceiling=0.6)
    assert 0.2 <= voice <= 0.6
    assert voice == pytest.approx(0.6, abs=0.02), "a lane this sharp reaches the ceiling it was given"


def test_a_lane_falling_away_evenly_lands_in_the_middle_of_the_band():
    voice, judged = lane_voice(lane(1.0, 0.8, 0.6, 0.4, 0.2), floor=0.0, ceiling=1.0)
    assert judged["separation"] == pytest.approx(0.5)
    assert voice == pytest.approx(0.5)


def test_the_separation_is_read_from_the_middle_not_the_mean():
    """One candidate far below the rest drags a mean down and would read
    as a confident top; the middle of the lane does not move."""
    heavy_tail = lane(0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.0, 0.0, 0.0, 0.0)
    voice, judged = lane_voice(heavy_tail, floor=0.25)
    assert judged["separation"] < 0.1, "six candidates within a hair of each other is not a confident lane"
    assert voice < 0.35
    raw = [score for _, score in heavy_tail]
    assert (raw[0] - sum(raw) / len(raw)) / (raw[0] - min(raw)) > 0.4, "which is what a mean would have said"


def test_a_score_that_is_not_a_number_at_all_is_refused_rather_than_ranked():
    with pytest.raises(TypeError):
        lane_voice([(1, "high"), (2, "low"), (3, "lower")], floor=0.25)  # type: ignore[list-item]


def test_an_infinite_score_is_no_shape_to_judge():
    voice, judged = lane_voice(lane(math.inf, 0.2, 0.1), floor=0.25)
    assert (voice, judged["reason"]) == (0.25, "no scores to judge")


# -- through a real engine ---------------------------------------------------------


class FlatEmbedder:
    """Every text the same vector: a lane that can only rank by the order
    the index returns, with no distance between its candidates."""

    id = "flat-1"
    dim = 8

    async def embed(self, texts):
        return [[1.0] * self.dim for _ in texts]


class OneStandsOut:
    """A vector for the number of times a text says 'billing', so one
    passage sits far above the rest and the shape is real."""

    id = "billing-1"
    dim = 2

    async def embed(self, texts):
        return [[float(text.lower().count("billing")), 1.0] for text in texts]


async def engine_with(embedder, **kw):
    from scone_memory import InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
                                events=InMemoryEventLog(), vector_weight=0.25, **kw).open()
    for text in ("The billing run went out late and the invoices were wrong.",
                 "Invoices again, about nothing in particular at all.",
                 "A shed roof before winter, and the gutters.",
                 "Notes from the Tuesday meeting about the office move."):
        await engine.remember("s", text)
    return engine


async def recall_event(engine, query="billing run invoices"):
    await engine.recall("s", query, limit=3)
    [event] = await engine.events.query("s", kind="recall")
    return event.payload


async def test_a_flat_vector_lane_keeps_the_configured_voice_and_the_event_says_why():
    payload = await recall_event(await engine_with(FlatEmbedder(), lane_trust=True))
    assert payload["fusion_weights"]["vector"] == 0.25
    assert payload["lane_trust"]["vector"]["reason"] == "no spread"
    assert payload["lane_trust"]["vector"]["separation"] == 0.0


async def test_a_vector_lane_with_a_real_shape_earns_a_fuller_voice():
    payload = await recall_event(await engine_with(OneStandsOut(), lane_trust=True))
    judged = payload["lane_trust"]["vector"]
    assert judged["separation"] > 0.5, judged
    assert 0.25 < payload["fusion_weights"]["vector"] <= 1.0
    assert payload["fusion_weights"]["vector"] == judged["voice"]


async def test_switched_off_nothing_is_read_and_nothing_is_recorded():
    payload = await recall_event(await engine_with(OneStandsOut()))
    assert payload["fusion_weights"]["vector"] == 0.25, "the configured voice, on every query"
    assert "lane_trust" not in payload


async def test_the_engine_refuses_anything_but_a_boolean():
    from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.core.errors import InvalidInput

    for bad in (1, "on", None):
        with pytest.raises(InvalidInput, match="lane_trust"):
            MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), FlatEmbedder(), lane_trust=bad)  # type: ignore[arg-type]
