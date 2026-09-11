"""Temporal questions answered by computation, with the working shown.

Each event phrase is grounded against memory: the passage holding most
of its words, on the day that passage records. The arithmetic is then
exact, and every step is cited, so the answer can be checked instead of
believed. When an event cannot be pinned down, nothing is computed.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.retrieval.temporal import temporal_answer

NOW = "2023-04-20T10:12:00Z"
DIARY = [
    ("2023-03-11T09:00:00Z", "I sold homemade baked goods at the farmers market with my neighbour."),
    ("2023-04-01T18:30:00Z", "I ran the charity bake-off at the village hall and we raised plenty."),
    ("2023-04-11T12:00:00Z", "I met Emma for coffee near the river and we talked about her new job."),
    ("2023-02-02T08:00:00Z", "I started the Spanish course at the community centre this morning."),
    ("2023-01-05T20:00:00Z", "I finished reading The Nightingale, which took me most of December."),
]


async def diary(*extra: tuple[str, str]) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for when, text in (*DIARY, *extra):
        await engine.remember("alpha", text, created_at=when)
    return engine


async def test_the_days_between_two_events_are_counted_exactly():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "How many weeks passed between the time I sold homemade "
                                                    "baked goods and the time I ran the charity bake-off?", now=NOW)
    assert answer.status == "computed"
    assert answer.value == {"days": 21, "weeks": 3, "months": 0, "unit": "week", "asked": 3}
    assert "answer: 3 weeks (21 days)" in answer.text
    assert "2023-03-11 → 2023-04-01 is 21 days" in answer.text
    [first, second] = answer.anchors
    assert (first["date"], second["date"]) == ("2023-03-11", "2023-04-01")
    assert first["episode_id"] and first["quote"] in DIARY[0][1]


async def test_how_long_ago_counts_from_the_moment_asked():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "How many days ago did I meet Emma?", now=NOW)
    assert answer.status == "computed" and answer.value["days"] == 9
    assert "answer: 9 days ago" in answer.text and "2023-04-11 → 2023-04-20 is 9 days" in answer.text


async def test_an_answer_asked_for_no_unit_is_given_in_days():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "How long ago did I meet Emma?", now=NOW)
    assert answer.status == "computed" and "answer: 9 days ago" in answer.text


async def test_which_of_two_came_first_is_the_earlier_of_them():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "Which came first, the charity bake-off or the Spanish "
                                                    "course at the community centre?", now=NOW)
    assert answer.status == "computed"
    assert answer.value["first"] == "spanish course at the community centre"
    assert "answer: spanish course at the community centre, on 2023-02-02" in answer.text


async def test_events_are_put_in_the_order_their_days_give():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "In what order did these happen: 'I met Emma for coffee', "
                                                    "'I sold homemade baked goods', 'I ran the charity bake-off'?",
                                   now=NOW)
    assert answer.status == "computed"
    assert [event["date"] for event in answer.value["order"]] == ["2023-03-11", "2023-04-01", "2023-04-11"]


async def test_an_event_memory_does_not_hold_is_not_computed_around():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "How many days ago did I climb Mount Fuji?", now=NOW)
    assert answer.status == "ungrounded" and answer.value == {}
    assert "coverage: limited: ungrounded" in answer.text and "climb mount fuji" in answer.text


async def test_an_event_told_on_two_days_is_said_to_be_ambiguous():
    """Two passages hold the phrase equally well and record different days,
    so which day is meant is not for this to guess."""
    engine = await diary(("2023-04-05T09:00:00Z", "I met Emma for coffee near the river again."))
    answer = await temporal_answer(engine, "alpha", "How many days ago did I meet Emma for coffee near the river?",
                                   now=NOW)
    assert answer.status == "ambiguous" and answer.value == {}
    assert "2023-04-05" in answer.text and "2023-04-11" in answer.text


async def test_a_question_that_is_not_temporal_is_left_to_the_caller():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "What did I bake for the charity event?", now=NOW)
    assert answer.status == "not_temporal" and answer.anchors == () and answer.value == {}


async def test_the_answer_records_what_it_read_and_what_it_computed():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "How many days ago did I meet Emma?", now=NOW)
    record = answer.record("alpha")
    assert record["schema_version"] == 1 and record["space"] == "alpha"
    assert record["now"] == "2023-04-20T10:12:00.000Z", "the moment asked, as the ledger writes moments"
    assert record["plan"] == {"kind": "since", "unit": "day", "events": ["meet emma"]}
    assert record["status"] == "computed" and record["value"]["days"] == 9
    assert record["anchors"][0]["chunk_id"] and record["anchors"][0]["support"] >= 0.5


@pytest.mark.parametrize("options, message", [
    ({"limit": 0}, "limit"), ({"limit": 51}, "limit"), ({"max_bytes": 10}, "max_bytes"),
])
async def test_bounds_are_refused_before_anything_is_read(options, message):
    from scone_memory.retrieval.temporal import TemporalError

    engine = await diary()
    with pytest.raises(TemporalError, match=message):
        await temporal_answer(engine, "alpha", "How many days ago did I meet Emma?", now=NOW, **options)


async def test_between_is_the_distance_either_way_round():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "How many days passed between the time I ran the charity "
                                                    "bake-off and the time I sold homemade baked goods?", now=NOW)
    assert answer.status == "computed" and answer.value["days"] == 21


async def test_whole_months_are_counted_by_the_day_of_the_month():
    """On 20 April, something on 25 January was two months ago, not three."""
    engine = await diary(("2023-01-25T09:00:00Z", "I booked the airbnb in San Francisco for the summer."))
    answer = await temporal_answer(engine, "alpha", "How many months ago did I book the airbnb in San Francisco?",
                                   now=NOW)
    assert answer.status == "computed" and answer.value["months"] == 2 and answer.value["asked"] == 2
    assert answer.value["days"] == 85


async def test_the_passage_holding_most_of_the_words_grounds_the_event():
    """Ranking answers questions; grounding a date needs the passage that
    actually holds the event, so the words decide and the rank breaks ties."""
    from scone_memory.core.models import RecallItem, RecallResult
    from scone_memory.retrieval.temporal import _anchor

    class Recalls:
        """Returns the passages given, in the order given."""

        def __init__(self, *items: tuple[int, str, str]) -> None:
            self.items = [RecallItem(chunk_id=n, episode_id=n, text=text, score=1.0, created_at=when)
                          for n, when, text in items]

        async def recall(self, space, query, limit=5, as_of=None):
            return RecallResult(items=self.items[:limit], query=query)

    engine = Recalls((1, "2023-03-02T09:00:00Z", "The community centre course fair was busy this morning."),
                     (2, "2023-02-02T08:00:00Z", "I started the Spanish course at the community centre."))
    anchor = await _anchor(engine, "alpha", "spanish course at the community centre", limit=5, as_of=None)
    assert (anchor["status"], anchor["date"], anchor["episode_id"]) == ("found", "2023-02-02", 2)


async def test_a_day_another_passage_nearly_matches_is_undecided():
    """A passage from another day holding nearly as much of a long phrase
    leaves which day is meant open, and both days are shown: here one
    word in fourteen separates them."""
    engine = await diary(
        ("2023-04-11T12:00:00Z", "I meet Emma for coffee near the river to talk about her new job in Lisbon "
                                 "on this rainy Tuesday morning."),
        ("2023-04-05T09:00:00Z", "Emma and I had coffee near the river to talk about her new job in Lisbon; "
                                 "a rainy Tuesday morning."))
    answer = await temporal_answer(engine, "alpha", "How many days ago did I meet Emma for coffee near the river "
                                                    "to talk about her new job in Lisbon on a rainy Tuesday "
                                                    "morning?", now=NOW)
    assert answer.status == "ambiguous" and "2023-04-05" in answer.text and "2023-04-11" in answer.text


async def test_one_passage_cannot_date_two_events_apart():
    """A passage that holds both events records one day, so the distance
    between them would be an artefact of that, and is not computed."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("alpha", "Looking back: I sold baked goods at the market, and later I ran the "
                                   "charity bake-off at the village hall.", created_at="2023-04-18T09:00:00Z")
    answer = await temporal_answer(engine, "alpha", "How many days passed between the time I sold baked goods and "
                                                    "the time I ran the charity bake-off?", now=NOW)
    assert answer.status == "ambiguous" and answer.value == {}
    assert "one passage holds more than one of the events" in answer.text


async def test_what_was_recorded_on_a_day_is_returned_with_the_day_it_read():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "What did I do 9 days ago?", now=NOW)
    assert answer.status == "recalled"
    assert answer.value["window"] == {"start": "2023-04-11", "end": "2023-04-12", "words": "9 days ago"}
    assert answer.value["passages"][0]["date"] == "2023-04-11" and "Emma" in answer.value["passages"][0]["quote"]
    assert "on: 2023-04-11, from \"9 days ago\"" in answer.text


async def test_a_day_with_nothing_recorded_says_so():
    engine = await diary()
    answer = await temporal_answer(engine, "alpha", "What did I do 3 days ago?", now=NOW)
    assert answer.status == "ungrounded" and answer.value == {}
    assert "nothing recorded" in answer.text


async def ledger() -> MemoryEngine:
    """Alice worked at Acme for a while, then at Globex; the Lisbon office
    opened and has not closed."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from="2021-03-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Globex", valid_from="2023-06-30T00:00:00Z")
    await engine.assert_fact("alpha", "lisbon office", "status", "open", valid_from="2022-01-15T00:00:00Z")
    return engine


async def test_how_long_a_claim_held_is_the_length_of_its_valid_time():
    answer = await temporal_answer(await ledger(), "alpha", "How long did Alice work at Acme Robotics?", now=NOW)
    assert answer.status == "computed"
    assert answer.value["days"] == 851 and answer.value["months"] == 27
    assert "answer: 851 days" in answer.text and "2021-03-01 → 2023-06-30" in answer.text
    assert answer.anchors[0]["claim"] == "alice chen works_at Acme Robotics"
    assert answer.anchors[0]["fact_ids"] and answer.anchors[0]["status"] == "found"


async def test_a_claim_that_still_holds_is_counted_up_to_the_moment_asked():
    answer = await temporal_answer(await ledger(), "alpha", "How long has the Lisbon office been open?", now=NOW)
    assert answer.status == "computed" and answer.value["holds"] is True
    assert "so far" in answer.text and answer.value["days"] == (
        __import__("datetime").date(2023, 4, 20) - __import__("datetime").date(2022, 1, 15)).days


async def test_when_a_claim_began_and_ended_are_both_named():
    answer = await temporal_answer(await ledger(), "alpha", "When did Alice work at Acme Robotics?", now=NOW)
    assert answer.status == "computed"
    assert answer.value == {"from": "2021-03-01", "until": "2023-06-30", "holds": False,
                            "days": 851, "months": 27, "years": 2}
    assert "answer: from 2021-03-01 until 2023-06-30" in answer.text


async def test_a_claim_the_ledger_does_not_hold_is_not_computed_around():
    answer = await temporal_answer(await ledger(), "alpha", "How long did Alice work at Initech?", now=NOW)
    assert answer.status == "ungrounded" and answer.value == {}
    assert "coverage: limited: ungrounded" in answer.text


async def test_two_claims_the_question_fits_alike_leave_it_undecided():
    engine = await ledger()
    await engine.assert_fact("alpha", "alice chen", "advises", "Acme Robotics", valid_from="2020-02-01T00:00:00Z")
    answer = await temporal_answer(engine, "alpha", "How long has Alice Chen Acme Robotics?", now=NOW)
    assert answer.status == "ambiguous" and answer.value == {}
