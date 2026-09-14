"""What people said about a passage, folded into a lesson recall can show beside it.

Feedback is recorded -- a person marks a returned passage useful or not --
and then nothing reads it but a metrics count. The reference keeps outcomes
and surfaces them as lessons with decay and corroboration. Here a
passage's judgements (the latest per recall) are folded into a signed score
that halves every ``half_life_days``, counts of each kind, and a state:
``preferred`` only when corroborated by ``min_corroboration`` useful
judgements and none against, ``dead_end`` when it was only ever judged not
useful, ``contested`` when both, ``tentative`` otherwise. Lessons are shown
beside recall items when asked for; ranking is untouched until a measurement
says a lesson should move it. The read is bounded and says when it bit.
"""

from __future__ import annotations

import json
import math

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.ports import Event
from scone_memory.retrieval.lessons import fold_lessons
from scone_memory.testing import Clock


def judged(event_id: int, ts: str, chunk: int, useful: bool, recall: int = 1) -> Event:
    return Event(event_id=event_id, space="default", kind="feedback", ts=ts, schema_version=1,
                 payload={"recall_event_id": recall, "chunk_id": chunk, "useful": useful, "note": None})


NOW = "2026-06-01T00:00:00Z"


def test_two_useful_judgements_forty_days_apart_decay_and_corroborate():
    folded = fold_lessons([judged(1, "2026-04-22T00:00:00Z", 7, True, recall=1),
                           judged(2, "2026-06-01T00:00:00Z", 7, True, recall=2)],
                          now=NOW, half_life_days=30, min_corroboration=2)
    lesson = folded[7]
    assert lesson.score == pytest.approx(1 + 0.5 ** (40 / 30)) and lesson.state == "preferred"
    assert (lesson.useful, lesson.not_useful, lesson.last_at) == (2, 0, "2026-06-01T00:00:00Z")
    one = fold_lessons([judged(2, "2026-06-01T00:00:00Z", 7, True, recall=2)], now=NOW, half_life_days=30,
                       min_corroboration=2)
    assert one[7].state == "tentative", "one person's word is not yet a preference"


def test_a_fresh_dead_end_weighs_more_than_an_old_success():
    folded = fold_lessons([judged(1, "2026-04-02T00:00:00Z", 1, True), judged(2, "2026-05-31T00:00:00Z", 2, False)],
                          now=NOW, half_life_days=30, min_corroboration=2)
    assert folded[2].state == "dead_end" and abs(folded[2].score) > abs(folded[1].score)
    assert folded[2].score == pytest.approx(-(0.5 ** (1 / 30)))


def test_the_latest_judgement_of_one_recall_counts_and_both_kinds_make_it_contested():
    changed_mind = fold_lessons([judged(1, "2026-05-01T00:00:00Z", 3, False, recall=9),
                                 judged(2, "2026-05-02T00:00:00Z", 3, True, recall=9)],
                                now=NOW, half_life_days=30, min_corroboration=2)
    assert (changed_mind[3].useful, changed_mind[3].not_useful) == (1, 0)
    split = fold_lessons([judged(1, "2026-05-01T00:00:00Z", 3, False, recall=1),
                          judged(2, "2026-05-02T00:00:00Z", 3, True, recall=2)],
                         now=NOW, half_life_days=30, min_corroboration=2)
    assert split[3].state == "contested"


def test_a_judgement_after_now_is_not_counted():
    folded = fold_lessons([judged(1, "2026-07-01T00:00:00Z", 3, True)], now=NOW, half_life_days=30, min_corroboration=2)
    assert folded == {}


async def engine_with_feedback(clock: Clock):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock,
                                events=InMemoryEventLog()).open()
    await engine.remember("default", "The crane survey was booked for the third of May.")
    await engine.remember("default", "The canteen serves soup on Tuesdays.")
    return engine


async def test_the_engine_reads_lessons_within_its_window_and_says_when_its_bound_bit():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with_feedback(clock)
    try:
        first = await engine.recall("default", "crane survey booked")
        crane = first.items[0].chunk_id
        await engine.feedback("default", first.event_id, crane, True)
        clock.now = "2026-05-20T00:00:00.000Z"
        second = await engine.recall("default", "crane survey booked")
        await engine.feedback("default", second.event_id, crane, True)
        found = await engine.lessons("default")
        assert found.lessons[crane].state == "preferred" and found.events_read == 2 and found.events_cut is False
        assert found.lessons[crane].evidence == "present"
        record = found.record()
        assert record["window_days"] == 90 and record["half_life_days"] == 30 and record["events_cut"] is False
        cut = await engine.lessons("default", max_events=1)
        assert cut.events_cut is True and cut.events_read == 1 and cut.lessons[crane].useful == 1
        clock.now = "2026-09-01T00:00:00.000Z"
        assert (await engine.lessons("default")).lessons == {}, "judgements older than the window are not read"
    finally:
        await engine.close()


async def test_a_judged_passage_whose_episode_was_forgotten_is_gone():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with_feedback(clock)
    try:
        result = await engine.recall("default", "crane survey booked")
        item = result.items[0]
        await engine.feedback("default", result.event_id, item.chunk_id, False)
        await engine.forget("default", item.episode_id)
        assert (await engine.lessons("default")).lessons[item.chunk_id].evidence == "gone"
    finally:
        await engine.close()


async def test_recall_shows_lessons_only_when_asked_and_leaves_the_order_alone():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with_feedback(clock)
    try:
        result = await engine.recall("default", "crane survey booked")
        await engine.feedback("default", result.event_id, result.items[0].chunk_id, False)
        plain = await engine.recall("default", "crane survey booked")
        shown = await engine.recall("default", "crane survey booked", lessons=True)
    finally:
        await engine.close()
    assert [item.chunk_id for item in shown.items] == [item.chunk_id for item in plain.items]
    assert "lessons" not in json.dumps(plain.items[0].model_dump(mode="json")), "the default item is as it was"
    assert shown.items[0].lessons["state"] == "dead_end" and shown.items[0].lessons["not_useful"] == 1
    assert all(item.lessons is None for item in shown.items[1:])


def test_bad_lesson_settings_are_refused():
    from scone_memory.core.errors import InvalidInput

    for settings in ({"half_life_days": 0}, {"min_corroboration": 0}, {"half_life_days": math.inf}):
        with pytest.raises(InvalidInput):
            fold_lessons([], now=NOW, **{"half_life_days": 30, "min_corroboration": 2, **settings})


def test_the_latest_judgement_wins_whatever_order_the_events_arrive_in():
    """The event log answers newest first."""
    newest_first = [judged(2, "2026-05-02T00:00:00Z", 3, True, recall=9), judged(1, "2026-05-01T00:00:00Z", 3, False, recall=9)]
    folded = fold_lessons(newest_first, now=NOW, half_life_days=30, min_corroboration=2)
    assert (folded[3].useful, folded[3].not_useful) == (1, 0)


async def test_a_cut_read_keeps_the_newest_judgements():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with_feedback(clock)
    try:
        first = await engine.recall("default", "crane survey booked")
        crane = first.items[0].chunk_id
        await engine.feedback("default", first.event_id, crane, True)
        clock.now = "2026-05-20T00:00:00.000Z"
        second = await engine.recall("default", "crane survey booked")
        await engine.feedback("default", second.event_id, crane, False)
        cut = await engine.lessons("default", max_events=1)
    finally:
        await engine.close()
    assert cut.events_cut and (cut.lessons[crane].useful, cut.lessons[crane].not_useful) == (0, 1)
