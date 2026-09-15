"""The semantic end-of-turn replay: what it models, and what it measured on the fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from scone_memory.bench.semantic_turn import (
    START_MS, STOP_MS, WORD_MS, events, judge_cost_ns, load, measure, replay, report, session_latency,
)

FIXTURE = Path(__file__).resolve().parents[2] / "benchmarks" / "semantic-turn-v1.json"


def utterance(fragments, pauses, kind="cut"):
    return {"id": "u", "kind": kind, "fragments": fragments, "pauses_ms": pauses}


def test_a_long_pause_ends_a_transcript_and_a_short_one_is_bridged():
    cut = utterance(["Send it to", "the team."], [700])
    three = 3 * WORD_MS
    assert events(cut) == [(three + STOP_MS, "transcript", "Send it to"),
                           (three + 700 + START_MS, "speech", ""),
                           (three + 700 + 2 * WORD_MS + STOP_MS, "transcript", "the team.")]
    bridged = utterance(["Order a pizza", "with cheese."], [300], kind="complete")
    assert events(bridged) == [(3 * WORD_MS + 300 + 2 * WORD_MS + STOP_MS, "transcript", "Order a pizza with cheese.")]


def test_silence_alone_answers_half_a_sentence_and_the_detector_waits_for_the_rest():
    cut = utterance(["Send it to", "the team."], [700])
    alone = replay(cut, semantic=False)
    assert alone.premature == 1 and [end.text for _, end in alone.released] == ["Send it to", "the team."]
    waited = replay(cut, semantic=True)
    assert waited.premature == 0 and waited.added_ms == 0
    [(_, end)] = waited.released
    assert end.text == "Send it to the team." and end.receipt.reason == "semantic_complete"


def test_a_pause_longer_than_the_hold_is_still_cut_and_says_so():
    late = replay(utterance(["Send it to", "the team."], [2500]), semantic=True)
    assert late.premature == 1 and late.released[0][1].receipt.reason == "semantic_incomplete_timeout"


def test_a_complete_turn_the_rules_mistake_for_open_is_held_and_charged():
    held = replay(utterance(["sure I'd love to"], [], kind="complete"), semantic=True)
    assert held.premature == 0 and held.added_ms == 1500


def test_a_complete_turn_ending_on_a_number_is_answered_at_the_pause():
    run = replay(utterance(["set a timer for 20"], [], kind="complete"), semantic=True)
    assert run.added_ms == 0 and run.released[0][1].receipt.reason == "silence"


def test_a_malformed_utterance_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"utterances": [{"id": "x", "kind": "cut", "fragments": ["a", "b"], "pauses_ms": []}]}')
    with pytest.raises(ValueError, match="x"):
        load(path)


def test_the_fixture_measurement():
    tally = measure(FIXTURE)
    assert (tally.utterances, tally.cut) == (32, 16)
    assert tally.premature == {"silence": 17, "semantic": 5}
    assert tally.answered_early["silence"] == [f"cut-{n:02}" for n in range(1, 17)]
    assert tally.answered_early["semantic"] == ["cut-04", "cut-09", "cut-11", "cut-15", "cut-16"]
    assert sum(value > 0 for value in tally.complete_added_ms["semantic"]) == 2
    assert all(value == 0 for value in tally.complete_added_ms["silence"])
    assert tally.reasons == {"silence": {"silence": 49},
                             "semantic": {"semantic_complete": 24, "silence": 10, "semantic_incomplete_timeout": 3}}
    [cut04] = [u for u in load(FIXTURE) if u["id"] == "cut-04"]
    first = replay(cut04, semantic=True).released[0][1].receipt
    assert (first.reason, first.cue) == ("silence", "no cue"), "cut-04 was released for want of a cue, not as complete"
    assert "premature endings" in report(tally)
    assert len(judge_cost_ns(FIXTURE, repeats=1, loops=1)) == 1


async def test_session_latency_is_measured_through_a_real_session():
    for semantic in (False, True):
        latencies = await session_latency(["What time is it?", "Turn off the lights."], semantic=semantic)
        assert len(latencies) == 2 and all(value > 0 for value in latencies)
    [held] = await session_latency(["Send it to"], semantic=True)
    assert held >= 1400, "the session measured with the detector really holds an open clause"
    [answered] = await session_latency(["Send it to"], semantic=False)
    assert answered < held
