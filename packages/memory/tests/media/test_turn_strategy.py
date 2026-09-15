"""When the bot may take its turn, as strategies over a turn's judgement, with plain numbers."""

from __future__ import annotations

import math

import pytest

from scone_memory.realtime.turn_end import COMPLETE, INCOMPLETE, UNSURE, Judgement, TurnHold
from scone_memory.realtime.turn_strategy import STRATEGIES, EndOfTurn, KeypadSubmit, MinSpeech

ASKED = Judgement(COMPLETE, "question mark")


def test_end_of_turn_is_today_and_changes_nothing():
    strategy = EndOfTurn()
    assert strategy.name == "end_of_turn" and strategy.submit is None
    for judgement in (ASKED, Judgement(UNSURE, "no cue"), Judgement(INCOMPLETE, "trailing 'for'")):
        assert strategy.decide(judgement, 50.0) is judgement
        assert strategy.decide(judgement, None) is judgement


def test_min_speech_holds_a_short_turn_and_says_both_numbers():
    strategy = MinSpeech(0.8)
    assert strategy.name == "min_speech" and strategy.submit is None
    held = strategy.decide(ASKED, 312.4)
    assert held.verdict == INCOMPLETE
    assert held.cue == "min_speech: 312 ms < 800 ms"


def test_min_speech_takes_a_turn_that_reached_the_minimum():
    strategy = MinSpeech(0.8)
    assert strategy.decide(ASKED, 800.0) is ASKED
    assert strategy.decide(Judgement(UNSURE, "no cue"), 2400.0).verdict == UNSURE


def test_min_speech_leaves_a_clause_the_detector_holds_as_it_is():
    open_clause = Judgement(INCOMPLETE, "trailing 'for'")
    assert MinSpeech(0.8).decide(open_clause, 10.0) is open_clause


def test_min_speech_with_no_sign_of_speech_takes_the_turn_and_says_it_could_not_measure():
    decided = MinSpeech(0.8).decide(Judgement(UNSURE, "semantic turn detection off"), None)
    assert decided.verdict == UNSURE
    assert decided.cue == "semantic turn detection off; min_speech: speech duration unknown"


@pytest.mark.parametrize("seconds", [0, -1, True, math.inf, math.nan, "1", 61])
def test_min_speech_that_cannot_work_is_refused(seconds):
    with pytest.raises(ValueError):
        MinSpeech(seconds)


def test_keypad_submit_holds_every_spoken_turn_for_its_key():
    strategy = KeypadSubmit()
    assert strategy.name == "keypad_submit" and strategy.submit == "#"
    for judgement in (ASKED, Judgement(UNSURE, "no cue")):
        assert strategy.decide(judgement, 5000.0) == Judgement(INCOMPLETE, "keypad_submit: waiting for #")
    assert KeypadSubmit("*").submit == "*"


@pytest.mark.parametrize("key", ["x", "", "##", None, "a"])
def test_a_submit_key_that_is_not_a_key_is_refused(key):
    with pytest.raises(ValueError):
        KeypadSubmit(key)


def test_strategies_are_named_for_settings():
    assert STRATEGIES == ("end_of_turn", "min_speech", "keypad_submit")


def test_a_turn_held_by_a_strategy_says_so_when_its_hold_runs_out():
    turns = TurnHold(hold=1.0, max_duration=10.0)
    assert turns.heard("yes", "user", MinSpeech(0.8).decide(ASKED, 300.0), 0.0,
                       timeout_reason="strategy_timeout") == []
    [end] = turns.expire(1.0)
    assert (end.receipt.reason, end.receipt.cue) == ("strategy_timeout", "min_speech: 300 ms < 800 ms")


def test_the_detector_s_own_hold_still_reads_as_semantic():
    turns = TurnHold(hold=1.0, max_duration=10.0)
    turns.heard("book a table for", "user", Judgement(INCOMPLETE, "trailing 'for'"), 0.0)
    assert turns.expire(1.0)[0].receipt.reason == "semantic_incomplete_timeout"


def test_keys_held_for_a_submit_join_the_turn_and_wait_for_its_bound():
    turns = TurnHold(hold=1.0, max_duration=10.0)
    waiting = Judgement(INCOMPLETE, "keypad_submit: waiting for #")
    turns.heard("my PIN is", "caller", waiting, 0.0)
    assert turns.keyed("[keypad] 12", 2.0, hold=waiting) == []
    assert turns.pending and turns.deadline == 10.0, "held to the turn's own bound, not the hold"
    [end] = turns.keyed("[keypad] #", 3.0)
    assert (end.text, end.speaker, end.receipt.reason, end.receipt.fragments) == \
        ("my PIN is [keypad] 12 [keypad] #", "caller", "keypad", 3)


def test_held_keys_with_nothing_said_start_a_turn_that_the_bound_releases():
    turns = TurnHold(hold=1.0, max_duration=10.0)
    assert turns.keyed("[keypad] 4", 5.0, hold=Judgement(INCOMPLETE, "keypad_submit: waiting for #")) == []
    assert turns.expire(14.9) == []
    [end] = turns.expire(15.0)
    assert (end.text, end.speaker, end.receipt.reason) == ("[keypad] 4", "user", "max_duration")
    assert (end.receipt.verdict, end.receipt.cue) == (INCOMPLETE, "keypad_submit: waiting for #")


def test_held_keys_that_would_outgrow_the_turn_release_it_first():
    turns = TurnHold(hold=1.0, max_duration=10.0, max_bytes=12)
    waiting = Judgement(INCOMPLETE, "keypad_submit: waiting for #")
    turns.heard("0123456789", "user", waiting, 0.0)
    [end] = turns.keyed("[keypad] 5", 1.0, hold=waiting)
    assert (end.text, end.receipt.reason) == ("0123456789", "max_bytes")
    assert turns.pending
