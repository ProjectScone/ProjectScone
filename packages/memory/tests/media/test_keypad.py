"""Keys pressed on a phone, as a conversation takes them."""

from __future__ import annotations

import pytest

from scone_memory.realtime.keypad import KEYS, Keypress, keypad_key


def test_a_keypress_is_one_of_sixteen_keys_and_says_how_it_arrived():
    assert sorted(KEYS) == sorted("0123456789*#ABCD")
    assert Keypress("5").source == "event"
    assert Keypress("#", "inband", offset_ms=120.0, tone_ms=51.2).tone_ms == 51.2
    for bad in ("", "55", "a", "E", " 5"):
        with pytest.raises(ValueError, match="keypress is one of"):
            Keypress(bad)
    with pytest.raises(ValueError, match="arrives by"):
        Keypress("5", "voice")
    assert [keypad_key(v) for v in ("d", " 7 ", 0, True, 1.0, "##")] == ["D", "7", "0", None, None, None]


from scone_memory.realtime.keypad import KeypadCollector, KeypadPolicy  # noqa: E402
from scone_memory.realtime.turn_end import COMPLETE, INCOMPLETE, Judgement, TurnHold  # noqa: E402


def press(collector, keys, *, at=0.0, step=0.5, source="event"):
    released = []
    for n, key in enumerate(keys):
        released += collector.press(Keypress(key, source), at + n * step)
    return released


def test_append_gives_every_key_to_the_turn_at_once():
    collector = KeypadCollector(KeypadPolicy("append"))
    [entry] = collector.press(Keypress("5"), 1.0)
    assert (entry.keys, entry.ended_by, entry.text) == ("5", "key", "[keypad] 5")
    assert not collector.pending and collector.deadline is None


def test_collect_holds_keys_until_the_terminator_and_the_terminator_is_part_of_the_entry():
    collector = KeypadCollector(KeypadPolicy("collect", timeout=3.0))
    assert collector.press(Keypress("1"), 10.0) == []
    assert collector.pending and collector.deadline == 13.0
    assert collector.press(Keypress("2", "inband", offset_ms=900.0, tone_ms=45.0), 12.5) == []
    assert collector.deadline == 15.5, "the timeout runs from the latest key"
    [entry] = collector.press(Keypress("#"), 13.0)
    assert (entry.keys, entry.ended_by, entry.text) == ("12#", "terminator", "[keypad] 12#")
    assert [p.source for p in entry.presses] == ["event", "inband", "event"] and entry.at == (10.0, 12.5, 13.0)
    assert not collector.pending and collector.deadline is None


def test_collect_releases_on_the_timeout_by_the_clock_it_is_given():
    collector = KeypadCollector(KeypadPolicy("collect", timeout=3.0))
    press(collector, "42", at=0.0, step=2.9)
    assert collector.expire(5.899) == [], "not yet: 2.9 + 3.0"
    [entry] = collector.expire(5.9)
    assert (entry.keys, entry.ended_by) == ("42", "timeout")
    assert collector.expire(100.0) == [] and collector.deadline is None

    press(collector, "7", at=200.0)
    [late] = collector.press(Keypress("8"), 203.0)
    assert (late.keys, late.ended_by) == ("7", "timeout"), "a key after the timeout, before anyone expired it, starts a new entry"
    assert collector.pending and collector.deadline == 206.0


def test_the_digit_bound_releases_the_entry_and_says_so():
    collector = KeypadCollector(KeypadPolicy("collect", max_digits=3))
    [entry] = press(collector, "1234")
    assert (entry.keys, entry.ended_by) == ("123", "max_digits")
    assert collector.pending, "the next key starts the next entry"
    [ended] = press(collector, "5#", at=2.0)
    assert (ended.keys, ended.ended_by) == ("45#", "terminator"), "a terminator that also fills the entry is a terminator"


def test_without_a_terminator_only_the_timeout_or_the_bound_ends_an_entry():
    collector = KeypadCollector(KeypadPolicy("collect", terminator=None))
    assert press(collector, "12#*") == []
    [entry] = collector.drain(9.0, "input_ended")
    assert (entry.keys, entry.ended_by) == ("12#*", "input_ended")
    assert collector.drain(9.0, "session_ended") == []


def test_an_entry_records_how_and_when_each_key_came_in_metadata_a_record_can_hold():
    collector = KeypadCollector(KeypadPolicy("collect", timeout=60.0, max_digits=32))
    released = []
    for n in range(32):
        released += collector.press(Keypress("9", "inband" if n % 2 else "event"), 100.0 + n * 59.999)
    [entry] = released
    metadata = entry.metadata(origin=40.0)
    assert metadata["keypad_keys"] == "9" * 32 and metadata["keypad_ended"] == "max_digits"
    assert metadata["keypad_sources"] == "ei" * 16, "e for the transport's event, i for tones heard in the audio"
    assert metadata["keypad_started_ms"] == "60000", "the first key, from the session's start"
    offsets = metadata["keypad_at_ms"].split(",")
    assert offsets[:3] == ["0", "59999", "119998"] and len(offsets) == 32
    assert all(1 <= len(value) <= 256 for value in metadata.values()), "the longest entry the policy allows still fits"


@pytest.mark.parametrize("options, match", [
    ({"mode": "off"}, "mode"), ({"mode": "collect", "timeout": 0}, "timeout"),
    ({"mode": "collect", "timeout": 61}, "timeout"), ({"mode": "collect", "timeout": float("nan")}, "timeout"),
    ({"mode": "collect", "max_digits": 0}, "max_digits"), ({"mode": "collect", "max_digits": 33}, "max_digits"),
    ({"mode": "collect", "max_digits": 4.0}, "max_digits"), ({"mode": "collect", "terminator": "##"}, "terminator"),
    ({"mode": "collect", "template": "keys"}, "template"), ({"mode": "collect", "template": "{keys} {other}"}, "template"),
])
def test_a_policy_is_checked_when_it_is_made(options, match):
    with pytest.raises(ValueError, match=match):
        KeypadPolicy(**options)


def test_keys_join_a_held_spoken_turn_and_release_it():
    turns = TurnHold(hold=1.5, max_duration=10)
    assert turns.heard("my account number is", "caller", Judgement(INCOMPLETE, "no cue"), 1.0) == []
    [end] = turns.keyed("[keypad] 1234#", 4.0)
    assert (end.text, end.speaker) == ("my account number is [keypad] 1234#", "caller")
    assert (end.receipt.reason, end.receipt.verdict, end.receipt.cue) == ("keypad", COMPLETE, "keypad")
    assert (end.receipt.fragments, end.receipt.held_ms, end.receipt.turn_ms) == (2, 0.0, 3000.0)
    assert not turns.pending and turns.deadline is None

    [alone] = turns.keyed("[keypad] 5", 9.0, reason="session_ended")
    assert (alone.text, alone.speaker, alone.receipt.reason, alone.receipt.fragments) == ("[keypad] 5", "user", "session_ended", 1)


def test_keys_that_would_outgrow_a_held_turn_release_it_first():
    turns = TurnHold(max_bytes=30)
    turns.heard("I would like to pay", "user", Judgement(INCOMPLETE, "no cue"), 0.0)
    first, second = turns.keyed("[keypad] 123456", 1.0)
    assert (first.text, first.receipt.reason) == ("I would like to pay", "max_bytes")
    assert (second.text, second.receipt.reason) == ("[keypad] 123456", "keypad")
