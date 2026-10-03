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
    ({"mode": "collect", "speech_wait": 0}, "speech_wait"), ({"mode": "collect", "speech_wait": 61}, "speech_wait"),
    ({"mode": "collect", "speech_wait": float("inf")}, "speech_wait"), ({"mode": "collect", "speech_wait": True}, "speech_wait"),
])
def test_a_policy_is_checked_when_it_is_made(options, match):
    with pytest.raises(ValueError, match=match):
        KeypadPolicy(**options)


def test_keys_pressed_after_the_caller_began_speaking_wait_for_those_words():
    """A recognizer gives the words only after the caller stops and it has
    waited to be sure; keys come at once. Keys pressed after speech began
    are held back until its words are given, then given after them."""
    collector = KeypadCollector(KeypadPolicy("collect", timeout=3.0, speech_wait=2.0))
    collector.speech(1.0)
    assert press(collector, "41#", at=1.5, step=0.1) == [], "finished, and waiting for the words"
    assert not collector.pending and collector.deadline == pytest.approx(1.7 + 2.0)
    first, after = collector.words(2.5)
    assert first == []
    [entry] = after
    assert (entry.keys, entry.ended_by, entry.waited) == ("41#", "terminator", "words")
    assert entry.metadata(origin=0.0)["keypad_waited"] == "words"
    assert collector.deadline is None
    [next_entry] = collector.press(Keypress("#"), 3.0)
    assert next_entry.waited is None and "keypad_waited" not in next_entry.metadata(origin=0.0), \
        "those words were given: the next key waits for nothing"


def test_keys_pressed_before_the_caller_began_speaking_are_given_before_the_words():
    collector = KeypadCollector(KeypadPolicy("collect"))
    press(collector, "77", at=1.0)
    [first], after = collector.words(3.0)
    assert (first.keys, first.ended_by, first.waited, after) == ("77", "speech", None, []), \
        "with no sign of when speech began, keys are given as they arrived"

    press(collector, "88", at=3.5)
    collector.speech(4.0)
    collector.speech(4.4)  # still the same speech: it began at 4.0
    [first], after = collector.words(5.0)
    assert (first.keys, first.ended_by, after) == ("88", "speech", [])

    collector.speech(6.0)
    press(collector, "9", at=6.0)
    assert collector.words(7.0) == ([], []), "a key pressed as speech began came after it"
    assert collector.pending


def test_speech_begins_once_until_its_words_come():
    """A partial transcript says the caller is still speaking; it does not
    make the speech begin again after keys pressed during it."""
    collector = KeypadCollector(KeypadPolicy("collect"))
    collector.speech(0.0)
    press(collector, "1", at=0.2)
    collector.speech(0.5)
    assert collector.press(Keypress("#"), 0.6) == [], "pressed after the speech began: waits for its words"


def test_keys_still_being_collected_after_the_words_carry_on():
    collector = KeypadCollector(KeypadPolicy("collect", timeout=3.0))
    collector.speech(1.0)
    press(collector, "41", at=1.2, step=0.1)
    assert collector.words(2.0) == ([], [])
    assert collector.pending and collector.deadline == pytest.approx(4.3)
    [entry] = collector.expire(4.3)
    assert (entry.keys, entry.ended_by, entry.waited) == ("41", "timeout", None)


def test_keys_wait_for_words_no_longer_than_speech_wait_and_say_so():
    collector = KeypadCollector(KeypadPolicy("append", speech_wait=0.5))
    collector.speech(0.0)
    assert collector.press(Keypress("5"), 0.2) == []
    assert collector.press(Keypress("6"), 0.4) == []
    assert collector.deadline == pytest.approx(0.7), "from the first key held back"
    assert collector.expire(0.699) == []
    five, six = collector.expire(0.7)
    assert [(e.keys, e.ended_by, e.waited) for e in (five, six)] == [("5", "key", "timeout"), ("6", "key", "timeout")]
    [seven] = collector.press(Keypress("7"), 0.8)
    assert seven.waited is None, "the words are given up on: the next key is not held for them"
    assert collector.words(0.9) == ([], []), "and words that come late find nothing waiting"


def test_speech_that_ends_without_words_lets_the_keys_go():
    collector = KeypadCollector(KeypadPolicy("append"))
    collector.speech(0.0)
    assert collector.press(Keypress("5"), 0.0) == [], "pressed as the speech began"
    [entry] = collector.no_words(0.3)
    assert (entry.keys, entry.waited) == ("5", "no_words")
    assert collector.no_words(0.4) == [] and collector.press(Keypress("6"), 0.5)[0].waited is None


def test_draining_gives_keys_waiting_for_words_before_keys_being_collected():
    collector = KeypadCollector(KeypadPolicy("collect"))
    collector.speech(0.0)
    assert press(collector, "1#2", at=0.1, step=0.1) == []
    waiting, collecting = collector.drain(1.0, "session_ended")
    assert (waiting.keys, waiting.ended_by, waiting.waited) == ("1#", "terminator", "no_words")
    assert (collecting.keys, collecting.ended_by, collecting.waited) == ("2", "session_ended", None)
    assert collector.drain(1.0, "session_ended") == [] and collector.deadline is None


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


def test_the_entries_a_turn_was_given_have_one_receipt_with_the_latest_keys():
    from scone_memory.core.validation import MAX_METADATA_VALUE
    from scone_memory.realtime.keypad import MAX_DIGITS, KeypadEntry, joined

    first = KeypadEntry("[keypad] 12", (Keypress("1"), Keypress("2", "inband")), (10.0, 10.5), "timeout")
    last = KeypadEntry("[keypad] #", (Keypress("#"),), (14.0,), "terminator", "words")
    assert joined([last]) == last
    entry = joined([first, last])
    assert (entry.text, entry.keys, entry.at, entry.ended_by, entry.waited, entry.dropped) == \
        ("[keypad] 12 [keypad] #", "12#", (10.0, 10.5, 14.0), "terminator", "words", 0)
    assert entry.metadata(9.0) == {"keypad_keys": "12#", "keypad_ended": "terminator", "keypad_sources": "eie",
                                   "keypad_started_ms": "1000", "keypad_at_ms": "0,500,4000", "keypad_waited": "words"}
    many = [KeypadEntry(f"[keypad] {n % 10}", (Keypress(str(n % 10)),), (n * 3.0,), "key") for n in range(MAX_DIGITS + 3)]
    cut = joined(many)
    assert (cut.dropped, len(cut.presses), cut.at[0]) == (3, MAX_DIGITS, 9.0), "the latest keys are kept"
    metadata = cut.metadata(0.0)
    assert metadata["keypad_dropped"] == "3" and metadata["keypad_started_ms"] == "9000"
    assert all(len(value) <= MAX_METADATA_VALUE for value in metadata.values())
