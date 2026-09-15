"""Whether a speaker has finished, from what was said as well as from the pause.

The rules are read against scripted transcripts and the hold is driven with
plain numbers for a clock, so nothing here waits on real time."""

from __future__ import annotations

import pytest

from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.realtime.turn_end import (
    COMPLETE, INCOMPLETE, UNSURE, ChatEndOfTurn, Judgement, LexicalEndOfTurn, TurnHold, judge_text,
)


@pytest.mark.parametrize("text, cue", [
    ("I want to book a table and", "trailing 'and'"),
    ("I was going to say but", "trailing 'but'"),
    ("Could you send it to", "trailing 'to'"),
    ("Put it on the", "trailing 'the'"),
    ("I need a flight from", "trailing 'from'"),
    ("She said \"call me when you land", "open quote"),
    ("The meeting (the one on Friday", "open parenthesis"),
    ("She wrote " + chr(0x201C) + "see you", "open quote"),
    ("I think, um", "filler 'um'"),
    ("Let me see, uh.", "filler 'uh'"),
    ("So what I mean is,", "trailing ','"),
    ("Well I was thinking...", "trailing '...'"),
    ("When I get home remind me to", "trailing 'to'"),
    ("How do I get from", "trailing 'from'"),
    ("why don't you add it to", "trailing 'to'"),
])
def test_a_transcript_cut_mid_clause_is_incomplete(text, cue):
    assert judge_text(text) == Judgement(INCOMPLETE, cue)


@pytest.mark.parametrize("text, cue", [
    ("Book a table for four people.", "terminal punctuation"),
    ("That is wonderful!", "terminal punctuation"),
    ("Where does Juniper point?", "question mark"),
    ("He said \"go home.\"", "terminal punctuation"),
    ("Can you tell me what it is for?", "question mark"),
    ("I went to the shop and.", "terminal punctuation"),
])
def test_a_finished_sentence_or_a_question_is_complete(text, cue):
    assert judge_text(text) == Judgement(COMPLETE, cue)


@pytest.mark.parametrize("text", ["turn it on", "I think that is all", "   ", "okay", "what I want is the"])
def test_no_evidence_either_way_is_unsure_unless_a_clause_is_open(text):
    verdict = judge_text(text)
    if text == "what I want is the":
        assert verdict == Judgement(INCOMPLETE, "trailing 'the'"), "an article never ends a question"
    else:
        assert verdict.verdict == UNSURE


@pytest.mark.parametrize("text", [
    "set a timer for 20", "Book a table for 2", "set the temperature to 72", "remind me at 5",
    "send it to " + chr(0x6771) + chr(0x4EAC), "put it in the " + chr(0x0444) + chr(0x0430) + chr(0x0439) + chr(0x043B),
])
def test_the_last_word_may_be_a_number_or_in_any_script(text):
    assert judge_text(text) == Judgement(UNSURE, "no cue"), "the word before it is not the last word"


@pytest.mark.parametrize("text", [
    "what's on my calendar today", "how long does it take", "can you help me", "what I really need is",
    "What's the weather going to be like in",
])
def test_a_question_word_first_is_not_evidence_that_the_turn_is_finished(text):
    assert judge_text(text) == Judgement(UNSURE, "no cue")


@pytest.mark.parametrize("text, cue", [
    ("what are you waiting for", "'for' may be stranded by 'what'"),
    ("where did you send it to", "'to' may be stranded by 'where'"),
    ("where's the party at", "'at' may be stranded by 'where'"),
    ("who is this for", "'for' may be stranded by 'who'"),
    ("how much is it for", "'for' may be stranded by 'how much'"),
    ("How many people is the table for", "'for' may be stranded by 'how many'"),
])
def test_a_preposition_a_wh_phrase_can_strand_is_neither_held_nor_called_complete(text, cue):
    assert judge_text(text) == Judgement(UNSURE, cue)


@pytest.mark.parametrize("verdict, cue", [("maybe", "no cue"), (COMPLETE, None)])
def test_a_judgement_is_a_known_verdict_with_a_cue(verdict, cue):
    with pytest.raises(ValueError):
        Judgement(verdict, cue)


async def test_the_lexical_detector_is_the_rules_behind_the_seam():
    detector = LexicalEndOfTurn()
    assert await detector.judge("Send it to") == Judgement(INCOMPLETE, "trailing 'to'")
    assert await detector.aclose() is None


async def test_a_chat_model_can_stand_behind_the_seam():
    chat = FakeChat(["incomplete", " Complete.\n", "maybe"])
    detector = ChatEndOfTurn(chat)
    assert await detector.judge("I want to book a table for") == Judgement(INCOMPLETE, "model")
    assert await detector.judge("Book it.") == Judgement(COMPLETE, "model")
    assert await detector.judge("hmm") == Judgement(UNSURE, "unreadable model answer")
    assert chat.calls[0][1] == "I want to book a table for" and "complete" in chat.calls[0][0]
    with pytest.raises(ChatError):
        await detector.judge("one more")


def hold(**options):
    return TurnHold(**{"hold": 1.5, "max_duration": 10.0, **options})


def test_a_complete_transcript_ends_the_turn_when_it_is_heard():
    turns = hold()
    [end] = turns.heard("Book a table.", "user", Judgement(COMPLETE, "terminal punctuation"), now=3.0)
    assert end.text == "Book a table." and end.speaker == "user"
    assert end.receipt.reason == "semantic_complete" and end.receipt.cue == "terminal punctuation"
    assert (end.receipt.fragments, end.receipt.held_ms, end.receipt.turn_ms) == (1, 0.0, 0.0)
    assert not turns.pending and turns.deadline is None


def test_no_evidence_ends_the_turn_at_the_pause_the_recognizer_already_waited():
    [end] = hold().heard("okay", "user", Judgement(UNSURE, "no cue"), now=1.0)
    assert end.receipt.reason == "silence" and end.receipt.verdict == UNSURE


def test_an_incomplete_transcript_is_held_and_the_continuation_joins_it():
    turns = hold()
    assert turns.heard("I want a table for", "user", Judgement(INCOMPLETE, "trailing 'for'"), now=2.0) == []
    assert turns.pending and turns.deadline == 3.5
    assert turns.text_with("four people.", "user") == "I want a table for four people."
    assert turns.expire(now=3.4) == [], "nothing is released before the deadline"
    turns.speech_started(now=2.6)
    assert turns.deadline == 12.0, "while the speaker talks only the turn's bound stands"
    [end] = turns.heard("four people.", "user", Judgement(COMPLETE, "terminal punctuation"), now=3.9)
    assert end.text == "I want a table for four people."
    assert end.receipt.reason == "semantic_complete" and end.receipt.fragments == 2
    assert end.receipt.held_ms == 0.0 and end.receipt.turn_ms == 1900.0


def test_a_hold_that_runs_out_releases_the_turn_and_says_so():
    turns = hold(hold=0.5)
    turns.heard("Send it to", "user", Judgement(INCOMPLETE, "trailing 'to'"), now=1.0)
    [end] = turns.expire(now=1.5)
    assert end.text == "Send it to" and end.receipt.reason == "semantic_incomplete_timeout"
    assert end.receipt.held_ms == 500.0 and end.receipt.verdict == INCOMPLETE
    assert not turns.pending and turns.expire(now=9.0) == []


def test_the_turn_bound_caps_the_hold_and_is_reported_when_it_bites():
    turns = hold(hold=1.5, max_duration=2.0)
    turns.heard("I need a", "user", Judgement(INCOMPLETE, "trailing 'a'"), now=0.0)
    turns.speech_started(now=0.8)
    turns.heard("flight and", "user", Judgement(INCOMPLETE, "trailing 'and'"), now=1.0)
    assert turns.deadline == 2.0, "the hold would run to 2.5; the turn's bound is 2.0"
    [end] = turns.expire(now=2.0)
    assert end.receipt.reason == "max_duration" and end.text == "I need a flight and"


def test_a_fragment_past_the_turn_bound_ends_the_turn_at_once():
    turns = hold(hold=1.5, max_duration=2.0)
    turns.heard("I need a", "user", Judgement(INCOMPLETE, "trailing 'a'"), now=0.0)
    turns.speech_started(now=0.5)
    [end] = turns.heard("flight to", "user", Judgement(INCOMPLETE, "trailing 'to'"), now=2.5)
    assert end.receipt.reason == "max_duration" and end.receipt.turn_ms == 2500.0


def test_speech_that_never_becomes_a_transcript_is_released_at_the_turn_bound():
    turns = hold(max_duration=4.0)
    turns.heard("and then", "user", Judgement(INCOMPLETE, "trailing 'and'"), now=1.0)
    turns.speech_started(now=1.2)
    assert turns.expire(now=4.9) == []
    [end] = turns.expire(now=5.0)
    assert end.receipt.reason == "max_duration"


def test_speech_started_with_nothing_held_changes_nothing():
    turns = hold()
    turns.speech_started(now=1.0)
    assert not turns.pending and turns.deadline is None


def test_another_speaker_releases_the_held_turn_before_their_own():
    turns = hold()
    turns.heard("I was going to", "alice", Judgement(INCOMPLETE, "trailing 'to'"), now=1.0)
    assert turns.text_with("Hello.", "bob") == "Hello."
    first, second = turns.heard("Hello.", "bob", Judgement(COMPLETE, "terminal punctuation"), now=1.2)
    assert (first.speaker, first.text, first.receipt.reason) == ("alice", "I was going to", "speaker_changed")
    assert (second.speaker, second.text, second.receipt.reason) == ("bob", "Hello.", "semantic_complete")


def test_a_held_turn_that_would_outgrow_its_byte_bound_is_released_first():
    turns = hold(max_bytes=32)
    turns.heard("Twenty bytes of it and", "user", Judgement(INCOMPLETE, "trailing 'and'"), now=1.0)
    assert turns.text_with("more words to", "user") == "more words to"
    first, = turns.heard("more words to", "user", Judgement(INCOMPLETE, "trailing 'to'"), now=1.1)
    assert first.receipt.reason == "max_bytes" and first.text == "Twenty bytes of it and"
    assert turns.pending and turns.text_with("x", "user") == "more words to x"


def test_a_stopping_session_releases_what_is_held_under_its_own_reason():
    turns = hold()
    turns.heard("Cancel my card and", "user", Judgement(INCOMPLETE, "trailing 'and'"), now=1.0)
    [end] = turns.drain(now=1.5, reason="session_ended")
    assert end.receipt.reason == "session_ended" and end.text == "Cancel my card and" and not turns.pending


def test_input_ending_releases_what_is_held():
    turns = hold()
    assert turns.drain(now=1.0) == []
    turns.heard("I think", "user", Judgement(INCOMPLETE, "filler"), now=1.0)
    [end] = turns.drain(now=1.25)
    assert end.receipt.reason == "input_ended" and end.receipt.held_ms == 250.0


@pytest.mark.parametrize("options", [{"hold": 0}, {"hold": float("inf")}, {"max_duration": -1},
                                     {"hold": 3.0, "max_duration": 2.0}, {"hold": True}, {"max_bytes": 0},
                                     {"max_bytes": 2.5}, {"max_duration": float("inf")}])
def test_bounds_must_be_sensible(options):
    with pytest.raises(ValueError):
        hold(**options)


def test_a_receipt_reads_as_strings_for_a_record():
    [end] = hold().heard("Done.", "user", Judgement(COMPLETE, "terminal punctuation"), now=0.0)
    assert end.receipt.as_dict() == {"reason": "semantic_complete", "verdict": COMPLETE, "cue": "terminal punctuation",
                                     "fragments": 1, "held_ms": 0.0, "turn_ms": 0.0}
