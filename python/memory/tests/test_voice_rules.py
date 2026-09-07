"""The rules a voice provider has to keep, written once.

An adapter is somebody else's software, and the ways one can be wrong do
not depend on who is running the turn. These are the checks a voice
session and a pipeline both apply, so the two cannot drift into
disagreeing about what a well-behaved provider is.
"""

from __future__ import annotations

import pytest

from scone_memory.realtime.audio import (AudioChunk, Reply, SpeechStarted, Transcript, check_audio,
                                         is_question)
from scone_memory.realtime.events import ReplyCompleted, TextDelta

SOUND = AudioChunk(pcm=b"\x00\x01" * 80, sample_rate=16000)


def test_audio_is_handed_back_so_the_check_cannot_be_forgotten():
    assert check_audio(SOUND, 1000) is SOUND


@pytest.mark.parametrize("chunk", [b"\x00\x01" * 80, None, "sound", 7])
def test_something_that_is_not_audio_is_not_audio(chunk):
    """Bytes are not a chunk: a chunk carries its rate and its channels,
    and guessing those is how a call ends up sounding like a chipmunk."""
    with pytest.raises(ValueError, match="invalid or oversized"):
        check_audio(chunk, 1000)


def test_more_audio_than_was_agreed_is_refused():
    with pytest.raises(ValueError, match="invalid or oversized"):
        check_audio(SOUND, len(SOUND.pcm) - 1)


def test_a_finished_question_is_one_and_a_partial_one_is_not_yet():
    assert is_question(Transcript(text="where are my keys?", final=True)) is True
    assert is_question(Transcript(text="where are my", final=False)) is False
    assert is_question(Transcript(text="   ", final=True)) is False


@pytest.mark.parametrize("event, complaint", [
    (SpeechStarted(), "unsupported speech event"),
    ("words", "unsupported speech event"),
    (Transcript(text="hello", final="yes"), "invalid transcript event"),
    (Transcript(text=b"hello", final=True), "invalid transcript event"),
    (Transcript(text="hello", final=True, speaker=""), "invalid transcript speaker"),
])
def test_a_recognizer_reporting_nonsense_is_broken_rather_than_quiet(event, complaint):
    """Going on would mean trusting the rest of what it says."""
    with pytest.raises(ValueError, match=complaint):
        is_question(event)


def test_a_transcript_too_long_to_be_a_turn_is_refused_before_it_is_finished():
    """The size check comes first, so an enormous partial transcript is
    caught rather than accumulated in the hope it ends."""
    with pytest.raises(ValueError, match="byte limit"):
        is_question(Transcript(text="a" * 40000, final=False))


def test_an_answer_within_the_rules_is_accepted_and_knows_it_has_ended():
    reply = Reply(1000)
    reply.accept(TextDelta(text="all done. "))
    assert reply.done is False
    reply.accept(ReplyCompleted())
    assert reply.done is True


def test_speaking_after_saying_it_has_finished_is_refused():
    """What follows an ending is not a longer answer. It is whatever the
    adapter had lying around, which is exactly what must not be spoken
    or stored as something the assistant said."""
    reply = Reply(1000)
    reply.accept(ReplyCompleted())
    with pytest.raises(RuntimeError, match="after completion"):
        reply.accept(TextDelta(text="and another thing"))


@pytest.mark.parametrize("event, complaint", [
    (TextDelta(text=""), "invalid public text delta"),
    (TextDelta(text=None), "invalid public text delta"),
    (SpeechStarted(), "unsupported model event"),
])
def test_a_model_event_that_is_not_an_answer_is_refused(event, complaint):
    with pytest.raises(ValueError, match=complaint):
        Reply(1000).accept(event)


def test_an_answer_is_measured_across_the_whole_stream_not_one_piece():
    """Each piece is small and the answer is not: the count has to carry
    across events or a limit means nothing."""
    reply = Reply(10)
    reply.accept(TextDelta(text="12345"))
    with pytest.raises(RuntimeError, match="byte limit"):
        reply.accept(TextDelta(text="678901"))
