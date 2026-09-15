"""A voice session whose turn strategy decides when the bot may take its turn.

The strategies' rules are driven with plain numbers in test_turn_strategy.py;
here the session runs them against a scripted recognizer, with short real waits."""

from __future__ import annotations

import asyncio
import time

import pytest

from scone_memory.realtime.audio import SpeechStarted, Transcript
from scone_memory.realtime.keypad import KeypadPolicy, Keypress
from scone_memory.realtime.turn_strategy import EndOfTurn, KeypadSubmit, MinSpeech
from scone_memory.realtime.voice import VoiceSession

from .test_voice_turn_end import SID, SPACE, Rig, memory  # noqa: F401 - memory is a fixture


async def keys(rig, *presses):
    for press in presses:
        await rig.transport.input.put(Keypress(press))


async def test_end_of_turn_given_explicitly_is_today(memory):
    async with Rig(memory, turn_strategy=EndOfTurn()) as rig:
        await rig.say(SpeechStarted(), Transcript("Yes."))
        [user] = await rig.users(1)
        assert (user.content, user.metadata["turn_end"]) == ("Yes.", "silence")
        assert rig.session.last_turn_receipt.cue == "semantic turn detection off"
        await rig.finish()


async def test_min_speech_holds_a_short_turn_and_the_next_words_join_it(memory):
    async with Rig(memory, turn_strategy=MinSpeech(.3), turn_hold=1.0) as rig:
        await rig.say(SpeechStarted(), Transcript("Yes."))
        await asyncio.sleep(.2)
        assert rig.session.stored_count == 0 and rig.model.contexts == [], "too short to answer yet"
        await rig.say(SpeechStarted())
        await asyncio.sleep(.15)
        await rig.say(Transcript("the second one."))  # timed from the turn's first speech, not this speech
        [user] = await rig.users(1)
        assert user.content == "Yes. the second one."
        receipt = rig.session.last_turn_receipt
        assert (receipt.reason, receipt.fragments) == ("silence", 2)
        assert len(rig.model.contexts) == 1
        await rig.finish()


async def test_min_speech_answers_a_short_turn_when_its_hold_runs_out_and_says_why(memory):
    async with Rig(memory, turn_strategy=MinSpeech(.5), turn_hold=.2) as rig:
        began = time.perf_counter()
        await rig.say(SpeechStarted(), Transcript("Yes."))
        [user] = await rig.users(1)
        assert time.perf_counter() - began >= .2
        assert user.metadata["turn_end"] == "strategy_timeout"
        receipt = rig.session.last_turn_receipt
        assert receipt.verdict == "incomplete" and receipt.cue.startswith("min_speech: ") and receipt.cue.endswith(" < 500 ms")
        await rig.finish()


async def test_min_speech_takes_a_long_enough_turn_at_once_and_times_the_next_turn_afresh(memory):
    async with Rig(memory, turn_strategy=MinSpeech(.25), turn_hold=2.0) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.3)
        began = time.perf_counter()
        await rig.say(Transcript("I need help with my order."))
        [user] = await rig.users(1)
        assert time.perf_counter() - began < 1.0, "taken when it was heard, not after the hold"
        assert user.metadata["turn_end"] == "silence"
        await rig.say(SpeechStarted(), Transcript("Thanks."))
        await asyncio.sleep(.15)
        assert rig.session.stored_count == 2, "the short second turn is held: its speech is timed from its own start"
        await rig.finish()


async def test_min_speech_does_not_count_speech_that_ended_with_no_words(memory):
    async with Rig(memory, turn_strategy=MinSpeech(.3), turn_hold=2.0) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.4)
        await rig.say(Transcript(""))  # a cough, and nothing is held
        await rig.say(SpeechStarted(), Transcript("Yes."))
        await asyncio.sleep(.15)
        assert rig.session.stored_count == 0, "the cough is not part of the turn's speech"
        await rig.finish()


async def test_min_speech_keeps_timing_a_held_turn_through_a_cough(memory):
    async with Rig(memory, turn_strategy=MinSpeech(.3), turn_hold=1.0) as rig:
        await rig.say(SpeechStarted(), Transcript("Yes."))
        await asyncio.sleep(.35)
        await rig.say(SpeechStarted(), Transcript(""))  # a cough while the short turn is held
        await rig.say(SpeechStarted(), Transcript("the second one."))
        [user] = await rig.users(1)
        receipt = rig.session.last_turn_receipt
        assert (user.content, receipt.reason, receipt.fragments) == ("Yes. the second one.", "silence", 2)
        await rig.finish()


async def test_min_speech_with_no_sign_of_speech_takes_the_turn_and_says_it_could_not_measure(memory):
    async with Rig(memory, turn_strategy=MinSpeech(.5), turn_hold=2.0) as rig:
        await rig.say(Transcript("Yes."))
        [user] = await rig.users(1)
        assert user.metadata["turn_end"] == "silence"
        assert rig.session.last_turn_receipt.cue == "semantic turn detection off; min_speech: speech duration unknown"
        await rig.finish()


async def test_keypad_submit_needs_a_keypad(memory):
    with pytest.raises(ValueError, match="keypad"):
        Rig(memory, turn_strategy=KeypadSubmit())


async def test_keypad_submit_waits_for_the_key_whatever_was_said_and_keyed_before_it(memory):
    async with Rig(memory, keypad=KeypadPolicy("append"), turn_strategy=KeypadSubmit(), turn_max_duration=5) as rig:
        await rig.say(SpeechStarted(), Transcript("My account number is"))
        await rig.say(SpeechStarted(), Transcript("four one one."))
        await asyncio.sleep(.1)
        await keys(rig, "5")
        await asyncio.sleep(.2)
        assert rig.session.stored_count == 0 and rig.model.contexts == [], "a finished sentence and a key do not submit"
        await keys(rig, "#")
        [user] = await rig.users(1)
        assert user.content == "My account number is four one one. [keypad] 5 [keypad] #"
        assert (user.metadata["turn_end"], user.metadata["keypad_keys"]) == ("keypad", "#")
        assert rig.session.last_turn_receipt.fragments == 4
        assert len(rig.model.contexts) == 1
        await rig.finish()


async def test_keypad_submit_holds_collected_keys_that_timed_out_and_takes_the_terminated_entry(memory):
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=.1), turn_strategy=KeypadSubmit(),
                   turn_max_duration=5) as rig:
        await keys(rig, "1", "2")
        await asyncio.sleep(.3)
        assert rig.session.stored_count == 0, "an entry the timeout ended has no submit key"
        await keys(rig, "3", "#")
        [user] = await rig.users(1)
        assert user.content == "[keypad] 12 [keypad] 3#"
        assert user.metadata["keypad_ended"] == "terminator"
        await rig.finish()


async def test_keypad_submit_is_bounded_by_the_turn_s_own_bound_and_a_cough_does_not_cut_it_short(memory):
    async with Rig(memory, keypad=KeypadPolicy("append"), turn_strategy=KeypadSubmit(), turn_hold=.05,
                   turn_max_duration=.5) as rig:
        began = time.perf_counter()
        await rig.say(SpeechStarted(), Transcript("Hello."))
        await asyncio.sleep(.1)
        await rig.say(SpeechStarted(), Transcript(""))  # speech with no words
        [user] = await rig.users(1)
        assert time.perf_counter() - began >= .5
        assert (user.content, user.metadata["turn_end"]) == ("Hello.", "max_duration")
        await rig.finish()


async def test_keys_held_for_a_submit_that_would_outgrow_the_turn_release_it_first(memory):
    async with Rig(memory, keypad=KeypadPolicy("append"), turn_strategy=KeypadSubmit(), turn_max_duration=5) as rig:
        await rig.say(Transcript("a" * 31995))
        await asyncio.sleep(.05)
        await keys(rig, "5")
        [user] = await rig.users(1)
        assert (len(user.content), user.metadata["turn_end"]) == (31995, "max_bytes")
        await rig.finish()


async def test_a_strategy_that_does_not_judge_is_refused(memory):
    class Broken:
        name, submit = "broken", None

        def decide(self, judgement, speech_ms):
            return "complete"

    async with Rig(memory, turn_strategy=Broken()) as rig:
        await rig.say(Transcript("Hello."))
        with pytest.raises(ValueError, match="Judgement"):
            await asyncio.wait_for(rig.running, 3)


@pytest.mark.parametrize("strategy", [object(), type("Nameless", (), {"decide": lambda *a: None})()])
async def test_something_that_is_not_a_strategy_is_refused_before_resources(memory, strategy):
    with pytest.raises(ValueError, match="turn_strategy"):
        VoiceSession(memory, SPACE, SID, transport_factory=object, stt_factory=object, model_factory=object,
                     tts_factory=object, capture=True, turn_strategy=strategy)
