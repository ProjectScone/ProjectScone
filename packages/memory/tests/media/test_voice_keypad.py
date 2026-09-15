"""A voice session that takes keys from the phone as well as speech.

The transport here yields ``Keypress`` events among its audio, as a
carrier transport with its keypad on does. Timeouts are short so the
waits the session makes are real but brief."""

from __future__ import annotations

import asyncio

import pytest

from scone_memory.realtime.audio import SpeechStarted, TextDelta, Transcript
from scone_memory.realtime.keypad import KeypadPolicy, Keypress
from scone_memory.realtime.turn_end import LexicalEndOfTurn

from .test_voice_turn_end import PCM, Model, Resource, Rig, memory  # noqa: F401 - memory is a fixture


async def keys(rig, *presses):
    for press in presses:
        await rig.transport.input.put(press if isinstance(press, Keypress) else Keypress(press))


async def test_off_by_default_keys_are_counted_and_do_nothing(memory):
    async with Rig(memory) as rig:
        await keys(rig, "5", "#")
        await rig.transport.input.put(PCM)
        await rig.say(None)  # the recognizer takes the next chunk: the keys did not stop the audio
        async with asyncio.timeout(2):
            while rig.session.keypad_ignored < 2:
                await asyncio.sleep(.005)
        await rig.finish()
    assert rig.session.stored_count == 0 and rig.model.contexts == []
    assert rig.session.last_keypad_receipt is None


async def test_append_makes_a_key_the_callers_turn(memory):
    rig = Rig(memory, keypad=KeypadPolicy("append"))
    await asyncio.sleep(.3)  # made well before it runs: keys are timed from the run
    async with rig:
        await asyncio.sleep(.1)
        await keys(rig, Keypress("5", "inband", offset_ms=1200.0, tone_ms=44.8))
        [user] = await rig.users(1)
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        assert user.content == "[keypad] 5"
        assert {k: user.metadata[k] for k in ("turn_end", "keypad_keys", "keypad_sources", "keypad_ended")} == \
            {"turn_end": "keypad", "keypad_keys": "5", "keypad_sources": "i", "keypad_ended": "key"}
        assert 90 <= int(user.metadata["keypad_started_ms"]) < 300, "from the session's start, not its making"
        assert user.metadata["keypad_at_ms"] == "0"
        assert rig.model.contexts[0][-1] == {"role": "user", "content": "[keypad] 5"}
        receipt = rig.session.last_keypad_receipt
        assert receipt.presses[0].tone_ms == 44.8, "the session keeps the whole receipt, audio offsets included"
        await rig.finish()


async def test_a_key_finishes_what_the_caller_was_saying(memory):
    async with Rig(memory, keypad=KeypadPolicy("append"), turn_detector_factory=LexicalEndOfTurn, turn_hold=5) as rig:
        await rig.say(Transcript("the option I want is, um"))
        await asyncio.sleep(.05)
        assert rig.session.stored_count == 0, "held: the clause is open"
        await keys(rig, "3")
        [user] = await rig.users(1)
        assert (user.content, user.metadata["turn_end"]) == ("the option I want is, um [keypad] 3", "keypad")
        assert rig.session.last_turn_receipt.fragments == 2
        await rig.finish()


async def test_collect_gives_the_turn_one_entry_ended_by_the_terminator(memory):
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=5)) as rig:
        await keys(rig, "1", Keypress("2", "inband", offset_ms=500.0, tone_ms=50.0))
        await asyncio.sleep(.05)
        assert rig.session.stored_count == 0 and rig.model.contexts == [], "still collecting"
        await keys(rig, "#")
        [user] = await rig.users(1)
        assert user.content == "[keypad] 12#"
        assert (user.metadata["keypad_ended"], user.metadata["keypad_sources"]) == ("terminator", "eie")
        at = [int(ms) for ms in user.metadata["keypad_at_ms"].split(",")]
        assert at[0] == 0 and at == sorted(at) and len(at) == 3, "each key's time from the first"
        assert len(rig.model.contexts) == 1
        await rig.finish()


async def test_collect_gives_the_turn_what_it_has_when_the_keys_stop(memory):
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=.1)) as rig:
        await keys(rig, "4", "2")
        [user] = await rig.users(1)
        assert (user.content, user.metadata["keypad_ended"], user.metadata["turn_end"]) == ("[keypad] 42", "timeout", "keypad")
        await rig.finish()


async def test_collecting_keys_holds_the_callers_open_clause_for_them(memory):
    policy = KeypadPolicy("collect", timeout=.4)
    async with Rig(memory, keypad=policy, turn_detector_factory=LexicalEndOfTurn, turn_hold=.15, turn_max_duration=5) as rig:
        await rig.say(Transcript("my card number is um"))
        await asyncio.sleep(.05)  # heard and held before the keys come
        assert rig.session.stored_count == 0
        await keys(rig, "4", "1")
        await asyncio.sleep(.25)  # past the clause's own hold
        assert rig.session.stored_count == 0, "the caller is keying the rest of the sentence"
        [user] = await rig.users(1)
        assert user.content == "my card number is um [keypad] 41" and user.metadata["keypad_ended"] == "timeout"
        await rig.finish()


async def test_the_first_key_stops_a_reply_that_is_playing(memory):
    class Endless(Model):
        async def respond(self, messages):
            self.contexts.append(messages)
            yield TextDelta("Let me read you the whole menu.")
            await asyncio.Future()

    model = Endless()
    async with Rig(memory, model_factory=lambda: model, keypad=KeypadPolicy("collect", timeout=5)) as rig:
        await rig.say(Transcript("What are my options?"))
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        [question] = await rig.users(1)
        await keys(rig, "9")
        async with asyncio.timeout(2):
            while not rig.transport.cleared:
                await asyncio.sleep(.005)
        assert rig.transport.cleared == [question.metadata["turn_id"]]
        assert rig.session.stored_count == 1, "the key interrupted; its entry is not finished"


async def test_speech_while_keys_are_held_gives_the_keys_to_the_turn_first(memory):
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=5)) as rig:
        await keys(rig, "7", "7")
        await asyncio.sleep(.05)
        await rig.say(SpeechStarted(), Transcript("Actually, can I talk to someone?"))
        users = await rig.users(2)
        keyed, spoken = sorted(users, key=lambda e: e.episode_id)
        assert (keyed.content, keyed.metadata["keypad_ended"]) == ("[keypad] 77", "speech")
        assert (spoken.content, spoken.metadata["turn_end"]) == ("Actually, can I talk to someone?", "silence")
        assert "keypad_keys" not in spoken.metadata
        await rig.finish()


def by_time(episodes):
    return sorted(episodes, key=lambda e: e.episode_id)


async def test_keys_pressed_after_the_caller_spoke_are_given_after_their_words(memory):
    """The recognizer gives the words after the caller has stopped and it has
    waited; the keys come at once. Speech had begun before the keys, so the
    words are the caller's turn first and the keys come after them."""
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=.3)) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "4", "1", "1", "1")
        await asyncio.sleep(.05)
        await rig.say(Transcript("my card number is um"))
        spoken, keyed = by_time(await rig.users(2))
        assert (spoken.content, spoken.metadata["turn_end"]) == ("my card number is um", "silence")
        assert (keyed.content, keyed.metadata["keypad_ended"]) == ("[keypad] 4111", "timeout")
        assert "keypad_waited" not in keyed.metadata, "still being collected when the words came"
        await rig.finish()


async def test_keys_that_finish_a_clause_still_on_its_way_join_it(memory):
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=5), turn_detector_factory=LexicalEndOfTurn,
                   turn_hold=.2) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "4", "1", "1", "1", "#")
        await asyncio.sleep(.1)
        assert rig.session.stored_count == 0, "the keys wait for the words the caller said first"
        await rig.say(Transcript("my card number is um"))
        [user] = await rig.users(1)
        assert (user.content, user.metadata["turn_end"]) == ("my card number is um [keypad] 4111#", "keypad")
        assert (user.metadata["keypad_ended"], user.metadata["keypad_waited"]) == ("terminator", "words")
        await asyncio.sleep(.3)  # past the clause's hold: nothing is left to time out
        assert len(await rig.users(1)) == 1, "one turn, not the keys and then the clause alone"
        await rig.finish()


async def test_keys_still_being_collected_when_the_words_come_hold_the_clause(memory):
    policy = KeypadPolicy("collect", timeout=.4)
    async with Rig(memory, keypad=policy, turn_detector_factory=LexicalEndOfTurn, turn_hold=.15, turn_max_duration=5) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "4", "1")
        await asyncio.sleep(.05)
        await rig.say(Transcript("my card number is um"))
        await asyncio.sleep(.25)  # past the clause's own hold
        assert rig.session.stored_count == 0, "the caller is keying the rest of the sentence"
        [user] = await rig.users(1)
        assert user.content == "my card number is um [keypad] 41" and user.metadata["keypad_ended"] == "timeout"
        await rig.finish()


async def test_a_partial_transcript_or_the_sessions_own_detector_also_says_speech_began(memory):
    class Detector(Resource):
        def __init__(self):
            super().__init__()
            self.frames = 0

        async def detect(self, chunk):
            self.frames += 1
            return self.frames > 1

    detector = Detector()
    async with Rig(memory, keypad=KeypadPolicy("append"), activity_factory=lambda: detector) as rig:
        await rig.say(None)  # nothing in the first chunk
        await rig.transport.input.put(PCM)  # the detector hears speech in the second
        async with asyncio.timeout(2):
            while detector.frames < 2:
                await asyncio.sleep(.005)
        await asyncio.sleep(.05)
        await keys(rig, "5")
        await asyncio.sleep(.05)
        assert rig.session.stored_count == 0, "held for the words the detector heard begin"
        await rig.say(Transcript("my extension is"))
        await rig.say(Transcript("and my", final=False))
        await asyncio.sleep(.05)
        await keys(rig, "6")
        await asyncio.sleep(.05)
        await rig.say(Transcript("room is"))
        spoken, five, later, six = by_time(await rig.users(4))
        assert [e.content for e in (spoken, five, later, six)] == \
            ["my extension is", "[keypad] 5", "room is", "[keypad] 6"]
        assert five.metadata["keypad_waited"] == six.metadata["keypad_waited"] == "words"
        await rig.finish()


async def test_with_no_keys_being_collected_an_open_clause_waits_only_its_hold(memory):
    async with Rig(memory, keypad=KeypadPolicy("collect", timeout=5), turn_detector_factory=LexicalEndOfTurn,
                   turn_hold=.05, turn_max_duration=5) as rig:
        await rig.say(Transcript("Send it to"))
        [user] = await rig.users(1)
        assert user.metadata["turn_end"] == "semantic_incomplete_timeout", "not held to the turn's bound for keys"
        await rig.finish()


async def test_keys_wait_for_the_words_no_longer_than_speech_wait_and_say_so(memory):
    async with Rig(memory, keypad=KeypadPolicy("append", speech_wait=.1)) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "5")
        [user] = await rig.users(1)
        assert (user.content, user.metadata["keypad_ended"], user.metadata["keypad_waited"]) == \
            ("[keypad] 5", "key", "timeout")
        await rig.finish()


async def test_speech_that_ends_without_words_lets_the_waiting_keys_go(memory):
    async with Rig(memory, keypad=KeypadPolicy("append")) as rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "5")
        await asyncio.sleep(.05)
        assert rig.session.stored_count == 0
        await rig.say(Transcript("", final=True))
        [user] = await rig.users(1)
        assert (user.content, user.metadata["keypad_waited"]) == ("[keypad] 5", "no_words")
        await rig.finish()


async def test_keys_held_when_input_ends_are_answered(memory):
    rig = Rig(memory, keypad=KeypadPolicy("collect", timeout=5))
    async with rig:
        await keys(rig, "3", "3")
        await asyncio.sleep(.05)
        await rig.finish()
    [user] = await rig.users(1)
    assert (user.content, user.metadata["keypad_ended"]) == ("[keypad] 33", "input_ended")
    assert rig.model.contexts[0][-1]["content"] == "[keypad] 33" and rig.transport.sent


async def test_keys_held_when_the_session_stops_are_stored_and_not_answered(memory):
    rig = Rig(memory, keypad=KeypadPolicy("collect", timeout=5))
    async with rig:
        await keys(rig, "8")
        await asyncio.sleep(.05)
    [user] = await rig.users(1)
    assert (user.content, user.metadata["keypad_ended"], user.metadata["turn_end"]) == \
        ("[keypad] 8", "session_ended", "session_ended")
    assert rig.model.contexts == []


async def test_keys_waiting_for_words_when_the_session_stops_are_stored_and_not_answered(memory):
    rig = Rig(memory, keypad=KeypadPolicy("collect", timeout=5))
    async with rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "8", "#")
        await asyncio.sleep(.05)
        assert rig.session.stored_count == 0
    [user] = await rig.users(1)
    assert (user.content, user.metadata["turn_end"]) == ("[keypad] 8#", "session_ended")
    assert (user.metadata["keypad_ended"], user.metadata["keypad_waited"]) == ("terminator", "no_words")
    assert rig.model.contexts == []


async def test_keys_waiting_for_words_when_input_ends_are_answered(memory):
    rig = Rig(memory, keypad=KeypadPolicy("append"))
    async with rig:
        await rig.say(SpeechStarted())
        await asyncio.sleep(.05)
        await keys(rig, "3")
        await asyncio.sleep(.05)
        await rig.finish()
    [user] = await rig.users(1)
    assert (user.content, user.metadata["keypad_waited"]) == ("[keypad] 3", "no_words")
    assert rig.model.contexts[0][-1]["content"] == "[keypad] 3"


async def test_a_keypad_policy_is_checked_when_the_session_is_made(memory):
    with pytest.raises(ValueError, match="keypad"):
        Rig(memory, keypad="collect")


async def test_keys_that_would_outgrow_a_held_turn_release_it_first_without_their_receipt(memory):
    long_clause = "word " * 6398 + "and"  # just under the 32 000 bytes a held turn may reach
    async with Rig(memory, keypad=KeypadPolicy("append"), turn_detector_factory=LexicalEndOfTurn, turn_hold=5) as rig:
        await rig.say(Transcript(long_clause))
        await asyncio.sleep(.05)
        await keys(rig, "1")
        spoken, keyed = sorted(await rig.users(2), key=lambda e: e.episode_id)
        assert (spoken.metadata["turn_end"], "keypad_keys" in spoken.metadata) == ("max_bytes", False)
        assert (keyed.content, keyed.metadata["keypad_keys"]) == ("[keypad] 1", "1")
        await rig.finish()
