"""A voice session that waits for a speaker who paused mid-clause.

The recognizer here is a script the test feeds one event at a time, so a
pause is whatever the test leaves between two transcripts. Holds are short
so the waits the session makes are real but brief."""

from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.audio import AudioChunk, ReplyCompleted, SpeechStarted, TextDelta, Transcript
from scone_memory.realtime.turn_end import COMPLETE, INCOMPLETE, UNSURE, Judgement, LexicalEndOfTurn
from scone_memory.realtime.voice import VoiceSession

SPACE, SID = "voice-turns", "turn-session"
PCM = AudioChunk(b"\x02\x00" * 320, 16000)


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


class Resource:
    def __init__(self):
        self.closes = 0

    async def aclose(self):
        self.closes += 1


class Transport(Resource):
    def __init__(self):
        super().__init__()
        self.input: asyncio.Queue = asyncio.Queue()
        self.sent, self.cleared = [], []
        self.delivered = asyncio.Event()

    async def receive(self):
        while (packet := await self.input.get()) is not None:
            yield packet

    async def send(self, audio, turn_id):
        self.sent.append((audio, turn_id))
        self.delivered.set()

    async def clear(self, turn_id):
        self.cleared.append(turn_id)


class Script(Resource):
    """Per audio chunk, yields the events the test puts until a None."""

    def __init__(self):
        super().__init__()
        self.events: asyncio.Queue = asyncio.Queue()

    async def transcribe(self, audio):
        async for _ in audio:
            while (event := await self.events.get()) is not None:
                yield event


class Model(Resource):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def respond(self, messages):
        self.contexts.append(messages)
        yield TextDelta("Noted.")
        yield ReplyCompleted()


class Synthesizer(Resource):
    async def synthesize(self, text):
        yield AudioChunk(b"\x01\x00" * 160, 16000)


class Rig:
    def __init__(self, memory, **options):
        self.transport, self.stt, self.model, self.tts = Transport(), Script(), Model(), Synthesizer()
        self.resources = [self.transport, self.stt, self.model, self.tts]
        settings = dict(transport_factory=lambda: self.transport, stt_factory=lambda: self.stt,
                        model_factory=lambda: self.model, tts_factory=lambda: self.tts,
                        capture=True, session_timeout=10, turn_timeout=3)
        settings.update(options)
        self.session = VoiceSession(memory, SPACE, SID, **settings)
        self.memory = memory

    async def __aenter__(self):
        self.running = asyncio.create_task(self.session.run())
        await asyncio.wait_for(self.session.started.wait(), 2)
        await self.transport.input.put(PCM)
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        await asyncio.gather(self.running, return_exceptions=True)

    async def say(self, *events):
        for event in events:
            await self.stt.events.put(event)

    async def users(self, count):
        async with asyncio.timeout(3):
            while True:
                found = [e for e in await self.memory.episodes(SPACE, {"session_id": SID}) if e.metadata["role"] == "user"]
                if len(found) >= count:
                    return found
                await asyncio.sleep(.005)

    async def finish(self):
        await self.stt.events.put(None)
        await self.transport.input.put(None)
        await asyncio.wait_for(self.running, 3)


async def test_off_by_default_every_final_transcript_is_a_turn_ended_by_silence(memory):
    async with Rig(memory) as rig:
        await rig.say(Transcript("I want to book a table for"))
        [user] = await rig.users(1)
        assert user.content == "I want to book a table for" and user.metadata["turn_end"] == "silence"
        receipt = rig.session.last_turn_receipt
        assert (receipt.reason, receipt.verdict, receipt.cue) == ("silence", UNSURE, "semantic turn detection off")
        await rig.finish()
        assert {task for task in asyncio.all_tasks() if not task.done()} == {asyncio.current_task()}, \
            "nothing the session started outlives it"
    [assistant] = [e for e in await memory.episodes(SPACE, {"session_id": SID}) if e.metadata["role"] == "assistant"]
    assert "turn_end" not in assistant.metadata


async def test_a_pause_mid_clause_is_waited_out_and_the_turn_is_one_question(memory):
    async with Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=5) as rig:
        await rig.say(Transcript("I want to book a table for"))
        await asyncio.sleep(.1)
        assert rig.session.stored_count == 0 and rig.model.contexts == [], "the clause is still open"
        await rig.say(SpeechStarted(), Transcript("four people."))
        [user] = await rig.users(1)
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        assert user.content == "I want to book a table for four people."
        assert user.metadata["turn_end"] == "semantic_complete"
        assert rig.model.contexts[0][-1] == {"role": "user", "content": "I want to book a table for four people."}
        receipt = rig.session.last_turn_receipt
        assert (receipt.reason, receipt.fragments, receipt.cue) == ("semantic_complete", 2, "terminal punctuation")
        await rig.finish()
    assert rig.session.state == "ended"


async def test_a_hold_that_runs_out_answers_what_was_said(memory):
    async with Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=.05) as rig:
        await rig.say(Transcript("Send it to"))
        [user] = await rig.users(1)
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        assert user.content == "Send it to" and user.metadata["turn_end"] == "semantic_incomplete_timeout"
        assert rig.session.last_turn_receipt.held_ms >= 50
        await rig.say(Transcript("And copy in the"))
        [later] = [e for e in await rig.users(2) if e.content == "And copy in the"]
        assert later.metadata["turn_end"] == "semantic_incomplete_timeout", "a second hold is timed too"
        await rig.finish()


async def test_speech_starting_again_keeps_the_turn_open_past_the_hold(memory):
    async with Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=.5, turn_max_duration=5) as rig:
        await rig.say(Transcript("I need a flight to"), SpeechStarted())
        await asyncio.sleep(1.0)
        assert rig.session.stored_count == 0, "the speaker is talking; the hold does not cut them off"
        await rig.say(Transcript("Boston."))
        [user] = await rig.users(1)
        assert user.content == "I need a flight to Boston."
        await rig.finish()


async def test_local_activity_keeps_the_turn_open_too(memory):
    class Detector(Resource):
        def __init__(self):
            super().__init__()
            self.frames = 0

        async def detect(self, chunk):
            self.frames += 1
            return self.frames > 1

    detector = Detector()
    async with Rig(memory, activity_factory=lambda: detector, turn_detector_factory=LexicalEndOfTurn,
                   turn_hold=.5, turn_max_duration=5) as rig:
        await rig.say(Transcript("I need a flight to"), None)
        await rig.transport.input.put(PCM)
        async with asyncio.timeout(2):
            while detector.frames < 2:
                await asyncio.sleep(.005)
        await asyncio.sleep(1.0)
        assert rig.session.stored_count == 0
        await rig.say(Transcript("Boston."))
        [user] = await rig.users(1)
        assert user.content == "I need a flight to Boston."
        await rig.finish()
    assert detector.closes == 1


async def test_speech_that_never_becomes_words_is_released_at_the_turn_bound(memory):
    async with Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=.05, turn_max_duration=.3) as rig:
        await rig.say(Transcript("I was going to"), SpeechStarted())
        [user] = await rig.users(1)
        assert user.metadata["turn_end"] == "max_duration" and rig.session.last_turn_receipt.turn_ms >= 300
        await rig.finish()


async def test_input_ending_answers_the_held_turn(memory):
    rig = Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=5)
    async with rig:
        await rig.say(Transcript("I was thinking about"))
        await asyncio.sleep(.05)
        assert rig.session.stored_count == 0
        await rig.finish()
    [user] = await rig.users(1)
    assert user.metadata["turn_end"] == "input_ended" and rig.model.contexts[0][-1]["content"] == "I was thinking about"
    assert rig.transport.sent, "the drained turn was answered before the session ended"


class Slow(Resource):
    async def judge(self, text):
        await asyncio.sleep(5)
        return Judgement(INCOMPLETE, "never")


class Broken(Resource):
    async def judge(self, text):
        raise RuntimeError("classifier unreachable")


@pytest.mark.parametrize("detector, cue", [(Slow, "detector timed out"), (Broken, "detector failed: RuntimeError")])
async def test_a_detector_that_cannot_answer_leaves_the_turn_to_silence(memory, detector, cue):
    made = detector()
    async with Rig(memory, turn_detector_factory=lambda: made, turn_judge_timeout=.05) as rig:
        await rig.say(Transcript("I want to"))
        [user] = await rig.users(1)
        assert user.metadata["turn_end"] == "silence"
        assert (rig.session.last_turn_receipt.cue, rig.session.last_turn_receipt.verdict) == (cue, UNSURE)
        await rig.finish()
    assert made.closes == 1


async def test_a_model_detector_sees_the_whole_held_turn(memory):
    class Seen(Resource):
        def __init__(self):
            super().__init__()
            self.texts = []

        async def judge(self, text):
            self.texts.append(text)
            return Judgement(COMPLETE if text.endswith("please") else INCOMPLETE, "model")

    made = Seen()
    async with Rig(memory, turn_detector_factory=lambda: made, turn_hold=5) as rig:
        await rig.say(Transcript("a coffee"), Transcript("please"))
        [user] = await rig.users(1)
        assert user.content == "a coffee please" and made.texts == ["a coffee", "a coffee please"]
        await rig.finish()


async def test_a_detector_that_answers_nonsense_fails_the_session(memory):
    class Nonsense(Resource):
        async def judge(self, text):
            return "complete"

    made = Nonsense()
    rig = Rig(memory, turn_detector_factory=lambda: made)
    await rig.transport.input.put(PCM)
    await rig.stt.events.put(Transcript("Hello."))
    with pytest.raises(ValueError, match="Judgement"):
        await rig.session.run()
    assert made.closes == 1 and all(resource.closes == 1 for resource in rig.resources)
    assert await memory.episodes(SPACE, {"session_id": SID}) == []


async def test_a_detector_without_judge_is_refused(memory):
    rig = Rig(memory, turn_detector_factory=Resource)
    with pytest.raises(TypeError, match="protocol"):
        await rig.session.run()
    assert all(resource.closes == 1 for resource in rig.resources)


@pytest.mark.parametrize("options", [{"turn_detector_factory": "lexical"}, {"turn_hold": 0},
                                     {"turn_hold": 20, "turn_max_duration": 10}, {"turn_judge_timeout": float("nan")}])
async def test_turn_options_are_checked_when_the_session_is_made(memory, options):
    with pytest.raises(ValueError):
        Rig(memory, **options)


async def test_new_words_stop_a_playing_reply_while_their_turn_is_still_held(memory):
    class Endless(Model):
        async def respond(self, messages):
            self.contexts.append(messages)
            yield TextDelta("Let me tell you all about it.")
            await asyncio.Future()

    model = Endless()
    async with Rig(memory, model_factory=lambda: model, turn_detector_factory=LexicalEndOfTurn, turn_hold=5) as rig:
        await rig.say(Transcript("What is Juniper?"))
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        [first] = await rig.users(1)
        await rig.say(Transcript("And also the"))
        async with asyncio.timeout(2):
            while not rig.transport.cleared:
                await asyncio.sleep(.005)
        assert rig.transport.cleared == [first.metadata["turn_id"]] and rig.session.stored_count == 1
        await rig.say(Transcript("moon."))
        await rig.users(2)


async def test_another_speaker_releases_the_held_turn_and_only_the_last_is_answered(memory):
    async with Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=5) as rig:
        await rig.say(Transcript("I was going to", speaker="alice"), Transcript("Hello.", speaker="bob"))
        alice, bob = sorted(await rig.users(2), key=lambda e: e.metadata["speaker"])
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        await rig.finish()
    assert (alice.metadata["turn_end"], bob.metadata["turn_end"]) == ("speaker_changed", "semantic_complete")
    assert {turn for _, turn in rig.transport.sent} == {bob.metadata["turn_id"]}


async def test_a_released_turn_that_cannot_be_stored_fails_the_session(memory):
    async def refuse(*args, **kwargs):
        raise RuntimeError("store down")

    rig = Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=.05)
    memory.remember_many = refuse
    await rig.transport.input.put(PCM)
    await rig.stt.events.put(Transcript("Send it to"))
    running = asyncio.create_task(rig.session.run())
    done, _ = await asyncio.wait({running}, timeout=1.5)
    assert done, "the failure ends the session when it happens, not when the session is closed"
    with pytest.raises(RuntimeError, match="store down"):
        await running
    assert all(resource.closes == 1 for resource in rig.resources)


def spy_on_timing(memory):
    latencies = []
    record_turn = memory.record_turn

    async def spy(space, **options):
        latencies.append(options["latency_ms"])
        return await record_turn(space, **options)

    memory.record_turn = spy
    return latencies


async def first_timing(latencies):
    async with asyncio.timeout(3):
        while not latencies:
            await asyncio.sleep(.005)
    return latencies[0]


async def test_a_held_turn_s_recorded_latency_counts_the_hold(memory):
    latencies = spy_on_timing(memory)
    async with Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=.3) as rig:
        await rig.say(Transcript("sure I'd love to"))
        timing = await first_timing(latencies)
        assert rig.session.last_turn_receipt.reason == "semantic_incomplete_timeout"
        assert timing["first_audio"] >= 300 and timing["total"] >= 300, \
            "a person waited through the hold; the turn's timing starts when their words were heard"
        await rig.finish()


async def test_a_detector_s_own_time_counts_in_the_turn_s_latency(memory):
    class Deliberate(Resource):
        async def judge(self, text):
            await asyncio.sleep(.2)
            return Judgement(COMPLETE, "model")

    made = Deliberate()
    latencies = spy_on_timing(memory)
    async with Rig(memory, turn_detector_factory=lambda: made, turn_judge_timeout=1) as rig:
        await rig.say(Transcript("Book it."))
        timing = await first_timing(latencies)
        assert rig.session.last_turn_receipt.reason == "semantic_complete"
        assert timing["first_audio"] >= 200, "the detector judged after the words were heard"
        await rig.finish()


async def held(rig, text):
    await rig.say(Transcript(text))
    async with asyncio.timeout(3):
        while not rig.session._turns.pending:  # the transcript has reached the hold
            await asyncio.sleep(.005)
    assert rig.session.stored_count == 0


async def _hold_then_stop(rig, how):
    await held(rig, "Please cancel my card ending in four two and")
    if how == "close":
        await rig.session.close()
    await asyncio.gather(rig.running, return_exceptions=True)


@pytest.mark.parametrize("how, options", [("close", {}), ("session_timeout", {"session_timeout": 1})])
async def test_a_session_that_stops_keeps_the_transcript_it_was_holding(memory, how, options):
    rig = Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=5, **options)
    async with rig:
        await _hold_then_stop(rig, how)
    [user] = await rig.users(1)
    assert user.content == "Please cancel my card ending in four two and"
    assert user.metadata["turn_end"] == "session_ended" and rig.session.last_turn_receipt.reason == "session_ended"
    assert rig.model.contexts == [] and rig.transport.sent == [], "a stopping session records the words, answers nothing"


async def test_a_held_transcript_that_cannot_be_kept_at_close_is_not_lost_quietly(memory):
    async def refuse(*args, **kwargs):
        raise RuntimeError("store down")

    rig = Rig(memory, turn_detector_factory=LexicalEndOfTurn, turn_hold=5)
    async with rig:
        await held(rig, "Please cancel my card and")
        memory.remember_many = refuse
        with pytest.raises(RuntimeError, match="store down"):
            await rig.session.close()


async def test_a_failing_provider_keeps_its_own_error_and_names_a_held_transcript_it_could_not_keep(memory):
    async def refuse(*args, **kwargs):
        raise RuntimeError("store down")

    class Dies(Script):
        async def transcribe(self, audio):
            async for _ in audio:
                while (event := await self.events.get()) is not None:
                    if isinstance(event, Exception):
                        raise event
                    yield event

    stt = Dies()
    rig = Rig(memory, stt_factory=lambda: stt, turn_detector_factory=LexicalEndOfTurn, turn_hold=5)
    rig.stt = stt
    async with rig:
        await held(rig, "Please cancel my card and")
        memory.remember_many = refuse
        await rig.say(ConnectionError("recognizer lost"))
        with pytest.raises(ConnectionError, match="recognizer lost") as failed:
            await asyncio.wait_for(rig.running, 3)
    assert any("held transcript was not stored" in note for note in getattr(failed.value, "__notes__", []))
