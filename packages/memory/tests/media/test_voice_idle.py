"""A voice session whose user has gone quiet.

The rules are driven with plain numbers in test_idle.py. Here the session
runs them against a scripted recognizer, so the waits are real and short,
and each test says what the wait is measured against."""

from __future__ import annotations

import asyncio
import time

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.audio import AudioChunk, SpeechStarted, Transcript
from scone_memory.realtime.idle import IdlePolicy
from scone_memory.realtime.keypad import KeypadPolicy, Keypress
from scone_memory.realtime.turn_end import LexicalEndOfTurn
from scone_memory.realtime.voice import VoiceSession

from .test_voice_turn_end import PCM, SID, SPACE, Model, Resource, Rig, Script, Synthesizer, Transport


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog()).open()
    yield engine
    await engine.close()


class Timed(Transport):
    """Keeps when each piece of audio went out, and for which turn."""

    def __init__(self):
        super().__init__()
        self.at: list[tuple[float, str]] = []

    async def send(self, audio, turn_id):
        self.at.append((time.perf_counter(), turn_id))
        await super().send(audio, turn_id)


class Slow(Synthesizer):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.started = asyncio.Event()

    async def synthesize(self, text):
        self.started.set()
        await asyncio.sleep(self.seconds)
        yield AudioChunk(b"\x01\x00" * 160, 16000)


class IdleRig(Rig):
    """The turn-end rig with a transport that keeps times, and a choice of model and synthesizer."""

    def __init__(self, memory, *, model=None, tts=None, **options):
        self.transport, self.stt = Timed(), Script()
        self.model, self.tts = model or Model(), tts or Synthesizer()
        self.resources = [self.transport, self.stt, self.model, self.tts]
        settings = dict(transport_factory=lambda: self.transport, stt_factory=lambda: self.stt,
                        model_factory=lambda: self.model, tts_factory=lambda: self.tts,
                        capture=True, session_timeout=10, turn_timeout=3)
        settings.update(options)
        self.session = VoiceSession(memory, SPACE, SID, **settings)
        self.memory = memory


def rig(memory, **options):
    return IdleRig(memory, **options)


def slow_rig(memory, seconds, **options):
    return IdleRig(memory, tts=Slow(seconds), **options)


async def until(condition, seconds=3.0):
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(.005)


async def episodes(memory, role):
    return [e for e in await memory.episodes(SPACE, {"session_id": SID}) if e.metadata["role"] == role]


async def idle_events(memory):
    return sorted(await memory.events.query(SPACE, kind="conversation_idle", limit=50), key=lambda e: e.event_id)


async def test_off_by_default_a_quiet_user_is_only_waited_on(memory):
    async with rig(memory) as r:
        await asyncio.sleep(.3)
        assert r.transport.sent == [] and r.session.last_idle_receipt is None and r.session.idles == 0
        await r.finish()
    assert r.session.end_reason == "input_ended"
    assert await idle_events(memory) == []


async def test_a_quiet_user_is_asked_and_the_prompt_is_said_recorded_and_remembered(memory):
    began = time.perf_counter()
    async with rig(memory, idle=IdlePolicy(.2, prompt="Still there?", end_after=None)) as r:
        await asyncio.wait_for(r.transport.delivered.wait(), 3)
        assert r.transport.at[0][0] - began >= .2, "not before the timeout"
        await until(lambda: r.session.stored_count == 1)
        [said] = await episodes(memory, "assistant")
        assert said.content == "Still there?"
        assert {k: said.metadata[k] for k in ("idle_count", "idle_action", "turn_id")} == \
            {"idle_count": "1", "idle_action": "prompt", "turn_id": r.transport.at[0][1]}
        assert int(said.metadata["idle_silent_ms"]) >= 200
        receipt = r.session.last_idle_receipt
        assert (receipt.count, receipt.action) == (1, "prompt") and receipt.silent_ms >= 200
        await r.say(SpeechStarted(), Transcript("Yes, sorry."))
        await until(lambda: r.model.contexts)
        roles = [(m["role"], m["content"]) for m in r.model.contexts[0] if m["role"] != "system"]
        assert roles[-2:] == [("assistant", "Still there?"), ("user", "Yes, sorry.")], \
            "the model hears what the bot said while it waited"
        [event, *_] = await idle_events(memory)
        assert {k: event.payload[k] for k in ("session_id", "count", "action", "turn_id")} == \
            {"session_id": SID, "count": 1, "action": "prompt", "turn_id": said.metadata["turn_id"]}
        assert event.payload["silent_ms"] >= 200
        await r.finish()


async def test_the_wait_is_paused_while_the_bot_speaks_and_counted_from_when_it_stops(memory):
    async with slow_rig(memory, .5, idle=IdlePolicy(.2, end_after=None)) as r:
        await r.say(SpeechStarted(), Transcript("Hello."))
        await asyncio.wait_for(r.transport.delivered.wait(), 3)
        reply_done = r.transport.at[0][0]
        assert r.session.last_idle_receipt is None, "no idle while the reply was being spoken"
        await until(lambda: len(r.transport.at) == 2)
        prompt_at = r.transport.at[1][0]
        assert prompt_at - reply_done >= .2 + .5, "a timeout after the reply, then the prompt's own synthesis"
        assert r.session.last_idle_receipt.count == 1 and r.session.last_idle_receipt.silent_ms >= 200
        assert r.transport.cleared[-1] == r.transport.at[0][1], "the reply's queued audio is cleared for the prompt"
        await r.finish()


async def test_a_user_still_speaking_is_not_idle(memory):
    async with rig(memory, idle=IdlePolicy(.15, end_after=None)) as r:
        await r.say(SpeechStarted())
        await asyncio.sleep(.45)
        assert r.session.last_idle_receipt is None and r.transport.sent == []
        await r.say(Transcript(""))  # the speech ended with no words: the wait starts again
        await until(lambda: r.session.last_idle_receipt is not None)
        await r.finish()


async def test_the_activity_detector_hearing_speech_stop_starts_the_wait(memory):
    class Activity(Resource):
        def __init__(self):
            super().__init__()
            self.loud = True

        async def detect(self, chunk):
            return self.loud

    activity = Activity()
    async with rig(memory, idle=IdlePolicy(.15, end_after=None), activity_factory=lambda: activity) as r:
        await r.say(None)  # the first chunk was loud; the recognizer takes the next one
        await asyncio.sleep(.4)
        assert r.session.last_idle_receipt is None, "speech began and has not ended"
        activity.loud = False
        await r.transport.input.put(PCM)
        await until(lambda: r.session.last_idle_receipt is not None)
        await r.finish()


async def test_a_clause_held_open_is_not_idle(memory):
    async with rig(memory, idle=IdlePolicy(.15, end_after=None), turn_detector_factory=LexicalEndOfTurn,
                   turn_hold=.6) as r:
        await r.say(SpeechStarted(), Transcript("I want to book a table for"))
        await asyncio.sleep(.4)
        assert r.session.last_idle_receipt is None and r.session.stored_count == 0, "the user is mid-sentence"
        await until(lambda: r.session.stored_count == 2)  # released at the hold, and answered
        await until(lambda: r.session.last_idle_receipt is not None)
        await r.finish()


async def test_keys_being_collected_are_not_idle(memory):
    async with rig(memory, idle=IdlePolicy(.15, end_after=None), keypad=KeypadPolicy("collect", timeout=.6)) as r:
        await r.transport.input.put(Keypress("4"))
        await asyncio.sleep(.4)
        assert r.session.last_idle_receipt is None and r.session.stored_count == 0
        await until(lambda: r.session.last_idle_receipt is not None)
        await r.finish()


async def test_speech_between_idles_clears_the_count(memory):
    async with rig(memory, idle=IdlePolicy(.2, end_after=2)) as r:
        await until(lambda: r.session.idles == 1)
        await until(lambda: r.session.stored_count == 1)  # the prompt was said
        await r.say(SpeechStarted(), Transcript("Hold on."))
        await until(lambda: r.session.idles == 2)
        assert r.session.last_idle_receipt.count == 1 and r.session.last_idle_receipt.action == "prompt"
        assert not r.running.done(), "the user spoke between the two idles: they are not two in a row"
        await r.finish()


async def test_a_key_between_idles_clears_the_count(memory):
    async with rig(memory, idle=IdlePolicy(.2, end_after=2), keypad=KeypadPolicy("append")) as r:
        await until(lambda: r.session.stored_count == 1)  # the first idle's prompt was said
        await r.transport.input.put(Keypress("5"))
        await until(lambda: r.session.idles == 2)
        assert r.session.last_idle_receipt.count == 1 and not r.running.done()
        await r.finish()


async def test_a_clause_held_open_does_not_wake_the_session_over_and_over(memory):
    async with rig(memory, idle=IdlePolicy(.05, end_after=None), turn_detector_factory=LexicalEndOfTurn,
                   turn_hold=.5) as r:
        expire, wakes = r.session._turns.expire, []

        def counted(now):
            wakes.append(now)
            return expire(now)

        r.session._turns.expire = counted
        await r.say(SpeechStarted(), Transcript("I want to book a table for"))
        await until(lambda: r.session.stored_count == 2)
        assert len(wakes) < 40, f"the holder woke {len(wakes)} times while the user was mid-sentence"
        await r.finish()


async def test_the_last_idle_in_a_row_ends_the_session_and_says_why(memory):
    r = rig(memory, idle=IdlePolicy(.1, prompt="Are you there?", end_after=2))
    async with r:
        await asyncio.wait_for(r.running, 3)
    assert r.session.state == "ended" and r.session.end_reason == "idle"
    assert r.session.last_idle_receipt.count == 2 and r.session.last_idle_receipt.action == "end"
    assert [(e.payload["count"], e.payload["action"]) for e in await idle_events(memory)] == [(1, "prompt"), (2, "end")]
    assert [e.content for e in await episodes(memory, "assistant")] == ["Are you there?"]
    assert all(resource.closes == 1 for resource in r.resources)


async def test_an_idle_with_no_prompt_is_noted_and_nothing_is_said(memory):
    r = rig(memory, idle=IdlePolicy(.08, prompt=None, end_after=3))
    async with r:
        await asyncio.wait_for(r.running, 3)
    assert r.transport.sent == [] and r.session.stored_count == 0
    events = await idle_events(memory)
    assert [e.payload["action"] for e in events] == ["noted", "noted", "end"]
    assert all("turn_id" not in e.payload for e in events), "nothing was said, so no turn"
    assert r.session.end_reason == "idle" and r.session.idles == 3


async def test_a_prompt_the_user_talks_over_is_cleared_not_recorded_and_clears_the_count(memory):
    async with slow_rig(memory, .5, idle=IdlePolicy(.15, end_after=2)) as r:
        await asyncio.wait_for(r.tts.started.wait(), 3)
        await r.say(SpeechStarted())
        await until(lambda: r.transport.cleared)
        assert r.session.stored_count == 0 and r.transport.sent == []
        await r.say(Transcript(""))
        await until(lambda: r.session.idles == 2)
        assert r.session.last_idle_receipt.count == 1, "talking over the prompt was the user coming back"
        await r.finish()


async def test_a_prompt_that_would_outgrow_the_history_fails_the_session(memory):
    r = rig(memory, idle=IdlePolicy(.05, prompt="Hello? " * 80, end_after=None), max_history_bytes=512)
    async with r:
        with pytest.raises(RuntimeError, match="history byte limit"):
            await asyncio.wait_for(r.running, 3)
    assert r.session.stored_count == 0 and r.session.end_reason == "failed"


async def test_a_prompt_is_bounded_by_the_turn_deadline(memory):
    r = slow_rig(memory, 1.0, idle=IdlePolicy(.05, end_after=None), turn_timeout=.2)
    async with r:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(r.running, 3)
    assert r.session.end_reason == "failed"


async def test_an_event_log_that_stalls_delays_the_prompt_only_to_its_bound(memory, monkeypatch):
    from scone_memory.realtime import voice

    async def stall(*args, **options):
        await asyncio.sleep(30)

    monkeypatch.setattr(memory, "record_idle", stall)
    monkeypatch.setattr(voice, "NOTE_TIMEOUT", .1)
    async with rig(memory, idle=IdlePolicy(.05, end_after=None)) as r:
        await until(lambda: r.session.stored_count == 1, 2)
        await r.finish()


async def test_an_event_log_that_fails_does_not_stop_the_prompt(memory, monkeypatch):
    async def refuse(*args, **options):
        raise RuntimeError("log down")

    monkeypatch.setattr(memory, "record_idle", refuse)
    async with rig(memory, idle=IdlePolicy(.1, end_after=None)) as r:
        await until(lambda: r.session.stored_count == 1)
        assert r.session.last_idle_receipt.count == 1
        await r.finish()


UNDECIDED = type("Undecided", (), {"name": "undecided", "submit": None})()
ODD_KEY = type("OddKey", (), {"name": "odd", "submit": "x", "decide": lambda self, judgement, speech_ms: judgement})()


@pytest.mark.parametrize("options", [{"idle": 5}, {"idle": "0.2"}, {"turn_strategy": "min_speech"},
                                     {"turn_strategy": UNDECIDED},
                                     {"turn_strategy": ODD_KEY, "keypad": KeypadPolicy("append")}])
async def test_an_idle_policy_or_strategy_of_the_wrong_kind_is_refused(memory, options):
    made = []
    with pytest.raises(ValueError):
        VoiceSession(memory, SPACE, SID, transport_factory=lambda: made.append(1), stt_factory=lambda: made.append(1),
                     model_factory=lambda: made.append(1), tts_factory=lambda: made.append(1), capture=True, **options)
    assert made == []


async def test_why_a_session_ended_when_it_was_closed_before_it_ran(memory):
    session = rig(memory).session
    await session.close()
    assert session.state == "interrupted" and session.end_reason == "closed"


async def test_why_a_session_ended_when_a_resource_could_not_be_made(memory):
    def refuse():
        raise RuntimeError("no transport")

    session = VoiceSession(memory, SPACE, SID, transport_factory=refuse, stt_factory=Script, model_factory=Model,
                           tts_factory=Synthesizer, capture=True)
    with pytest.raises(RuntimeError, match="no transport"):
        await session.run()
    assert session.end_reason == "failed"


async def test_why_a_session_ended_when_it_was_closed(memory):
    async with rig(memory, idle=IdlePolicy(5)) as r:
        pass
    assert r.session.state == "interrupted" and r.session.end_reason == "closed"


async def test_why_a_session_ended_when_its_deadline_came(memory):
    r = rig(memory, session_timeout=.2)
    async with r:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(r.running, 3)
    assert r.session.state == "failed" and r.session.end_reason == "session_timeout"


async def test_why_a_session_ended_when_it_failed(memory):
    class Broken(Model):
        async def respond(self, messages):
            raise RuntimeError("model down")
            yield

    r = rig(memory, model=Broken())
    async with r:
        await r.say(Transcript("Hello."))
        with pytest.raises(RuntimeError, match="model down"):
            await asyncio.wait_for(r.running, 3)
    assert r.session.state == "failed" and r.session.end_reason == "failed"


async def test_an_idle_event_is_validated_like_a_turn_s_timing(memory):
    from scone_memory.core.errors import InvalidInput

    event = await memory.record_idle(SPACE, session_id=SID, count=2, action="prompt", silent_ms=8000, turn_id="t1")
    assert (event.kind, event.payload) == ("conversation_idle", {"session_id": SID, "count": 2, "action": "prompt",
                                                                 "silent_ms": 8000.0, "turn_id": "t1"})
    assert type(event.payload["silent_ms"]) is float
    ended = await memory.record_idle(SPACE, session_id=SID, count=3, action="end", silent_ms=0)
    assert "turn_id" not in ended.payload
    for bad in ({"session_id": ""}, {"session_id": None}, {"turn_id": ""}, {"turn_id": "x" * 129}, {"count": 0},
                {"count": True}, {"count": 1.0}, {"action": "hangup"}, {"silent_ms": -1}, {"silent_ms": float("nan")},
                {"silent_ms": True}):
        options = {"session_id": SID, "count": 1, "action": "end", "silent_ms": 1.0, **bad}
        with pytest.raises(InvalidInput):
            await memory.record_idle(SPACE, **options)
    unlogged = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        assert await unlogged.record_idle(SPACE, session_id=SID, count=1, action="end", silent_ms=1.0) is None
    finally:
        await unlogged.close()
