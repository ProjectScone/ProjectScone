"""Scone-owned audio runtime: scripted services, native memory, no Pipecat."""

import asyncio
import copy
import importlib.util

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record


def api():
    assert importlib.util.find_spec("scone_memory.voice_types") is not None, (
        "Voice needs Scone-owned protocols, not third-party frame classes"
    )
    from scone_memory import voice_types
    from scone_memory.voice import VoiceSession
    return voice_types, VoiceSession


@pytest.fixture
async def memory():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


class Resource:
    def __init__(self):
        self.closes = 0

    async def aclose(self):
        self.closes += 1


class Transport(Resource):
    def __init__(self):
        super().__init__()
        self.input = asyncio.Queue()
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


class Recognizer(Resource):
    def __init__(self):
        super().__init__()
        self.received = []

    async def transcribe(self, audio):
        types, _ = api()
        async for chunk in audio:
            self.received.append(chunk)
            yield types.SpeechStarted()
            yield types.Transcript(
                "Where does Juniper point?" if len(self.received) == 1 else "What is its name?",
                speaker="speaker-1",
            )


class Model(Resource):
    def __init__(self):
        super().__init__()
        self.contexts = []
        self.finish = asyncio.Event()
        self.finish.set()

    async def respond(self, messages):
        types, _ = api()
        self.contexts.append(copy.deepcopy(messages))
        yield types.TextDelta("Juniper points to Polaris." if len(self.contexts) == 1 else "Its name is Juniper.")
        await self.finish.wait()
        yield types.ReplyCompleted()


class Synthesizer(Resource):
    def __init__(self):
        super().__init__()
        self.texts = []

    async def synthesize(self, text):
        types, _ = api()
        self.texts.append(text)
        yield types.AudioChunk(b"\x01\x00" * 160, 16000)


class Rig:
    def __init__(self, memory, **overrides):
        types, session_type = api()
        self.transport, self.stt, self.model, self.tts = Transport(), Recognizer(), Model(), Synthesizer()
        self.resources = [self.transport, self.stt, self.model, self.tts]
        options = dict(transport_factory=lambda: self.transport, stt_factory=lambda: self.stt,
                       model_factory=lambda: self.model, tts_factory=lambda: self.tts,
                       capture=True, session_timeout=5, turn_timeout=3)
        options.update(overrides)
        self.session = session_type(memory, "voice-test", "native-voice", **options)

    async def audio(self):
        types, _ = api()
        await self.transport.input.put(types.AudioChunk(b"\x02\x00" * 320, 16000))


async def records(memory, count):
    async with asyncio.timeout(2):
        while True:
            found = await memory.episodes("voice-test", {"session_id": "native-voice"})
            if len(found) >= count:
                return found
            await asyncio.sleep(.005)


async def test_native_voice_streams_audio_before_model_end_and_keeps_scoped_history(memory):
    await memory.remember_many("voice-test", [
        Record("Juniper points to Polaris.", metadata={"collection": "manual"}),
        Record("Juniper secret points to Mars.", metadata={"collection": "private"}),
    ])
    await memory.remember_many("another-space", [Record("Juniper points to Venus.")])
    rig = Rig(memory, where={"collection": "manual"})
    rig.model.finish.clear()
    running = asyncio.create_task(rig.session.run())
    try:
        await asyncio.wait_for(rig.session.started.wait(), 2)
        await rig.audio()
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        assert not running.done()
        assert len(await memory.episodes("voice-test", {"session_id": "native-voice"})) == 1
        rig.model.finish.set()
        await records(memory, 2)
        await rig.audio()
        saved = await records(memory, 4)
        await rig.transport.input.put(None)
        await asyncio.wait_for(running, 2)
    finally:
        await rig.session.close()
        await asyncio.gather(running, return_exceptions=True)
    assert rig.session.state == "ended"
    assert all(resource.closes == 1 for resource in rig.resources)
    assert [e.content for e in saved] == [
        "Where does Juniper point?", "Juniper points to Polaris.",
        "What is its name?", "Its name is Juniper.",
    ]
    assert [e.metadata["role"] for e in saved] == ["user", "assistant", "user", "assistant"]
    assert all(e.metadata["integration"] == "scone-voice" for e in saved)
    assert all(e.metadata["representation"] == "aggregated_text" for e in saved)
    assert rig.session.stored_count == 4
    assert "Mars" not in str(rig.model.contexts) and "Venus" not in str(rig.model.contexts)
    assert sum("Scone retrieved source material" in str(m) for m in rig.model.contexts[1]) == 1
    assert {"role": "assistant", "content": "Juniper points to Polaris."} in rig.model.contexts[1]
    assert [a.pcm for a in rig.stt.received] == [b"\x02\x00" * 320] * 2
    assert [a.pcm for a, _ in rig.transport.sent] == [b"\x01\x00" * 160] * 2


async def test_barge_in_clears_old_audio_and_does_not_save_partial_reply(memory):
    rig = Rig(memory)
    rig.model.finish.clear()
    running = asyncio.create_task(rig.session.run())
    try:
        await asyncio.wait_for(rig.session.started.wait(), 2)
        await rig.audio()
        await asyncio.wait_for(rig.transport.delivered.wait(), 2)
        first_turn = rig.transport.sent[0][1]
        await rig.audio()
        async with asyncio.timeout(2):
            while first_turn not in rig.transport.cleared:
                await asyncio.sleep(.005)
        rig.model.finish.set()
        saved = await records(memory, 3)
        await rig.transport.input.put(None)
        await asyncio.wait_for(running, 2)
    finally:
        await rig.session.close()
        await asyncio.gather(running, return_exceptions=True)
    assert [e.content for e in saved] == [
        "Where does Juniper point?", "What is its name?", "Its name is Juniper.",
    ]
    assert len({turn for _, turn in rig.transport.sent}) == 2


@pytest.mark.parametrize("options", [
    {"capture": False}, {"capture": "yes"}, {"session_timeout": 0},
    {"turn_timeout": float("inf")}, {"audio_queue_size": 0},
    {"max_history_bytes": True}, {"max_reply_bytes": 0},
    {"where": {"project": 42}}, {"stt_factory": None},
])
async def test_invalid_configuration_never_constructs_resources(memory, options):
    _, session_type = api()
    calls = []
    def factory():
        calls.append(True)
    args = dict(transport_factory=factory, stt_factory=factory, model_factory=factory,
                tts_factory=factory, capture=True)
    args.update(options)
    with pytest.raises(ValueError):
        session_type(memory, "voice-test", "native-voice", **args)
    assert not calls


@pytest.mark.parametrize("action", ["close", "cancel", "timeout"])
async def test_single_use_lifecycle_closes_every_resource(memory, action):
    rig = Rig(memory, session_timeout=.05 if action == "timeout" else 5)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    if action == "close":
        await rig.session.close()
    elif action == "cancel":
        running.cancel()
    with pytest.raises(TimeoutError if action == "timeout" else asyncio.CancelledError):
        await running
    assert all(r.closes == 1 for r in rig.resources)
    assert rig.session.state == ("failed" if action == "timeout" else "interrupted")
    with pytest.raises(RuntimeError, match="single-use"):
        await rig.session.run()


async def test_failed_construction_closes_partial_resources(memory):
    def broken():
        raise RuntimeError("provider construction failed")
    rig = Rig(memory, model_factory=broken)
    with pytest.raises(RuntimeError, match="construction"):
        await rig.session.run()
    assert rig.transport.closes == rig.stt.closes == 1
    assert rig.tts.closes == 0


async def test_missing_model_completion_never_saves_assistant(memory):
    class IncompleteModel(Model):
        async def respond(self, messages):
            types, _ = api()
            yield types.TextDelta("Unfinished reply.")
    rig = Rig(memory, model_factory=IncompleteModel)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    with pytest.raises(RuntimeError, match="completion"):
        await asyncio.wait_for(running, 2)
    assert [e.metadata["role"] for e in await records(memory, 1)] == ["user"]
    assert rig.transport.cleared


async def test_capture_failure_stops_before_model(memory, monkeypatch):
    async def broken(*args, **kwargs):
        raise OSError("store unavailable")
    monkeypatch.setattr(memory, "remember_many", broken)
    rig = Rig(memory)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    with pytest.raises(OSError, match="store unavailable"):
        await asyncio.wait_for(running, 2)
    assert not rig.model.contexts
    assert all(r.closes == 1 for r in rig.resources)


async def test_repeated_close_waits_for_same_cleanup(memory):
    entered, release = asyncio.Event(), asyncio.Event()
    class SlowModel(Model):
        async def aclose(self):
            self.closes += 1
            entered.set()
            await release.wait()
    model = SlowModel()
    rig = Rig(memory, model_factory=lambda: model)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    first = asyncio.create_task(rig.session.close())
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(rig.session.close())
    try:
        await asyncio.sleep(.03)
        assert not first.done() and not second.done()
    finally:
        release.set()
        outcomes = await asyncio.gather(first, second, running, return_exceptions=True)
    assert outcomes[:2] == [None, None]
    assert isinstance(outcomes[2], asyncio.CancelledError)
    assert model.closes == 1


def test_pcm_validation_rejects_malformed_frames():
    types, _ = api()
    for args in [(b"x", 16000), (b"", 16000), (b"xx", 0), (b"xx", True), ("text", 16000)]:
        with pytest.raises(ValueError):
            types.AudioChunk(*args)


async def test_empty_synthesis_cannot_claim_audio_output(memory):
    class SilentTTS(Synthesizer):
        async def synthesize(self, text):
            if False:
                yield
    rig = Rig(memory, tts_factory=SilentTTS)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    await rig.transport.input.put(None)
    with pytest.raises(RuntimeError, match="audio"):
        await asyncio.wait_for(running, 2)
    assert [e.metadata["role"] for e in await records(memory, 1)] == ["user"]


async def test_recognizer_cannot_silently_drop_queued_input(memory):
    class EarlySTT(Recognizer):
        async def transcribe(self, audio):
            if False:
                yield
    rig = Rig(memory, stt_factory=EarlySTT)
    await rig.audio()
    await rig.transport.input.put(None)
    with pytest.raises(RuntimeError, match="audio input"):
        await rig.session.run()


async def test_audio_input_is_backpressured_while_recognizer_waits(memory):
    entered = asyncio.Event()
    class BlockedSTT(Recognizer):
        async def transcribe(self, audio):
            entered.set()
            await asyncio.Event().wait()
            if False:
                yield
    class CountingTransport(Transport):
        received = 0
        async def receive(self):
            types, _ = api()
            for _ in range(100):
                self.received += 1
                yield types.AudioChunk(b"\x01\x00", 16000)
    transport = CountingTransport()
    rig = Rig(memory, transport_factory=lambda: transport, stt_factory=BlockedSTT, audio_queue_size=2)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.sleep(.02)
    assert transport.received == 3  # two queued plus one producer-held packet
    await rig.session.close()
    with pytest.raises(asyncio.CancelledError):
        await running


async def test_provider_cleanup_failure_is_visible_after_all_resources_close(memory):
    class BrokenModel(Model):
        async def aclose(self):
            await super().aclose()
            raise RuntimeError("cleanup broke")
    model = BrokenModel()
    rig = Rig(memory, model_factory=lambda: model)
    await rig.transport.input.put(None)
    with pytest.raises(RuntimeError, match="cleanup"):
        await rig.session.run()
    assert rig.session.state == "failed"
    assert model.closes == rig.stt.closes == rig.tts.closes == rig.transport.closes == 1


async def test_late_output_after_swallowed_cancellation_is_not_sent_or_saved(memory):
    types, _ = api()
    class LateModel(Model):
        async def respond(self, messages):
            self.contexts.append(messages)
            if len(self.contexts) == 1:
                yield types.TextDelta("First answer.")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    yield types.TextDelta("Stale leaked reply.")
            else:
                yield types.TextDelta("New answer.")
            yield types.ReplyCompleted()
    model = LateModel()
    rig = Rig(memory, model_factory=lambda: model)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    await asyncio.wait_for(rig.transport.delivered.wait(), 2)
    await rig.audio()
    saved = await records(memory, 3)
    await rig.transport.input.put(None)
    await asyncio.wait_for(running, 2)
    assert rig.tts.texts == ["First answer.", "New answer."]
    assert [e.content for e in saved][-1] == "New answer."
    assert all("Stale" not in e.content for e in saved)


async def test_interruption_during_uncertain_capture_fails_closed(memory, monkeypatch):
    committed = asyncio.Event()
    remember = memory.remember_many
    async def uncertain(space, batch):
        result = await remember(space, batch)
        if batch[0].metadata.get("role") == "assistant":
            committed.set()
            await asyncio.Event().wait()  # commit happened, acknowledgment has not
        return result
    monkeypatch.setattr(memory, "remember_many", uncertain)
    rig = Rig(memory)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    await asyncio.wait_for(committed.wait(), 2)
    await rig.audio()
    await rig.transport.input.put(None)
    with pytest.raises(RuntimeError, match="capture"):
        await asyncio.wait_for(running, 2)
    assert rig.session.state == "failed"
    assert len(rig.model.contexts) == 1
    assert len(await records(memory, 2)) == 2


async def test_close_during_barge_in_cleanup_does_not_resume_listener(memory):
    entered, release, escape = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class SlowModel(Model):
        async def respond(self, messages):
            types, _ = api()
            yield types.TextDelta("Partial reply.")
            try:
                await asyncio.Event().wait()
            finally:
                entered.set()
                await release.wait()
    class WaitingSTT(Recognizer):
        async def transcribe(self, audio):
            types, _ = api()
            async for chunk in audio:
                self.received.append(chunk)
                yield types.SpeechStarted()
                if len(self.received) == 1:
                    yield types.Transcript("First question")
                else:
                    await escape.wait()
                    return
    rig = Rig(memory, model_factory=SlowModel, stt_factory=WaitingSTT)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    await asyncio.wait_for(rig.transport.delivered.wait(), 2)
    await rig.audio()
    await asyncio.wait_for(entered.wait(), 2)
    closing = asyncio.create_task(rig.session.close())
    await asyncio.sleep(.03)
    release.set()
    try:
        done, _ = await asyncio.wait({closing}, timeout=.15)
        assert closing in done, "cancelled listener must not resume waiting for speech"
    finally:
        escape.set()
        await asyncio.gather(closing, running, return_exceptions=True)


async def test_sync_cleanup_exception_still_closes_other_resources(memory):
    class BrokenModel(Model):
        def aclose(self):
            self.closes += 1
            raise RuntimeError("synchronous cleanup failure")
    model = BrokenModel()
    rig = Rig(memory, model_factory=lambda: model)
    await rig.transport.input.put(None)
    with pytest.raises(RuntimeError, match="cleanup"):
        await rig.session.run()
    assert model.closes == rig.transport.closes == rig.stt.closes == rig.tts.closes == 1


@pytest.mark.parametrize("mode", ["unsupported", "after_end", "oversize"])
async def test_malformed_model_output_cannot_become_saved_reply(memory, mode):
    class BadModel(Model):
        async def respond(self, messages):
            types, _ = api()
            if mode == "unsupported":
                yield {"reasoning": "not public text"}
            elif mode == "after_end":
                yield types.ReplyCompleted()
                yield types.TextDelta("Too late.")
            else:
                yield types.TextDelta("x" * 513)
                yield types.ReplyCompleted()
    rig = Rig(memory, model_factory=BadModel, max_reply_bytes=512)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    with pytest.raises((ValueError, RuntimeError)):
        await asyncio.wait_for(running, 2)
    assert len(await records(memory, 1)) == 1
    assert rig.tts.texts == []


async def test_scope_is_copied_and_provider_mutation_never_rewrites_history(memory):
    class MutatingModel(Model):
        async def respond(self, messages):
            async for event in super().respond(messages):
                messages[0]["content"] = "provider mutation"
                yield event
    await memory.remember_many("voice-test", [Record("Juniper manual source.", metadata={"collection": "manual"})])
    scope = {"collection": "manual"}
    model = MutatingModel()
    rig = Rig(memory, model_factory=lambda: model, where=scope)
    scope["collection"] = "other"
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    await records(memory, 2)
    await rig.audio()
    await records(memory, 4)
    await rig.transport.input.put(None)
    await asyncio.wait_for(running, 2)
    assert "Juniper manual source." in str(model.contexts[0])
    assert "provider mutation" not in str(model.contexts[1])


@pytest.mark.parametrize("second_stop", ["session_deadline", "host_close"])
async def test_overlapping_deadlines_do_not_recancel_iterator_cleanup(memory, second_stop):
    entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class SlowIterator(Model):
        async def respond(self, messages):
            try:
                await asyncio.Event().wait()
                if False:
                    yield
            finally:
                entered.set()
                await release.wait()
                completed.set()
    model = SlowIterator()
    rig = Rig(memory, model_factory=lambda: model, turn_timeout=.03,
              session_timeout=.08 if second_stop == "session_deadline" else 5)
    running = asyncio.create_task(rig.session.run())
    await asyncio.wait_for(rig.session.started.wait(), 2)
    await rig.audio()
    await asyncio.wait_for(entered.wait(), 2)
    closing = asyncio.create_task(rig.session.close()) if second_stop == "host_close" else None
    try:
        await asyncio.sleep(.12)
        assert not running.done() and model.closes == 0
    finally:
        release.set()
        results = await asyncio.gather(*([running, closing] if closing else [running]), return_exceptions=True)
    assert completed.is_set()
    assert isinstance(results[0], TimeoutError)
    assert model.closes == 1
