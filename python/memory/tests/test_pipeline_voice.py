"""A voice conversation as a pipeline.

Audio goes in at the far end and comes back out of it. Everything
between is an ordinary stage, so the same interruption, the same
failure reporting and the same record of what happened apply to a phone
call as to anything else built this way.
"""

from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.pipeline import Pipeline
from scone_memory.pipeline.memory import MemoryStage
from scone_memory.pipeline.voice import (CallerStage, ModelStage, RecognizerStage, Spoken,
                                         SynthesizerStage)
from scone_memory.realtime.audio import AudioChunk, SpeechStarted, Transcript
from scone_memory.realtime.events import ReplyCompleted, TextDelta

SOUND = AudioChunk(pcm=b"\x00\x01" * 80, sample_rate=16000)


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Mark takes his coffee black", tags=["habits"])
    return memory


async def until(predicate, what, timeout=2.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


class FarEnd:
    """The other end of the call, under the test's control: what it says,
    what it is told to play, and what it is told to throw away."""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[tuple[str, AudioChunk]] = []
        self.cleared: list[str] = []
        self.broken: Exception | None = None

    async def says(self, chunk=SOUND):
        await self.inbox.put(chunk)

    def hangs_up(self):
        self.inbox.put_nowait(None)

    def breaks(self, error):
        self.inbox.put_nowait(error)

    async def _incoming(self):
        while (item := await self.inbox.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def receive(self):
        return self._incoming()

    async def send(self, audio, turn_id):
        self.sent.append((turn_id, audio))

    async def clear(self, turn_id):
        self.cleared.append(turn_id)

    async def aclose(self):
        pass


class Ear:
    """Hears whatever the test says each chunk of audio means."""

    def __init__(self, script):
        self.script = list(script)
        self.heard: list[AudioChunk] = []

    def transcribe(self, audio):
        async def events():
            async for chunk in audio:
                self.heard.append(chunk)
                if self.script:
                    step = self.script.pop(0)
                    if step is not None:
                        yield step
        return events()

    async def aclose(self):
        pass


class Mind:
    """Says what the test told it to say, a word at a time."""

    def __init__(self, replies, hold=None):
        self.replies = list(replies)
        self.asked: list[list[dict]] = []
        self.hold = hold

    def respond(self, messages):
        self.asked.append(messages)
        text = self.replies.pop(0) if self.replies else "yes"

        async def words():
            for word in text.split():
                yield TextDelta(text=word + " ")
                if self.hold is not None:
                    await self.hold.wait()
                await asyncio.sleep(0)
            yield ReplyCompleted()

        return words()

    async def aclose(self):
        pass


class Mouth:
    """Turns each sentence into two chunks of audio."""

    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text):
        async def audio():
            self.spoken.append(text)
            for _ in range(2):
                yield SOUND
                await asyncio.sleep(0)

        return audio()

    async def aclose(self):
        pass


def line(far, ear, mind, mouth, memory=None):
    stages = [CallerStage(far), RecognizerStage(ear)]
    if memory is not None:
        stages.append(memory)
    stages += [ModelStage(mind), SynthesizerStage(mouth)]
    return Pipeline(stages, session="call-1")


async def test_a_turn_goes_in_as_audio_and_comes_back_as_audio(engine):
    """The whole point, end to end: the person is heard, what is already
    known about them reaches the model, and the answer is played back
    under the turn it belongs to."""
    far, ear = FarEnd(), Ear([Transcript(text="how do I take my coffee?", final=True)])
    mind, mouth = Mind(["black, always"]), Mouth()
    pipeline = line(far, ear, mind, mouth, MemoryStage(engine, "alpha", session="call-1"))
    await pipeline.start()
    await far.says()
    await until(lambda: far.sent, "the answer to be played")

    assert any("black" in m["content"] for m in mind.asked[0] if m["role"] == "system"), \
        "what is remembered about the person was put in front of the model"
    assert mouth.spoken == ["black, always"]
    assert {turn for turn, _ in far.sent} == {"2"}, \
        "played under one turn, the one the question opened: asking ends whatever came before"
    await pipeline.stop()


async def test_someone_interrupting_stops_the_answer_and_it_is_never_remembered(engine):
    """Barge-in: the person starts talking over the answer. What is
    queued to play is dropped by name, and words nobody heard the end of
    belong to nobody, so they are not learned."""
    hold = asyncio.Event()
    far = FarEnd()
    ear = Ear([Transcript(text="how do I take my coffee?", final=True), SpeechStarted()])
    mind, mouth = Mind(["black. always black."], hold=hold), Mouth()
    pipeline = line(far, ear, mind, mouth, MemoryStage(engine, "alpha", session="call-2"))
    await pipeline.start()
    await far.says()
    await until(lambda: far.sent, "the first sentence to play")
    await far.says()
    await until(lambda: "2" in far.cleared, "the queued answer to be dropped")
    hold.set()
    await asyncio.sleep(0.05)

    assert far.cleared == ["1", "2"], \
        "each turn is ended by name: the empty one the first question closed, then the one cut off"
    assert mouth.spoken == ["black."], "the rest of the answer was never even spoken"
    found = await engine.recall("alpha", "always black", limit=10)
    assert not any("always black" in item.text for item in found.items), \
        "an answer cut off part way is not something that happened"
    await pipeline.stop()


async def test_a_line_that_dies_ends_the_call_and_says_why():
    """The far end is the call. When it goes, the run is over, and
    whoever started it is told what happened rather than left waiting."""
    far = FarEnd()
    pipeline = line(far, Ear([]), Mind([]), Mouth())
    await pipeline.start()
    far.breaks(OSError("the carrier went away"))
    with pytest.raises(OSError, match="carrier went away"):
        await asyncio.wait_for(pipeline.wait(), 2)
    await pipeline.stop()


async def test_the_same_line_without_memory_still_answers():
    """Memory is a stage, so it can be left out. What remains is a
    conversation that answers and forgets, which is a useful thing to be
    able to run beside one that does not."""
    far, ear = FarEnd(), Ear([Transcript(text="what time is it?", final=True)])
    mind, mouth = Mind(["nearly one."]), Mouth()
    pipeline = line(far, ear, mind, mouth)
    await pipeline.start()
    await far.says()
    await until(lambda: far.sent, "the answer to be played")

    assert mouth.spoken == ["nearly one."]
    assert [m["role"] for m in mind.asked[0]] == ["system", "user"], \
        "the question reaches the model on its own, with nothing recalled in front of it"
    await pipeline.stop()


async def test_the_assistant_is_never_transcribed_as_the_caller():
    """The answer travels back past the ear on its way to the far end.
    If it arrived as plain audio the assistant would be transcribed as
    the person and answer itself."""
    far, ear = FarEnd(), Ear([Transcript(text="hello?", final=True)])
    mind, mouth = Mind(["hello there."]), Mouth()
    pipeline = line(far, ear, mind, mouth)
    await pipeline.start()
    await far.says()
    await until(lambda: far.sent, "the answer to be played")

    assert len(ear.heard) == 1, "the ear heard the person once and the assistant not at all"
    assert far.sent, "and the answer did reach the far end, so it really did travel back past the ear"
    await pipeline.stop()


async def test_a_model_that_gives_up_does_not_end_the_call():
    """A provider failing part way through an answer is a bad turn, not
    a dropped call: the line stays up and the next question is heard."""

    class Sulks:
        def __init__(self):
            self.tries = 0

        def respond(self, messages):
            self.tries += 1
            attempt = self.tries

            async def words():
                if attempt == 1:
                    raise RuntimeError("the model gave up")
                yield TextDelta(text="still here. ")
                yield ReplyCompleted()

            return words()

        async def aclose(self):
            pass

    far = FarEnd()
    ear = Ear([Transcript(text="hello?", final=True), Transcript(text="still there?", final=True)])
    mind, mouth = Sulks(), Mouth()
    pipeline = line(far, ear, mind, mouth)
    await pipeline.start()
    await far.says()
    await until(lambda: pipeline.failures, "the bad turn to be reported")
    assert not pipeline.ended.is_set(), "a bad answer does not drop the call"
    await far.says()
    await until(lambda: far.sent, "the next answer to be played")

    assert mouth.spoken == ["still here."]
    await pipeline.stop()
