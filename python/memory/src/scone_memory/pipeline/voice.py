"""A voice conversation as a pipeline.

The far end is one stage. Audio arrives there and answers go back there,
so a call is an ordinary line of stages: the same interruption, the same
failure reporting, the same record of what happened as anything else
built this way. Adding memory is adding a stage; leaving it out leaves a
conversation that answers without remembering, which is a useful thing
to be able to run.

The providers are the ones the rest of Scone already speaks to, the
transport, recognizer, model and synthesizer of ``realtime.audio``, so
an adapter written for a voice session works here unchanged. Each stage
does one job and validates only what it is responsible for.

Nobody closes what they were handed: a stage that closed a provider it
borrowed would be deciding the lifetime of something it does not own.
The caller opens them and the caller closes them.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional

from ..realtime.audio import (AudioChunk, AudioTransport, Reply, SpeechRecognizer, SpeechStarted,
                              SpeechSynthesizer, Transcript, VoiceModel, check_audio, is_question,
                              sentences)
from ..realtime.events import ReplyCompleted, TextDelta
from .core import Interrupted
from .memory import Prompt

#: What the assistant is told it is, when nothing else says.
PERSONA = "You are a helpful voice assistant. Answer in one or two short sentences."


@dataclass(frozen=True)
class Spoken:
    """Audio the assistant is saying, on its way back to the far end.

    Wrapped rather than sent as bare PCM because it travels back past the
    stages that heard the person, and the ear must not hear the mouth."""

    audio: AudioChunk


class CallerStage:
    """The far end of the call: where the person is heard and where the
    answer is played.

    Essential, because a call without it is not a call. It reads on a
    task of its own, so the run learns of a dead line through the same
    path as any other fault."""

    essential = True

    def __init__(self, transport: AudioTransport, *, max_bytes: int = 64000) -> None:
        self.transport = transport
        self.max_bytes = max_bytes
        self._feed = None
        self._task: Optional[asyncio.Task] = None

    async def attach(self, feed) -> None:
        self._feed = feed

    async def start(self) -> None:
        self._task = asyncio.create_task(self._listen(), name="voice-caller")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _listen(self) -> None:
        try:
            async with contextlib.aclosing(self.transport.receive()) as incoming:
                async for chunk in incoming:
                    await self._feed(check_audio(chunk, self.max_bytes))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - a dead line ends the call, it does not raise into nothing
            await self._feed.fail(error)

    async def handle(self, frame: object, emit) -> None:
        if isinstance(frame, Spoken):
            await self.transport.send(frame.audio, str(emit.turn))
        elif isinstance(frame, Interrupted):
            # Whatever is queued for the abandoned turn stops playing. The
            # turn is named, so this cannot silence the answer that has
            # just replaced it.
            await self.transport.clear(str(frame.turn))


class RecognizerStage:
    """Words out of audio.

    A recognizer reads a stream rather than a frame, so the stage feeds
    it one and passes on what comes back. It is the stage nearest the
    person, so it is the one that says a turn is over: when it hears
    speech begin, the answer in flight stops being wanted."""

    essential = True

    def __init__(self, recognizer: SpeechRecognizer, *, backlog: int = 8) -> None:
        self.recognizer = recognizer
        # Built here rather than on start, so audio arriving from a stage
        # in front of this one before its own turn to start has somewhere
        # to wait. Whether that happens is the other stage's business.
        self._audio: asyncio.Queue = asyncio.Queue(backlog)
        self._feed = None
        self._task: Optional[asyncio.Task] = None

    async def attach(self, feed) -> None:
        self._feed = feed

    async def start(self) -> None:
        self._task = asyncio.create_task(self._read(), name="voice-recognizer")

    async def stop(self) -> None:
        # Cancelled rather than sent a last marker. A marker needs room in
        # a queue that may be full of audio nobody is reading, and a call
        # has to be able to end even then.
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _chunks(self):
        while True:
            yield await self._audio.get()

    async def _read(self) -> None:
        try:
            async with contextlib.aclosing(self.recognizer.transcribe(self._chunks())) as events:
                async for event in events:
                    await self._heard(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the same path a raise on a frame takes
            await self._feed.fail(error)

    async def _heard(self, event: object) -> None:
        if isinstance(event, SpeechStarted):
            await self._feed.interrupt()
            return
        if not is_question(event):
            return
        # A finished question ends whatever was being said: answering the
        # last one over the top of the new one helps nobody.
        await self._feed.interrupt()
        await self._feed(event)

    async def handle(self, frame: object, emit) -> None:
        # Only what the person said. Audio going the other way is Spoken,
        # so the assistant is never transcribed as if it were the caller.
        if isinstance(frame, AudioChunk):
            await self._audio.put(frame)


class ModelStage:
    """The answer, a piece at a time.

    Buffered because it waits on a network, with a short backlog: more
    than a few questions waiting on the model means the conversation has
    run away from the person having it. What it says goes forward to be
    spoken and back to be remembered."""

    buffered = True
    capacity = 4

    def __init__(self, model: VoiceModel, *, persona: str = PERSONA, max_bytes: int = 64000) -> None:
        self.model = model
        self.persona = persona
        self.max_bytes = max_bytes

    def _messages(self, frame: object) -> Optional[list[dict]]:
        if isinstance(frame, Prompt):
            messages = list(frame.messages)
        elif isinstance(frame, Transcript) and frame.final and frame.text.strip():
            messages = [{"role": "user", "content": frame.text.strip()}]
        else:
            return None
        if self.persona and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.persona})
        return messages

    async def handle(self, frame: object, emit) -> None:
        messages = self._messages(frame)
        if messages is None:
            return
        reply, said = Reply(self.max_bytes), False
        # Read to the end of the stream rather than stopping at the
        # ending it declares: what a provider says afterwards is the
        # thing worth catching, and it can only be seen by looking.
        async with contextlib.aclosing(self.model.respond(messages)) as stream:
            async for event in stream:
                if emit.cut_off:
                    return
                reply.accept(event)
                said = said or isinstance(event, TextDelta)
                await emit(event)
                # Memory sits in front of the model, so what the model
                # says has to travel back to reach it.
                await emit.up(event)
        if not reply.done or not said:
            raise RuntimeError("the model stopped without a finished answer")


class SynthesizerStage:
    """Sentences into audio, sent back to the far end.

    A sentence at a time, so the first words are heard while the rest of
    the answer is still arriving."""

    buffered = True
    capacity = 8

    def __init__(self, synthesizer: SpeechSynthesizer) -> None:
        self.synthesizer = synthesizer
        self._pending = ""
        self._turn: Optional[int] = None

    async def handle(self, frame: object, emit) -> None:
        if isinstance(frame, Interrupted):
            self._pending = ""
            return
        if isinstance(frame, TextDelta):
            if self._turn != emit.turn:
                self._turn, self._pending = emit.turn, ""
            self._pending += frame.text
            done, self._pending = sentences(self._pending)
            for sentence in done:
                await self._say(sentence, emit)
        elif isinstance(frame, ReplyCompleted):
            rest, self._pending = self._pending, ""
            if rest.strip():
                await self._say(rest, emit)

    async def _say(self, text: str, emit) -> None:
        if emit.cut_off:
            return
        async with contextlib.aclosing(self.synthesizer.synthesize(text.strip())) as audio:
            async for chunk in audio:
                if emit.cut_off:
                    return
                await emit.up(Spoken(audio=chunk))
