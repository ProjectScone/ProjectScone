"""Scone Voice: native, provider-independent audio conversation orchestration.

Only asyncio and Scone memory are required (Python 3.11+ for structured deadlines).
The host supplies authorized transport/provider factories and participant consent.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping
from contextlib import aclosing
from contextvars import ContextVar
from uuid import uuid4

from ..memory.engine import MemoryEngine, Record, check_space
from ..retrieval.recall_scope import RecallScope
from .context import MemoryContext
from .lifecycle import cancel_once as _cancel_once, settle as _settle
from .audio import (
    AudioChunk, AudioTransport, SpeechRecognizer, VoiceModel, SpeechSynthesizer,
    SpeechStarted, Transcript, TextDelta, ReplyCompleted, SpeechActivityDetector,
)

_OWNER: ContextVar[object | None] = ContextVar("scone_voice_owner", default=None)
_EOF = object()


def _bytes(value) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


async def _close(resource):
    # Invoke inside its own coroutine so even a synchronous adapter exception
    # cannot prevent neighboring resources from receiving their close call.
    await resource.aclose()


class VoiceSession:
    """Single-use Scone turn controller with bounded audio and public memory.

    Factories are synchronous, nonblocking and return fresh resources implementing
    realtime.audio protocols. Stream methods return async iterators with aclose().
    The host owns authentication, consent, device permissions and signaling.
    Deadlines request cancellation; provider cleanup must cooperate. No audio,
    hidden reasoning, tools or retrieved context is persisted as transcript.
    """

    def __init__(
        self, memory: MemoryEngine, space: str, session_id: str, *,
        transport_factory: Callable[[], AudioTransport],
        stt_factory: Callable[[], SpeechRecognizer],
        model_factory: Callable[[], VoiceModel],
        tts_factory: Callable[[], SpeechSynthesizer],
        activity_factory: Callable[[], SpeechActivityDetector] | None = None,
        capture: bool, system_prompt: str = "You are a helpful voice assistant.",
        where: Mapping[str, str] | None = None, kind: str | None = None,
        source_prefix: str | None = None, since: str | None = None, until: str | None = None,
        session_timeout: float = 1800, turn_timeout: float = 30,
        audio_queue_size: int = 8, max_audio_bytes: int = 64000,
        max_history_bytes: int = 128000, max_reply_bytes: int = 64000,
    ):
        check_space(space)
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", session_id):
            raise ValueError("session_id must be an opaque identifier of 1..128 characters")
        if capture is not True:
            raise ValueError("capture=True is required for public transcript retention")
        factories = (transport_factory, stt_factory, model_factory, tts_factory)
        if activity_factory is not None:
            factories += (activity_factory,)
        if any(not callable(factory) for factory in factories):
            raise ValueError("factories must supply fresh transport/provider resources")
        for timeout in (session_timeout, turn_timeout):
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("deadlines must be finite and positive")
        for value in (max_audio_bytes, max_history_bytes, max_reply_bytes):
            if type(value) is not int or not 512 <= value <= 1_000_000:
                raise ValueError("byte limits must be integers in 512..1000000")
        if type(audio_queue_size) is not int or not 1 <= audio_queue_size <= 256:
            raise ValueError("audio_queue_size must be in 1..256")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty")
        self._history = [{"role": "system", "content": system_prompt}]
        if _bytes(self._history) > max_history_bytes:
            raise ValueError("system prompt exceeds history limit")
        self._scope = RecallScope.validated(where=where, kind=kind, source_prefix=source_prefix, since=since, until=until)
        self._memory, self._space, self._session_id = memory, space, session_id
        self._memory_context = MemoryContext(memory, space, session_id, **self._scope.kwargs())
        self._factories = factories
        self._session_timeout, self._turn_timeout = session_timeout, turn_timeout
        self._queue_size, self._max_audio = audio_queue_size, max_audio_bytes
        self._max_history, self._max_reply = max_history_bytes, max_reply_bytes
        self._active = self._reply = None
        self._generation = 0
        self._output_turn = None
        self._capture_id = uuid4().hex
        self._state = "new"
        self._stop = asyncio.Event()
        self.started = asyncio.Event()
        self.stored_count = 0
        self.last_memory_receipt: dict | None = None

    @property
    def state(self) -> str:
        return self._state

    async def run(self) -> None:
        if self._state != "new":
            raise RuntimeError("voice sessions are single-use")
        self._state = "starting"
        self._active = asyncio.create_task(self._run())
        try:
            await asyncio.shield(self._active)
        except asyncio.CancelledError:
            self._stop.set()
            outcome, _ = await _settle(self._active)
            if isinstance(outcome, Exception):
                self._state = "failed"
                raise outcome
            self._state = "interrupted"
            raise
        except BaseException:
            self._state = "failed"
            raise
        else:
            self._state = "ended"
        finally:
            self._active = None

    async def close(self) -> None:
        """Host-only cancellation; repeated calls join the same owned cleanup."""
        if _OWNER.get() is self:
            raise RuntimeError("close must be called outside voice providers and callbacks")
        if self._state == "new":
            self._state = "interrupted"
        if self._active is not None:
            self._stop.set()
            outcome, cancelled = await _settle(self._active)
            if isinstance(outcome, Exception):
                raise outcome
            if cancelled:
                raise asyncio.CancelledError()

    async def _run(self):
        token = _OWNER.set(self)
        resources, tasks = [], []
        try:
            if self._stop.is_set():
                raise asyncio.CancelledError()
            requirements = (("receive", "send", "clear"), ("transcribe",), ("respond",), ("synthesize",))
            if len(self._factories) == 5:
                requirements += (("detect",),)
            for factory, methods in zip(self._factories, requirements):
                resource = factory()
                if any(resource is existing for existing in resources):
                    raise ValueError("voice resources must be distinct")
                if not callable(getattr(resource, "aclose", None)):
                    raise TypeError("voice resources must implement aclose")
                resources.append(resource)
                if any(not callable(getattr(resource, name, None)) for name in methods):
                    raise TypeError("voice resource does not implement its Scone protocol")
            transport, stt, model, tts = resources[:4]
            activity = resources[4] if len(resources) == 5 else None
            work = asyncio.create_task(self._conversation(transport, stt, model, tts, activity))
            stop = asyncio.create_task(self._stop.wait())
            tasks = [work, stop]
            self._state = "running"
            self.started.set()
            async with asyncio.timeout(self._session_timeout):
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if self._stop.is_set():
                    raise asyncio.CancelledError()
                await work
        finally:
            try:
                for task in tasks:
                    _cancel_once(task)
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                # Close every admitted resource even when a neighbor fails.
                closed = await asyncio.gather(*(_close(r) for r in reversed(resources)), return_exceptions=True)
                failure = next((e for e in closed if isinstance(e, BaseException)), None)
                if failure is not None:
                    raise RuntimeError("voice resource cleanup failed") from failure
                failure = next((e for e in outcomes if isinstance(e, Exception)), None)
                if failure is not None:
                    raise failure
            finally:
                _OWNER.reset(token)

    def _check_audio(self, chunk):
        if not isinstance(chunk, AudioChunk) or len(chunk.pcm) > self._max_audio:
            raise ValueError("invalid or oversized audio chunk")

    async def _conversation(self, transport, stt, model, tts, activity=None):
        queue = asyncio.Queue(self._queue_size)
        control = asyncio.Lock()
        input_drained = False

        async def feed():
            speaking = False
            async with aclosing(transport.receive()) as incoming:
                async for chunk in incoming:
                    self._check_audio(chunk)
                    if activity is not None:
                        detected = await activity.detect(chunk)
                        if type(detected) is not bool:
                            raise ValueError("speech activity detector must return bool")
                        if detected and not speaking:
                            await queue.put(SpeechStarted())
                        speaking = detected
                    await queue.put(chunk)
            await queue.put(_EOF)

        async def audio():
            nonlocal input_drained
            while (chunk := await queue.get()) is not _EOF:
                if isinstance(chunk, SpeechStarted):
                    # Duplex recognizers may consume PCM in a separate task.
                    # A shared controller keeps old cleanup from clearing the
                    # output of a newer transcript while either task awaits.
                    async with control:
                        await self._interrupt(transport)
                    continue
                yield chunk
            input_drained = True

        async def listen():
            async with aclosing(audio()) as chunks, aclosing(stt.transcribe(chunks)) as events:
                async for event in events:
                    async with control:
                        if isinstance(event, SpeechStarted):
                            await self._interrupt(transport)
                        elif isinstance(event, Transcript):
                            if type(event.final) is not bool or not isinstance(event.text, str):
                                raise ValueError("invalid transcript event")
                            if len(event.text.encode("utf-8")) > 32000:
                                raise ValueError("transcript exceeds byte limit")
                            if not event.final or not event.text.strip():
                                continue
                            if not isinstance(event.speaker, str) or not 1 <= len(event.speaker) <= 128:
                                raise ValueError("invalid transcript speaker")
                            await self._interrupt(transport)
                            messages = [*self._history, {"role": "user", "content": event.text}]
                            if _bytes(messages) > self._max_history:
                                raise RuntimeError("voice history byte limit reached")
                            turn_id = uuid4().hex
                            await self._record(turn_id, "user", event.text, speaker=event.speaker)
                            self._history = messages
                            self._output_turn = turn_id
                            self._reply = asyncio.create_task(self._respond(transport, model, tts, turn_id, self._generation))
                        else:
                            raise ValueError("unsupported speech event")

        feeder, listener = asyncio.create_task(feed()), asyncio.create_task(listen())
        try:
            while True:
                watched = {t for t in (feeder, listener, self._reply) if t is not None and not t.done()}
                for task in (feeder, listener, self._reply):
                    if task is not None and task.done():
                        task.result()
                if listener.done() and (self._reply is None or self._reply.done()):
                    if not input_drained:
                        raise RuntimeError("speech recognizer ended before audio input")
                    break
                await asyncio.wait(watched, timeout=.01, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._generation += 1
            pending = [t for t in (feeder, listener, self._reply) if t is not None]
            for task in pending:
                _cancel_once(task)
            results = await asyncio.gather(*pending, return_exceptions=True)
            if self._output_turn is not None and (self._reply is None or self._reply.cancelled()
                                                 or self._reply.exception() is not None):
                await transport.clear(self._output_turn)
            failure = next((e for e in results if isinstance(e, Exception)), None)
            if failure is not None:
                raise failure

    async def _interrupt(self, transport):
        self._generation += 1
        reply, self._reply = self._reply, None
        cancelled = False
        if reply is not None:
            _cancel_once(reply)
            outcome, cancelled = await _settle(reply)
            if isinstance(outcome, Exception):
                raise outcome
        if self._output_turn is not None:
            await transport.clear(self._output_turn)
            self._output_turn = None
        if cancelled:
            raise asyncio.CancelledError()

    async def _record(self, turn_id, role, text, *, speaker=None):
        metadata = {"integration": "scone-voice", "session_id": self._session_id,
                    "capture_id": self._capture_id, "turn_id": turn_id, "role": role,
                    "representation": "aggregated_text",
                    "capture_status": "submitted" if role == "user" else "aggregated"}
        if speaker is not None:
            metadata["speaker"] = speaker
        if role == "assistant":
            metadata["completion_evidence"] = "adapter_end_and_output_accepted"
            metadata["playback"] = "unverified"
        try:
            async with asyncio.timeout(5):
                await self._memory.remember_many(self._space, [Record(
                    text, kind="conversation", source=self._session_id, metadata=metadata,
                    dedup_key=f"scone-voice:{self._capture_id}:{turn_id}:{role}",
                )])
        except (asyncio.CancelledError, TimeoutError) as exc:
            raise RuntimeError("voice capture is unconfirmed; inspect stored records before retrying") from exc
        self.stored_count += 1

    async def _context(self):
        messages, self.last_memory_receipt = await self._memory_context.prepare(self._history)
        return messages

    async def _respond(self, transport, model, tts, turn_id, generation):
        def current():
            if generation != self._generation or self._stop.is_set():
                raise asyncio.CancelledError()

        async def speak(text):
            current()
            emitted = False
            async with aclosing(tts.synthesize(text)) as output:
                async for chunk in output:
                    current()
                    self._check_audio(chunk)
                    await transport.send(chunk, turn_id)
                    emitted = True
                    current()
            if not emitted:
                raise RuntimeError("speech synthesis ended without audio output")

        async with asyncio.timeout(self._turn_timeout):
            messages = await self._context()
            current()
            parts, pending, completed, size = [], "", False, 0
            async with aclosing(model.respond(messages)) as response:
                async for event in response:
                    current()
                    if completed:
                        raise RuntimeError("model emitted data after completion")
                    if isinstance(event, TextDelta):
                        if not isinstance(event.text, str) or not event.text:
                            raise ValueError("invalid public text delta")
                        size += len(event.text.encode("utf-8"))
                        if size > self._max_reply:
                            raise RuntimeError("voice reply byte limit reached")
                        parts.append(event.text)
                        pending += event.text
                        while pending:
                            sentence = re.search(r"[.!?](?:\s|$)", pending)
                            end = sentence.end() if sentence else 240 if len(pending) >= 240 else 0
                            if not end:
                                break
                            await speak(pending[:end])
                            pending = pending[end:]
                    elif isinstance(event, ReplyCompleted):
                        completed = True
                    else:
                        raise ValueError("unsupported model event; only public text and completion are allowed")
            if not completed or not "".join(parts).strip():
                raise RuntimeError("model ended without explicit nonempty reply completion")
            if pending.strip():
                await speak(pending)
            current()
            text = "".join(parts)
            history = [*self._history, {"role": "assistant", "content": text}]
            if _bytes(history) > self._max_history:
                raise RuntimeError("voice history byte limit reached")
            await self._record(turn_id, "assistant", text)
            self._history = history
