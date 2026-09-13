"""Bounded, payload-free observation of one native agent invocation."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Literal
from uuid import uuid4

OperationKind = Literal['model', 'memory', 'custom']
EventKind = Literal[
    'turn_started',
    'turn_completed',
    'turn_paused',
    'turn_failed',
    'turn_cancelled',
    'operation_started',
    'operation_completed',
    'operation_failed',
    'operation_reused',
    'tool_proposed',
    'tool_result',
]
TerminalKind = Literal['turn_completed', 'turn_paused', 'turn_failed', 'turn_cancelled']


@dataclass(frozen=True)
class AgentProgressEvent:
    sequence: int
    invocation_id: str
    agent_id: str
    model_id: str
    binding: str
    kind: EventKind
    occurred_at: str
    elapsed_s: float
    operation_id: int | None = None
    operation_kind: OperationKind | None = None
    duration_s: float | None = None
    tool_index: int | None = None
    tool_name: str | None = None
    status: Literal['prepared', 'empty', 'unavailable'] | None = None
    error: str | None = None
    output_bytes: int | None = None
    origin: Literal['host', 'model'] | None = None
    reused: bool | None = None
    journal_reused: bool | None = None
    presentation_reused: bool | None = None


@dataclass(frozen=True)
class AgentProgressGap:
    invocation_id: str
    first_sequence: int
    last_sequence: int


class AgentEventStream:
    """One producer and one active reader; observation never controls execution.

    A slow reader loses the oldest buffered events and receives explicit sequence
    gaps. Closing/cancelling an iterator detaches that reader, not the agent. A new
    iterator can consume the remaining buffer. This stream is not durable storage.
    """

    def __init__(self, *, max_events: int = 128) -> None:
        if type(max_events) is not int or not 1 <= max_events <= 1024:
            raise ValueError('event capacity must be an integer in 1..1024')
        self._capacity = max_events
        self._buffer: deque[AgentProgressEvent] = deque()
        self._ready = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._emitter: ProgressEmitter | None = None
        self._reader = self._done = False
        self._next_read, self._dropped = 1, 0

    @property
    def dropped_events(self) -> int:
        return self._dropped

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError('agent event stream belongs to another event loop')
        self._loop = loop

    def _begin(self, agent_id: str, model_id: str, binding: str) -> ProgressEmitter:
        self._check_loop()
        if self._emitter is not None:
            raise ValueError('agent event stream already belongs to an invocation')
        self._emitter = ProgressEmitter(self, agent_id, model_id, binding)
        self._emitter.emit('turn_started')
        return self._emitter

    def _append(self, event: AgentProgressEvent) -> None:
        self._check_loop()
        if self._done:
            raise RuntimeError('agent event stream is finished')
        if len(self._buffer) == self._capacity:
            self._buffer.popleft()
            self._dropped += 1
        self._buffer.append(event)
        self._ready.set()

    async def __aiter__(self) -> AsyncIterator[AgentProgressEvent | AgentProgressGap]:
        self._check_loop()
        if self._reader:
            raise RuntimeError('agent event stream already has an active reader')
        self._reader = True
        try:
            while True:
                self._check_loop()
                if self._buffer:
                    first = self._buffer[0]
                    if first.sequence > self._next_read:
                        gap = AgentProgressGap(first.invocation_id, self._next_read, first.sequence - 1)
                        self._next_read = first.sequence
                        yield gap
                        continue
                    event = self._buffer.popleft()
                    self._next_read = event.sequence + 1
                    yield event
                elif self._done:
                    return
                else:
                    self._ready.clear()
                    await self._ready.wait()
        finally:
            self._reader = False


class ProgressEmitter:
    """Internal producer: receives host metadata, never provider payloads."""

    def __init__(self, stream: AgentEventStream, agent_id: str, model_id: str, binding: str) -> None:
        self._stream = stream
        self._agent, self._model, self._binding = agent_id, model_id, binding
        self._invocation = uuid4().hex
        self._entered = time.monotonic()
        self._sequence = self._operations = 0

    def emit(
        self,
        kind: EventKind,
        *,
        operation_id: int | None = None,
        operation_kind: OperationKind | None = None,
        duration_s: float | None = None,
        tool_index: int | None = None,
        tool_name: str | None = None,
        status: Literal['prepared', 'empty', 'unavailable'] | None = None,
        error: str | None = None,
        output_bytes: int | None = None,
        origin: Literal['host', 'model'] | None = None,
        reused: bool | None = None,
        journal_reused: bool | None = None,
        presentation_reused: bool | None = None,
    ) -> None:
        self._sequence += 1
        self._stream._append(
            AgentProgressEvent(
                self._sequence,
                self._invocation,
                self._agent,
                self._model,
                self._binding,
                kind,
                datetime.now(timezone.utc).isoformat(),
                max(0.0, time.monotonic() - self._entered),
                operation_id,
                operation_kind,
                duration_s,
                tool_index,
                tool_name,
                status,
                error,
                output_bytes,
                origin,
                reused,
                journal_reused,
                presentation_reused,
            )
        )

    def finish(self, kind: TerminalKind) -> None:
        self.emit(kind)
        self._stream._done = True
        self._stream._ready.set()


class _Operation:
    def __init__(
        self, emitter: ProgressEmitter | None, kind: OperationKind, name: str | None, tool_index: int | None
    ) -> None:
        self._emitter, self._kind, self._name, self._tool_index = emitter, kind, name, tool_index
        self._started: float | None = None
        self._id = 0
        if emitter is not None:
            emitter._operations += 1
            self._id = emitter._operations

    @property
    def dispatched(self) -> bool:
        return self._started is not None

    def started(self) -> None:
        if self._started is not None:
            raise RuntimeError('agent operation already started')
        self._started = time.monotonic()
        self._emit('operation_started')

    def _emit(self, kind: EventKind, duration: float | None = None) -> None:
        if self._emitter is not None:
            self._emitter.emit(
                kind,
                operation_id=self._id,
                operation_kind=self._kind,
                tool_name=self._name,
                tool_index=self._tool_index,
                duration_s=duration,
            )

    def finish(self, failed: bool) -> None:
        if self._started is None:
            if not failed:
                self._emit('operation_reused')
            return
        self._emit(
            'operation_failed' if failed else 'operation_completed',
            max(0.0, time.monotonic() - self._started),
        )


@contextmanager
def operation_scope(
    emitter: ProgressEmitter | None,
    kind: OperationKind,
    tool_name: str | None = None,
    tool_index: int | None = None,
) -> Iterator[_Operation]:
    operation = _Operation(emitter, kind, tool_name, tool_index)
    try:
        yield operation
    except BaseException:
        operation.finish(True)
        raise
    else:
        operation.finish(False)
