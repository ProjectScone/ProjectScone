"""Ordered agent operation receipts over an active encrypted step checkpoint."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import hashlib
import json
import math
import sqlite3
import time
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .workflow import JSONValue, StepCheckpoints, WorkflowError, WorkflowPaused, _encode, _name

OperationKind = Literal['model', 'custom', 'memory']
_KEY = 'agent-turn-v1'


class TurnJournalError(RuntimeError):
    """A fixed diagnostic code, never callback messages or checkpoint contents."""


class TurnJournalPaused(RuntimeError):
    """A host scheduling yield at a sealed operation boundary, not approval."""
    def __init__(self, checkpoint: str):
        super().__init__('turn_paused')
        self.pause = WorkflowPaused(checkpoint)


class _Event(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)
    kind: OperationKind
    request: str = Field(pattern=r'^[0-9a-f]{64}$')
    status: Literal['started', 'completed']
    result: JsonValue = None

    @model_validator(mode='after')
    def result_presence(self) -> Self:
        if ('result' in self.model_fields_set) != (self.status == 'completed'):
            raise ValueError('invalid operation result')
        return self


class _State(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)
    version: int = Field(ge=1, le=1)
    binding: str = Field(pattern=r'^[0-9a-f]{64}$')
    configuration: str | None
    elapsed: float = Field(ge=0, le=86400)
    finished: bool
    events: list[_Event] = Field(max_length=64)


def _bytes(value: object, maximum: int) -> bytes:
    try:
        return _encode(cast(JSONValue, value), maximum)
    except (WorkflowError, ValueError, TypeError, RecursionError):
        raise TurnJournalError('invalid_journal_payload') from None


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


class ToolTurnJournal:
    """One activation of a trusted native turn; construct anew after each pause.

    Binding must identify the host-selected model and its revision. The loop
    additionally binds its messages, tools, scope and limits. Completed operations
    replay in order; started operations cannot be automatically retried. Elapsed
    time counts active execution, excluding time between explicit activations.
    """

    def __init__(self, checkpoints: StepCheckpoints, *, binding: JSONValue,
                 timeout_s: float = 600.0, max_operations: int = 64,
                 max_bytes: int = 4 * 1024 * 1024, checkpoint: str = _KEY,
                 max_new_operations: int | None = None) -> None:
        if (not isinstance(checkpoints, StepCheckpoints) or type(timeout_s) not in (float, int)
                or not 0.01 <= timeout_s <= 600 or not math.isfinite(timeout_s)
                or type(max_operations) is not int or not 1 <= max_operations <= 64
                or type(max_bytes) is not int or not 1024 <= max_bytes <= 8 * 1024 * 1024):
            raise TurnJournalError('invalid_journal_configuration')
        if max_new_operations is not None and (type(max_new_operations) is not int or not 1 <= max_new_operations <= 64):
            raise TurnJournalError('invalid_journal_configuration')
        self._quantum, self._new_operations = max_new_operations, 0
        _name(checkpoint)
        self._checkpoints, self._key = checkpoints, checkpoint
        self._maximum, self._limit, self._timeout = max_bytes, max_operations, float(timeout_s)
        self._cursor, self._busy, self._failed = 0, False, False
        identity = hashlib.sha256(_bytes({'binding': binding, 'timeout_s': self._timeout,
            'max_operations': max_operations, 'max_bytes': max_bytes}, max_bytes)).hexdigest()
        self._saved = self._read()
        if self._saved is None:
            self._state = _State(version=1, binding=identity, configuration=None,
                                 elapsed=0.0, finished=False, events=[])
        else:
            try:
                if len(self._saved) > max_bytes:
                    raise ValueError()
                decoded = json.loads(self._saved, object_pairs_hook=_pairs)
                _bytes(decoded, max_bytes)
                self._state = _State.model_validate(decoded)
                if (len(self._state.events) > max_operations
                        or any(row.status != 'completed' for row in self._state.events[:-1])
                        or (self._state.configuration is not None and
                            (len(self._state.configuration) != 64 or any(c not in '0123456789abcdef' for c in self._state.configuration)))):
                    raise ValueError()
            except (ValueError, TypeError, RecursionError, TurnJournalError):
                raise TurnJournalError('journal_integrity') from None
            if self._state.binding != identity:
                raise TurnJournalError('binding_mismatch')
            if any(row.status == 'started' for row in self._state.events):
                raise TurnJournalError('outcome_unknown')
        self._base_elapsed = self._state.elapsed
        self._entered = time.monotonic()

    def _read(self) -> bytes | None:
        try:
            return self._checkpoints.get(self._key)
        except (OSError, sqlite3.Error, WorkflowError):
            raise TurnJournalError('journal_unavailable') from None

    def _check(self) -> None:
        active = asyncio.current_task()
        if active is not None and active.cancelling():
            raise asyncio.CancelledError
        if self._failed:
            raise TurnJournalError('journal_inactive')
        if self._read() != self._saved:
            self._failed = True
            raise TurnJournalError('journal_changed')
        if self.remaining_s <= 0:
            raise TurnJournalError('deadline')

    def check_current(self) -> None:
        """Recheck deadline, cancellation and checkpoint lease, even inside an operation."""
        self._check()

    @property
    def elapsed_s(self) -> float:
        return self._base_elapsed + time.monotonic() - self._entered

    @property
    def remaining_s(self) -> float:
        return self._timeout - (self._base_elapsed + time.monotonic() - self._entered)

    def _write(self, state: _State) -> None:
        self._check()
        candidate = state.model_copy(deep=True)
        candidate.elapsed = self._base_elapsed + time.monotonic() - self._entered
        payload = _bytes(candidate.model_dump(exclude_unset=True), self._maximum)
        try:
            self._checkpoints.put(self._key, payload)
        except BaseException as error:
            self._failed = True
            if isinstance(error, (OSError, sqlite3.Error, WorkflowError)):
                raise TurnJournalError('journal_unavailable') from None
            raise
        self._state, self._saved = candidate, payload
        self._check()

    def bind_configuration(self, value: JSONValue) -> None:
        self._check()
        if self._busy or self._cursor:
            raise TurnJournalError('journal_busy')
        digest = hashlib.sha256(_bytes(value, self._maximum)).hexdigest()
        if self._state.configuration is not None:
            if self._state.configuration != digest:
                raise TurnJournalError('binding_mismatch')
            return
        if self._state.events or self._state.finished:
            raise TurnJournalError('binding_mismatch')
        self._write(self._state.model_copy(update={'configuration': digest}, deep=True))

    def operation_identity(self, kind: OperationKind, request: JSONValue) -> tuple[str, bool]:
        """Inspect the next exact operation without admitting it or advancing replay."""
        self._check()
        if self._busy:
            raise TurnJournalError('journal_busy')
        if kind not in ('model', 'custom', 'memory'):
            raise TurnJournalError('invalid_operation')
        digest = hashlib.sha256(_bytes(request, self._maximum)).hexdigest()
        replay = self._cursor < len(self._state.events)
        if replay:
            event = self._state.events[self._cursor]
            if event.kind != kind or event.request != digest:
                raise TurnJournalError('operation_mismatch')
        else:
            if self._state.finished:
                raise TurnJournalError('turn_finished')
            if self._cursor >= self._limit:
                raise TurnJournalError('operation_limit')
            if self._quantum is not None and self._new_operations >= self._quantum:
                self.pause()
                raise TurnJournalPaused(self._key)
        identity = hashlib.sha256(_bytes(['tool-operation-v1', self._state.binding,
            self._state.configuration, self._cursor, kind, digest], self._maximum)).hexdigest()
        self._check()
        return identity, replay

    async def execute(self, kind: OperationKind, request: JSONValue,
                      operation: Callable[[], Awaitable[JSONValue]]) -> JSONValue:
        self._check()
        if self._busy:
            raise TurnJournalError('journal_busy')
        if kind not in ('model', 'custom', 'memory') or not callable(operation):
            raise TurnJournalError('invalid_operation')
        digest = hashlib.sha256(_bytes(request, self._maximum)).hexdigest()
        if self._cursor < len(self._state.events):
            event = self._state.events[self._cursor]
            if event.kind != kind or event.request != digest:
                raise TurnJournalError('operation_mismatch')
            saved_result = cast(JSONValue, json.loads(_bytes(event.result, self._maximum)))
            self._check()
            self._cursor += 1
            return saved_result
        if self._state.finished:
            raise TurnJournalError('turn_finished')
        if self._cursor >= self._limit:
            raise TurnJournalError('operation_limit')
        if self._quantum is not None and self._new_operations >= self._quantum:
            self.pause()
            raise TurnJournalPaused(self._key)
        self._busy = True
        try:
            events = [*self._state.events, _Event(kind=kind, request=digest, status='started')]
            self._write(self._state.model_copy(update={'events': events}, deep=True))
            try:
                async with asyncio.timeout(self.remaining_s):
                    result = await operation()
                self._check()
                clean = cast(JSONValue, json.loads(_bytes(result, self._maximum)))
            except asyncio.CancelledError:
                raise
            except Exception:
                raise TurnJournalError('operation_failed') from None
            events[-1] = _Event(kind=kind, request=digest, status='completed', result=clean)
            self._write(self._state.model_copy(update={'events': events}, deep=True))
            self._cursor += 1
            self._new_operations += 1
            detached = cast(JSONValue, json.loads(_bytes(clean, self._maximum)))
            self._check()
            return detached
        except BaseException:
            self._failed = True
            raise
        finally:
            self._busy = False

    def _boundary(self) -> None:
        self._check()
        if self._busy:
            raise TurnJournalError('journal_busy')
        if self._cursor != len(self._state.events):
            raise TurnJournalError('unread_operations')

    def pause(self) -> WorkflowPaused:
        """Seal an operation boundary; this does not authorize any tool."""
        self._boundary()
        self._write(self._state)
        return WorkflowPaused(self._key)

    def finish(self) -> None:
        self._boundary()
        self._write(self._state.model_copy(update={'finished': True}, deep=True))
