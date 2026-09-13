"""Host-registered application functions, separate from retained memory evidence."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import inspect
import json
import math
import re
import threading
import time
from types import MappingProxyType

from ..core.validation import check_space
from ..realtime.output_schema import accepts_schema, compile_schema
from ..retrieval.recall_scope import RecallScope

_RESERVED = {'search_memory', 'trace_memory', 'read_memory', 'compute_memory',
             'answer', 'unknown_tool', 'custom_tool'}
_dispatch_abort: ContextVar[threading.Event | None] = ContextVar('scone_tool_dispatch_abort', default=None)


def _result_packet(result: object) -> dict[str, object]:
    return {'ok': True, 'status': 'prepared', 'verified_accuracy': False,
            'source_status': 'unverified', 'result': result}


_MIN_RESULT_BYTES = len(json.dumps(_result_packet(0), separators=(',', ':')).encode())


def _encoded(value: object, limit: int) -> str:
    pending: list[tuple[object, int]] = [(value, 0)]
    count = size = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 32 or count > 4096:
            raise ValueError('application tool JSON structure limit')
        if type(item) is dict:
            assert isinstance(item, dict)
            if len(item) > 4096 or any(type(key) is not str for key in item):
                raise ValueError('application tool JSON keys must be strings')
            size += 2 + max(0, len(item) - 1)
            for key in item:
                if len(key) > limit:
                    raise ValueError('application tool JSON byte limit')
                size += len(json.dumps(key, ensure_ascii=False).encode()) + 1
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            assert isinstance(item, list)
            if len(item) > 4096:
                raise ValueError('application tool JSON structure limit')
            size += 2 + max(0, len(item) - 1)
            pending.extend((child, depth + 1) for child in item)
        elif type(item) not in (str, bool, int, float, type(None)):
            raise ValueError('application tool result must contain JSON values')
        else:
            if isinstance(item, str) and len(item) > limit:
                raise ValueError('application tool JSON byte limit')
            size += len(json.dumps(item, ensure_ascii=False, allow_nan=False).encode())
        if size > limit:
            raise ValueError('application tool JSON byte limit')
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(encoded.encode()) > limit:
        raise ValueError('application tool JSON byte limit')
    return encoded


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: object) -> object:
    count = 0

    def visit(item: object, depth: int) -> object:
        nonlocal count
        count += 1
        if depth > 32 or count > 4096:
            raise ValueError('application tool schema structure limit')
        if isinstance(item, (dict, MappingProxyType, list, tuple)) and len(item) > 4096:
            raise ValueError('application tool schema structure limit')
        if isinstance(item, (dict, MappingProxyType)):
            return {key: visit(child, depth + 1) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child, depth + 1) for child in item]
        return item

    return visit(value, 0)


def _dispose(value: object) -> None:
    if inspect.iscoroutine(value):
        value.close()
    elif isinstance(value, asyncio.Future):
        value.cancel()


@dataclass(frozen=True)
class _WorkerOutcome:
    value: object = None
    error: BaseException | None = None


async def _run_worker(function: Callable[[], object]) -> _WorkerOutcome:
    try:
        return _WorkerOutcome(value=await asyncio.to_thread(function))
    except BaseException as error:
        # A shielded failed task can log its exception after its caller cancels.
        return _WorkerOutcome(error=error)


def _discard_worker(task: asyncio.Task[_WorkerOutcome]) -> None:
    if task.cancelled():
        return
    try:
        _dispose(task.result().value)
    except Exception:
        pass


def _check_context(context: ToolContext) -> None:
    active = asyncio.current_task()
    if active is not None and active.cancelling():
        raise asyncio.CancelledError()
    if time.monotonic() >= context.deadline:
        raise RuntimeError('application tool deadline exceeded')


def _check_dispatch(context: ToolContext) -> None:
    """Check a nested synchronous dispatch without changing public context data."""
    aborted = _dispatch_abort.get()
    if (aborted is not None and aborted.is_set()) or time.monotonic() >= context.deadline:
        raise RuntimeError('application tool turn ended before worker dispatch')


@dataclass(frozen=True)
class ToolContext:
    """Detached invocation scope; trusted handlers enforce their own resources.

    This contains no memory engine, credential or implicit authorization for an
    external service. The original turn deadline also bounds handler execution.
    """

    space: str
    scope: RecallScope
    exclude_session_id: str | None
    deadline: float

    def __post_init__(self) -> None:
        check_space(self.space)
        if not isinstance(self.scope, RecallScope) or type(self.deadline) not in (float, int) or not math.isfinite(self.deadline):
            raise ValueError('invalid application tool context')
        object.__setattr__(self, 'scope', RecallScope.validated(**self.scope.kwargs()))


@dataclass(frozen=True)
class AgentTool:
    """An explicit application function and its versioned JSON argument contract.

    Handlers accept (arguments, context) and return a JSON value, synchronously
    or asynchronously. Hosts must change revision when implementation or external
    configuration changes. Handler code is trusted, not sandboxed by this class.
    """

    name: str
    description: str
    revision: str
    parameters: Mapping[str, object]
    handler: Callable[[dict[str, object], ToolContext], object] = field(repr=False, compare=False)
    max_output_bytes: int = 16000
    _compiled: dict[str, object] = field(init=False, repr=False, compare=False)
    _parameters_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.name) is None
                or self.name in _RESERVED or not isinstance(self.revision, str)
                or re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', self.revision) is None):
            raise ValueError('invalid application tool name or revision')
        if (not isinstance(self.description, str) or not self.description.strip()
                or len(self.description) > 4000 or len(self.description.encode()) > 4000 or not callable(self.handler)
                or type(self.max_output_bytes) is not int or not 1 <= self.max_output_bytes <= 64000):
            raise ValueError('invalid application tool metadata or output budget')
        try:
            parameters = _thaw(self.parameters) if isinstance(self.parameters, MappingProxyType) else self.parameters
            encoded = _encoded(parameters, 32768)
            compiled = compile_schema(parameters)
        except ImportError:
            raise ValueError('application tool schemas require the structured-output extra') from None
        object.__setattr__(self, 'parameters', _freeze(json.loads(encoded)))
        object.__setattr__(self, '_parameters_json', encoded)
        object.__setattr__(self, '_compiled', compiled)

    def snapshot(self) -> AgentTool:
        return AgentTool(self.name, self.description, self.revision, json.loads(self._parameters_json),
                         self.handler, self.max_output_bytes)

    def info(self) -> dict[str, object]:
        return {'name': self.name, 'description': self.description, 'revision': self.revision,
                'parameters': json.loads(self._parameters_json), 'max_output_bytes': self.max_output_bytes}

    def openai(self) -> dict[str, object]:
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description,
                                               'parameters': deepcopy(self._compiled)}}

    async def invoke(self, arguments: dict[str, object], context: ToolContext, *,
                     remaining_output_bytes: int | None = None) -> str:
        """Reject invalid arguments before execution; ambiguous handler errors stop.

        Sync handlers run in a worker thread. Cancellation cannot undo effects or
        force-stop a thread. No error here triggers an automatic handler retry.
        """
        _check_context(context)
        if remaining_output_bytes is not None and (type(remaining_output_bytes) is not int or remaining_output_bytes < 0):
            raise ValueError('invalid application tool remaining output budget')
        output_limit = min(self.max_output_bytes, remaining_output_bytes) if remaining_output_bytes is not None else self.max_output_bytes
        if output_limit < _MIN_RESULT_BYTES:
            raise RuntimeError('application tool output budget exhausted before execution')
        try:
            encoded = _encoded(arguments, 16000)
        except (ValueError, RecursionError):
            encoded = ''
        accepted = bool(encoded) and accepts_schema(encoded, self._compiled)
        _check_context(context)
        if not accepted:
            return '{"ok":false,"status":"unavailable","error":"invalid_arguments","verified_accuracy":false}'
        detached: dict[str, object] = json.loads(encoded)
        try:
            async with asyncio.timeout_at(context.deadline):
                if inspect.iscoroutinefunction(self.handler):
                    result: object = self.handler(detached, context)
                else:
                    aborted = threading.Event()

                    def execute() -> object:
                        token = _dispatch_abort.set(aborted)
                        try:
                            _check_dispatch(context)
                            return self.handler(detached, context)
                        finally:
                            _dispatch_abort.reset(token)

                    worker = asyncio.create_task(_run_worker(execute))
                    try:
                        outcome = await asyncio.shield(worker)
                    except BaseException:
                        aborted.set()
                        worker.add_done_callback(_discard_worker)
                        raise
                    if outcome.error is not None:
                        raise outcome.error
                    result = outcome.value
                try:
                    _check_context(context)
                except BaseException:
                    _dispose(result)
                    raise
                if inspect.isawaitable(result):
                    result = await result
                _check_context(context)
                payload = _encoded(_result_packet(result), output_limit)
                _check_context(context)
                return payload
        except asyncio.CancelledError:
            raise
        except Exception:
            raise RuntimeError('application tool execution failed; outcome may be unknown') from None


def snapshot_tools(tools: Sequence[AgentTool]) -> tuple[AgentTool, ...]:
    if len(tools) > 32 or any(not isinstance(tool, AgentTool) for tool in tools):
        raise ValueError('application tools require at most 32 registrations')
    copied = tuple(tool.snapshot() for tool in tools)
    if len({tool.name for tool in copied}) != len(copied):
        raise ValueError('duplicate application tool name')
    if sum(len(json.dumps(tool.openai(), ensure_ascii=False).encode()) for tool in copied) > 128000:
        raise ValueError('application tool metadata byte limit')
    return copied
