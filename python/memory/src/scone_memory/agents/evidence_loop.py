"""Bounded, model-neutral search/trace turns with final source revalidation.

No text is published during tool selection. Providers implement one complete
turn; the host owns tools, scope, budgets, and the final publication boundary.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from ..integrations.scoped_tools import ScopedMemoryTools
from ..integrations.read_memory import ReadMemoryArgs
from .tool_evidence import PreparedToolEvidence


class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9._:-]+$')
    name: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    arguments: dict[str, object]


class ToolStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    content: str = Field(default='', max_length=64000)
    calls: tuple[ToolCall, ...] = Field(default=(), max_length=8)


class ToolLoopLimits(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)
    max_tool_calls: int = Field(default=4, ge=1, le=16)
    max_tool_rounds: int = Field(default=4, ge=1, le=16)
    timeout_s: float = Field(default=120.0, ge=0.01, le=600.0)
    max_transcript_bytes: int = Field(default=256000, ge=1024, le=1000000)
    max_tool_bytes: int = Field(default=128000, ge=512, le=512000)
    max_reply_bytes: int = Field(default=16000, ge=1, le=64000)


class ToolModel(Protocol):
    async def complete(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ToolStep: ...


class ToolOutcome(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    call_id: str
    name: Literal['search_memory', 'trace_memory', 'read_memory', 'unknown_tool']
    status: Literal['prepared', 'empty', 'unavailable']
    error: str | None
    output_bytes: int
    origin: Literal['host', 'model'] = 'model'
    reused: bool = False


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    model_calls: int
    tool_calls: int
    evidence_ids: tuple[str, ...]
    source_status: str
    evidence_packets: tuple[str, ...]
    tool_outcomes: tuple[ToolOutcome, ...]
    _validator: Callable[[], Awaitable[bool]] = field(repr=False, compare=False)
    verified_accuracy: bool = False
    deadline: float | None = field(default=None, repr=False, compare=False)

    async def validate(self) -> bool:
        """Recheck before later capture, within the original turn deadline."""
        return await self._validator()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _denied(reason: str) -> str:
    return _json({'ok': False, 'status': 'unavailable', 'error': reason, 'verified_accuracy': False})


def _read_key(call: ToolCall) -> tuple[int, int, int] | None:
    if call.name != 'read_memory':
        return None
    try:
        args = ReadMemoryArgs.model_validate(call.arguments)
    except ValueError:
        return None
    return args.chunk_id, args.before, args.after


def _covers_same_source(evidence: PreparedToolEvidence, key: tuple[int, int, int]) -> bool:
    packet = json.loads(evidence.payload)
    coverage = packet.get('coverage', {})
    items = packet.get('items', [])
    if (not isinstance(coverage, dict) or coverage.get('mode') != 'chunk_window'
            or coverage.get('truncated') is not False or coverage.get('has_more_after') is not False
            or not isinstance(items, list) or not 1 <= len(items) <= 9):
        return False
    ordinals = coverage.get('returned_ordinals')
    if (not isinstance(ordinals, list) or any(type(value) is not int for value in ordinals)
            or ordinals != list(range(len(items)))):
        return False
    for ordinal, item in enumerate(items):
        if isinstance(item, dict) and type(item.get('chunk_id')) is int and item['chunk_id'] == key[0]:
            return key[1] >= ordinal and ordinal + key[2] >= len(items) - 1
    return False


class EvidenceToolLoop:
    """One bounded turn. Call IDs are unique across rounds; tools are read-only.

    At most max_tool_rounds tool-bearing requests plus one final request are
    sent. Unexecuted calls receive explicit denials, preserving protocol pairs.
    The final request has no tools. Tool/call limits count attempts, not just
    successful retrievals. Deadline cancellation is cooperative with adapters.
    """

    def __init__(self, model: ToolModel, tools: ScopedMemoryTools, *, limits: ToolLoopLimits | None = None,
                 initial_search: bool = False) -> None:
        if type(initial_search) is not bool:
            raise ValueError('initial_search must be a boolean')
        self._initial_search = initial_search
        self._model, self._tools = model, tools
        self._limits = ToolLoopLimits.model_validate((limits or ToolLoopLimits()).model_dump())

    async def run(self, messages: list[dict[str, str]]) -> ToolLoopResult:
        limits = self._limits
        # Only ordinary host messages may seed a turn. Models cannot inject a
        # preexisting tool result or select tools by altering the input history.
        if not messages or any(set(row) != {'role', 'content'} or row['role'] not in ('system', 'user', 'assistant')
                               or not isinstance(row['content'], str) for row in messages):
            raise ValueError('invalid tool conversation messages')
        if self._initial_search and (messages[-1]['role'] != 'user' or not messages[-1]['content'].strip()
                or len(messages[-1]['content'].encode()) > 8000):
            raise ValueError('initial memory query requires a final user message of 1..8000 bytes')
        transcript = cast(list[dict[str, object]], json.loads(_json(messages)))
        seen: set[str] = set()
        known_facts: set[int] = set()
        known_chunks: set[int] = set()
        prepared: list[PreparedToolEvidence] = []
        reads: dict[tuple[int, int, int], tuple[PreparedToolEvidence, int]] = {}
        outcomes: list[ToolOutcome] = []
        calls = rounds = model_calls = tool_bytes = 0
        deadline = time.monotonic() + limits.timeout_s

        def check_deadline() -> None:
            active = asyncio.current_task()
            if active is not None and active.cancelling():
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError('tool loop deadline exceeded')

        async def record_call(call: ToolCall, origin: Literal['host', 'model']) -> None:
            nonlocal calls, tool_bytes
            reused = False
            if calls >= limits.max_tool_calls:
                payload = _denied('tool_budget')
            else:
                calls += 1
                seed = call.arguments.get('seed_fact_id')
                if call.name == 'trace_memory' and (type(seed) is not int or seed not in known_facts):
                    payload = _denied('search_for_seed_first')
                elif call.name == 'read_memory' and (type(call.arguments.get('chunk_id')) is not int
                        or call.arguments['chunk_id'] not in known_chunks):
                    payload = _denied('search_for_chunk_first')
                else:
                    key = _read_key(call)
                    cached = reads.get(key) if key is not None else None
                    if cached is None and key is not None:
                        cached = next((entry for entry in reads.values() if _covers_same_source(entry[0], key)), None)
                    if cached is not None:
                        valid = await asyncio.create_task(cached[0].validate())
                        check_deadline()
                        if not valid:
                            raise RuntimeError('tool evidence changed before reuse')
                        reused = True
                        payload = _json({'ok':True, 'status':'prepared', 'verified_accuracy':False,
                            'reuse':{'tool_result_number':cached[1], 'new_evidence_count':0},
                            'message':'These passages are already in an earlier tool result above. '
                                      'Use that evidence, choose a different read for missing context, or answer.'})
                        if tool_bytes + len(payload.encode()) > limits.max_tool_bytes:
                            payload = _denied('tool_output_budget')
                            reused = False
                    else:
                        evidence = await asyncio.create_task(self._tools.prepare(call.name, call.arguments))
                        check_deadline()
                        size = len(evidence.payload.encode())
                        if tool_bytes + size > limits.max_tool_bytes:
                            payload = _denied('tool_output_budget')
                        else:
                            payload = evidence.payload
                            prepared.append(evidence)
                            if key is not None and evidence.evidence_ids:
                                reads[key] = (evidence, len(outcomes) + 1)
                            known_facts.update(int(item.partition(':')[2]) for item in evidence.evidence_ids
                                               if item.startswith('fact:'))
                            known_chunks.update(int(item.partition(':')[2]) for item in evidence.evidence_ids
                                                if item.startswith('chunk:'))
            tool_bytes += len(payload.encode())
            if tool_bytes > limits.max_tool_bytes:
                raise RuntimeError('tool output byte limit')
            transcript.append({'role': 'tool', 'tool_call_id': call.id, 'content': payload})
            packet = json.loads(payload)
            error = packet.get('error')
            codes = {'unknown_tool', 'invalid_arguments', 'timeout', 'store_error', 'retrieval_failed',
                     'output_bytes', 'evidence_unavailable', 'tool_budget', 'tool_output_budget',
                     'search_for_seed_first', 'search_for_chunk_first', 'unsupported_chunk_window'}
            # Model-authored IDs/unknown names can contain source text.
            # Keep them only in the transient provider protocol.
            name = cast(Literal['search_memory', 'trace_memory', 'read_memory', 'unknown_tool'],
                        call.name if call.name in ('search_memory', 'trace_memory', 'read_memory') else 'unknown_tool')
            outcomes.append(ToolOutcome(call_id=f'tool-{len(outcomes) + 1}', name=name, status=packet['status'],
                error=error if isinstance(error, str) and error in codes else None,
                output_bytes=len(payload.encode()), origin=origin, reused=reused))

        async with asyncio.timeout(limits.timeout_s):
            if self._initial_search:
                check_deadline()
                if len(_json(transcript).encode()) > limits.max_transcript_bytes:
                    raise RuntimeError('tool transcript byte limit')
                initial = ToolCall(id='host-initial-search', name='search_memory',
                    arguments={'query':messages[-1]['content'], 'limit':5})
                seen.add(initial.id)
                transcript.append({'role':'assistant', 'content':None, 'tool_calls':[
                    {'id':initial.id, 'type':'function', 'function':{
                        'name':initial.name, 'arguments':_json(initial.arguments)}}]})
                await record_call(initial, 'host')
                if outcomes[-1].status == 'unavailable':
                    raise RuntimeError('initial memory retrieval unavailable')
            while True:
                check_deadline()
                if len(_json(transcript).encode()) > limits.max_transcript_bytes:
                    raise RuntimeError('tool transcript byte limit')
                enabled = calls < limits.max_tool_calls and rounds < limits.max_tool_rounds
                schemas = self._tools.openai() if enabled else []
                if not known_facts:
                    schemas = [row for row in schemas if cast(dict[str, object], row['function'])['name'] != 'trace_memory']
                if not known_chunks:
                    schemas = [row for row in schemas if cast(dict[str, object], row['function'])['name'] != 'read_memory']
                try:
                    # Independent copies stop an adapter from mutating the
                    # authoritative transcript, schemas, or paired call IDs.
                    step = await asyncio.create_task(self._model.complete(json.loads(_json(transcript)), schemas))
                    check_deadline()
                except (asyncio.CancelledError, TimeoutError):
                    raise
                except Exception:
                    raise RuntimeError('tool model unavailable') from None
                model_calls += 1
                try:
                    step = ToolStep.model_validate(step.model_dump())
                    if len(_json(step.model_dump()).encode()) > 128000:
                        raise ValueError()
                    ids = [call.id for call in step.calls]
                    if len(set(ids)) != len(ids) or seen.intersection(ids) or (step.calls and not enabled):
                        raise ValueError()
                    if any(len(_json(call.arguments).encode()) > 16000 for call in step.calls):
                        raise ValueError()
                except Exception:
                    raise RuntimeError('invalid tool protocol') from None
                if not step.calls:
                    if not step.content.strip() or len(step.content.encode()) > limits.max_reply_bytes:
                        raise RuntimeError('tool reply byte limit or empty reply')
                    retained = tuple(prepared)

                    async def validate() -> bool:
                        async with asyncio.timeout_at(deadline):
                            check_deadline()
                            for evidence in retained:
                                valid = await asyncio.create_task(evidence.validate())
                                check_deadline()
                                if not valid:
                                    return False
                            return True

                    if not await validate():
                        raise RuntimeError('tool evidence changed before publication')
                    evidence_ids = tuple(dict.fromkeys(key for evidence in prepared for key in evidence.evidence_ids))
                    return ToolLoopResult(step.content, model_calls, calls, evidence_ids,
                                          'retained' if evidence_ids else 'none',
                                          tuple(evidence.payload for evidence in retained if evidence.evidence_ids),
                                          tuple(outcomes), validate, deadline=deadline)
                rounds += 1
                seen.update(ids)
                transcript.append({'role': 'assistant', 'content': step.content or None, 'tool_calls': [
                    {'id': call.id, 'type': 'function', 'function': {'name': call.name, 'arguments': _json(call.arguments)}}
                    for call in step.calls]})
                for call in step.calls:
                    await record_call(call, 'model')
