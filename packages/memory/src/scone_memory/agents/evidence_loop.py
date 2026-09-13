"""Bounded, model-neutral search/trace turns with final source revalidation.

No text is published during tool selection. Providers implement one complete
turn; the host owns tools, scope, budgets, and the final publication boundary.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from ..integrations.scoped_tools import ScopedMemoryTools
from ..integrations.read_memory import ReadMemoryArgs
from ..realtime.answer_requirements import AnswerRequirements, validated_requirements
from ..retrieval.computation import ComputeMemoryArgs
from .tool_evidence import PreparedToolEvidence
from .custom_tools import AgentTool, snapshot_tools
from .turn_journal import ToolTurnJournal, TurnJournalError, TurnJournalPaused
from .workflow import JSONValue
from .usage import ModelTokenUsage, ToolTokenUsage


if TYPE_CHECKING:
    from .approval_context import BoundApproval


class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9._:-]+$')
    name: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    arguments: dict[str, object]


class ToolStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    content: str = Field(default='', max_length=64000)
    calls: tuple[ToolCall, ...] = Field(default=(), max_length=8)
    usage: ModelTokenUsage = Field(default_factory=ModelTokenUsage)


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


class ConstrainedToolModel(ToolModel, Protocol):
    """Optional provider capability; the host still validates returned text."""

    async def complete_with_requirements(self, messages: list[dict[str, object]],
        tools: list[dict[str, object]], requirements: AnswerRequirements) -> ToolStep: ...


class ToolOutcome(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    call_id: str
    name: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
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
    usage: ToolTokenUsage = field(default_factory=ToolTokenUsage)

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


def _computation_denial(call: ToolCall, known_chunks: set[int]) -> str | None:
    if call.name != 'compute_memory':
        return None
    try:
        args = ComputeMemoryArgs.model_validate(call.arguments)
    except ValueError:
        return 'invalid_arguments'
    if any(row.chunk_id not in known_chunks for row in args.left + args.right):
        return 'search_for_chunk_first'
    return None


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


def _accepted_step(step: ToolStep, seen: set[str], enabled: bool) -> ToolStep:
    try:
        clean = ToolStep.model_validate(step.model_dump(warnings='error'))
        if len(_json(clean.model_dump(exclude={'usage'})).encode()) > 128000:
            raise ValueError()
        ids = [call.id for call in clean.calls]
        if len(set(ids)) != len(ids) or seen.intersection(ids) or (clean.calls and not enabled):
            raise ValueError()
        if any(len(_json(call.arguments).encode()) > 16000 for call in clean.calls):
            raise ValueError()
        return clean
    except Exception:
        raise RuntimeError('invalid tool protocol') from None


class EvidenceToolLoop:
    """One bounded turn. Memory tools are read-only; application tools are host code.

    At most max_tool_rounds tool-bearing requests plus one final request are
    sent. Unexecuted calls receive explicit denials, preserving protocol pairs.
    The final request has no tools. Tool/call limits count attempts, not just
    successful retrievals. Deadline cancellation is cooperative with adapters.
    Optional host answer requirements are supplied on every model request and
    checked before returning text. This checks format, not factual correctness.
    """

    def __init__(self, model: ToolModel, tools: ScopedMemoryTools, *, limits: ToolLoopLimits | None = None,
                 initial_search: bool = False, answer_requirements: AnswerRequirements | None = None,
                 compact_search_results: bool = False,
                 custom_tools: Sequence[AgentTool] = (), journal: ToolTurnJournal | None = None,
                 approval: BoundApproval | None = None) -> None:
        if journal is not None and not isinstance(journal, ToolTurnJournal):
            raise ValueError('invalid turn journal')
        self._journal = journal
        if type(initial_search) is not bool:
            raise ValueError('initial_search must be a boolean')
        if type(compact_search_results) is not bool:
            raise ValueError('compact_search_results must be a boolean')
        self._custom_tools = {tool.name: tool for tool in snapshot_tools(custom_tools)}
        if any(tool.requires_approval for tool in self._custom_tools.values()) and (approval is None or journal is None):
            raise ValueError('guarded tools require bound approval and journal')
        if approval is not None:
            from .approval_context import BoundApproval
            if not isinstance(approval, BoundApproval) or journal is None:
                raise ValueError('invalid bound approval')
            approval.check(tools, tuple(self._custom_tools.values()))
        self._approval = approval
        self._initial_search = initial_search
        self._compact_search_results = compact_search_results
        self._model, self._tools = model, tools
        self._limits = ToolLoopLimits.model_validate((limits or ToolLoopLimits()).model_dump())
        self._answer_requirements = validated_requirements(answer_requirements)

    async def _complete(self, transcript: list[dict[str, object]], schemas: list[dict[str, object]]) -> ToolStep:
        messages, tools = json.loads(_json(transcript)), json.loads(_json(schemas))
        constrained = getattr(self._model, 'complete_with_requirements', None)
        if self._answer_requirements is None or not callable(constrained):
            return await self._model.complete(messages, tools)
        requirements = validated_requirements(self._answer_requirements)
        assert requirements is not None
        return await cast(ConstrainedToolModel, self._model).complete_with_requirements(messages, tools, requirements)

    async def run(self, messages: list[dict[str, str]]) -> ToolLoopResult:
        limits = self._limits
        journal = self._journal
        memory_binding = _json(self._tools.journal_binding()) if journal is not None else None
        usage: list[ModelTokenUsage] = []
        # Only ordinary host messages may seed a turn. Models cannot inject a
        # preexisting tool result or select tools by altering the input history.
        if not messages or any(set(row) != {'role', 'content'} or row['role'] not in ('system', 'user', 'assistant')
                               or not isinstance(row['content'], str) for row in messages):
            raise ValueError('invalid tool conversation messages')
        if self._initial_search and (messages[-1]['role'] != 'user' or not messages[-1]['content'].strip()
                or len(messages[-1]['content'].encode()) > 8000):
            raise ValueError('initial memory query requires a final user message of 1..8000 bytes')
        transcript = cast(list[dict[str, object]], json.loads(_json(messages)))
        if self._answer_requirements is not None:
            transcript.insert(0, {'role':'system', 'content':self._answer_requirements.prompt()})
        seen: set[str] = set()
        known_facts: set[int] = set()
        known_chunks: set[int] = set()
        prepared: list[PreparedToolEvidence] = []
        reads: dict[tuple[int, int, int], tuple[PreparedToolEvidence, int]] = {}
        searches: dict[str, tuple[PreparedToolEvidence, int]] = {}
        outcomes: list[ToolOutcome] = []
        calls = rounds = model_calls = tool_bytes = 0
        if journal is not None:
            journal.bind_configuration(cast(JSONValue, {'messages': messages,
                'memory': json.loads(memory_binding) if memory_binding is not None else None, 'tools': [tool.info() for tool in self._custom_tools.values()],
                'limits': limits.model_dump(mode='json'), 'initial_search': self._initial_search,
                'compact_search_results': self._compact_search_results,
                'answer_requirements': self._answer_requirements.model_dump(mode='json') if self._answer_requirements else None}))
        duration = limits.timeout_s if journal is None else min(limits.timeout_s - journal.elapsed_s, journal.remaining_s)
        if duration <= 0:
            raise TurnJournalError('deadline')
        deadline = time.monotonic() + duration

        def check_deadline() -> None:
            if self._approval is not None:
                self._approval.check(self._tools, tuple(self._custom_tools.values()))
            if memory_binding is not None and _json(self._tools.journal_binding()) != memory_binding:
                raise TurnJournalError('binding_mismatch')
            active = asyncio.current_task()
            if active is not None and active.cancelling():
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError('tool loop deadline exceeded')

        async def finish(text: str) -> ToolLoopResult:
            check_deadline()
            if not text.strip() or len(text.encode()) > limits.max_reply_bytes:
                raise RuntimeError('tool reply byte limit or empty reply')
            if self._answer_requirements is not None and not self._answer_requirements.accepts(text):
                raise RuntimeError('tool answer format rejected')
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
            if journal is not None:
                journal.finish()
                check_deadline()
            evidence_ids = tuple(dict.fromkeys(key for evidence in retained for key in evidence.evidence_ids))
            return ToolLoopResult(text, model_calls, calls, evidence_ids,
                                  'retained' if evidence_ids else 'none',
                                  tuple(evidence.payload for evidence in retained if evidence.evidence_ids),
                                  tuple(outcomes), validate, deadline=deadline,
                                  usage=ToolTokenUsage(calls=tuple(usage)))

        async def record_call(call: ToolCall, origin: Literal['host', 'model'], *, skipped: bool = False) -> str | None:
            nonlocal calls, tool_bytes
            reused = False
            direct = False
            if skipped:
                payload = _denied('direct_return')
            elif calls >= limits.max_tool_calls:
                payload = _denied('tool_budget')
            else:
                calls += 1
                seed = call.arguments.get('seed_fact_id')
                if call.name in self._custom_tools:
                    check_deadline()
                    if len(_json(transcript).encode()) > limits.max_transcript_bytes:
                        raise RuntimeError('tool transcript byte limit')
                    registered = self._custom_tools[call.name]
                    async def validate_custom() -> None:
                        check_deadline()
                        if journal is not None:
                            for evidence in prepared:
                                if not await asyncio.create_task(evidence.validate()):
                                    raise RuntimeError('tool evidence changed before application dispatch')
                                check_deadline()
                    approval_record = None
                    if registered.requires_approval:
                        assert self._approval is not None and journal is not None
                        identity, replay = journal.operation_identity('custom', cast(JSONValue, call.model_dump(mode='json')))
                        arguments_json = registered.prepare_arguments(call.arguments)
                        check_deadline()
                        if not replay and arguments_json is not None:
                            await validate_custom()
                            journal.operation_identity('custom', cast(JSONValue, call.model_dump(mode='json')))
                            approval_record = self._approval.request(registered, arguments_json, identity)
                            if (approval_record.activation_id is None
                                    or approval_record.activation_id != self._approval.context.activation_id):
                                boundary = journal.pause()
                                from .approval_context import ToolApprovalPaused
                                raise ToolApprovalPaused(boundary.checkpoint, approval_record.request_id)
                    async def invoke_custom() -> JSONValue:
                        await validate_custom()
                        if journal is not None:
                            journal.check_current()
                        arguments = call.arguments
                        if approval_record is not None:
                            assert self._approval is not None
                            check_deadline()
                            admitted = self._approval.claim(approval_record)
                            if admitted.decision == 'deny':
                                return _denied('approval_denied')
                            arguments = admitted.call.arguments()
                        return await registered.invoke(arguments,
                            self._tools.invocation_context(deadline),
                            remaining_output_bytes=limits.max_tool_bytes - tool_bytes)
                    if journal is None:
                        custom_result = await invoke_custom()
                    else:
                        custom_result = await journal.execute('custom', cast(JSONValue, call.model_dump(mode='json')), invoke_custom)
                    if not isinstance(custom_result, str):
                        raise TurnJournalError('journal_integrity')
                    payload = custom_result
                    check_deadline()
                    direct = self._custom_tools[call.name].return_direct
                elif call.name == 'trace_memory' and (type(seed) is not int or seed not in known_facts):
                    payload = _denied('search_for_seed_first')
                elif call.name == 'read_memory' and (type(call.arguments.get('chunk_id')) is not int
                        or call.arguments['chunk_id'] not in known_chunks):
                    payload = _denied('search_for_chunk_first')
                elif denial := _computation_denial(call, known_chunks):
                    payload = _denied(denial)
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
                        if journal is None:
                            evidence = await asyncio.create_task(self._tools.prepare(call.name, call.arguments))
                        else:
                            fresh: PreparedToolEvidence | None = None
                            async def prepare_memory() -> JSONValue:
                                nonlocal fresh
                                check_deadline()
                                fresh = await asyncio.create_task(self._tools.prepare(call.name, call.arguments))
                                check_deadline()
                                if fresh.source_digest is None:
                                    raise RuntimeError('memory tool has no durable evidence receipt')
                                return {'payload': fresh.payload, 'source_digest': fresh.source_digest}
                            saved = await journal.execute('memory', cast(JSONValue, call.model_dump(mode='json')), prepare_memory)
                            if (not isinstance(saved, dict) or set(saved) != {'payload', 'source_digest'}
                                    or not isinstance(saved['payload'], str) or not isinstance(saved['source_digest'], str)):
                                raise TurnJournalError('journal_integrity')
                            evidence = fresh if fresh is not None else await self._tools.restore(saved['payload'], saved['source_digest'])
                        check_deadline()
                        payload = evidence.payload
                        repeated = (searches.get(payload) if self._compact_search_results
                                    and call.name == 'search_memory' and evidence.evidence_ids else None)
                        if repeated is not None:
                            valid = await asyncio.create_task(repeated[0].validate())
                            check_deadline()
                            if not valid:
                                raise RuntimeError('tool evidence changed before reuse')
                            reference = _json({'ok':True, 'status':'prepared', 'verified_accuracy':False,
                                'reuse':{'tool_result_number':repeated[1], 'new_evidence_count':0, 'searched_again':True},
                                'message':'Fresh search returned the identical result shown earlier. '
                                          'Read relevant context, target a missing fact with a different query, '
                                          'or answer from the available evidence.'})
                            if len(reference.encode()) < len(payload.encode()):
                                payload, reused = reference, True
                        size = len(payload.encode())
                        if tool_bytes + size > limits.max_tool_bytes:
                            payload = _denied('tool_output_budget')
                            reused = False
                        else:
                            # Keep each fresh snapshot even when its presentation
                            # references an earlier result. Retrieval was executed.
                            prepared.append(evidence)
                            if self._compact_search_results and call.name == 'search_memory' and evidence.evidence_ids:
                                searches.setdefault(evidence.payload, (evidence, len(outcomes) + 1))
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
                     'search_for_seed_first', 'search_for_chunk_first', 'unsupported_chunk_window',
                     'invalid_computation', 'ambiguous_quote', 'numeric_literal_required', 'direct_return', 'approval_denied'}
            # Model-authored IDs/unknown names can contain source text.
            # Keep them only in the transient provider protocol.
            name = (call.name if call.name in ('search_memory', 'trace_memory', 'read_memory', 'compute_memory')
                    or call.name in self._custom_tools else 'unknown_tool')
            outcomes.append(ToolOutcome(call_id=f'tool-{len(outcomes) + 1}', name=name, status=packet['status'],
                error=error if isinstance(error, str) and error in codes else None,
                output_bytes=len(payload.encode()), origin=origin, reused=reused))
            if direct and packet.get('ok') is True and packet['status'] == 'prepared':
                result = packet['result']
                return result if isinstance(result, str) else _json(result)
            return None

        async with asyncio.timeout(duration):
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
                schemas = (self._tools.openai() + [tool.openai() for tool in self._custom_tools.values()]) if enabled else []
                if not known_facts:
                    schemas = [row for row in schemas if cast(dict[str, object], row['function'])['name'] != 'trace_memory']
                if not known_chunks:
                    schemas = [row for row in schemas if cast(dict[str, object], row['function'])['name'] not in ('read_memory', 'compute_memory')]
                try:
                    # Independent copies stop an adapter from mutating the
                    # authoritative transcript, schemas, or paired call IDs.
                    if journal is None:
                        step = await asyncio.create_task(self._complete(transcript, schemas))
                    else:
                        async def complete_model() -> JSONValue:
                            check_deadline()
                            response = await asyncio.create_task(self._complete(transcript, schemas))
                            check_deadline()
                            return cast(JSONValue, _accepted_step(response, seen, enabled).model_dump(mode='json'))
                        saved_step = await journal.execute('model', cast(JSONValue,
                            {'messages': transcript, 'tools': schemas}), complete_model)
                        step = ToolStep.model_validate_json(_json(saved_step))
                    check_deadline()
                except (asyncio.CancelledError, TimeoutError, TurnJournalError, TurnJournalPaused):
                    raise
                except Exception:
                    raise RuntimeError('tool model unavailable') from None
                model_calls += 1
                step = _accepted_step(step, seen, enabled)
                ids = [call.id for call in step.calls]
                usage.append(step.usage)
                if not step.calls:
                    return await finish(step.content)
                rounds += 1
                seen.update(ids)
                transcript.append({'role': 'assistant', 'content': step.content or None, 'tool_calls': [
                    {'id': call.id, 'type': 'function', 'function': {'name': call.name, 'arguments': _json(call.arguments)}}
                    for call in step.calls]})
                direct_result: str | None = None
                for call in step.calls:
                    result = await record_call(call, 'model', skipped=direct_result is not None)
                    if result is not None:
                        direct_result = result
                if direct_result is not None:
                    if len(_json(transcript).encode()) > limits.max_transcript_bytes:
                        raise RuntimeError('tool transcript byte limit')
                    return await finish(direct_result)
