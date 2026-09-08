"""Bounded, model-neutral search/trace turns with final source revalidation.

No text is published during tool selection. Providers implement one complete
turn; the host owns tools, scope, budgets, and the final publication boundary.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from ..integrations.scoped_tools import ScopedMemoryTools
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


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    model_calls: int
    tool_calls: int
    evidence_ids: tuple[str, ...]
    source_status: str
    evidence_packets: tuple[str, ...]
    verified_accuracy: bool = False


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _denied(reason: str) -> str:
    return _json({'ok': False, 'status': 'unavailable', 'error': reason, 'verified_accuracy': False})


class EvidenceToolLoop:
    """One bounded turn. Call IDs are unique across rounds; tools are read-only.

    At most max_tool_rounds tool-bearing requests plus one final request are
    sent. Unexecuted calls receive explicit denials, preserving protocol pairs.
    The final request has no tools. Tool/call limits count attempts, not just
    successful retrievals. Deadline cancellation is cooperative with adapters.
    """

    def __init__(self, model: ToolModel, tools: ScopedMemoryTools, *, limits: ToolLoopLimits | None = None) -> None:
        self._model, self._tools = model, tools
        self._limits = ToolLoopLimits.model_validate((limits or ToolLoopLimits()).model_dump())

    async def run(self, messages: list[dict[str, str]]) -> ToolLoopResult:
        limits = self._limits
        # Only ordinary host messages may seed a turn. Models cannot inject a
        # preexisting tool result or select tools by altering the input history.
        if not messages or any(set(row) != {'role', 'content'} or row['role'] not in ('system', 'user', 'assistant')
                               or not isinstance(row['content'], str) for row in messages):
            raise ValueError('invalid tool conversation messages')
        transcript = cast(list[dict[str, object]], json.loads(_json(messages)))
        seen: set[str] = set()
        known_facts: set[int] = set()
        prepared: list[PreparedToolEvidence] = []
        calls = rounds = model_calls = tool_bytes = 0
        deadline = time.monotonic() + limits.timeout_s

        def check_deadline() -> None:
            active = asyncio.current_task()
            if active is not None and active.cancelling():
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError('tool loop deadline exceeded')

        async with asyncio.timeout(limits.timeout_s):
            while True:
                check_deadline()
                if len(_json(transcript).encode()) > limits.max_transcript_bytes:
                    raise RuntimeError('tool transcript byte limit')
                enabled = calls < limits.max_tool_calls and rounds < limits.max_tool_rounds
                schemas = self._tools.openai() if enabled else []
                if not known_facts:
                    schemas = [row for row in schemas if cast(dict[str, object], row['function'])['name'] != 'trace_memory']
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
                    for evidence in prepared:
                        valid = await asyncio.create_task(evidence.validate())
                        check_deadline()
                        if not valid:
                            raise RuntimeError('tool evidence changed before publication')
                    evidence_ids = tuple(dict.fromkeys(key for evidence in prepared for key in evidence.evidence_ids))
                    return ToolLoopResult(step.content, model_calls, calls, evidence_ids,
                                          'retained' if evidence_ids else 'none',
                                          tuple(evidence.payload for evidence in prepared if evidence.evidence_ids))
                rounds += 1
                seen.update(ids)
                transcript.append({'role': 'assistant', 'content': step.content or None, 'tool_calls': [
                    {'id': call.id, 'type': 'function', 'function': {'name': call.name, 'arguments': _json(call.arguments)}}
                    for call in step.calls]})
                for call in step.calls:
                    if calls >= limits.max_tool_calls:
                        payload = _denied('tool_budget')
                    else:
                        calls += 1
                        seed = call.arguments.get('seed_fact_id')
                        if call.name == 'trace_memory' and (type(seed) is not int or seed not in known_facts):
                            payload = _denied('search_for_seed_first')
                            tool_bytes += len(payload.encode())
                            if tool_bytes > limits.max_tool_bytes:
                                raise RuntimeError('tool output byte limit')
                            transcript.append({'role': 'tool', 'tool_call_id': call.id, 'content': payload})
                            continue
                        evidence = await asyncio.create_task(self._tools.prepare(call.name, call.arguments))
                        check_deadline()
                        size = len(evidence.payload.encode())
                        if tool_bytes + size > limits.max_tool_bytes:
                            payload = _denied('tool_output_budget')
                        else:
                            payload = evidence.payload
                            prepared.append(evidence)
                            known_facts.update(int(key.partition(':')[2]) for key in evidence.evidence_ids
                                               if key.startswith('fact:'))
                    tool_bytes += len(payload.encode())
                    if tool_bytes > limits.max_tool_bytes:
                        raise RuntimeError('tool output byte limit')
                    transcript.append({'role': 'tool', 'tool_call_id': call.id, 'content': payload})
