"""Validated metadata envelopes for encrypted native agent event history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .approval_models import Digest, Name
from .progress import AgentProgressEvent, AgentProgressGap

MAX_POSITION = 2**53 - 1
_ERRORS = frozenset(
    (
        'unknown_tool',
        'invalid_arguments',
        'timeout',
        'store_error',
        'retrieval_failed',
        'output_bytes',
        'evidence_unavailable',
        'tool_budget',
        'tool_output_budget',
        'search_for_seed_first',
        'search_for_chunk_first',
        'unsupported_chunk_window',
        'invalid_computation',
        'ambiguous_quote',
        'numeric_literal_required',
        'direct_return',
        'approval_denied',
    )
)


def _positive(value: int) -> None:
    if type(value) is not int or not 1 <= value <= MAX_POSITION:
        raise ValueError('invalid event integer')


class AgentHistoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    position: int = Field(ge=1, le=MAX_POSITION)
    step_id: Name
    selection_id: Name
    event: AgentProgressEvent | AgentProgressGap

    @model_validator(mode='after')
    def bounded_event(self) -> Self:
        event = self.event
        if not re.fullmatch(r'[0-9a-f]{32}', event.invocation_id):
            raise ValueError('invalid event invocation')
        if isinstance(event, AgentProgressGap):
            _positive(event.first_sequence)
            _positive(event.last_sequence)
            if event.first_sequence > event.last_sequence:
                raise ValueError('invalid event gap')
            return self
        _positive(event.sequence)
        for value in (event.agent_id, event.model_id):
            if not re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', value):
                raise ValueError('invalid event identity')
        if not re.fullmatch(r'[0-9a-f]{64}', event.binding):
            raise ValueError('invalid event binding')
        if len(event.occurred_at) > 64 or datetime.fromisoformat(event.occurred_at).tzinfo is None:
            raise ValueError('invalid event timestamp')
        for duration in (event.elapsed_s, event.duration_s):
            if duration is not None and (not math.isfinite(duration) or duration < 0):
                raise ValueError('invalid event timing')
        for ordinal in (event.operation_id, event.tool_index):
            if ordinal is not None:
                _positive(ordinal)
        if event.tool_name is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', event.tool_name):
            raise ValueError('invalid event tool')
        if event.error is not None and event.error not in _ERRORS:
            raise ValueError('invalid event error')
        if event.output_bytes is not None and (
            type(event.output_bytes) is not int or not 0 <= event.output_bytes <= MAX_POSITION
        ):
            raise ValueError('invalid event byte count')
        operations = (event.operation_id, event.operation_kind, event.duration_s)
        tool = (event.tool_index, event.tool_name, event.origin)
        result = (
            event.status,
            event.error,
            event.output_bytes,
            event.reused,
            event.journal_reused,
            event.presentation_reused,
        )
        if event.kind.startswith('turn_'):
            if any(value is not None for value in (*operations, *tool, *result)):
                raise ValueError('turn event has operation metadata')
        elif event.kind.startswith('operation_'):
            if event.operation_id is None or event.operation_kind is None or event.origin is not None:
                raise ValueError('operation identity required')
            timed = event.kind in ('operation_completed', 'operation_failed')
            if timed != (event.duration_s is not None) or any(value is not None for value in result):
                raise ValueError('invalid operation result metadata')
            if event.operation_kind == 'model':
                if event.tool_index is not None or event.tool_name is not None:
                    raise ValueError('model operation cannot identify a tool')
            elif event.tool_index is None or event.tool_name is None:
                raise ValueError('tool operation identity required')
        else:
            if any(value is not None for value in operations) or any(value is None for value in tool):
                raise ValueError('invalid tool event identity')
            if event.kind == 'tool_proposed':
                if any(value is not None for value in result):
                    raise ValueError('proposal cannot report a result')
            elif (
                event.status is None
                or event.output_bytes is None
                or event.journal_reused is None
                or event.presentation_reused is None
                or event.reused is not (event.journal_reused or event.presentation_reused)
            ):
                raise ValueError('invalid tool result metadata')
        return self


class HistoryManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    invocation_digest: Digest
    generation: str = Field(pattern=r'^[0-9a-f]{32}$')
    retained_from: int = Field(ge=1, le=MAX_POSITION)
    next_position: int = Field(ge=2, le=MAX_POSITION)

    @model_validator(mode='after')
    def bounded(self) -> Self:
        if not 1 <= self.next_position - self.retained_from <= 4096:
            raise ValueError('invalid history range')
        return self


@dataclass(frozen=True)
class AgentHistoryPage:
    available: bool
    items: tuple[AgentHistoryEntry, ...]
    next_after: str | None
    retained_from: int | None
    omitted: tuple[int, int] | None = None
