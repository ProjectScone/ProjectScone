"""Strict metadata events, independent of native runtime and provider packages."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re
from typing import Optional, Union

from .agent_traps import TrapGraph
from ._wire import boolean, digest, identifier, integer, invalid, record, text, timestamp

MAX_POSITION = 2**53 - 1
TERMINALS = ('turn_completed', 'turn_paused', 'turn_failed', 'turn_cancelled')
KINDS = (
    'trap_detected',
    'turn_started',
    *TERMINALS,
    'operation_started',
    'operation_completed',
    'operation_failed',
    'operation_reused',
    'tool_proposed',
    'tool_result',
)
ERRORS = (
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


def invocation(value: object) -> str:
    result = text(value, 32, 'invocation')
    if re.fullmatch(r'[0-9a-f]{32}', result) is None:
        raise invalid('invocation')
    return result


def choice(value: object, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise invalid('event kind')
    return value


def event_tool_name(value: object) -> str:
    result = text(value, 128, 'event tool')
    if re.fullmatch(r'[A-Za-z0-9_-]{1,128}', result) is None:
        raise invalid('event tool')
    return result


def duration(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise invalid('event timing')
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise invalid('event timing') from None
    if not math.isfinite(result) or result < 0:
        raise invalid('event timing')
    return result


@dataclass(frozen=True)
class ProgressEvent:
    sequence: int
    invocation_id: str
    agent_id: str
    model_id: str
    binding: str
    kind: str
    occurred_at: str
    elapsed_s: float
    operation_id: Optional[int]
    operation_kind: Optional[str]
    duration_s: Optional[float]
    tool_index: Optional[int]
    tool_name: Optional[str]
    status: Optional[str]
    error: Optional[str]
    output_bytes: Optional[int]
    origin: Optional[str]
    reused: Optional[bool]
    journal_reused: Optional[bool]
    presentation_reused: Optional[bool]
    trap_graph: Optional[TrapGraph] = None

    @classmethod
    def from_json(cls, value: object) -> ProgressEvent:
        row = record(value)
        allowed = {field.name for field in fields(cls)}
        if not allowed - {'trap_graph'} <= set(row) <= allowed:
            raise invalid('progress fields')
        result = cls(
            integer(row['sequence'], 1, MAX_POSITION),
            invocation(row['invocation_id']),
            identifier(row['agent_id']),
            identifier(row['model_id']),
            digest(row['binding']),
            choice(row['kind'], KINDS),
            timestamp(row['occurred_at']),
            duration(row['elapsed_s']),
            integer(row['operation_id'], 1, MAX_POSITION) if row['operation_id'] is not None else None,
            (
                choice(row['operation_kind'], ('model', 'memory', 'custom'))
                if row['operation_kind'] is not None
                else None
            ),
            duration(row['duration_s']) if row['duration_s'] is not None else None,
            integer(row['tool_index'], 1, MAX_POSITION) if row['tool_index'] is not None else None,
            event_tool_name(row['tool_name']) if row['tool_name'] is not None else None,
            (
                choice(row['status'], ('prepared', 'empty', 'unavailable'))
                if row['status'] is not None
                else None
            ),
            choice(row['error'], ERRORS) if row['error'] is not None else None,
            integer(row['output_bytes'], 0, MAX_POSITION) if row['output_bytes'] is not None else None,
            choice(row['origin'], ('host', 'model')) if row['origin'] is not None else None,
            boolean(row['reused']) if row['reused'] is not None else None,
            boolean(row['journal_reused']) if row['journal_reused'] is not None else None,
            boolean(row['presentation_reused']) if row['presentation_reused'] is not None else None,
            TrapGraph.from_json(row['trap_graph']) if row.get('trap_graph') is not None else None,
        )
        operations = (result.operation_id, result.operation_kind, result.duration_s)
        tool = (result.tool_index, result.tool_name, result.origin)
        outcome = (
            result.status,
            result.error,
            result.output_bytes,
            result.reused,
            result.journal_reused,
            result.presentation_reused,
        )
        if result.kind == 'trap_detected':
            if result.trap_graph is None or any(
                item is not None for item in (*operations, *tool, *outcome)
            ):
                raise invalid('trap event metadata')
            return result
        if result.trap_graph is not None:
            raise invalid('unexpected trap graph')
        if result.kind.startswith('turn_'):
            if any(item is not None for item in (*operations, *tool, *outcome)):
                raise invalid('turn metadata')
        elif result.kind.startswith('operation_'):
            timed = result.kind in ('operation_completed', 'operation_failed')
            if (
                result.operation_id is None
                or result.operation_kind is None
                or result.origin is not None
                or timed != (result.duration_s is not None)
                or any(item is not None for item in outcome)
            ):
                raise invalid('operation metadata')
            if result.operation_kind == 'model':
                if result.tool_index is not None or result.tool_name is not None:
                    raise invalid('model operation')
            elif result.tool_index is None or result.tool_name is None:
                raise invalid('tool operation')
        elif any(item is not None for item in operations) or any(item is None for item in tool):
            raise invalid('tool metadata')
        elif result.kind == 'tool_proposed':
            if any(item is not None for item in outcome):
                raise invalid('tool proposal')
        elif (
            result.status is None
            or result.output_bytes is None
            or result.journal_reused is None
            or result.presentation_reused is None
            or result.reused is not (result.journal_reused or result.presentation_reused)
        ):
            raise invalid('tool outcome')
        return result


@dataclass(frozen=True)
class ProgressGap:
    invocation_id: str
    first_sequence: int
    last_sequence: int

    @classmethod
    def from_json(cls, value: object) -> ProgressGap:
        row = record(value)
        if set(row) != {field.name for field in fields(cls)}:
            raise invalid('gap fields')
        first = integer(row['first_sequence'], 1, MAX_POSITION)
        return cls(
            invocation(row['invocation_id']), first, integer(row['last_sequence'], first, MAX_POSITION)
        )


@dataclass(frozen=True)
class CollectionEvent:
    kind: str
    collection_id: str
    occurred_at: str
    invocation_id: Optional[str]
    last_sequence: int
    observed_events: int
    lost_events: int
    terminal_kind: Optional[str]
    error: Optional[str]

    @classmethod
    def from_json(cls, value: object) -> CollectionEvent:
        row = record(value)
        if set(row) != {field.name for field in fields(cls)}:
            raise invalid('collection fields')
        result = cls(
            choice(row['kind'], ('collection_started', 'collection_finished', 'collection_failed')),
            invocation(row['collection_id']),
            timestamp(row['occurred_at']),
            invocation(row['invocation_id']) if row['invocation_id'] is not None else None,
            integer(row['last_sequence'], 0, MAX_POSITION),
            integer(row['observed_events'], 0, MAX_POSITION),
            integer(row['lost_events'], 0, MAX_POSITION),
            choice(row['terminal_kind'], TERMINALS) if row['terminal_kind'] is not None else None,
            (
                choice(row['error'], ('history_unavailable', 'collection_interrupted'))
                if row['error'] is not None
                else None
            ),
        )
        if result.observed_events + result.lost_events != result.last_sequence:
            raise invalid('collection counts')
        if result.kind == 'collection_started':
            if result.last_sequence or any(
                item is not None for item in (result.invocation_id, result.terminal_kind, result.error)
            ):
                raise invalid('collection start')
        elif result.kind == 'collection_finished':
            if (
                result.invocation_id is None
                or result.terminal_kind is None
                or not result.last_sequence
                or result.error is not None
            ):
                raise invalid('collection completion')
        elif result.error is None:
            raise invalid('collection failure')
        return result


HistoryEvent = Union[ProgressEvent, ProgressGap, CollectionEvent]


def parse_event(value: object) -> HistoryEvent:
    row = record(value)
    if 'kind' not in row:
        return ProgressGap.from_json(row)
    if isinstance(row['kind'], str) and row['kind'].startswith('collection_'):
        return CollectionEvent.from_json(row)
    return ProgressEvent.from_json(row)
