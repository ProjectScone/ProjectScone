"""Detached, request-bound history pages with explicit retention continuity."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
import re
from typing import Iterator, Optional

from .errors import SconeError
from ._wire import StreamedLines
from ._wire import boolean, identifier, integer, invalid, items, record, text
from .agent_events import CollectionEvent, HistoryEvent, MAX_POSITION, ProgressEvent, invocation, parse_event
from .agent_models import HandoffPlan, ModelTask, RunRequest


def history_cursor(value: object) -> str:
    result = text(value, 114, 'history cursor')
    if re.fullmatch(r'[0-9a-f]{32}\.[0-9a-f]{16}\.[0-9a-f]{64}', result) is None:
        raise invalid('history cursor')
    integer(int(result.split('.')[1], 16), 0, MAX_POSITION)
    return result


def _position(cursor: str) -> int:
    return int(cursor.split('.')[1], 16)


@dataclass(frozen=True)
class HistoryEntry:
    position: int
    step_id: str
    selection_id: str
    event: HistoryEvent
    collection_id: Optional[str] = None
    activation_id: Optional[str] = None

    @classmethod
    def from_json(cls, value: object, *, request: RunRequest) -> HistoryEntry:
        row = record(value)
        required = {'position', 'step_id', 'selection_id', 'event'}
        if not required <= row.keys() or row.keys() - required - {'collection_id', 'activation_id'}:
            raise invalid('history entry fields')
        result = cls(
            integer(row['position'], 1, MAX_POSITION),
            identifier(row['step_id']),
            identifier(row['selection_id']),
            parse_event(row['event']),
            invocation(row['collection_id']) if row.get('collection_id') is not None else None,
            identifier(row['activation_id']) if row.get('activation_id') is not None else None,
        )
        if isinstance(result.event, CollectionEvent) and result.collection_id != result.event.collection_id:
            raise invalid('collection identity')
        result.match(request)
        return result

    def match(self, request: RunRequest) -> None:
        plan = request.plan.plan
        if isinstance(plan, HandoffPlan):
            steps = tuple('hop-' + format(index, '02d') for index in range(1, plan.max_handoffs + 2))
            if self.step_id not in steps:
                raise invalid('history hop')
            reachable = {plan.root_agent}
            policies = {agent.agent_id: agent for agent in plan.agents}
            for _ in range(steps.index(self.step_id)):
                reachable = {target for name in reachable for target in policies[name].can_handoff_to}
            if self.selection_id not in reachable:
                raise invalid('history route')
            selected = policies[self.selection_id]
            agent_id, model_id = selected.agent_id, selected.model_id
        else:
            task = next((task for task in plan.tasks if task.task_id == self.step_id), None)
            if not isinstance(task, ModelTask) or self.selection_id != self.step_id:
                raise invalid('history task')
            agent_id, model_id = task.agent_id, task.model_id
        binding = request.plan.bindings.get(self.selection_id)
        if binding is None:
            raise invalid('history selection')
        if isinstance(self.event, ProgressEvent) and (
            self.event.agent_id != agent_id
            or self.event.model_id != model_id
            or self.event.binding != binding
        ):
            raise invalid('history model binding')


@dataclass(frozen=True)
class HistoryPage:
    space: str
    run_id: str
    available: bool
    items: tuple[HistoryEntry, ...]
    next_after: Optional[str]
    retained_from: Optional[int]
    omitted: Optional[tuple[int, int]]

    @classmethod
    def from_json(
        cls, value: object, *, request: RunRequest, after: Optional[str] = None, limit: int = 50
    ) -> HistoryPage:
        integer(limit, 1, 100)
        previous = history_cursor(after) if after is not None else None
        row = record(value)
        if set(row) != {field.name for field in fields(cls)}:
            raise invalid('history page fields')
        if row['space'] != request.space or row['run_id'] != request.run_id:
            raise invalid('history identity')
        available = boolean(row['available'])
        entries = tuple(HistoryEntry.from_json(item, request=request) for item in items(row['items'], limit))
        if not available:
            if (
                entries
                or previous is not None
                or any(row[name] is not None for name in ('next_after', 'retained_from', 'omitted'))
            ):
                raise invalid('unavailable history')
            return cls(request.space, request.run_id, False, (), None, None, None)
        next_after = history_cursor(row['next_after'])
        floor = integer(row['retained_from'], 1, MAX_POSITION)
        wanted = _position(previous) + 1 if previous is not None else 1
        start = max(wanted, floor)
        omitted = None
        if row['omitted'] is not None:
            gap = items(row['omitted'], 2)
            if len(gap) != 2:
                raise invalid('retention gap')
            omitted = (integer(gap[0], 1, MAX_POSITION), integer(gap[1], 1, MAX_POSITION))
        if omitted != ((wanted, start - 1) if start > wanted else None):
            raise invalid('retention continuity')
        if tuple(entry.position for entry in entries) != tuple(range(start, start + len(entries))):
            raise invalid('history continuity')
        expected = entries[-1].position if entries else start - 1
        if _position(next_after) != expected or (not entries and previous != next_after):
            raise invalid('history cursor position')
        if previous is not None and next_after.split('.')[0] != previous.split('.')[0]:
            raise invalid('history generation')
        observed: dict[str, tuple[Optional[str], Optional[str], str, str]] = {}
        for entry in entries:
            if entry.collection_id is None:
                continue
            current = (entry.event.invocation_id, entry.activation_id, entry.step_id, entry.selection_id)
            earlier = observed.get(entry.collection_id)
            if earlier is not None and (
                earlier[1:] != current[1:]
                or (earlier[0] is not None and current[0] is not None and earlier[0] != current[0])
            ):
                raise invalid('collection continuity')
            observed[entry.collection_id] = (current[0] or (earlier[0] if earlier else None), *current[1:])
        return cls(request.space, request.run_id, True, entries, next_after, floor, omitted)


class HistoryStream:
    """Verified pages as they are published, read frame by frame.

    Each ``history`` frame is validated exactly as a page read is -- same
    identity, same shape, same cursor continuity -- and its SSE ``id`` must
    name that page's ``next_after``, because the ``id`` is what a
    reconnect will send back. An ``error`` frame is a refusal and is
    raised; an ``end`` frame stops the iteration. Comments are ignored.
    Use as a context manager so the connection is released whether the
    stream ended, was refused, or the caller stopped early.
    """

    def __init__(self, lines: "StreamedLines", *, request: "RunRequest", after: Optional[str],
                 limit: int) -> None:
        self._lines = lines
        self._request = request
        self._after = after
        self._limit = limit
        self._closed = False
        self.cursor: Optional[str] = after

    def __enter__(self) -> "HistoryStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._lines.close()

    def __iter__(self) -> Iterator[HistoryPage]:
        kind: Optional[str] = None
        identifier: Optional[str] = None
        data: list[bytes] = []
        for raw in self._lines:
            if raw == b"":
                if kind is not None or data:
                    page = self._frame(kind, identifier, b"\n".join(data))
                    kind, identifier, data = None, None, []
                    if page is None:
                        return
                    yield page
                continue
            if raw.startswith(b":"):
                continue
            field, _, value = raw.partition(b":")
            value = value[1:] if value.startswith(b" ") else value
            if field == b"event":
                kind = value.decode("utf-8", errors="strict")
            elif field == b"id":
                identifier = value.decode("utf-8", errors="strict")
            elif field == b"data":
                data.append(value)
        if kind is not None or data:
            raise invalid("history stream ended inside a frame")

    def _frame(self, kind: Optional[str], identifier: Optional[str], data: bytes) -> Optional[HistoryPage]:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise invalid("history frame") from None
        if kind == "end":
            return None
        if kind == "error":
            row = record(payload)
            reason = row.get("reason")
            raise SconeError("history stream refused: " + (reason if isinstance(reason, str) else "unknown"))
        if kind != "history":
            raise invalid("history frame kind")
        page = HistoryPage.from_json(payload, request=self._request, after=self.cursor, limit=self._limit)
        if identifier is None or identifier != page.next_after:
            raise invalid("history frame id does not name its page")
        self.cursor = page.next_after
        return page
