"""Immutable directory run records; no dependency on the native framework."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Optional, TypeVar, Callable

from ._wire import boolean, cursor, digest, identifier, integer, invalid, items, record, timestamp

T = TypeVar('T')
TERMINAL = ('completed', 'partial')


def optional(value: object, parse: Callable[[object], T]) -> Optional[T]:
    return None if value is None else parse(value)


def name(value: object, maximum: int, *, empty: bool = False, unicode_diagnostic: bool = False) -> str:
    if not isinstance(value, str) or not (0 if empty else 1) <= len(value) <= maximum:
        raise invalid('sync text')
    if not unicode_diagnostic:
        try:
            value.encode('utf-8')
        except UnicodeError:
            raise invalid('sync text') from None
    return value


def choice(value: object, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise invalid('sync state')
    return value


@dataclass(frozen=True)
class SyncCollection:
    collection_id: str
    label: str
    allow_delete_missing: bool
    configuration: str

    @classmethod
    def from_json(cls, value: object) -> SyncCollection:
        row = record(value)
        return cls(identifier(row.get('collection_id')), name(row.get('label'), 160, unicode_diagnostic=True),
                   boolean(row.get('allow_delete_missing')), digest(row.get('configuration')))


@dataclass(frozen=True)
class SyncSpec:
    collection_id: str
    configuration: str
    delete_missing: bool
    deadline_s: float
    max_attempts: int

    @classmethod
    def from_json(cls, value: object) -> SyncSpec:
        row = record(value)
        deadline = row.get('deadline_s')
        if type(deadline) not in (int, float) or not isinstance(deadline, (int, float)) or not 0 < deadline <= 3600 or not math.isfinite(deadline):
            raise invalid('sync deadline')
        return cls(identifier(row.get('collection_id')), digest(row.get('configuration')), boolean(row.get('delete_missing')),
                   float(deadline), integer(row.get('max_attempts'), 1, 4))


@dataclass(frozen=True)
class SyncRecord:
    space: str
    run_id: str
    spec: SyncSpec
    created_at: str
    revision: int
    attempt: int
    status: str
    last_started_at: Optional[str]
    cancel_requested_at: Optional[str]
    finished_at: Optional[str]
    error_code: Optional[str]
    collection_instance: Optional[str]
    source_count: int
    issue_count: int
    outcome_count: int
    skipped: int

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: Optional[str] = None) -> SyncRecord:
        row = record(value)
        result = cls(name(row.get('space'), 256), identifier(row.get('run_id')), SyncSpec.from_json(row.get('spec')),
            timestamp(row.get('created_at')), integer(row.get('revision'), 0, 2**31-1), integer(row.get('attempt'), 0, 4),
            choice(row.get('status'), ('registered', 'running', 'completed', 'partial', 'failed', 'cancelled')),
            optional(row.get('last_started_at'), timestamp), optional(row.get('cancel_requested_at'), timestamp),
            optional(row.get('finished_at'), timestamp), optional(row.get('error_code'), identifier),
            optional(row.get('collection_instance'), lambda value: name(value, 32)), integer(row.get('source_count'), 0, 20000),
            integer(row.get('issue_count'), 0, 300000), integer(row.get('outcome_count'), 0, 300000), integer(row.get('skipped'), 0))
        if result.space != expected_space or (run_id is not None and result.run_id != run_id):
            raise invalid('sync identity')
        if (result.revision < result.attempt or result.attempt > result.spec.max_attempts
                or (result.attempt == 0) != (result.last_started_at is None)
                or (result.status == 'registered') != (result.attempt == 0)
                or (result.status in TERMINAL) != (result.collection_instance is not None)
                or (result.collection_instance is not None and re.fullmatch('[a-f0-9]{32}', result.collection_instance) is None)
                or result.outcome_count != result.source_count + result.issue_count
                or (result.status not in TERMINAL and (result.outcome_count or result.skipped))
                or (result.status == 'completed' and result.issue_count)
                or (result.status in (*TERMINAL, 'failed', 'cancelled')) != (result.finished_at is not None)):
            raise invalid('sync record consistency')
        created = datetime.fromisoformat(result.created_at.replace('Z', '+00:00'))
        if any(datetime.fromisoformat(stamp.replace('Z', '+00:00')) < created
               for stamp in (result.last_started_at, result.cancel_requested_at, result.finished_at) if stamp is not None):
            raise invalid('sync timestamp order')
        return result


@dataclass(frozen=True)
class SyncStatus:
    record: SyncRecord
    status: str
    active_local: bool
    active_elsewhere: bool
    outcome_unknown: bool

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: Optional[str] = None) -> SyncStatus:
        row = record(value)
        saved = SyncRecord.from_json(row.get('record'), expected_space=expected_space, run_id=run_id)
        local, elsewhere, unknown = (boolean(row.get(key)) for key in ('active_local', 'active_elsewhere', 'outcome_unknown'))
        observed = saved.status
        if not local and not elsewhere:
            if observed == 'running' or (observed == 'failed' and saved.error_code == 'sync_interrupted'):
                observed = 'interrupted'
            if saved.cancel_requested_at is not None:
                observed = 'cancelled'
        if (local and elsewhere or row.get('status') != observed
                or unknown != (saved.attempt > 0 and not local and not elsewhere and saved.status not in TERMINAL)):
            raise invalid('sync ownership status')
        return cls(saved, observed, local, elsewhere, unknown)


@dataclass(frozen=True)
class SyncPage:
    items: tuple[SyncStatus, ...]
    next_after: Optional[str]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, limit: int, after: Optional[str]) -> SyncPage:
        row = record(value)
        values = tuple(SyncStatus.from_json(value, expected_space=expected_space) for value in items(row.get('items'), limit))
        next_after = cursor(row.get('next_after'))
        if (len({value.record.run_id for value in values}) != len(values)
                or (next_after is not None and (not values or (after is not None and next_after <= after)))):
            raise invalid('sync history page')
        return cls(values, next_after)


@dataclass(frozen=True)
class SyncSourceOutcome:
    path: str
    status: str
    episode_id: Optional[int]
    previous_episode_id: Optional[int]
    code: Optional[str]

    @classmethod
    def from_json(cls, value: object) -> SyncSourceOutcome:
        row = record(value)
        result = cls(name(row.get('path'), 1024), choice(row.get('status'), ('added', 'updated', 'unchanged', 'deleted', 'absent', 'suppressed', 'failed')),
            optional(row.get('episode_id'), lambda value: integer(value, 1)), optional(row.get('previous_episode_id'), lambda value: integer(value, 1)),
            optional(row.get('code'), identifier))
        if (re.search(r'[\x00-\x1f\x7f\\]', result.path) or any(part in ('', '.', '..') for part in result.path.split('/'))
                or (result.status in ('added', 'updated', 'unchanged', 'absent') and result.episode_id is None)
                or (result.status in ('updated', 'deleted')) != (result.previous_episode_id is not None)
                or (result.status == 'updated' and result.episode_id == result.previous_episode_id)
                or (result.status in ('deleted', 'failed') and result.episode_id is not None)
                or (result.status == 'failed') != (result.code is not None)):
            raise invalid('sync source outcome')
        return result


@dataclass(frozen=True)
class SyncScanIssue:
    path: str
    path_escaped: bool
    code: str

    @classmethod
    def from_json(cls, value: object) -> SyncScanIssue:
        row = record(value)
        return cls(name(row.get('path'), 49154, empty=True), boolean(row.get('path_escaped')), identifier(row.get('code')))


@dataclass(frozen=True)
class SyncOutcome:
    index: int
    source: Optional[SyncSourceOutcome]
    issue: Optional[SyncScanIssue]


@dataclass(frozen=True)
class SyncOutcomePage:
    space: str
    run_id: str
    items: tuple[SyncOutcome, ...]
    next_after: Optional[int]

    @classmethod
    def from_json(cls, value: object, *, request: SyncRecord, limit: int, after: Optional[int]) -> SyncOutcomePage:
        row = record(value)
        if row.get('space') != request.space or row.get('run_id') != request.run_id or request.status not in TERMINAL:
            raise invalid('sync result identity')
        start = 0 if after is None else after + 1
        if start > request.outcome_count:
            raise invalid('sync result cursor')
        values = []
        for index, value in enumerate(items(row.get('items'), limit), start):
            saved = record(value)
            source = optional(saved.get('source'), SyncSourceOutcome.from_json)
            issue = optional(saved.get('issue'), SyncScanIssue.from_json)
            if (integer(saved.get('index'), 0, 299999) != index or (source is None) == (issue is None)
                    or (index < request.source_count) != (source is not None)
                    or (request.status == 'completed' and source is not None and source.status == 'failed')):
                raise invalid('sync result ordering')
            values.append(SyncOutcome(index, source, issue))
        next_after = optional(row.get('next_after'), lambda value: integer(value, 0, 299999))
        expected_after = start + len(values) - 1 if start + len(values) < request.outcome_count else None
        if len(values) != min(limit, request.outcome_count - start) or next_after != expected_after:
            raise invalid('sync result page')
        return cls(request.space, request.run_id, tuple(values), next_after)
