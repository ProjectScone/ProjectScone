"""Encrypted, bounded observation history; reading never executes an agent."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import hmac
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from pydantic import ValidationError

from ..core.errors import InvalidInput
from ._encrypted_store import EncryptedRecordStore
from .approval_store import invocation_digest
from .handoff_workflow import AgentHandoffPlan
from .plan_store import _selections
from .history_models import AgentHistoryEntry, AgentHistoryPage, HistoryManifest, MAX_POSITION
from .progress import AgentProgressEvent, AgentProgressGap
from .run_store import AgentRunRequest
from .workflow import WorkflowError, _integer


class AgentEventHistoryStore:
    """Host-owned metadata storage, separate from authoritative work receipts.

    Callers must authorize the saved request and current space/scope before use.
    This store persists observations supplied by trusted native collectors; it
    does not prove collection completeness, authorize effects or restore sources.
    """

    def __init__(
        self, path: str | Path, *, key: bytes, max_events: int = 512, max_histories: int = 4096
    ) -> None:
        _integer(max_events, 1, 4096)
        _integer(max_histories, 1, 100000)
        self._capacity, self._maximum, self._key = max_events, max_histories, key
        self._storage = EncryptedRecordStore(
            path,
            key=key,
            table='agent_history',
            metadata='agent_history_meta',
            application_id=0x53434548,
            domain='scone-agent-event-history-v1',
            label='history',
        )

    @contextmanager
    def _access(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            with self._storage._access(write=write) as db:
                yield db
        except OSError:
            raise WorkflowError('history_store_unavailable') from None

    def _identity(self, request: AgentRunRequest) -> tuple[str, str, AgentRunRequest]:
        try:
            # Frozen Pydantic instances can still be constructed/copied unchecked.
            checked = AgentRunRequest.model_validate(request.model_dump(warnings=False))
        except (AttributeError, ValidationError, ValueError, TypeError, RecursionError, InvalidInput):
            raise WorkflowError('invalid_history_request') from None
        token = hmac.new(
            self._key,
            b'agent-history:' + checked.space.encode() + b'\0' + checked.run_id.encode(),
            hashlib.sha256,
        ).hexdigest()
        return token, invocation_digest(checked), checked

    @staticmethod
    def _row_token(token: str, position: int) -> str:
        return f'event:{token}:{position:016x}'

    def _manifest(self, db: sqlite3.Connection, token: str, digest: str) -> HistoryManifest | None:
        row = db.execute('SELECT payload FROM agent_history WHERE token=?', ('state:' + token,)).fetchone()
        if row is None:
            prefix = f'event:{token}:'
            if (
                db.execute(
                    'SELECT 1 FROM agent_history WHERE token>=? AND token<? LIMIT 1', (prefix, prefix + '~')
                ).fetchone()
                is not None
            ):
                raise WorkflowError('history_key_or_integrity')
            return None
        try:
            state = HistoryManifest.model_validate_json(self._storage._unseal('state:' + token, row[0]))
        except ValidationError:
            raise WorkflowError('history_key_or_integrity') from None
        if state.invocation_digest != digest:
            raise WorkflowError('history_run_conflict')
        return state

    def _cursor(self, token: str, state: HistoryManifest, position: int) -> str:
        prefix = f'{state.generation}.{position:016x}'
        signature = hmac.new(
            self._key,
            ('agent-history-cursor:' + token + ':' + state.invocation_digest + ':' + prefix).encode(),
            hashlib.sha256,
        ).hexdigest()
        return prefix + '.' + signature

    def _after(self, token: str, state: HistoryManifest, cursor: str | None) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or not re.fullmatch(
            r'[0-9a-f]{32}\.[0-9a-f]{16}\.[0-9a-f]{64}', cursor
        ):
            raise WorkflowError('invalid_history_cursor')
        position = int(cursor.split('.')[1], 16)
        if position >= state.next_position or not hmac.compare_digest(
            cursor, self._cursor(token, state, position)
        ):
            raise WorkflowError('invalid_history_cursor')
        return position

    @staticmethod
    def _selection(request: AgentRunRequest, entry: AgentHistoryEntry) -> None:
        plan = request.plan.plan
        selected = _selections(plan).get(entry.selection_id)
        if isinstance(plan, AgentHandoffPlan):
            steps = [f'hop-{index + 1:02d}' for index in range(plan.max_handoffs + 1)]
            valid_step = entry.step_id in steps
            if valid_step:
                reachable = {plan.root_agent}
                targets = {agent.agent_id: agent.can_handoff_to for agent in plan.agents}
                for _ in range(steps.index(entry.step_id)):
                    reachable = {target for source in reachable for target in targets[source]}
                valid_step = entry.selection_id in reachable
        else:
            valid_step = entry.step_id == entry.selection_id
        event = entry.event
        if (
            selected is None
            or not valid_step
            or (
                isinstance(event, AgentProgressEvent)
                and (
                    selected != (event.agent_id, event.model_id)
                    or request.plan.bindings.get(entry.selection_id) != event.binding
                )
            )
        ):
            raise WorkflowError('history_selection_mismatch')

    def append(
        self,
        request: AgentRunRequest,
        *,
        step_id: str,
        selection_id: str,
        event: AgentProgressEvent | AgentProgressGap,
    ) -> AgentHistoryEntry:
        token, digest, request = self._identity(request)
        try:
            if any(
                type(value) is not str or not re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', value)
                for value in (step_id, selection_id)
            ):
                raise ValueError('invalid event selection')
            if type(event) not in (AgentProgressEvent, AgentProgressGap):
                raise ValueError('invalid native event')
            values: dict[str, object] = {field.name: getattr(event, field.name) for field in fields(event)}
            for value in values.values():
                if type(value) not in (str, int, float, bool, type(None)):
                    raise ValueError('event fields must be primitive')
                if isinstance(value, str) and len(value) > 512:
                    raise ValueError('event field too large')
                if type(value) is int and not 0 <= value <= MAX_POSITION:
                    raise ValueError('event integer out of bounds')
            encoded = json.dumps(
                {'position': 1, 'step_id': step_id, 'selection_id': selection_id, 'event': values},
                allow_nan=False,
            )
            if len(encoded.encode()) > 4096:
                raise ValueError('event too large')
            entry = AgentHistoryEntry.model_validate_json(encoded)
        except (ValidationError, ValueError, TypeError, OverflowError):
            raise WorkflowError('invalid_history_event') from None
        self._selection(request, entry)
        with self._access(write=True) as db:
            prior = self._manifest(db, token, digest)
            if prior is None:
                if (
                    db.execute("SELECT COUNT(*) FROM agent_history WHERE token LIKE 'state:%'").fetchone()[0]
                    >= self._maximum
                ):
                    raise WorkflowError('history_store_limit')
                position, generation, floor = 1, uuid4().hex, 1
            else:
                position, generation, floor = prior.next_position, prior.generation, prior.retained_from
            if position >= MAX_POSITION:
                raise WorkflowError('history_position_limit')
            entry = entry.model_copy(update={'position': position})
            row_token = self._row_token(token, position)
            payload = entry.model_dump_json().encode()
            if len(payload) > 4096:
                raise WorkflowError('invalid_history_event')
            db.execute(
                'INSERT INTO agent_history VALUES (?,?)',
                (row_token, self._storage._seal(row_token + ':' + generation, payload)),
            )
            state = HistoryManifest(
                invocation_digest=digest,
                generation=generation,
                retained_from=max(floor, position + 1 - self._capacity),
                next_position=position + 1,
            )
            db.execute(
                'DELETE FROM agent_history WHERE token>=? AND token<?',
                (f'event:{token}:', self._row_token(token, state.retained_from)),
            )
            db.execute(
                'INSERT INTO agent_history VALUES (?,?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload',
                ('state:' + token, self._storage._seal('state:' + token, state.model_dump_json().encode())),
            )
            return entry

    def read(
        self, request: AgentRunRequest, *, after: str | None = None, limit: int = 50
    ) -> AgentHistoryPage:
        _integer(limit, 1, 100)
        token, digest, request = self._identity(request)
        with self._access() as db:
            state = self._manifest(db, token, digest)
            if state is None:
                if after is not None:
                    raise WorkflowError('invalid_history_cursor')
                return AgentHistoryPage(False, (), None, None)
            position = self._after(token, state, after)
            start = max(position + 1, state.retained_from)
            stop = min(start + limit, state.next_position)
            rows = db.execute(
                'SELECT token,payload FROM agent_history WHERE token>=? AND token<? ORDER BY token LIMIT ?',
                (self._row_token(token, start), self._row_token(token, stop), limit + 1),
            ).fetchall()
            if len(rows) != stop - start:
                raise WorkflowError('history_key_or_integrity')
            entries = []
            for expected, (row_token, payload) in zip(range(start, stop), rows, strict=True):
                try:
                    entry = AgentHistoryEntry.model_validate_json(
                        self._storage._unseal(row_token + ':' + state.generation, payload)
                    )
                except ValidationError:
                    raise WorkflowError('history_key_or_integrity') from None
                if row_token != self._row_token(token, expected) or entry.position != expected:
                    raise WorkflowError('history_key_or_integrity')
                try:
                    self._selection(request, entry)
                except WorkflowError:
                    raise WorkflowError('history_key_or_integrity') from None
                entries.append(entry)
            omitted = (position + 1, state.retained_from - 1) if position + 1 < state.retained_from else None
            return AgentHistoryPage(
                True, tuple(entries), self._cursor(token, state, stop - 1), state.retained_from, omitted
            )

    def purge(self, request: AgentRunRequest) -> None:
        """Forget this history generation; old cursors cannot read later history."""
        token, digest, request = self._identity(request)
        with self._access(write=True) as db:
            self._manifest(db, token, digest)
            prefix = f'event:{token}:'
            db.execute('DELETE FROM agent_history WHERE token>=? AND token<?', (prefix, prefix + '~'))
            db.execute('DELETE FROM agent_history WHERE token=?', ('state:' + token,))

    def close(self) -> None:
        self._storage.close()
