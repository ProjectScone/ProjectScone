"""Encrypted sync requests and atomically published, paged historical results."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3

from ..agents._encrypted_store import EncryptedRecordStore
from ..agents.workflow import WorkflowError, _integer, _name
from ..core.validation import check_space
from .directory_run_types import (MAX_OUTCOMES, TERMINAL_RESULTS, SyncOutcome, SyncRunRecord,
                                  SyncRunSpec, SyncScanIssue, SyncSourceOutcome)
from .directory_sync import DirectorySyncResult


@dataclass(frozen=True)
class SyncRunPage:
    items: tuple[SyncRunRecord, ...]
    next_after: str | None


@dataclass(frozen=True)
class SyncOutcomePage:
    items: tuple[SyncOutcome, ...]
    next_after: int | None


class DirectoryRunStore:
    """Control revisions are conditional writes, not an execution lease.

    A service must own the run and collection locks before starting an attempt.
    Result rows and the terminal summary commit together. History never scans a
    directory or establishes that an old episode is still readable.
    """
    def __init__(self, path: str | Path, *, key: bytes, max_runs: int = 4096) -> None:
        _integer(max_runs, 1, 100000)
        self._key, self._maximum = key, max_runs
        self._storage = EncryptedRecordStore(path, key=key, table='directory_runs',
            metadata='directory_run_meta', application_id=0x53434452,
            domain='scone-directory-runs-v1', label='sync')

    def _prefix(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, b'sync-space:' + space.encode(), hashlib.sha256).hexdigest() + ':'

    def _token(self, space: str, run_id: str) -> str:
        _name(run_id)
        return self._prefix(space) + hmac.new(self._key, b'sync-run:' + run_id.encode(), hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, space: str) -> SyncRunRecord:
        try:
            record = SyncRunRecord.model_validate_json(self._storage._unseal(token, payload))
        except ValueError:
            raise WorkflowError('sync_key_or_integrity') from None
        if record.space != space or self._token(record.space, record.run_id) != token:
            raise WorkflowError('sync_key_or_integrity')
        return record

    def _read(self, db: sqlite3.Connection, token: str, space: str) -> SyncRunRecord:
        row = db.execute('SELECT payload FROM directory_runs WHERE token=?', (token,)).fetchone()
        if row is None:
            raise WorkflowError('sync_not_found')
        return self._decode(token, row[0], space)

    def _write(self, db: sqlite3.Connection, token: str, record: SyncRunRecord) -> None:
        payload = self._storage._seal(token, record.model_dump_json().encode())
        db.execute('UPDATE directory_runs SET payload=? WHERE token=?', (payload, token))

    def get(self, space: str, run_id: str) -> SyncRunRecord | None:
        token = self._token(space, run_id)
        with self._storage._access() as db:
            row = db.execute('SELECT payload FROM directory_runs WHERE token=?', (token,)).fetchone()
            return self._decode(token, row[0], space) if row else None

    def register(self, space: str, run_id: str, spec: SyncRunSpec) -> SyncRunRecord:
        token = self._token(space, run_id)
        saved = SyncRunRecord(space=space, run_id=run_id, spec=SyncRunSpec.model_validate(spec),
                              created_at=datetime.now(timezone.utc))
        with self._storage._access(write=True) as db:
            row = db.execute('SELECT payload FROM directory_runs WHERE token=?', (token,)).fetchone()
            if row:
                prior = self._decode(token, row[0], space)
                if prior.spec != saved.spec:
                    raise WorkflowError('sync_request_conflict')
                return prior
            if db.execute('SELECT COUNT(*) FROM directory_runs WHERE length(token)=129').fetchone()[0] >= self._maximum:
                raise WorkflowError('sync_store_limit')
            db.execute('INSERT INTO directory_runs VALUES (?, ?)',
                       (token, self._storage._seal(token, saved.model_dump_json().encode())))
        return saved

    def _change(self, space: str, run_id: str, revision: int,
                change: Callable[[SyncRunRecord], dict[str, object]]) -> SyncRunRecord:
        _integer(revision, 0, 2**31 - 1)
        token = self._token(space, run_id)
        with self._storage._access(write=True) as db:
            prior = self._read(db, token, space)
            if prior.revision != revision:
                raise WorkflowError('sync_request_conflict')
            updated = SyncRunRecord.model_validate({**prior.model_dump(), **change(prior), 'revision': revision + 1})
            self._write(db, token, updated)
            return updated

    def start_attempt(self, space: str, run_id: str, *, expected_revision: int,
                      resume: bool = False) -> SyncRunRecord:
        if type(resume) is not bool:
            raise WorkflowError('invalid_resume_flag')
        def change(prior: SyncRunRecord) -> dict[str, object]:
            if prior.status in TERMINAL_RESULTS:
                raise WorkflowError('sync_result_terminal')
            if prior.cancel_requested_at is not None and not resume:
                raise WorkflowError('sync_cancelled')
            if prior.attempt and not resume:
                raise WorkflowError('sync_resume_required')
            if prior.attempt >= prior.spec.max_attempts:
                raise WorkflowError('sync_attempt_limit')
            return {'attempt': prior.attempt + 1, 'status': 'running', 'last_started_at': datetime.now(timezone.utc),
                    'finished_at': None, 'cancel_requested_at': None, 'error_code': None}
        return self._change(space, run_id, expected_revision, change)

    def request_cancel(self, space: str, run_id: str, *, expected_revision: int) -> SyncRunRecord:
        def change(prior: SyncRunRecord) -> dict[str, object]:
            if prior.status in TERMINAL_RESULTS:
                raise WorkflowError('sync_result_terminal')
            return {'cancel_requested_at': prior.cancel_requested_at or datetime.now(timezone.utc)}
        return self._change(space, run_id, expected_revision, change)

    def fail(self, space: str, run_id: str, *, expected_revision: int, error_code: str) -> SyncRunRecord:
        _name(error_code)
        def change(prior: SyncRunRecord) -> dict[str, object]:
            if prior.status != 'running':
                raise WorkflowError('sync_not_running')
            return {'status': 'cancelled' if prior.cancel_requested_at is not None else 'failed',
                    'finished_at': datetime.now(timezone.utc), 'error_code': error_code}
        return self._change(space, run_id, expected_revision, change)

    def finish(self, space: str, run_id: str, *, expected_revision: int,
               result: DirectorySyncResult) -> SyncRunRecord:
        _integer(expected_revision, 0, 2**31 - 1)
        if type(result.complete) is not bool or len(result.receipts) + len(result.issues) > MAX_OUTCOMES:
            raise WorkflowError('sync_result_invalid')
        if result.complete and (result.issues or any(item.status == 'failed' for item in result.receipts)):
            raise WorkflowError('sync_result_invalid')
        if len({item.path for item in result.receipts}) != len(result.receipts):
            raise WorkflowError('sync_result_invalid')
        token = self._token(space, run_id)
        with self._storage._access(write=True) as db:
            prior = self._read(db, token, space)
            if prior.revision != expected_revision:
                raise WorkflowError('sync_request_conflict')
            if prior.cancel_requested_at is not None:
                raise WorkflowError('sync_cancelled')
            if prior.status != 'running':
                raise WorkflowError('sync_not_running')
            updated = SyncRunRecord.model_validate({**prior.model_dump(), 'revision': prior.revision + 1,
                'status': 'completed' if result.complete else 'partial', 'finished_at': datetime.now(timezone.utc),
                'collection_instance': result.collection_id, 'source_count': len(result.receipts),
                'issue_count': len(result.issues), 'outcome_count': len(result.receipts) + len(result.issues),
                'skipped': result.skipped})
            for index, source in enumerate(result.receipts):
                outcome = SyncOutcome(index=index, source=SyncSourceOutcome.model_validate(asdict(source)))
                self._insert_outcome(db, token, outcome)
            for index, issue in enumerate(result.issues, len(result.receipts)):
                path = issue.path
                if not isinstance(path, str) or len(path) > 8192:
                    raise WorkflowError('sync_result_invalid')
                escaped = False
                try:
                    path.encode('utf-8')
                except UnicodeError:
                    path, escaped = json.dumps(path, ensure_ascii=True), True
                outcome = SyncOutcome(index=index, issue=SyncScanIssue(path=path, path_escaped=escaped, code=issue.code))
                self._insert_outcome(db, token, outcome)
            self._write(db, token, updated)
            return updated

    def _insert_outcome(self, db: sqlite3.Connection, token: str, outcome: SyncOutcome) -> None:
        key = f'{token}:o:{outcome.index:06d}'
        payload = self._storage._seal(key, outcome.model_dump_json().encode())
        db.execute('INSERT INTO directory_runs VALUES (?, ?)', (key, payload))

    def list(self, space: str, *, limit: int = 20, after: str | None = None) -> SyncRunPage:
        _integer(limit, 1, 100)
        prefix = self._prefix(space)
        if after is not None and (not isinstance(after, str) or not re.fullmatch(prefix + r'[a-f0-9]{64}', after)):
            raise WorkflowError('invalid_sync_cursor')
        with self._storage._access() as db:
            rows = db.execute('SELECT token,payload FROM directory_runs WHERE token>? AND token<? '
                              'AND length(token)=129 ORDER BY token LIMIT ?',
                              (after or prefix, prefix + '~', limit + 1)).fetchall()
            items = tuple(self._decode(token, payload, space) for token, payload in rows[:limit])
            return SyncRunPage(items, rows[limit - 1][0] if len(rows) > limit else None)

    def outcomes(self, space: str, run_id: str, *, limit: int = 20,
                 after: int | None = None) -> SyncOutcomePage:
        _integer(limit, 1, 100)
        if after is not None:
            _integer(after, 0, MAX_OUTCOMES - 1)
        token = self._token(space, run_id)
        start = 0 if after is None else after + 1
        with self._storage._access() as db:
            record = self._read(db, token, space)
            if record.status not in TERMINAL_RESULTS:
                raise WorkflowError('sync_result_unavailable')
            if start > record.outcome_count:
                raise WorkflowError('invalid_sync_cursor')
            prefix = token + ':o:'
            rows = db.execute('SELECT token,payload FROM directory_runs WHERE token>=? AND token<? '
                              'ORDER BY token LIMIT ?', (f'{prefix}{start:06d}', prefix + '~', limit + 1)).fetchall()
            expected = min(limit + 1, record.outcome_count - start)
            if len(rows) != expected:
                raise WorkflowError('sync_key_or_integrity')
            items: list[SyncOutcome] = []
            for index, (key, payload) in enumerate(rows, start):
                try:
                    outcome = SyncOutcome.model_validate_json(self._storage._unseal(key, payload))
                except ValueError:
                    raise WorkflowError('sync_key_or_integrity') from None
                if key != f'{prefix}{index:06d}' or outcome.index != index:
                    raise WorkflowError('sync_key_or_integrity')
                if (index < record.source_count) != (outcome.source is not None):
                    raise WorkflowError('sync_key_or_integrity')
                if len(items) < limit:
                    items.append(outcome)
            next_after = items[-1].index if len(rows) > limit else None
            return SyncOutcomePage(tuple(items), next_after)

    def close(self) -> None:
        self._storage.close()
