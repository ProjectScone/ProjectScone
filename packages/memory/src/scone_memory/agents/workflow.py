"""Explicit local async steps with authenticated, revision-bound checkpoints."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import inspect
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import TypeAlias, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

JSONValue: TypeAlias = 'None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]'
_NAME = re.compile(r'[A-Za-z0-9._:-]{1,128}\Z')
_APP_ID = 0x53435731


class WorkflowError(Exception):
    """Safe public reason; callback messages and payloads are never included."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class StepCheckpoints:
    """Opaque work receipts valid only while their owning step attempt runs.

    The journal encrypts bytes and binds keys to the run, invocation and step.
    Callbacks must validate receipt contents before reuse. This is intermediate
    work, not a completed step or evidence that an external write succeeded.
    """
    get: Callable[[str], bytes | None]
    put: Callable[[str, bytes], None]


@dataclass(frozen=True)
class StepContext:
    run_id: str
    space: str
    scope: dict[str, JSONValue]
    inputs: JSONValue
    completed: dict[str, JSONValue]
    checkpoints: StepCheckpoints | None = None


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    version: str
    run: Callable[[StepContext], Awaitable[JSONValue]]
    idempotent: bool = False
    retryable: bool = False


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    status: str
    results: dict[str, JSONValue]
    reused_steps: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowStatus:
    status: str
    completed_steps: tuple[str, ...]
    inflight: str | None
    attempts: dict[str, int]
    error_class: str | None
    checkpoint_count: int = 0


def _name(value: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise WorkflowError('invalid_identifier')


def _integer(value: int, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise WorkflowError('invalid_budget')


def _encode(value: object, maximum: int) -> bytes:
    """Bound structure before JSON encoding; reject non-JSON and non-finite data."""
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes, size = 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > 20000:
            raise WorkflowError('payload_limit')
        if item is None or type(item) is bool:
            size += 5
        elif type(item) is str:
            try:
                size += len(item.encode('utf-8'))
            except UnicodeError:
                raise WorkflowError('invalid_payload') from None
        elif type(item) is int:
            if item.bit_length() > 256:
                raise WorkflowError('invalid_payload')
            size += 1
        elif type(item) is float:
            if not math.isfinite(item):
                raise WorkflowError('invalid_payload')
            size += 1
        elif type(item) is list:
            children = cast(list[object], item)
            if len(children) > 20000:
                raise WorkflowError('payload_limit')
            pending.extend((child, depth + 1) for child in children)
        elif type(item) is dict:
            mapping = cast(dict[object, object], item)
            if len(mapping) > 20000 or any(type(key) is not str for key in mapping):
                raise WorkflowError('invalid_payload')
            for key, child in mapping.items():
                pending.extend(((key, depth + 1), (child, depth + 1)))
        else:
            raise WorkflowError('invalid_payload')
        if size > maximum:
            raise WorkflowError('payload_limit')
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise WorkflowError('invalid_payload') from None
    if len(encoded) > maximum:
        raise WorkflowError('payload_limit')
    return encoded


def _copy(value: JSONValue, maximum: int) -> JSONValue:
    return cast(JSONValue, json.loads(_encode(value, maximum)))


def _private_file(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or info.st_nlink != 1):
        os.close(fd)
        raise WorkflowError('private_file_required')
    return fd


class WorkflowRunner:
    """Caller-owned POSIX local journal; one active run per journal file.

    Steps and verifier are trusted application code. Scope is an immutable
    invocation binding, not an authorization engine. The verifier must check
    current retained evidence and authorization, including completed outputs.
    Storage OSError/SQLite operational failures pause verification without
    discarding receipts. FileNotFoundError and explicit false results invalidate
    the run; adapters should report confirmed missing evidence accordingly.
    Async deadlines are cooperative; callbacks must propagate cancellation.
    """
    def __init__(
        self, path: str | Path, *, key: bytes, steps: Sequence[WorkflowStep],
        source_verifier: Callable[[StepContext], Awaitable[bool]],
        max_payload_bytes: int = 256000, deadline: float = 30.0, max_retries: int = 1,
        verify_before_step: bool = False,
    ):
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        if type(verify_before_step) is not bool:
            raise WorkflowError('invalid_verification_policy')
        self._verify_before_step = verify_before_step
        _integer(max_payload_bytes, 512, 1000000)
        _integer(max_retries, 0, 3)
        if type(deadline) not in (float, int) or not math.isfinite(deadline) or not 0 < deadline <= 300:
            raise WorkflowError('invalid_budget')
        if (not isinstance(steps, Sequence) or not 1 <= len(steps) <= 32
                or any(not isinstance(step, WorkflowStep) for step in steps)
                or len({s.step_id for s in steps}) != len(steps)):
            raise WorkflowError('invalid_steps')
        for step in steps:
            _name(step.step_id)
            _name(step.version)
            if (type(step.idempotent) is not bool or type(step.retryable) is not bool
                    or (step.retryable and not step.idempotent) or not inspect.iscoroutinefunction(step.run)):
                raise WorkflowError('invalid_step')
        if not inspect.iscoroutinefunction(source_verifier):
            raise WorkflowError('invalid_verifier')
        self._steps = tuple(steps)
        self._verifier = source_verifier
        self._maximum = max_payload_bytes
        self._deadline = float(deadline)
        self._retries = max_retries
        self._cipher = AESGCM(key)
        self._key = key
        self._running = False
        self._closed = False
        self._checkpoint_attempt: object | None = None
        revision: JSONValue = [{'id': s.step_id, 'version': s.version, 'idempotent': s.idempotent, 'retryable': s.retryable} for s in steps]
        if verify_before_step:
            revision = {'steps': revision, 'verify_before_step': True}
        self._revision = hashlib.sha256(_encode(revision, 32768)).hexdigest()
        self._lock_fd = -1
        self._db: sqlite3.Connection
        try:
            target = Path(path).absolute()
            fd = _private_file(target)
            os.close(fd)
            self._lock_fd = _private_file(Path(str(target) + '.lock'))
            self._db = sqlite3.connect(target, timeout=0, isolation_level=None)
            self._db.execute('PRAGMA synchronous=FULL')
            self._initialize()
        except WorkflowError:
            if hasattr(self, '_db'):
                self._db.close()
            if self._lock_fd >= 0:
                os.close(self._lock_fd)
            raise
        except (OSError, sqlite3.Error):
            if hasattr(self, '_db'):
                self._db.close()
            if self._lock_fd >= 0:
                os.close(self._lock_fd)
            raise WorkflowError('journal_unavailable') from None

    def _seal(self, token: str, payload: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._cipher.encrypt(nonce, payload, ('scone-workflow-v1:' + token).encode())

    def _unseal(self, token: str, payload: bytes) -> bytes:
        try:
            return self._cipher.decrypt(payload[:12], payload[12:], ('scone-workflow-v1:' + token).encode())
        except (InvalidTag, ValueError):
            raise WorkflowError('journal_key_or_integrity') from None

    def _initialize(self) -> None:
        self._db.execute('BEGIN IMMEDIATE')
        try:
            app = self._db.execute('PRAGMA application_id').fetchone()[0]
            version = self._db.execute('PRAGMA user_version').fetchone()[0]
            names = {r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'")}
            if not app and not version and not names:
                self._db.execute('CREATE TABLE workflow_runs (token TEXT PRIMARY KEY, payload BLOB NOT NULL)')
                self._db.execute('CREATE TABLE workflow_meta (payload BLOB NOT NULL)')
                self._db.execute('INSERT INTO workflow_meta VALUES (?)', (self._seal('key-check', b'scone-workflow-v1'),))
                self._db.execute(f'PRAGMA application_id={_APP_ID}')
                self._db.execute('PRAGMA user_version=1')
            elif app != _APP_ID or version != 1 or names != {'workflow_runs', 'workflow_meta'}:
                raise WorkflowError('foreign_journal')
            check = self._db.execute('SELECT payload FROM workflow_meta').fetchone()
            if check is None or self._unseal('key-check', check[0]) != b'scone-workflow-v1':
                raise WorkflowError('journal_key_or_integrity')
            self._db.execute('COMMIT')
        except BaseException:
            self._db.execute('ROLLBACK')
            raise

    def close(self) -> None:
        if self._running:
            raise WorkflowError('busy')
        if not self._closed:
            self._db.close()
            os.close(self._lock_fd)
            self._closed = True

    def _save(self, token: str, state: dict[str, JSONValue]) -> None:
        payload = self._seal(token, _encode(state, self._maximum + 32768))
        self._db.execute('INSERT INTO workflow_runs VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload', (token, payload))

    def _context(self, run_id: str, space: str, scope: dict[str, JSONValue], inputs: JSONValue, state: dict[str, JSONValue],
                 checkpoints: StepCheckpoints | None = None) -> StepContext:
        # Fresh copies stop callbacks from mutating future steps or bindings.
        payload: JSONValue = {'scope': scope, 'inputs': inputs, 'completed': state['results']}
        clean = cast(dict[str, JSONValue], _copy(payload, self._maximum))
        return StepContext(run_id, space, cast(dict[str, JSONValue], clean['scope']), clean['inputs'], cast(dict[str, JSONValue], clean['completed']), checkpoints)

    def _checkpoint_count(self, token: str) -> int:
        return int(self._db.execute('SELECT COUNT(*) FROM workflow_runs WHERE token LIKE ?',
                                    (token + ':checkpoint:%',)).fetchone()[0])

    def _clear_checkpoints(self, token: str) -> None:
        self._db.execute('DELETE FROM workflow_runs WHERE token LIKE ?', (token + ':checkpoint:%',))

    def _checkpoints(self, token: str, binding: str, step_id: str) -> StepCheckpoints:
        attempt = object()
        self._checkpoint_attempt = attempt

        def identity(key: str) -> str:
            if not self._running or self._checkpoint_attempt is not attempt:
                raise WorkflowError('checkpoint_inactive')
            _name(key)
            encoded = _encode([binding, step_id, key], 1024)
            return token + ':checkpoint:' + hmac.new(self._key, encoded, hashlib.sha256).hexdigest()

        def get(key: str) -> bytes | None:
            identifier = identity(key)
            row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (identifier,)).fetchone()
            return None if row is None else self._unseal(identifier, row[0])

        def put(key: str, value: bytes) -> None:
            identifier = identity(key)
            if type(value) is not bytes:
                raise WorkflowError('invalid_checkpoint')
            if len(value) > 16 * 1024 * 1024:
                raise WorkflowError('checkpoint_limit')
            # The run owns the file lock. Budget checks and the following write
            # cannot race another supported writer to this journal.
            count, total = self._db.execute(
                'SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM workflow_runs WHERE token LIKE ? AND token != ?',
                (token + ':checkpoint:%', identifier)).fetchone()
            if count >= 4096 or total + len(value) + 28 > 128 * 1024 * 1024:
                raise WorkflowError('checkpoint_limit')
            payload = self._seal(identifier, value)
            self._db.execute('INSERT INTO workflow_runs VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload',
                             (identifier, payload))

        return StepCheckpoints(get, put)

    def status(self, run_id: str, *, space: str, scope: Mapping[str, JSONValue], inputs: JSONValue) -> WorkflowStatus | None:
        """Read committed progress with the same binding as run; expose no source text."""
        _name(run_id)
        _name(space)
        if self._closed:
            raise WorkflowError('closed')
        if not isinstance(scope, Mapping):
            raise WorkflowError('invalid_payload')
        binding: JSONValue = {'space': space, 'scope': dict(scope), 'inputs': inputs, 'revision': self._revision}
        signature = hashlib.sha256(_encode(binding, self._maximum)).hexdigest()
        token = hmac.new(self._key, run_id.encode(), hashlib.sha256).hexdigest()
        try:
            row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
        except sqlite3.Error:
            raise WorkflowError('journal_unavailable') from None
        if row is None:
            return None
        state = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
        if state['binding'] != signature:
            raise WorkflowError('binding_mismatch')
        results = cast(dict[str, JSONValue], state['results'])
        return WorkflowStatus(str(state['status']), tuple(s.step_id for s in self._steps if s.step_id in results),
                              cast(str | None, state['inflight']), dict(cast(dict[str, int], state['attempts'])),
                              cast(str | None, state['error_class']), self._checkpoint_count(token))

    async def read_result(self, run_id: str, *, space: str, scope: Mapping[str, JSONValue],
                          inputs: JSONValue) -> WorkflowResult | None:
        """Revalidate completed results without executing any workflow step.

        Missing runs return None. Incomplete, failed or interrupted work is never
        resumed by this read. Verification outages preserve completed receipts;
        confirmed invalid evidence is invalidated under the normal verifier policy.
        """
        _name(run_id)
        _name(space)
        if self._closed:
            raise WorkflowError('closed')
        if self._running:
            raise WorkflowError('busy')
        if not isinstance(scope, Mapping):
            raise WorkflowError('invalid_payload')
        clean_scope = cast(dict[str, JSONValue], _copy(dict(scope), self._maximum))
        clean_input = _copy(inputs, self._maximum)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkflowError('busy') from None
        self._running = True
        try:
            progress = self.status(run_id, space=space, scope=clean_scope, inputs=clean_input)
            if progress is None:
                return None
            expected = tuple(step.step_id for step in self._steps)
            if (progress.status not in {'completed', 'verification_unavailable'}
                    or progress.inflight is not None or progress.completed_steps != expected):
                raise WorkflowError('not_completed')
            token = hmac.new(self._key, run_id.encode(), hashlib.sha256).hexdigest()
            row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
            if row is None:
                raise WorkflowError('journal_unavailable')
            state = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
            await asyncio.wait_for(self._verify(token, self._context(run_id, space, clean_scope, clean_input, state), state),
                                   timeout=self._deadline)
            state.update(status='completed', error_class=None)
            self._save(token, state)
            self._clear_checkpoints(token)
            return WorkflowResult(run_id, 'completed', cast(dict[str, JSONValue], _copy(state['results'], self._maximum)), expected)
        except asyncio.TimeoutError:
            raise WorkflowError('verification_unavailable') from None
        except sqlite3.Error:
            raise WorkflowError('journal_unavailable') from None
        finally:
            self._running = False
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    async def run(self, run_id: str, *, space: str, scope: Mapping[str, JSONValue], inputs: JSONValue) -> WorkflowResult:
        _name(run_id)
        _name(space)
        if self._closed:
            raise WorkflowError('closed')
        if self._running:
            raise WorkflowError('busy')
        if not isinstance(scope, Mapping):
            raise WorkflowError('invalid_payload')
        clean_scope = cast(dict[str, JSONValue], _copy(dict(scope), self._maximum))
        clean_input = _copy(inputs, self._maximum)
        binding: JSONValue = {'space': space, 'scope': clean_scope, 'inputs': clean_input, 'revision': self._revision}
        signature = hashlib.sha256(_encode(binding, self._maximum)).hexdigest()
        token = hmac.new(self._key, run_id.encode(), hashlib.sha256).hexdigest()
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkflowError('busy') from None
        self._running = True
        state: dict[str, JSONValue] | None = None
        try:
            row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
            if row is None:
                state = {'binding': signature, 'results': {}, 'attempts': {}, 'inflight': None, 'status': 'created', 'error_class': None}
                self._save(token, state)
            else:
                state = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
                if state['binding'] != signature:
                    state = None
                    raise WorkflowError('binding_mismatch')
            return await asyncio.wait_for(self._execute(token, state, run_id, space, clean_scope, clean_input), timeout=self._deadline)
        except asyncio.CancelledError:
            if state is not None:
                state['status'] = 'cancelled'
                state['error_class'] = 'CancelledError'
                self._save(token, state)
            raise
        except asyncio.TimeoutError:
            if state is not None:
                state['status'] = 'deadline'
                state['error_class'] = 'TimeoutError'
                self._save(token, state)
            raise WorkflowError('deadline') from None
        except sqlite3.Error:
            raise WorkflowError('journal_unavailable') from None
        finally:
            self._checkpoint_attempt = None
            self._running = False
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    async def _verify(self, token: str, context: StepContext, state: dict[str, JSONValue]) -> None:
        try:
            valid = await self._verifier(context)
        except FileNotFoundError:
            valid = False
        except (OSError, sqlite3.OperationalError) as exc:
            state.update(status='verification_unavailable', error_class=type(exc).__name__[:80])
            self._save(token, state)
            raise WorkflowError('verification_unavailable') from None
        except Exception:
            valid = False
        if valid is not True:
            state.update(status='sources_invalid', results={}, error_class=None)
            self._save(token, state)
            self._clear_checkpoints(token)
            raise WorkflowError('sources_invalid')

    async def _execute(self, token: str, state: dict[str, JSONValue], run_id: str, space: str, scope: dict[str, JSONValue], inputs: JSONValue) -> WorkflowResult:
        if state['status'] == 'sources_invalid':
            self._clear_checkpoints(token)
            raise WorkflowError('sources_invalid')
        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
        results = cast(dict[str, JSONValue], state['results'])
        attempts = cast(dict[str, JSONValue], state['attempts'])
        reused = tuple(step.step_id for step in self._steps if step.step_id in results)
        for step in self._steps:
            if step.step_id in results:
                continue
            if self._verify_before_step:
                await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
            previous = cast(int, attempts.get(step.step_id, 0))
            if previous and not (step.idempotent and step.retryable):
                code = 'outcome_unknown' if not step.idempotent else 'retry_not_allowed'
                state['status'] = code
                self._save(token, state)
                raise WorkflowError(code)
            maximum = 1 + self._retries if step.retryable else 1
            if previous >= maximum:
                raise WorkflowError('retries_exhausted')
            for attempt in range(previous + 1, maximum + 1):
                state.update(status='running', inflight=step.step_id, error_class=None)
                attempts[step.step_id] = attempt
                self._save(token, state)
                try:
                    checkpoints = self._checkpoints(token, cast(str, state['binding']), step.step_id)
                    value = await step.run(self._context(run_id, space, scope, inputs, state, checkpoints))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    state.update(status='failed', error_class=type(exc).__name__[:80])
                    self._save(token, state)
                    if not step.retryable or attempt == maximum:
                        raise WorkflowError('step_failed') from None
                    continue
                finally:
                    self._checkpoint_attempt = None
                try:
                    clean = _copy(value, self._maximum)
                    # Enforce an aggregate input/scope/results budget.
                    _encode({'scope': scope, 'inputs': inputs, 'completed': {**results, step.step_id: clean}}, self._maximum)
                except WorkflowError:
                    state.update(status='failed', error_class='InvalidPayload')
                    self._save(token, state)
                    raise
                results[step.step_id] = clean
                state.update(inflight=None, status='running')
                self._save(token, state)
                break
        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
        state.update(status='completed', inflight=None, error_class=None)
        self._save(token, state)
        self._clear_checkpoints(token)
        return WorkflowResult(run_id, 'completed', cast(dict[str, JSONValue], _copy(results, self._maximum)), reused)
