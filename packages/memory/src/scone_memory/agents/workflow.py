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

from .workflow_pause import (PauseReceipt, WorkflowPauseSnapshot as WorkflowPauseSnapshot,
                             WorkflowPaused as WorkflowPaused,
                             WorkflowPausableStep as WorkflowPausableStep,
                             pause_receipt, paused_state, validate_pause)

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
    deadline: float | None = None


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    version: str
    run: Callable[[StepContext], Awaitable[JSONValue]]
    idempotent: bool = False
    retryable: bool = False


@dataclass(frozen=True)
class WorkflowInputValue:
    """A value authorized for consumption by the host's continuation policy."""
    value: JSONValue


@dataclass(frozen=True)
class WorkflowInputStep:
    """Idempotent request reconciliation; None means a durable request is pending.

    Poll callbacks must not wait for a human. The host owns response storage and
    activation authorization; the scheduler only consumes returned values.
    """
    step_id: str
    version: str
    poll: Callable[[StepContext], Awaitable[WorkflowInputValue | None]]


@dataclass(frozen=True)
class WorkflowCompletion:
    """Pure synchronous completion policy, revision-bound like a step.

    The predicate must depend only on its detached invocation and results.
    Change version whenever that policy changes. It must never invoke models,
    perform external writes or use mutable external state.
    """
    version: str
    when: Callable[[StepContext], bool]

    def __post_init__(self) -> None:
        _name(self.version)
        if not callable(self.when) or inspect.iscoroutinefunction(self.when):
            raise WorkflowError('invalid_completion')


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
    inflight_steps: tuple[str, ...] = ()
    waiting_steps: tuple[str, ...] = ()
    paused_steps: tuple[str, ...] = ()


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


def _private_file(path: Path, *, read_only: bool = False) -> int:
    flags = os.O_RDONLY if read_only else os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or info.st_nlink != 1):
        os.close(fd)
        raise WorkflowError('private_file_required')
    return fd


def _dependencies(steps: Sequence[WorkflowStep | WorkflowInputStep | WorkflowPausableStep], values: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(values, Mapping) or set(values) != {step.step_id for step in steps}:
        raise WorkflowError('invalid_dependencies')
    result: dict[str, tuple[str, ...]] = {}
    for name, items in values.items():
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or len(items) > 31:
            raise WorkflowError('invalid_dependencies')
        if any(not isinstance(item, str) or item not in values or item == name for item in items) or len(set(items)) != len(items):
            raise WorkflowError('invalid_dependencies')
        result[name] = tuple(items)
    done: set[str] = set()
    while len(done) < len(result):
        ready = {name for name, required in result.items() if name not in done and set(required) <= done}
        if not ready:
            raise WorkflowError('invalid_dependencies')
        done.update(ready)
    return result


class WorkflowRunner:
    """Caller-owned POSIX local journal; one active run per journal file.

    Steps and verifier are trusted application code. Scope is an immutable
    invocation binding, not an authorization engine. The verifier must check
    current retained evidence and authorization, including completed outputs.
    Storage OSError/SQLite operational failures pause verification without
    discarding receipts. FileNotFoundError and explicit false results invalidate
    the run; adapters should report confirmed missing evidence accordingly.
    Async deadlines are cooperative; callbacks must propagate cancellation.
    With automatic_retries=False, failed steps wait for another explicit run;
    max_retries still bounds attempts across all runs and journal reopenings.
    """
    def __init__(
        self, path: str | Path, *, key: bytes, steps: Sequence[WorkflowStep | WorkflowInputStep | WorkflowPausableStep],
        source_verifier: Callable[[StepContext], Awaitable[bool]],
        max_payload_bytes: int = 256000, deadline: float = 30.0, max_retries: int = 1,
        verify_before_step: bool = False, automatic_retries: bool = True,
        dependencies: Mapping[str, Sequence[str]] | None = None, max_parallel: int = 1,
        completion: WorkflowCompletion | None = None, read_only: bool = False,
    ):
        if type(read_only) is not bool:
            raise WorkflowError('invalid_journal_mode')
        self._read_only = read_only
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        if type(verify_before_step) is not bool:
            raise WorkflowError('invalid_verification_policy')
        if type(automatic_retries) is not bool:
            raise WorkflowError('invalid_retry_policy')
        self._automatic_retries = automatic_retries
        self._verify_before_step = verify_before_step
        _integer(max_payload_bytes, 512, 1000000)
        _integer(max_retries, 0, 3)
        if type(deadline) not in (float, int) or not math.isfinite(deadline) or not 0 < deadline <= 300:
            raise WorkflowError('invalid_budget')
        if (not isinstance(steps, Sequence) or not 1 <= len(steps) <= 32
                or any(not isinstance(step, (WorkflowStep, WorkflowInputStep, WorkflowPausableStep)) for step in steps)
                or len({s.step_id for s in steps}) != len(steps)):
            raise WorkflowError('invalid_steps')
        for step in steps:
            _name(step.step_id)
            _name(step.version)
            if isinstance(step, WorkflowInputStep):
                if not inspect.iscoroutinefunction(step.poll):
                    raise WorkflowError('invalid_input_step')
                continue
            if isinstance(step, WorkflowPausableStep):
                _integer(step.max_resumes, 1, 128)
                if not inspect.iscoroutinefunction(step.run):
                    raise WorkflowError('invalid_pausable_step')
                continue
            if (type(step.idempotent) is not bool or type(step.retryable) is not bool
                    or (step.retryable and not step.idempotent) or not inspect.iscoroutinefunction(step.run)):
                raise WorkflowError('invalid_step')
        if not inspect.iscoroutinefunction(source_verifier):
            raise WorkflowError('invalid_verifier')
        _integer(max_parallel, 1, 8)
        if dependencies is None and any(isinstance(step, WorkflowInputStep) for step in steps):
            raise WorkflowError('input_dependencies_required')
        if dependencies is None and max_parallel != 1:
            raise WorkflowError('parallel_dependencies_required')
        self._dependencies = _dependencies(steps, dependencies) if dependencies is not None else None
        if dependencies is not None and any(isinstance(step, WorkflowStep) and step.retryable for step in steps):
            raise WorkflowError('parallel_retries_unsupported')
        if completion is not None and (not isinstance(completion, WorkflowCompletion) or dependencies is not None):
            raise WorkflowError('invalid_completion')
        self._completion = WorkflowCompletion(completion.version, completion.when) if completion is not None else None
        self._parallel = max_parallel
        self._steps = tuple(steps)
        self._verifier = source_verifier
        self._maximum = max_payload_bytes
        self._deadline = float(deadline)
        self._pause_expires_at: float | None = None
        self._retries = max_retries
        self._cipher = AESGCM(key)
        self._key = key
        self._running = False
        self._closed = False
        self._checkpoint_attempts: dict[str, object] = {}
        revision: JSONValue = [
            {'id': s.step_id, 'version': s.version, 'kind': 'input'} if isinstance(s, WorkflowInputStep)
            else {'id': s.step_id, 'version': s.version, 'kind': 'pausable', 'max_resumes': s.max_resumes}
            if isinstance(s, WorkflowPausableStep)
            else {'id': s.step_id, 'version': s.version, 'idempotent': s.idempotent, 'retryable': s.retryable}
            for s in steps]
        if not automatic_retries:
            revision = {'steps': revision, 'automatic_retries': False, 'max_retries': max_retries}
        if verify_before_step:
            revision = {'steps': revision, 'verify_before_step': True}
        if self._dependencies is not None:
            revision = {'steps': revision, 'dependencies': {name: list(values) for name, values in self._dependencies.items()},
                        'max_parallel': max_parallel}
        if self._completion is not None:
            revision = {'steps': revision, 'completion': self._completion.version}
        self._revision = hashlib.sha256(_encode(revision, 256000)).hexdigest()
        self._lock_fd = -1
        self._db: sqlite3.Connection
        try:
            target = Path(path).absolute()
            fd = _private_file(target, read_only=read_only)
            checked = os.fstat(fd)
            os.close(fd)
            if read_only:
                self._db = sqlite3.connect(target.as_uri() + '?mode=ro', uri=True, timeout=0, isolation_level=None)
                current = target.stat(follow_symlinks=False)
                if (checked.st_dev, checked.st_ino) != (current.st_dev, current.st_ino):
                    raise WorkflowError('journal_unavailable')
            else:
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
        self._db.execute('BEGIN' if self._read_only else 'BEGIN IMMEDIATE')
        try:
            app = self._db.execute('PRAGMA application_id').fetchone()[0]
            version = self._db.execute('PRAGMA user_version').fetchone()[0]
            names = {r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'")}
            if not app and not version and not names and not self._read_only:
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
            if self._lock_fd >= 0:
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
        return StepContext(run_id, space, cast(dict[str, JSONValue], clean['scope']), clean['inputs'], cast(dict[str, JSONValue], clean['completed']), checkpoints, self._pause_expires_at)

    def _checkpoint_count(self, token: str) -> int:
        return int(self._db.execute('SELECT COUNT(*) FROM workflow_runs WHERE token LIKE ?',
                                    (token + ':checkpoint:%',)).fetchone()[0])

    def _clear_checkpoints(self, token: str) -> None:
        self._db.execute('DELETE FROM workflow_runs WHERE token LIKE ?', (token + ':checkpoint:%',))

    def _checkpoints(self, token: str, binding: str, step_id: str) -> StepCheckpoints:
        attempt = object()
        self._checkpoint_attempts[step_id] = attempt

        def identity(key: str) -> str:
            if not self._running or self._checkpoint_attempts.get(step_id) is not attempt:
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
        pauses = paused_state(state, self._steps, self._dependencies)
        results = cast(dict[str, JSONValue], state['results'])
        return WorkflowStatus(str(state['status']), tuple(s.step_id for s in self._steps if s.step_id in results),
                              cast(str | None, state['inflight']), dict(cast(dict[str, int], state['attempts'])),
                              cast(str | None, state['error_class']), self._checkpoint_count(token),
                              tuple(cast(list[str], state.get('inflight_steps', [state['inflight']] if state['inflight'] else []))),
                              tuple(cast(list[str], state.get('waiting_steps', []))),
                              tuple(step.step_id for step in self._steps if step.step_id in pauses))

    async def inspect_completed(self, run_id: str, *, space: str, scope: Mapping[str, JSONValue],
                                inputs: JSONValue) -> dict[str, JSONValue]:
        """Verify a committed snapshot without executing, locking or changing it.

        An active sibling may publish additional results during verification;
        this read returns only its original snapshot. Neither verifier failures
        nor cancellation alter the executing owner's journal or checkpoint leases.
        """
        _name(run_id)
        _name(space)
        if self._closed:
            raise WorkflowError('closed')
        if not isinstance(scope, Mapping):
            raise WorkflowError('invalid_payload')
        clean_scope = cast(dict[str, JSONValue], _copy(dict(scope), self._maximum))
        clean_input = _copy(inputs, self._maximum)
        signature = hashlib.sha256(_encode({'space': space, 'scope': clean_scope,
            'inputs': clean_input, 'revision': self._revision}, self._maximum)).hexdigest()
        token = hmac.new(self._key, run_id.encode(), hashlib.sha256).hexdigest()

        def snapshot() -> dict[str, JSONValue] | None:
            if self._closed:
                raise WorkflowError('closed')
            try:
                row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
            except sqlite3.Error:
                raise WorkflowError('journal_unavailable') from None
            if row is None:
                return None
            state = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
            if state['binding'] != signature:
                raise WorkflowError('binding_mismatch')
            paused_state(state, self._steps, self._dependencies)
            if state['status'] == 'sources_invalid':
                raise WorkflowError('sources_invalid')
            return state

        state = snapshot()
        if state is None:
            return {}
        try:
            valid = await asyncio.wait_for(self._verifier(
                self._context(run_id, space, clean_scope, clean_input, state)), self._deadline)
        except FileNotFoundError:
            valid = False
        except (OSError, sqlite3.OperationalError):
            raise WorkflowError('verification_unavailable') from None
        except Exception:
            valid = False
        if valid is not True:
            raise WorkflowError('sources_invalid')
        if snapshot() is None:
            raise WorkflowError('journal_unavailable')
        return cast(dict[str, JSONValue], _copy(state['results'], self._maximum))

    async def inspect_pauses(self, run_id: str, *, space: str, scope: Mapping[str, JSONValue],
                             inputs: JSONValue) -> tuple[WorkflowPauseSnapshot, ...]:
        """Verify exact pause tickets without executing, writing or issuing a lease."""
        _name(run_id)
        _name(space)
        if not isinstance(scope, Mapping):
            raise WorkflowError('invalid_payload')
        clean_scope = cast(dict[str, JSONValue], _copy(dict(scope), self._maximum))
        clean_input = _copy(inputs, self._maximum)
        signature = hashlib.sha256(_encode({'space': space, 'scope': clean_scope,
            'inputs': clean_input, 'revision': self._revision}, self._maximum)).hexdigest()
        token = hmac.new(self._key, run_id.encode(), hashlib.sha256).hexdigest()
        expires = asyncio.get_running_loop().time() + self._deadline

        def read(identifier: str) -> bytes | None:
            if self._closed:
                raise WorkflowError('closed')
            try:
                row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (identifier,)).fetchone()
                return None if row is None else bytes(row[0])
            except (sqlite3.Error, ValueError, TypeError):
                raise WorkflowError('journal_unavailable') from None

        saved = read(token)
        if saved is None:
            return ()
        state = cast(dict[str, JSONValue], json.loads(self._unseal(token, saved)))
        if state['binding'] != signature:
            raise WorkflowError('binding_mismatch')
        if state['status'] == 'sources_invalid':
            raise WorkflowError('sources_invalid')
        tickets = paused_state(state, self._steps, self._dependencies)
        payloads: dict[str, tuple[str, bytes]] = {}
        for name, ticket in tickets.items():
            encoded = _encode([signature, name, ticket['checkpoint']], 1024)
            identifier = token + ':checkpoint:' + hmac.new(self._key, encoded, hashlib.sha256).hexdigest()
            encrypted = read(identifier)
            payload = None if encrypted is None else self._unseal(identifier, encrypted)
            if not payload or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), ticket['digest']):
                raise WorkflowError('pause_checkpoint_changed')
            payloads[name] = identifier, payload

        def current() -> None:
            active = asyncio.current_task()
            if active is not None and active.cancelling():
                raise asyncio.CancelledError
            if asyncio.get_running_loop().time() >= expires:
                raise WorkflowError('verification_unavailable')
            if read(token) != saved:
                raise WorkflowError('pause_snapshot_changed')
            for identifier, original in payloads.values():
                encrypted = read(identifier)
                if encrypted is None or self._unseal(identifier, encrypted) != original:
                    raise WorkflowError('pause_snapshot_changed')

        if not tickets:
            return ()
        context = self._context(run_id, space, clean_scope, clean_input, state)
        try:
            async with asyncio.timeout(self._deadline) as timer:
                valid = await self._verifier(context)
                if timer.expired():
                    raise WorkflowError('verification_unavailable')
                current()
        except WorkflowError:
            raise
        except FileNotFoundError:
            valid = False
        except (OSError, sqlite3.OperationalError):
            raise WorkflowError('verification_unavailable') from None
        except Exception:
            valid = False
        if valid is not True:
            raise WorkflowError('sources_invalid')
        current()
        return tuple(WorkflowPauseSnapshot(step.step_id, tickets[step.step_id]['checkpoint'],
            payloads[step.step_id][1], self._context(run_id, space, clean_scope, clean_input, state), current)
            for step in self._steps if step.step_id in tickets)

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
        if self._read_only:
            raise WorkflowError('read_only_journal')
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
            partial = progress.completed_steps != expected
            if (progress.status not in {'completed', 'verification_unavailable'}
                    or progress.inflight is not None or progress.paused_steps or (partial and (self._completion is None
                    or progress.completed_steps != expected[:len(progress.completed_steps)]))):
                raise WorkflowError('not_completed')
            token = hmac.new(self._key, run_id.encode(), hashlib.sha256).hexdigest()
            row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
            if row is None:
                raise WorkflowError('journal_unavailable')
            state = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
            await asyncio.wait_for(self._verify(token, self._context(run_id, space, clean_scope, clean_input, state), state),
                                   timeout=self._deadline)
            if partial and not self._finished(self._context(run_id, space, clean_scope, clean_input, state)):
                raise WorkflowError('not_completed')
            state.update(status='completed', error_class=None)
            self._save(token, state)
            self._clear_checkpoints(token)
            return WorkflowResult(run_id, 'completed', cast(dict[str, JSONValue], _copy(state['results'], self._maximum)), progress.completed_steps)
        except asyncio.TimeoutError:
            raise WorkflowError('verification_unavailable') from None
        except sqlite3.Error:
            raise WorkflowError('journal_unavailable') from None
        finally:
            self._running = False
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    async def run(self, run_id: str, *, space: str, scope: Mapping[str, JSONValue], inputs: JSONValue,
                  resume_steps: Sequence[str] | None = None) -> WorkflowResult:
        _name(run_id)
        _name(space)
        if self._closed:
            raise WorkflowError('closed')
        if self._read_only:
            raise WorkflowError('read_only_journal')
        if self._running:
            raise WorkflowError('busy')
        if not isinstance(scope, Mapping):
            raise WorkflowError('invalid_payload')
        selected: frozenset[str] | None = None
        if resume_steps is not None:
            if (not isinstance(resume_steps, (list, tuple)) or len(resume_steps) > 32
                    or any(type(name) is not str for name in resume_steps)
                    or len(set(resume_steps)) != len(resume_steps)
                    or set(resume_steps) - {step.step_id for step in self._steps if isinstance(step, WorkflowPausableStep)}):
                raise WorkflowError('invalid_resume_selection')
            selected = frozenset(resume_steps)
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
        self._pause_expires_at = asyncio.get_running_loop().time() + self._deadline
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
            paused_state(state, self._steps, self._dependencies)
            return await asyncio.wait_for(self._execute(token, state, run_id, space, clean_scope, clean_input, selected), timeout=self._deadline)
        except asyncio.CancelledError:
            if state is not None:
                self._record_interruption(token, 'cancelled', 'CancelledError')
            raise
        except asyncio.TimeoutError:
            if state is not None:
                self._record_interruption(token, 'deadline', 'TimeoutError')
            raise WorkflowError('deadline') from None
        except sqlite3.Error:
            raise WorkflowError('journal_unavailable') from None
        finally:
            self._checkpoint_attempts.clear()
            self._pause_expires_at = None
            self._running = False
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _record_interruption(self, token: str, status: str, error_class: str) -> None:
        # A write can commit before interruption reaches its caller. Reload the
        # durable state so cancellation never resurrects a consumed pause ticket.
        try:
            row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
            if row is not None:
                durable = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
                durable.update(status=status, error_class=error_class)
                self._save(token, durable)
        except (OSError, sqlite3.Error):
            pass  # Keep the committed attempt; storage failure cannot authorize replay.

    def _check_pause_deadline(self) -> None:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError
        if self._pause_expires_at is not None and asyncio.get_running_loop().time() >= self._pause_expires_at:
            raise asyncio.TimeoutError

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
            self._checkpoint_attempts.clear()
            state.pop('pauses', None)
            state.update(status='sources_invalid', results={}, error_class=None)
            self._save(token, state)
            self._clear_checkpoints(token)
            raise WorkflowError('sources_invalid')
        self._check_pause_deadline()

    def _finished(self, context: StepContext) -> bool:
        if self._completion is None:
            return False
        try:
            result = self._completion.when(context)
            if type(result) is not bool:
                if inspect.iscoroutine(result):
                    result.close()
                raise WorkflowError('completion_failed')
            return result
        except Exception:
            raise WorkflowError('completion_failed') from None

    def _admit_pause_step(self, token: str, state: dict[str, JSONValue], step: WorkflowPausableStep,
                          checkpoints: StepCheckpoints, pending: list[str] | None = None) -> None:
        self._check_pause_deadline()
        pauses = paused_state(state, self._steps, self._dependencies)
        attempts = dict(cast(dict[str, JSONValue], state['attempts']))
        previous = cast(int, attempts.get(step.step_id, 0))
        if previous:
            receipt = pauses.get(step.step_id)
            if receipt is None:
                raise WorkflowError('outcome_unknown')
            if previous >= step.max_resumes + 1:
                raise WorkflowError('resumes_exhausted')
            validate_pause(receipt, checkpoints)
        pauses.pop(step.step_id, None)
        attempts[step.step_id] = previous + 1
        candidate = {**state, 'attempts': cast(JSONValue, attempts), 'pauses': cast(JSONValue, pauses),
                     'status': 'running', 'error_class': None,
                     'inflight': pending[0] if pending else step.step_id}
        if pending is not None:
            candidate['inflight_steps'] = cast(JSONValue, pending)
        # Admission and ticket consumption are one durable write. No callback
        # may start if this write fails, or be retried if its outcome is unknown.
        self._save(token, candidate)
        state.update(candidate)

    async def _invoke_pause_step(self, step: WorkflowPausableStep, context: StepContext) -> JSONValue | PauseReceipt:
        assert context.checkpoints is not None
        try:
            self._check_pause_deadline()
            try:
                value = await step.run(context)
            except Exception:
                self._check_pause_deadline()
                raise WorkflowError('step_failed') from None
            self._check_pause_deadline()
            if isinstance(value, WorkflowPaused):
                return pause_receipt(value, context.checkpoints)
            return _copy(value, self._maximum)
        finally:
            self._checkpoint_attempts.pop(step.step_id, None)

    def _acknowledge_pause(self, token: str, state: dict[str, JSONValue], name: str,
                           receipt: PauseReceipt, pending: list[str] | None = None) -> None:
        self._check_pause_deadline()
        # Existing tickets belong to other nodes; this attempt is still in
        # flight until its newly authenticated ticket is durably published.
        pauses = dict(cast(dict[str, JSONValue], state.get('pauses', {})))
        pauses[name] = cast(JSONValue, receipt.payload())
        candidate = {**state, 'pauses': cast(JSONValue, pauses), 'error_class': None,
                     'inflight': pending[0] if pending else None}
        if pending is not None:
            candidate['inflight_steps'] = cast(JSONValue, pending)
        self._save(token, candidate)
        state.update(candidate)
        self._check_pause_deadline()

    async def _execute(self, token: str, state: dict[str, JSONValue], run_id: str, space: str, scope: dict[str, JSONValue], inputs: JSONValue,
                       resume_steps: frozenset[str] | None = None) -> WorkflowResult:
        if state['status'] == 'sources_invalid':
            self._clear_checkpoints(token)
            raise WorkflowError('sources_invalid')
        if self._dependencies is not None:
            return await self._execute_parallel(token, state, run_id, space, scope, inputs, resume_steps)
        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
        results = cast(dict[str, JSONValue], state['results'])
        attempts = cast(dict[str, JSONValue], state['attempts'])
        reused = tuple(step.step_id for step in self._steps if step.step_id in results)
        for step in self._steps:
            assert not isinstance(step, WorkflowInputStep)  # Input nodes require the dependency scheduler.
            if step.step_id in results:
                continue
            try:
                finished = self._completion is not None and self._finished(self._context(run_id, space, scope, inputs, state))
            except WorkflowError:
                state.update(status='failed', error_class='CompletionConditionError')
                self._save(token, state)
                raise
            if finished:
                if any(name not in results for name in attempts):
                    state.update(status='outcome_unknown', error_class=None)
                    self._save(token, state)
                    raise WorkflowError('outcome_unknown')
                break
            if self._verify_before_step:
                await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
            if isinstance(step, WorkflowPausableStep):
                if (resume_steps is not None and step.step_id not in resume_steps
                        and step.step_id in paused_state(state, self._steps, self._dependencies)):
                    state.update(status='paused')
                    self._save(token, state)
                    return WorkflowResult(run_id, 'paused', cast(dict[str, JSONValue], _copy(results, self._maximum)), reused)
                checkpoints = self._checkpoints(token, cast(str, state['binding']), step.step_id)
                try:
                    self._admit_pause_step(token, state, step, checkpoints)
                    attempts = cast(dict[str, JSONValue], state['attempts'])
                    try:
                        paused_value = await self._invoke_pause_step(step,
                            self._context(run_id, space, scope, inputs, state, checkpoints))
                    except asyncio.TimeoutError:
                        raise
                    except Exception as exc:
                        state.update(status='failed', error_class=type(exc).__name__[:80])
                        self._save(token, state)
                        raise WorkflowError(exc.code if isinstance(exc, WorkflowError) else 'step_failed') from None
                finally:
                    self._checkpoint_attempts.pop(step.step_id, None)
                if isinstance(paused_value, PauseReceipt):
                    await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
                    self._acknowledge_pause(token, state, step.step_id, paused_value)
                    state.update(status='paused')
                    self._save(token, state)
                    self._check_pause_deadline()
                    return WorkflowResult(run_id, 'paused', cast(dict[str, JSONValue], _copy(results, self._maximum)), reused)
                _encode({'scope': scope, 'inputs': inputs, 'completed': {**results, step.step_id: paused_value}}, self._maximum)
                self._check_pause_deadline()
                results[step.step_id] = paused_value
                state.update(inflight=None, status='running')
                self._save(token, state)
                continue
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
                self._check_pause_deadline()
                state.update(status='running', inflight=step.step_id, error_class=None)
                attempts[step.step_id] = attempt
                self._save(token, state)
                try:
                    self._check_pause_deadline()
                    checkpoints = self._checkpoints(token, cast(str, state['binding']), step.step_id)
                    value = await step.run(self._context(run_id, space, scope, inputs, state, checkpoints))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._check_pause_deadline()
                    state.update(status='failed', error_class=type(exc).__name__[:80])
                    self._save(token, state)
                    if not self._automatic_retries or not step.retryable or attempt == maximum:
                        raise WorkflowError('step_failed') from None
                    continue
                finally:
                    self._checkpoint_attempts.pop(step.step_id, None)
                self._check_pause_deadline()
                try:
                    clean = _copy(value, self._maximum)
                    # Enforce an aggregate input/scope/results budget.
                    _encode({'scope': scope, 'inputs': inputs, 'completed': {**results, step.step_id: clean}}, self._maximum)
                except WorkflowError:
                    state.update(status='failed', error_class='InvalidPayload')
                    self._save(token, state)
                    raise
                self._check_pause_deadline()
                results[step.step_id] = clean
                state.update(inflight=None, status='running')
                self._save(token, state)
                break
        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
        state.update(status='completed', inflight=None, error_class=None)
        self._save(token, state)
        self._clear_checkpoints(token)
        return WorkflowResult(run_id, 'completed', cast(dict[str, JSONValue], _copy(results, self._maximum)), reused)

    async def _execute_parallel(self, token: str, state: dict[str, JSONValue], run_id: str,
                                space: str, scope: dict[str, JSONValue], inputs: JSONValue,
                                resume_steps: frozenset[str] | None = None) -> WorkflowResult:
        dependencies = self._dependencies
        assert dependencies is not None
        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
        results = cast(dict[str, JSONValue], state['results'])
        attempts = cast(dict[str, JSONValue], state['attempts'])
        reused = tuple(step.step_id for step in self._steps if step.step_id in results)
        pauses = paused_state(state, self._steps, self._dependencies)
        if any(name not in results and name not in pauses for name in attempts):
            state.update(status='outcome_unknown', error_class=None)
            self._save(token, state)
            raise WorkflowError('outcome_unknown')
        active: dict[str, asyncio.Task[JSONValue | PauseReceipt]] = {}
        paused_this_run = set(pauses) - resume_steps if resume_steps is not None else set()
        polled: set[str] = set()
        waiting = set(cast(list[str], state.get('waiting_steps', [])))
        has_inputs = any(isinstance(step, WorkflowInputStep) for step in self._steps)

        def waiting_fields(names: set[str]) -> dict[str, JSONValue]:
            return {'waiting_steps': [step.step_id for step in self._steps if step.step_id in names]} if has_inputs else {}

        def publish(name: str, value: JSONValue) -> None:
            self._check_pause_deadline()
            completed = {**results, name: value}
            _encode({'scope': scope, 'inputs': inputs, 'completed': completed}, self._maximum)
            pending = [step.step_id for step in self._steps if step.step_id in active and step.step_id != name]
            candidate = {**state, 'results': cast(JSONValue, completed),
                         'inflight_steps': cast(JSONValue, pending), 'inflight': pending[0] if pending else None,
                         **waiting_fields(waiting - {name})}
            self._check_pause_deadline()
            self._save(token, candidate)
            results[name] = value
            state.update(candidate)
            state['results'] = results
            waiting.discard(name)
            active.pop(name, None)

        async def poll_input(step: WorkflowInputStep) -> None:
            try:
                value = await step.poll(self._context(run_id, space, scope, inputs, state))
                if value is not None and type(value) is not WorkflowInputValue:
                    if inspect.iscoroutine(value):
                        value.close()
                    raise WorkflowError('invalid_input_value')
                clean = _copy(value.value, self._maximum) if value is not None else None
            except (OSError, sqlite3.OperationalError) as exc:
                state.update(status='verification_unavailable', error_class=type(exc).__name__[:80])
                self._save(token, state)
                raise WorkflowError('verification_unavailable') from None
            except Exception as exc:
                state.update(status='failed', error_class=type(exc).__name__[:80])
                self._save(token, state)
                raise WorkflowError(exc.code if isinstance(exc, WorkflowError) else 'input_poll_failed') from None
            # Polling may await storage. Revalidate before acknowledging waiting
            # or committing consumption, just as before admitting a model.
            await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
            if value is None:
                candidate = {**state, **waiting_fields(waiting | {step.step_id})}
                self._save(token, candidate)
                waiting.add(step.step_id)
                state.update(candidate)
            else:
                try:
                    publish(step.step_id, clean)
                except WorkflowError:
                    state.update(status='failed', error_class='InvalidPayload')
                    self._save(token, state)
                    raise
            polled.add(step.step_id)

        async def invoke(step: WorkflowStep | WorkflowPausableStep, context: StepContext) -> JSONValue | PauseReceipt:
            if isinstance(step, WorkflowPausableStep):
                return await self._invoke_pause_step(step, context)
            try:
                self._check_pause_deadline()
                value = await step.run(context)
                self._check_pause_deadline()
                return _copy(value, self._maximum)
            finally:
                self._checkpoint_attempts.pop(step.step_id, None)

        try:
            while len(results) < len(self._steps):
                completed_before = len(results)
                for step in self._steps:
                    if any(task.done() for task in active.values()):
                        break
                    if step.step_id in results or step.step_id in active or step.step_id in paused_this_run or not set(dependencies[step.step_id]) <= results.keys():
                        continue
                    if isinstance(step, WorkflowInputStep):
                        if step.step_id not in polled:
                            await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
                            if any(task.done() for task in active.values()):
                                break
                            await poll_input(step)
                        continue
                    if len(active) >= self._parallel:
                        continue
                    await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
                    if any(task.done() for task in active.values()):
                        break
                    checkpoints = self._checkpoints(token, cast(str, state['binding']), step.step_id)
                    context = self._context(run_id, space, scope, inputs, state, checkpoints)
                    pending = [item.step_id for item in self._steps if item.step_id in active or item.step_id == step.step_id]
                    if isinstance(step, WorkflowPausableStep):
                        self._admit_pause_step(token, state, step, checkpoints, pending)
                        attempts = cast(dict[str, JSONValue], state['attempts'])
                    else:
                        attempts[step.step_id] = 1
                        state.update(status='running', error_class=None, inflight=pending[0], inflight_steps=cast(JSONValue, pending))
                        self._save(token, state)
                    active[step.step_id] = asyncio.create_task(invoke(step, context))
                if not active:
                    if len(results) > completed_before:
                        # Input values can unlock a node earlier in declaration order.
                        continue
                    if waiting or paused_this_run:
                        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
                        status = 'paused' if paused_this_run else 'awaiting_input'
                        state.update(status=status, inflight=None, inflight_steps=[], error_class=None)
                        self._save(token, state)
                        if paused_this_run:
                            self._check_pause_deadline()
                        return WorkflowResult(run_id, status, cast(dict[str, JSONValue],
                            _copy(results, self._maximum)), reused)
                    raise WorkflowError('invalid_dependencies')
                done, _ = await asyncio.wait(active.values(), return_when=asyncio.FIRST_COMPLETED)
                failure: str | None = None
                failure_code = 'step_failed'
                # Drain every ready result in plan order, even if a sibling failed.
                for step in self._steps:
                    task = active.get(step.step_id)
                    if task is None or task not in done:
                        continue
                    try:
                        value = task.result()
                    except asyncio.CancelledError:
                        failure = failure or 'CancelledError'
                        continue
                    except asyncio.TimeoutError:
                        raise
                    except WorkflowError as error:
                        failure, failure_code = 'InvalidPayload', error.code
                        continue
                    except Exception as error:
                        failure = failure or type(error).__name__[:80]
                        continue
                    if isinstance(value, PauseReceipt):
                        await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
                    try:
                        if isinstance(value, PauseReceipt):
                            pending = [item.step_id for item in self._steps if item.step_id in active and item.step_id != step.step_id]
                            self._acknowledge_pause(token, state, step.step_id, value, pending)
                            paused_this_run.add(step.step_id)
                            active.pop(step.step_id, None)
                        else:
                            publish(step.step_id, value)
                    except WorkflowError as error:
                        failure, failure_code = 'InvalidPayload', error.code
                if failure is not None:
                    state.update(status='failed', error_class=failure)
                    self._save(token, state)
                    raise WorkflowError(failure_code)
            await self._verify(token, self._context(run_id, space, scope, inputs, state), state)
            state.update(status='completed', inflight=None, inflight_steps=[], error_class=None)
            self._save(token, state)
            self._clear_checkpoints(token)
            return WorkflowResult(run_id, 'completed', cast(dict[str, JSONValue], _copy(results, self._maximum)), reused)
        finally:
            for task in active.values():
                task.cancel()
            drain = asyncio.gather(*active.values(), return_exceptions=True)
            interrupted = False
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    interrupted = True
            self._checkpoint_attempts.clear()
            # A sibling's admission write may have committed before raising.
            # Cleanup must merge results into that durable state, never revive
            # the stale pause tickets held by the interrupted scheduler.
            durable: dict[str, JSONValue] | None = None
            try:
                row = self._db.execute('SELECT payload FROM workflow_runs WHERE token=?', (token,)).fetchone()
                if row is not None:
                    durable = cast(dict[str, JSONValue], json.loads(self._unseal(token, row[0])))
            except (WorkflowError, OSError, sqlite3.Error):
                pass
            if durable is not None and durable['status'] != 'sources_invalid':
                state.update(durable)
                results = cast(dict[str, JSONValue], state['results'])
                for step in self._steps:
                    task = active.get(step.step_id)
                    if task is None or task.cancelled():
                        continue
                    try:
                        value = task.result()
                    except (Exception, asyncio.CancelledError):
                        continue
                    if isinstance(value, PauseReceipt):
                        continue
                    try:
                        publish(step.step_id, value)
                    except (WorkflowError, OSError, sqlite3.Error, asyncio.CancelledError):
                        # Preserve the original failure; an uncommitted attempt
                        # remains uncertain and cannot be replayed.
                        continue
            if interrupted:
                raise asyncio.CancelledError
