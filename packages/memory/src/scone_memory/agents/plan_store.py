"""Encrypted, revision-checked task plans with explicit host-model bindings."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Self

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .catalog import AgentCatalog
from .task_workflow import AgentTaskPlan
from .workflow import WorkflowError, _integer, _name, _private_file
from ..core.validation import check_space

_APP_ID = 0x5343504C
_MAX_BYTES = 128000
_MAX_REVISION = 2**63 - 1
_HEX = re.compile(r'[0-9a-f]{64}\Z')


class PlanConflict(WorkflowError):
    def __init__(self) -> None:
        super().__init__('plan_revision_conflict')


class PlanConfigurationChanged(WorkflowError):
    def __init__(self) -> None:
        super().__init__('plan_configuration_changed')


class SavedAgentPlan(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str
    revision: int = Field(ge=1, le=_MAX_REVISION)
    plan: AgentTaskPlan
    bindings: dict[str, str]
    updated_at: datetime

    @model_validator(mode='after')
    def valid(self) -> Self:
        check_space(self.space)
        if (set(self.bindings) != {task.task_id for task in self.plan.tasks}
                or any(not _HEX.fullmatch(binding) for binding in self.bindings.values())
                or any(task.model_id is None for task in self.plan.tasks)
                or self.updated_at.tzinfo is None):
            raise ValueError('invalid saved agent plan')
        return self

    def checked_plan(self, catalog: AgentCatalog) -> AgentTaskPlan:
        """Refuse changed host configuration; never silently adopt new defaults."""
        saved = SavedAgentPlan.model_validate(self.model_dump())
        try:
            for task in saved.plan.tasks:
                bound = catalog.bind(task.agent_id, model_id=task.model_id)
                if bound.fingerprint != saved.bindings[task.task_id]:
                    raise PlanConfigurationChanged()
        except (KeyError, ValueError):
            raise PlanConfigurationChanged() from None
        return saved.plan


@dataclass(frozen=True)
class AgentPlanPage:
    items: tuple[SavedAgentPlan, ...]
    next_after: str | None


class AgentPlanStore:
    """Local SQLite storage; authorization is enforced by the owning application.

    Space and plan names are HMAC-indexed; full plans are authenticated ciphertext.
    Saves use compare-and-swap revisions across local connections. The host owns
    key storage, backup retention and catalog permissions. The containing directory
    must be owned by this user and not writable by other users. As with SQLite
    generally, concurrent path replacement by the same OS user is unsupported.
    No model is invoked.
    """
    def __init__(self, path: str | Path, *, key: bytes, max_plans: int = 4096) -> None:
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        _integer(max_plans, 1, 100000)
        self._cipher, self._key, self._maximum = AESGCM(key), key, max_plans
        self._closed = False
        descriptor = -1
        try:
            target = Path(path).absolute()
            parent = target.parent.stat(follow_symlinks=False)
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
                    or parent.st_mode & 0o022):
                raise WorkflowError('private_directory_required')
            descriptor = _private_file(target)

            def unchanged_file() -> None:
                opened = os.fstat(descriptor)
                current = target.stat(follow_symlinks=False)
                if (not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid()
                        or current.st_mode & 0o077 or current.st_nlink != 1
                        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
                    raise WorkflowError('private_file_required')

            unchanged_file()
            self._db = sqlite3.connect(target, timeout=1, isolation_level=None)
            unchanged_file()
            self._db.execute('PRAGMA synchronous=FULL')
            with self._access(write=True) as db:
                app = db.execute('PRAGMA application_id').fetchone()[0]
                version = db.execute('PRAGMA user_version').fetchone()[0]
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'")}
                if not app and not version and not names:
                    db.execute('CREATE TABLE agent_plans (token TEXT PRIMARY KEY, payload BLOB NOT NULL)')
                    db.execute('CREATE TABLE agent_plan_meta (payload BLOB NOT NULL)')
                    db.execute('INSERT INTO agent_plan_meta VALUES (?)', (self._seal('key-check', b'scone-agent-plans-v1'),))
                    db.execute(f'PRAGMA application_id={_APP_ID}')
                    db.execute('PRAGMA user_version=1')
                elif app != _APP_ID or version != 1 or names != {'agent_plans', 'agent_plan_meta'}:
                    raise WorkflowError('foreign_plan_store')
                markers = db.execute('SELECT payload FROM agent_plan_meta LIMIT 2').fetchall()
                if len(markers) != 1 or self._unseal('key-check', markers[0][0]) != b'scone-agent-plans-v1':
                    raise WorkflowError('plan_key_or_integrity')
        except BaseException as error:
            if hasattr(self, '_db'):
                self._db.close()
            if isinstance(error, (OSError, sqlite3.Error)):
                raise WorkflowError('plan_store_unavailable') from None
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @contextmanager
    def _access(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise WorkflowError('plan_store_closed')
        try:
            self._db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield self._db
                self._db.execute('COMMIT')
            except BaseException:
                self._db.execute('ROLLBACK')
                raise
        except sqlite3.Error:
            raise WorkflowError('plan_store_unavailable') from None

    def _seal(self, token: str, payload: bytes) -> bytes:
        if len(payload) > _MAX_BYTES:
            raise WorkflowError('plan_payload_limit')
        nonce = os.urandom(12)
        return nonce + self._cipher.encrypt(nonce, payload, ('scone-agent-plans-v1:' + token).encode())

    def _unseal(self, token: str, payload: object) -> bytes:
        if not isinstance(payload, bytes) or not 28 <= len(payload) <= _MAX_BYTES + 28:
            raise WorkflowError('plan_key_or_integrity')
        try:
            return self._cipher.decrypt(payload[:12], payload[12:], ('scone-agent-plans-v1:' + token).encode())
        except (InvalidTag, ValueError):
            raise WorkflowError('plan_key_or_integrity') from None

    def _prefix(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, b'agent-plan-space:' + space.encode(), hashlib.sha256).hexdigest() + ':'

    def _token(self, space: str, workflow_id: str) -> str:
        _name(workflow_id)
        return self._prefix(space) + hmac.new(self._key, b'agent-plan-id:' + workflow_id.encode(), hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, space: str) -> SavedAgentPlan:
        try:
            saved = SavedAgentPlan.model_validate_json(self._unseal(token, payload))
        except ValidationError:
            raise WorkflowError('plan_key_or_integrity') from None
        if saved.space != space or self._token(saved.space, saved.plan.workflow_id) != token:
            raise WorkflowError('plan_key_or_integrity')
        return saved

    def get(self, space: str, workflow_id: str) -> SavedAgentPlan | None:
        token = self._token(space, workflow_id)
        with self._access() as db:
            row = db.execute('SELECT payload FROM agent_plans WHERE token=?', (token,)).fetchone()
            return None if row is None else self._decode(token, row[0], space)

    def save(self, space: str, plan: AgentTaskPlan, *, catalog: AgentCatalog,
             expected_revision: int) -> SavedAgentPlan:
        _integer(expected_revision, 0, _MAX_REVISION - 1)
        plan = AgentTaskPlan.model_validate(plan.model_dump())
        token = self._token(space, plan.workflow_id)
        agents = {task.task_id: catalog.bind(task.agent_id, model_id=task.model_id) for task in plan.tasks}
        explicit = AgentTaskPlan(workflow_id=plan.workflow_id, tasks=tuple(
            task.model_copy(update={'model_id': agents[task.task_id].model_id}) for task in plan.tasks))
        saved = SavedAgentPlan(space=space, revision=expected_revision + 1, plan=explicit,
            bindings={name: agent.fingerprint for name, agent in agents.items()}, updated_at=datetime.now(timezone.utc))
        payload = self._seal(token, saved.model_dump_json().encode())
        with self._access(write=True) as db:
            row = db.execute('SELECT payload FROM agent_plans WHERE token=?', (token,)).fetchone()
            revision = 0 if row is None else self._decode(token, row[0], space).revision
            if revision != expected_revision:
                raise PlanConflict()
            if row is None and db.execute('SELECT COUNT(*) FROM agent_plans').fetchone()[0] >= self._maximum:
                raise WorkflowError('plan_store_limit')
            db.execute('INSERT INTO agent_plans VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload',
                       (token, payload))
        return saved

    def list(self, space: str, *, limit: int = 50, after: str | None = None) -> AgentPlanPage:
        _integer(limit, 1, 100)
        prefix = self._prefix(space)
        if after is not None and (not isinstance(after, str) or not after.startswith(prefix)
                                  or not _HEX.fullmatch(after[len(prefix):])):
            raise WorkflowError('invalid_plan_cursor')
        with self._access() as db:
            rows = db.execute('SELECT token,payload FROM agent_plans WHERE token>? AND token<? ORDER BY token LIMIT ?',
                              (after or prefix, prefix + '~', limit + 1)).fetchall()
            items = tuple(self._decode(token, payload, space) for token, payload in rows[:limit])
            return AgentPlanPage(items, rows[limit - 1][0] if len(rows) > limit else None)

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True
