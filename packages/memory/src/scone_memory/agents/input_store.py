"""Encrypted, single-assignment human replies and explicit continuation receipts.

The service verifies current scope, sources and execution ownership before these
synchronous transactions. Storing an answer alone never authorizes execution.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import sqlite3
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .interactive_plan import HumanInputTask, InteractiveAgentPlan
from .run_store import AgentRunRequest, AgentRunStore
from .workflow import WorkflowError, _integer, _name


class AgentInputRecord(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str
    run_id: str
    task_id: str
    invocation_digest: str
    prompt: str
    context: str
    max_response_bytes: int = Field(ge=1, le=4000)
    revision: int = Field(ge=1, le=3)
    response: str | None = None
    activation_id: str | None = None
    created_at: datetime
    responded_at: datetime | None = None

    @model_validator(mode='after')
    def valid(self) -> Self:
        _name(self.run_id)
        _name(self.task_id)
        if self.activation_id is not None:
            _name(self.activation_id)
        if (self.created_at.tzinfo is None
                or (self.responded_at is not None and self.responded_at.tzinfo is None)
                or (self.revision == 1) != (self.response is None)
                or (self.response is None) != (self.responded_at is None)
                or (self.revision == 3) != (self.activation_id is not None)
                or len(self.context.encode()) > 32000
                or (self.response is not None and (not self.response.strip()
                    or len(self.response.encode()) > self.max_response_bytes))):
            raise ValueError('invalid input record')
        return self


class AgentInputActivation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str
    run_id: str
    continuation_id: str
    invocation_digest: str
    responses: dict[str, int] = Field(min_length=1, max_length=32)
    response_digests: dict[str, str]
    created_at: datetime

    @model_validator(mode='after')
    def valid(self) -> Self:
        _name(self.run_id)
        _name(self.continuation_id)
        for task_id, revision in self.responses.items():
            _name(task_id)
            if revision != 2:
                raise ValueError('activation requires answered revision')
        if self.responses.keys() != self.response_digests.keys() or self.created_at.tzinfo is None:
            raise ValueError('invalid activation receipt')
        return self


def _invocation_digest(request: AgentRunRequest) -> str:
    payload = request.model_dump(mode='json', exclude={'created_at', 'cancel_requested_at'})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _response_digest(record: AgentInputRecord) -> str:
    payload = record.model_dump(mode='json', exclude={'revision', 'activation_id'})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class AgentInputStore:
    """Shares the run store transaction, encryption key and lifetime."""
    def __init__(self, runs: AgentRunStore) -> None:
        self._runs = runs
        self._storage = runs._storage

    def _prefix(self, space: str, run_id: str) -> str:
        return 'input:' + self._runs._token(space, run_id) + ':'

    def _token(self, space: str, run_id: str, name: str, *, activation: bool = False) -> str:
        _name(name)
        domain = 'activation' if activation else 'input'
        suffix = hmac.new(self._runs._key, (domain + ':' + name).encode(), hashlib.sha256).hexdigest()
        return domain + ':' + self._runs._token(space, run_id) + ':' + suffix

    def _run(self, db: sqlite3.Connection, space: str, run_id: str, *, write: bool = False) -> AgentRunRequest:
        token = self._runs._token(space, run_id)
        row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
        if row is None:
            raise WorkflowError('run_not_found')
        request = self._runs._decode(token, row[0], space)
        if write and request.cancel_requested_at is not None:
            raise WorkflowError('run_cancelled')
        return request

    def _task(self, request: AgentRunRequest, task_id: str) -> HumanInputTask:
        _name(task_id)
        plan = request.plan.plan
        if isinstance(plan, InteractiveAgentPlan):
            for task in plan.tasks:
                if task.task_id == task_id and isinstance(task, HumanInputTask):
                    return task
        raise WorkflowError('input_task_not_found')

    def _decode(self, token: str, payload: object, request: AgentRunRequest) -> AgentInputRecord:
        try:
            record = AgentInputRecord.model_validate_json(self._storage._unseal(token, payload))
            task = self._task(request, record.task_id)
        except (ValidationError, ValueError):
            raise WorkflowError('input_key_or_integrity') from None
        if (record.space != request.space or record.run_id != request.run_id
                or self._token(record.space, record.run_id, record.task_id) != token
                or record.invocation_digest != _invocation_digest(request)
                or record.prompt != task.prompt or record.max_response_bytes != task.max_response_bytes):
            raise WorkflowError('input_key_or_integrity')
        return record

    def _get(self, db: sqlite3.Connection, request: AgentRunRequest, task_id: str) -> AgentInputRecord | None:
        token = self._token(request.space, request.run_id, task_id)
        row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
        return None if row is None else self._decode(token, row[0], request)

    def _put(self, db: sqlite3.Connection, record: AgentInputRecord) -> None:
        token = self._token(record.space, record.run_id, record.task_id)
        payload = self._storage._seal(token, record.model_dump_json().encode())
        db.execute('INSERT INTO agent_runs VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload',
                   (token, payload))

    def get(self, space: str, run_id: str, task_id: str) -> AgentInputRecord | None:
        self._token(space, run_id, task_id)
        with self._storage._access() as db:
            token = self._runs._token(space, run_id)
            if db.execute('SELECT 1 FROM agent_runs WHERE token=?', (token,)).fetchone() is None:
                return None
            return self._get(db, self._run(db, space, run_id), task_id)

    def list(self, space: str, run_id: str) -> tuple[AgentInputRecord, ...]:
        prefix = self._prefix(space, run_id)
        with self._storage._access() as db:
            request = self._run(db, space, run_id)
            rows = db.execute('SELECT token,payload FROM agent_runs WHERE token>? AND token<? ORDER BY token LIMIT 33',
                              (prefix, prefix + '~')).fetchall()
            if len(rows) > 32:
                raise WorkflowError('input_key_or_integrity')
            return tuple(self._decode(token, payload, request) for token, payload in rows)

    def activated(self, space: str, run_id: str) -> tuple[AgentInputRecord, ...]:
        return tuple(record for record in self.list(space, run_id) if record.activation_id is not None)

    def request(self, space: str, run_id: str, task_id: str, *, context: str) -> AgentInputRecord:
        if not isinstance(context, str) or len(context.encode()) > 32000:
            raise WorkflowError('invalid_input_context')
        try:
            json.loads(context)
        except (ValueError, RecursionError):
            raise WorkflowError('invalid_input_context') from None
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            task = self._task(request, task_id)
            prior = self._get(db, request, task_id)
            if prior is not None:
                if prior.context != context:
                    raise WorkflowError('input_request_conflict')
                return prior
            record = AgentInputRecord(space=space, run_id=run_id, task_id=task_id,
                invocation_digest=_invocation_digest(request), prompt=task.prompt, context=context,
                max_response_bytes=task.max_response_bytes, revision=1, created_at=datetime.now(timezone.utc))
            self._put(db, record)
            return record

    def respond(self, space: str, run_id: str, task_id: str, *, response: str,
                expected_revision: int) -> AgentInputRecord:
        _integer(expected_revision, 1, 3)
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            prior = self._get(db, request, task_id)
            if prior is None:
                raise WorkflowError('input_not_found')
            if expected_revision != 1:
                raise WorkflowError('input_revision_conflict')
            if not isinstance(response, str) or not response.strip() or len(response.encode()) > prior.max_response_bytes:
                raise WorkflowError('invalid_input_response')
            if prior.response is not None:
                if prior.response != response:
                    raise WorkflowError('input_response_conflict')
                return prior
            record = AgentInputRecord.model_validate({**prior.model_dump(), 'response': response,
                'revision': 2, 'responded_at': datetime.now(timezone.utc)})
            self._put(db, record)
            return record

    def activate(self, space: str, run_id: str, continuation_id: str, *,
                 responses: dict[str, int]) -> AgentInputActivation:
        token = self._token(space, run_id, continuation_id, activation=True)
        if not isinstance(responses, dict) or not 1 <= len(responses) <= 32:
            raise WorkflowError('invalid_input_activation')
        for name, revision in responses.items():
            _name(name)
            _integer(revision, 1, 3)
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
            if row is not None:
                try:
                    prior = AgentInputActivation.model_validate_json(self._storage._unseal(token, row[0]))
                except ValidationError:
                    raise WorkflowError('input_key_or_integrity') from None
                if (prior.space != space or prior.run_id != run_id or prior.continuation_id != continuation_id
                        or prior.invocation_digest != _invocation_digest(request)):
                    raise WorkflowError('input_key_or_integrity')
                if prior.responses != responses:
                    raise WorkflowError('input_activation_conflict')
                for task_id, digest in prior.response_digests.items():
                    record = self._get(db, request, task_id)
                    if record is None or record.activation_id != continuation_id or _response_digest(record) != digest:
                        raise WorkflowError('input_key_or_integrity')
                return prior
            records = []
            for task_id, revision in responses.items():
                record = self._get(db, request, task_id)
                if record is None or record.response is None:
                    raise WorkflowError('input_not_answered')
                if record.activation_id is not None:
                    raise WorkflowError('input_activation_conflict')
                if revision != record.revision:
                    raise WorkflowError('input_revision_conflict')
                records.append(record)
            activation = AgentInputActivation(space=space, run_id=run_id, continuation_id=continuation_id,
                invocation_digest=_invocation_digest(request), responses=dict(responses),
                response_digests={record.task_id: _response_digest(record) for record in records},
                created_at=datetime.now(timezone.utc))
            for record in records:
                self._put(db, AgentInputRecord.model_validate({**record.model_dump(),
                    'revision': 3, 'activation_id': continuation_id}))
            db.execute('INSERT INTO agent_runs VALUES (?, ?)',
                (token, self._storage._seal(token, activation.model_dump_json().encode())))
            return activation
