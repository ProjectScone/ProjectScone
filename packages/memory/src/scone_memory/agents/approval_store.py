"""Encrypted exact-call decisions, activation and single execution admission.

Transactions serialize against run cancellation. Workflow ownership, current
source validation and unknown-operation refusal remain the caller's boundaries.
A claim is admission, never proof that an external effect completed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import sqlite3

from pydantic import ValidationError

from .approval_models import ApprovalCall, Decision, ToolApprovalActivation, ToolApprovalRecord
from .plan_store import _selections
from .run_store import AgentRunRequest, AgentRunStore, _invocation_identity
from .workflow import JSONValue, WorkflowError, _encode, _integer, _name


def invocation_digest(request: AgentRunRequest) -> str:
    return hashlib.sha256(_invocation_identity(request).encode()).hexdigest()


def decision_digest(record: ToolApprovalRecord) -> str:
    return hashlib.sha256(record.model_dump_json(exclude={
        'revision', 'activation_id', 'activated_at', 'consumed_at'}).encode()).hexdigest()


class AgentApprovalStore:
    """Shares the run store's encrypted transactions and lifetime; no execution."""

    def __init__(self, runs: AgentRunStore) -> None:
        self._runs, self._storage = runs, runs._storage

    def _prefix(self, space: str, run_id: str, *, activation: bool = False) -> str:
        return ('tool-activation:' if activation else 'tool-approval:') + self._runs._token(space, run_id) + ':'

    def _token(self, space: str, run_id: str, name: str, *, activation: bool = False) -> str:
        _name(name)
        return self._prefix(space, run_id, activation=activation) + hmac.new(
            self._runs._key, name.encode(), hashlib.sha256).hexdigest()

    def _run(self, db: sqlite3.Connection, space: str, run_id: str, *, write: bool = False) -> AgentRunRequest:
        token = self._runs._token(space, run_id)
        row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
        if row is None:
            raise WorkflowError('run_not_found')
        request = self._runs._decode(token, row[0], space)
        if write and request.cancel_requested_at is not None:
            raise WorkflowError('run_cancelled')
        return request

    def _check_call(self, request: AgentRunRequest, call: ApprovalCall) -> None:
        selected = _selections(request.plan.plan).get(call.selection_id)
        if (selected != (call.agent_id, call.model_id)
                or request.plan.bindings.get(call.selection_id) != call.binding):
            raise WorkflowError('approval_selection_mismatch')

    def _request_id(self, request: AgentRunRequest, call: ApprovalCall) -> str:
        identity: JSONValue = ['tool-approval-v1', self._runs._token(request.space, request.run_id),
                               invocation_digest(request), call.step_id, call.operation_digest]
        return hmac.new(self._runs._key, _encode(identity, 2048), hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, request: AgentRunRequest) -> ToolApprovalRecord:
        try:
            record = ToolApprovalRecord.model_validate_json(self._storage._unseal(token, payload))
            self._check_call(request, record.call)
            if (record.space != request.space or record.run_id != request.run_id
                    or record.invocation_digest != invocation_digest(request)
                    or record.request_id != self._request_id(request, record.call)
                    or token != self._token(record.space, record.run_id, record.request_id)):
                raise ValueError()
            return record
        except (ValidationError, ValueError, WorkflowError, RecursionError):
            raise WorkflowError('approval_key_or_integrity') from None

    def _get(self, db: sqlite3.Connection, request: AgentRunRequest, request_id: str) -> ToolApprovalRecord | None:
        token = self._token(request.space, request.run_id, request_id)
        row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
        return None if row is None else self._decode(token, row[0], request)

    def _put(self, db: sqlite3.Connection, record: ToolApprovalRecord) -> None:
        token = self._token(record.space, record.run_id, record.request_id)
        payload = self._storage._seal(token, record.model_dump_json().encode())
        db.execute('INSERT INTO agent_runs VALUES (?, ?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload',
                   (token, payload))

    def _capacity(self, db: sqlite3.Connection, prefix: str) -> None:
        count = db.execute('SELECT COUNT(*) FROM agent_runs WHERE token>? AND token<?', (prefix, prefix + '~')).fetchone()[0]
        if count >= 512:
            raise WorkflowError('approval_store_limit')

    def get(self, space: str, run_id: str, request_id: str) -> ToolApprovalRecord | None:
        self._token(space, run_id, request_id)
        with self._storage._access() as db:
            if db.execute('SELECT 1 FROM agent_runs WHERE token=?', (self._runs._token(space, run_id),)).fetchone() is None:
                return None
            return self._get(db, self._run(db, space, run_id), request_id)

    def list(self, space: str, run_id: str) -> tuple[ToolApprovalRecord, ...]:
        prefix = self._prefix(space, run_id)
        with self._storage._access() as db:
            request = self._run(db, space, run_id)
            rows = db.execute('SELECT token,payload FROM agent_runs WHERE token>? AND token<? ORDER BY token LIMIT 513',
                              (prefix, prefix + '~')).fetchall()
            if len(rows) > 512:
                raise WorkflowError('approval_key_or_integrity')
            return tuple(self._decode(token, payload, request) for token, payload in rows)

    @staticmethod
    def _validated_call(call: ApprovalCall) -> ApprovalCall:
        try:
            return ApprovalCall.model_validate(call.model_dump(warnings=False))
        except (ValidationError, ValueError, AttributeError, WorkflowError, RecursionError):
            raise WorkflowError('invalid_approval_call') from None

    def request(self, space: str, run_id: str, call: ApprovalCall) -> ToolApprovalRecord:
        call = self._validated_call(call)
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            self._check_call(request, call)
            request_id = self._request_id(request, call)
            prior = self._get(db, request, request_id)
            if prior is not None:
                if prior.call != call:
                    raise WorkflowError('approval_request_conflict')
                return prior
            self._capacity(db, self._prefix(space, run_id))
            record = ToolApprovalRecord(space=space, run_id=run_id, request_id=request_id,
                invocation_digest=invocation_digest(request), call=call, revision=1, created_at=datetime.now(timezone.utc))
            self._put(db, record)
            return record

    def decide(self, space: str, run_id: str, request_id: str, *, decision: Decision,
               actor: str, expected_revision: int) -> ToolApprovalRecord:
        _integer(expected_revision, 1, 1)
        _name(actor)
        if decision not in ('approve', 'deny'):
            raise WorkflowError('invalid_approval_decision')
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            prior = self._get(db, request, request_id)
            if prior is None:
                raise WorkflowError('approval_not_found')
            if prior.decision is not None:
                if prior.decision != decision or prior.decided_by != actor:
                    raise WorkflowError('approval_decision_conflict')
                return prior
            record = ToolApprovalRecord.model_validate({**prior.model_dump(), 'decision': decision,
                'decided_by': actor, 'decided_at': datetime.now(timezone.utc), 'revision': 2})
            self._put(db, record)
            return record

    def _activation(self, db: sqlite3.Connection, request: AgentRunRequest, activation_id: str) -> ToolApprovalActivation | None:
        token = self._token(request.space, request.run_id, activation_id, activation=True)
        row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
        if row is None:
            return None
        try:
            saved = ToolApprovalActivation.model_validate_json(self._storage._unseal(token, row[0]))
            if (saved.space != request.space or saved.run_id != request.run_id or saved.activation_id != activation_id
                    or saved.invocation_digest != invocation_digest(request)):
                raise ValueError()
            for request_id, digest in saved.decision_digests.items():
                record = self._get(db, request, request_id)
                if (record is None or record.activation_id != activation_id or record.revision < 3
                        or decision_digest(record) != digest):
                    raise ValueError()
            return saved
        except (ValidationError, ValueError, WorkflowError, RecursionError):
            raise WorkflowError('approval_key_or_integrity') from None

    def activation(self, space: str, run_id: str, activation_id: str) -> ToolApprovalActivation | None:
        self._token(space, run_id, activation_id, activation=True)
        with self._storage._access() as db:
            return self._activation(db, self._run(db, space, run_id), activation_id)

    def activate(self, space: str, run_id: str, activation_id: str, *, decisions: dict[str, int]) -> ToolApprovalActivation:
        _name(activation_id)
        if not isinstance(decisions, dict) or not 1 <= len(decisions) <= 32:
            raise WorkflowError('invalid_approval_activation')
        decisions = dict(decisions)
        for request_id, revision in decisions.items():
            _name(request_id)
            _integer(revision, 2, 2)
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            prior = self._activation(db, request, activation_id)
            if prior is not None:
                if prior.decisions != decisions:
                    raise WorkflowError('approval_activation_conflict')
                return prior
            records = []
            for request_id in decisions:
                record = self._get(db, request, request_id)
                if record is None or record.decision is None:
                    raise WorkflowError('approval_not_decided')
                if record.revision != 2:
                    raise WorkflowError('approval_activation_conflict')
                records.append(record)
            self._capacity(db, self._prefix(space, run_id, activation=True))
            now = datetime.now(timezone.utc)
            activation = ToolApprovalActivation(space=space, run_id=run_id, activation_id=activation_id,
                invocation_digest=invocation_digest(request), decisions=decisions,
                decision_digests={record.request_id: decision_digest(record) for record in records}, created_at=now)
            for record in records:
                self._put(db, ToolApprovalRecord.model_validate({**record.model_dump(), 'revision': 3,
                    'activation_id': activation_id, 'activated_at': now}))
            token = self._token(space, run_id, activation_id, activation=True)
            db.execute('INSERT INTO agent_runs VALUES (?, ?)',
                (token, self._storage._seal(token, activation.model_dump_json().encode())))
            return activation

    def claim(self, space: str, run_id: str, request_id: str, *, activation_id: str, call: ApprovalCall) -> ToolApprovalRecord:
        _name(activation_id)
        call = self._validated_call(call)
        with self._storage._access(write=True) as db:
            request = self._run(db, space, run_id, write=True)
            prior = self._get(db, request, request_id)
            if prior is None or prior.activation_id is None or prior.activation_id != activation_id:
                raise WorkflowError('approval_not_activated')
            if prior.call != call:
                raise WorkflowError('approval_request_conflict')
            activation = self._activation(db, request, activation_id)
            if activation is None or request_id not in activation.decisions:
                raise WorkflowError('approval_key_or_integrity')
            if prior.revision == 4:
                raise WorkflowError('approval_already_claimed')
            record = ToolApprovalRecord.model_validate({**prior.model_dump(), 'revision': 4,
                                                        'consumed_at': datetime.now(timezone.utc)})
            self._put(db, record)
            return record
