"""Immutable encrypted invocation snapshots for locally managed agent runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ._encrypted_store import EncryptedRecordStore
from .catalog import Identifier
from .plan_store import SavedAgentPlan
from .workflow import WorkflowError, _integer, _name
from ..core.errors import InvalidInput
from ..core.validation import check_space
from ..retrieval.recall_scope import RecallScope

_HEX = re.compile(r'[0-9a-f]{64}\Z')


class RunConflict(WorkflowError):
    def __init__(self) -> None:
        super().__init__('run_request_conflict')


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str
    run_id: Identifier
    plan: SavedAgentPlan
    question: str = Field(min_length=1, max_length=4000)
    max_parallel: int = Field(default=1, ge=1, le=8)
    scope: dict[str, object]
    exclude_session_id: Identifier | None = None
    created_at: datetime
    cancel_requested_at: datetime | None = None

    @model_validator(mode='after')
    def valid(self) -> Self:
        check_space(self.space)
        if (self.plan.space != self.space or not self.question.strip()
                or len(self.question.encode()) > 4000 or self.created_at.tzinfo is None
                or (self.cancel_requested_at is not None and self.cancel_requested_at.tzinfo is None)):
            raise ValueError('invalid agent run request')
        try:
            canonical = RecallScope.from_mapping(self.scope).as_dict()
        except (ValueError, InvalidInput):
            raise ValueError('invalid agent run scope') from None
        if canonical != self.scope:
            raise ValueError('agent run scope must be canonical')
        return self

    def recall_scope(self) -> RecallScope:
        return RecallScope.from_mapping(self.scope)


@dataclass(frozen=True)
class AgentRunPage:
    items: tuple[AgentRunRequest, ...]
    next_after: str | None


def _request_bytes(request: AgentRunRequest) -> bytes:
    # Preserve the sequential record shape for existing local readers.
    return request.model_dump_json(exclude={'max_parallel'} if request.max_parallel == 1 else set()).encode()


class AgentRunStore:
    """Request registration is immutable and does not execute or resume a model.

    The caller authorizes the space and validates the saved plan against the host
    catalog before registration/execution. The store keeps a detached snapshot;
    changing a plan later cannot change a registered run. Execution receipts live
    separately in AgentWorkflow journals. A separate cancellation-intent timestamp
    may be added without changing the invocation. Closing this store never cancels runs.
    """
    def __init__(self, path: str | Path, *, key: bytes, max_runs: int = 4096) -> None:
        _integer(max_runs, 1, 100000)
        self._key, self._maximum = key, max_runs
        self._storage = EncryptedRecordStore(path, key=key, table='agent_runs',
            metadata='agent_run_meta', application_id=0x53435251, domain='scone-agent-runs-v1', label='run')

    def _prefix(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, b'agent-run-space:' + space.encode(), hashlib.sha256).hexdigest() + ':'

    def _token(self, space: str, run_id: str) -> str:
        _name(run_id)
        return self._prefix(space) + hmac.new(self._key, b'agent-run-id:' + run_id.encode(), hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, space: str) -> AgentRunRequest:
        try:
            request = AgentRunRequest.model_validate_json(self._storage._unseal(token, payload))
        except ValidationError:
            raise WorkflowError('run_key_or_integrity') from None
        if request.space != space or self._token(request.space, request.run_id) != token:
            raise WorkflowError('run_key_or_integrity')
        return request

    def get(self, space: str, run_id: str) -> AgentRunRequest | None:
        token = self._token(space, run_id)
        with self._storage._access() as db:
            row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
            return None if row is None else self._decode(token, row[0], space)

    def register(self, space: str, run_id: str, *, plan: SavedAgentPlan, question: str,
                 scope: RecallScope, exclude_session_id: str | None = None, max_parallel: int = 1) -> AgentRunRequest:
        token = self._token(space, run_id)
        if not isinstance(scope, RecallScope):
            raise ValueError('validated recall scope required')
        saved = AgentRunRequest(space=space, run_id=run_id,
            plan=SavedAgentPlan.model_validate(plan.model_dump()), question=question, max_parallel=max_parallel,
            scope=RecallScope.validated(**scope.kwargs()).as_dict(), exclude_session_id=exclude_session_id,
            created_at=datetime.now(timezone.utc))
        payload = self._storage._seal(token, _request_bytes(saved))
        with self._storage._access(write=True) as db:
            row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
            if row is not None:
                prior = self._decode(token, row[0], space)
                if prior.model_dump(exclude={'created_at', 'cancel_requested_at'}) != saved.model_dump(exclude={'created_at', 'cancel_requested_at'}):
                    raise RunConflict()
                return prior
            if db.execute('SELECT COUNT(*) FROM agent_runs').fetchone()[0] >= self._maximum:
                raise WorkflowError('run_store_limit')
            db.execute('INSERT INTO agent_runs VALUES (?, ?)', (token, payload))
        return saved

    def request_cancel(self, space: str, run_id: str) -> AgentRunRequest | None:
        """Persist cancellation intent without changing the bound invocation."""
        token = self._token(space, run_id)
        with self._storage._access(write=True) as db:
            row = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
            if row is None:
                return None
            request = self._decode(token, row[0], space)
            if request.cancel_requested_at is not None:
                return request
            request = AgentRunRequest.model_validate({**request.model_dump(),
                'cancel_requested_at': datetime.now(timezone.utc)})
            db.execute('UPDATE agent_runs SET payload=? WHERE token=?',
                       (self._storage._seal(token, _request_bytes(request)), token))
            return request

    def list(self, space: str, *, limit: int = 50, after: str | None = None) -> AgentRunPage:
        _integer(limit, 1, 100)
        prefix = self._prefix(space)
        if after is not None and (not isinstance(after, str) or not after.startswith(prefix)
                                  or not _HEX.fullmatch(after[len(prefix):])):
            raise WorkflowError('invalid_run_cursor')
        with self._storage._access() as db:
            rows = db.execute('SELECT token,payload FROM agent_runs WHERE token>? AND token<? ORDER BY token LIMIT ?',
                              (after or prefix, prefix + '~', limit + 1)).fetchall()
            items = tuple(self._decode(token, payload, space) for token, payload in rows[:limit])
            return AgentRunPage(items, rows[limit - 1][0] if len(rows) > limit else None)

    def close(self) -> None:
        self._storage.close()
