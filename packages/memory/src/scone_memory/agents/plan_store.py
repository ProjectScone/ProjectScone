"""Encrypted, revision-checked task plans with explicit host-model bindings."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
import re
import sqlite3
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ._encrypted_store import EncryptedRecordStore
from .catalog import AgentCatalog, BoundAgent
from .handoff_workflow import AgentHandoffPlan
from .interactive_plan import InteractiveAgentPlan
from .task_workflow import AgentTask, AgentTaskPlan
from .workflow import WorkflowError, _integer, _name
from ..core.validation import check_space

_APP_ID = 0x5343504C
_MAX_REVISION = 2**63 - 1
_HEX = re.compile(r'[0-9a-f]{64}\Z')
AgentPlan = AgentTaskPlan | AgentHandoffPlan | InteractiveAgentPlan


def _selections(plan: AgentPlan) -> dict[str, tuple[str, str | None]]:
    if isinstance(plan, (AgentTaskPlan, InteractiveAgentPlan)):
        return {task.task_id: (task.agent_id, task.model_id) for task in plan.tasks
                if isinstance(task, AgentTask)}
    return {agent.agent_id: (agent.agent_id, agent.model_id) for agent in plan.agents}


def _snapshot(plan: AgentPlan) -> AgentPlan:
    if isinstance(plan, AgentTaskPlan):
        return AgentTaskPlan.model_validate(plan.model_dump())
    if isinstance(plan, AgentHandoffPlan):
        return AgentHandoffPlan.model_validate(plan.model_dump())
    if isinstance(plan, InteractiveAgentPlan):
        return InteractiveAgentPlan.model_validate(plan.model_dump())
    raise ValueError('validated agent plan required')


def _explicit(plan: AgentPlan, agents: dict[str, BoundAgent]) -> AgentPlan:
    if isinstance(plan, InteractiveAgentPlan):
        return InteractiveAgentPlan(kind='interactive', workflow_id=plan.workflow_id, tasks=tuple(
            task.model_copy(update={'model_id': agents[task.task_id].model_id})
            if isinstance(task, AgentTask) else task for task in plan.tasks))
    if isinstance(plan, AgentTaskPlan):
        return AgentTaskPlan(workflow_id=plan.workflow_id, tasks=tuple(
            task.model_copy(update={'model_id': agents[task.task_id].model_id}) for task in plan.tasks))
    return AgentHandoffPlan(workflow_id=plan.workflow_id, root_agent=plan.root_agent,
        max_handoffs=plan.max_handoffs, agents=tuple(
            agent.model_copy(update={'model_id': agents[agent.agent_id].model_id}) for agent in plan.agents))


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
    plan: AgentPlan
    bindings: dict[str, str]
    updated_at: datetime

    @model_validator(mode='after')
    def valid(self) -> Self:
        check_space(self.space)
        selections = _selections(self.plan)
        if (set(self.bindings) != set(selections)
                or any(not _HEX.fullmatch(binding) for binding in self.bindings.values())
                or any(model is None for _, model in selections.values())
                or self.updated_at.tzinfo is None):
            raise ValueError('invalid saved agent plan')
        return self

    def checked_plan(self, catalog: AgentCatalog) -> AgentPlan:
        """Refuse changed host configuration; never silently adopt new defaults."""
        saved = SavedAgentPlan.model_validate(self.model_dump())
        try:
            for name, (agent_id, model_id) in _selections(saved.plan).items():
                bound = catalog.bind(agent_id, model_id=model_id)
                if bound.fingerprint != saved.bindings[name]:
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
        _integer(max_plans, 1, 100000)
        self._maximum, self._key = max_plans, key
        self._storage = EncryptedRecordStore(path, key=key, table='agent_plans',
            metadata='agent_plan_meta', application_id=_APP_ID, domain='scone-agent-plans-v1', label='plan')

    def _access(self, *, write: bool = False) -> AbstractContextManager[sqlite3.Connection]:
        return self._storage._access(write=write)

    def _seal(self, token: str, payload: bytes) -> bytes:
        return self._storage._seal(token, payload)

    def _unseal(self, token: str, payload: object) -> bytes:
        return self._storage._unseal(token, payload)

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

    def save(self, space: str, plan: AgentPlan, *, catalog: AgentCatalog,
             expected_revision: int) -> SavedAgentPlan:
        _integer(expected_revision, 0, _MAX_REVISION - 1)
        plan = _snapshot(plan)
        token = self._token(space, plan.workflow_id)
        agents = {name: catalog.bind(agent_id, model_id=model_id)
                  for name, (agent_id, model_id) in _selections(plan).items()}
        explicit = _explicit(plan, agents)
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
        self._storage.close()
