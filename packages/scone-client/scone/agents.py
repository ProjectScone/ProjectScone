"""Explicit agent operations; construction, reads and saves never resume work."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._wire import ResourceClient, address, bounded_body, cursor, identifier, integer, invalid, items, record, text
from .agent_models import (AgentChoice, HandoffPlan, Plan, RunPolicy, RunRequest, RunStatus,
                           SavedPlan, TaskPlan, parse_catalog)


@dataclass(frozen=True)
class PlanPage:
    items: tuple[SavedPlan, ...]
    next_after: Optional[str]


@dataclass(frozen=True)
class RunPage:
    items: tuple[RunStatus, ...]
    next_after: Optional[str]


class AgentClient(ResourceClient):
    def catalog(self) -> tuple[AgentChoice, ...]:
        self._check('agents.catalog')
        return parse_catalog(self._client._request('GET', '/v1/agents/catalog'))

    def policy(self) -> RunPolicy:
        self._check('agents.runs')
        return RunPolicy.from_json(self._client._request('GET', '/v1/agents/run-policy'),
                                   expected_space=self.expected_space)

    def plans(self, *, limit: int = 20, after: Optional[str] = None) -> PlanPage:
        params = {'limit': str(integer(limit, 1, 100))}
        if after is not None:
            params['after'] = cursor(after) or ''
        self._check('agents.plans')
        row = record(self._client._request('GET', '/v1/agent-plans', params=params))
        saved = tuple(SavedPlan.from_json(value, expected_space=self.expected_space)
                      for value in items(row.get('items'), limit))
        next_after = cursor(row.get('next_after'))
        if len({item.plan.workflow_id for item in saved}) != len(saved) or (after is not None and next_after == after):
            raise invalid('plan page')
        return PlanPage(saved, next_after)

    def plan(self, workflow_id: str) -> SavedPlan:
        path = '/v1/agent-plans/' + address(workflow_id)
        self._check('agents.plans')
        saved = SavedPlan.from_json(self._client._request('GET', path), expected_space=self.expected_space)
        if saved.plan.workflow_id != workflow_id:
            raise invalid('plan identity')
        return saved

    def save_plan(self, plan: Plan, *, expected_revision: int) -> SavedPlan:
        if not isinstance(plan, (TaskPlan, HandoffPlan)):
            raise invalid('plan type')
        integer(expected_revision, 0, 2**63 - 2)
        capability = 'agents.handoffs' if isinstance(plan, HandoffPlan) else 'agents.inputs' if plan.interactive else 'agents.plans'
        body = bounded_body({'plan': plan.to_json(), 'expected_revision': expected_revision}, 128000)
        self._check('agents.plans', mutation=True).require(capability)
        saved = SavedPlan.from_json(self._client._request('PUT', '/v1/agent-plans/' + address(plan.workflow_id),
            json=body), expected_space=self.expected_space)
        if saved.plan != plan or saved.revision != expected_revision + 1:
            raise invalid('saved plan acknowledgement')
        return saved

    def request(self, run_id: str) -> RunRequest:
        path = '/v1/agent-runs/' + address(run_id) + '/request'
        self._check('agents.runs')
        return RunRequest.from_json(self._client._request('GET', path), expected_space=self.expected_space, run_id=run_id)

    def status(self, run_id: str) -> RunStatus:
        path = '/v1/agent-runs/' + address(run_id)
        self._check('agents.runs')
        return RunStatus.from_json(self._client._request('GET', path), expected_space=self.expected_space, run_id=run_id)

    def runs(self, *, limit: int = 20, after: Optional[str] = None) -> RunPage:
        params = {'limit': str(integer(limit, 1, 100))}
        if after is not None:
            params['after'] = cursor(after) or ''
        self._check('agents.runs')
        row = record(self._client._request('GET', '/v1/agent-runs', params=params))
        values = tuple(RunStatus.from_json(value, expected_space=self.expected_space)
                       for value in items(row.get('items'), limit))
        next_after = cursor(row.get('next_after'))
        if len({item.run_id for item in values}) != len(values) or (after is not None and next_after == after):
            raise invalid('run page')
        return RunPage(values, next_after)

    def start(self, run_id: str, *, plan: SavedPlan, question: str, max_parallel: int = 1) -> RunStatus:
        identifier(run_id)
        text(question, 4000)
        integer(max_parallel, 1, 1 if isinstance(plan.plan, HandoffPlan) else 8)
        if plan.space != self.expected_space or plan.configuration_current is not True:
            raise invalid('current saved plan required')
        body = bounded_body({'run_id': run_id, 'workflow_id': plan.plan.workflow_id, 'plan_revision': plan.revision,
                'question': question, 'max_parallel': max_parallel})
        capabilities = self._check('agents.runs', mutation=True)
        if isinstance(plan.plan, HandoffPlan):
            capabilities.require('agents.handoffs')
        elif plan.plan.interactive:
            capabilities.require('agents.inputs')
        if max_parallel > 1:
            capabilities.require('agents.parallel')
        status = RunStatus.from_json(self._client._request('POST', '/v1/agent-runs', json=body),
                                     expected_space=self.expected_space, run_id=run_id)
        request = self.request(run_id)
        if (request.question != question or request.plan.plan != plan.plan or request.plan.revision != plan.revision
                or request.plan.bindings != plan.bindings or request.max_parallel != max_parallel):
            raise invalid('submitted run acknowledgement')
        status.match(request)
        return status

    def cancel(self, run_id: str) -> RunStatus:
        path = '/v1/agent-runs/' + address(run_id) + '/cancel'
        self._check('agents.runs', mutation=True)
        return RunStatus.from_json(self._client._request('POST', path, json={}),
                                   expected_space=self.expected_space, run_id=run_id)
