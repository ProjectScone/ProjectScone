"""Explicit agent operations; construction, reads and saves never resume work."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._wire import ResourceClient, address, bounded_body, cursor, identifier, integer, invalid, items, record, text
from .agent_inputs import InputRecord
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


    def inputs(self, run_id: str) -> tuple[InputRecord, ...]:
        path = '/v1/agent-runs/' + address(run_id) + '/inputs'
        self._check('agents.runs').require('agents.inputs')
        request = self.request(run_id)
        row = record(self._client._request('GET', path))
        if row.get('space') != self.expected_space or row.get('run_id') != run_id:
            raise invalid('input collection identity')
        values = tuple(InputRecord.from_json(value, expected_space=self.expected_space, run_id=run_id)
                       for value in items(row.get('items'), 32))
        if len({value.task_id for value in values}) != len(values):
            raise invalid('duplicate input records')
        for value in values:
            value.match(request)
        return values

    def respond(self, pending: InputRecord, *, response: str) -> InputRecord:
        if not isinstance(pending, InputRecord) or pending.space != self.expected_space:
            raise invalid('input identity')
        text(response, pending.max_response_bytes)
        path = '/v1/agent-runs/' + address(pending.run_id) + '/inputs/' + address(pending.task_id) + '/response'
        body = bounded_body({'response': response, 'expected_revision': 1})
        self._check('agents.runs', mutation=True).require('agents.inputs')
        pending.match(self.request(pending.run_id))
        current = next((value for value in self.inputs(pending.run_id) if value.task_id == pending.task_id), None)
        if (current is None or not current.same_input(pending)
                or (current.response is not None and current.response != response)):
            raise invalid('input changed before reply')
        saved = InputRecord.from_json(self._client._request('POST', path, json=body),
                                      expected_space=self.expected_space, run_id=pending.run_id)
        if not saved.same_input(pending) or saved.response != response or saved.revision < 2:
            raise invalid('reply acknowledgement')
        return saved

    def continue_run(self, run_id: str, *, continuation_id: str,
                     responses: tuple[InputRecord, ...]) -> RunStatus:
        path = '/v1/agent-runs/' + address(run_id) + '/continue'
        identifier(continuation_id)
        if not isinstance(responses, tuple) or not 1 <= len(responses) <= 32:
            raise invalid('input selection')
        for value in responses:
            if (not isinstance(value, InputRecord) or value.space != self.expected_space
                    or value.run_id != run_id or value.revision not in (2, 3)
                    or value.response is None or (value.revision == 3 and value.activation_id != continuation_id)):
                raise invalid('input selection')
        selected = {identifier(value.task_id): value for value in responses}
        if len(selected) != len(responses):
            raise invalid('duplicate input selection')
        body = bounded_body({'continuation_id': continuation_id, 'responses': {key: 2 for key in selected}})
        self._check('agents.runs', mutation=True).require('agents.inputs')
        request = self.request(run_id)
        for value in responses:
            value.match(request)
        current = {value.task_id: value for value in self.inputs(run_id)}
        for task_id, value in selected.items():
            latest = current.get(task_id)
            if (latest is None or not latest.same_reply(value) or latest.revision not in (2, 3)
                    or (latest.activation_id is not None and latest.activation_id != continuation_id)):
                raise invalid('activation acknowledgement: selected reply changed')
        prior_group = {value.task_id for value in current.values() if value.activation_id == continuation_id}
        if prior_group and prior_group != set(selected):
            raise invalid('activation acknowledgement: selection changed')
        status = RunStatus.from_json(self._client._request('POST', path, json=body),
                                     expected_space=self.expected_space, run_id=run_id)
        status.match(request)
        activated = {value.task_id: value for value in self.inputs(run_id) if value.activation_id == continuation_id}
        if set(activated) != set(selected) or any(not activated[key].same_reply(value) for key, value in selected.items()):
            raise invalid('activation acknowledgement')
        return status
