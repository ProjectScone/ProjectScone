"""Explicit agent operations; construction, reads and saves never resume work."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._wire import ResourceClient, address, boolean, bounded_body, cursor, identifier, integer, invalid, items, record, text
from .agent_approvals import AgentToolContinuation, ApprovalCall, ToolApprovalActivation, ToolApprovalRecord
from .agent_inputs import InputRecord
from .agent_results import AgentResult, parse_result
from .agent_models import (AgentChoice, HandoffPlan, ModelTask, Plan, RunPolicy, RunRequest, RunStatus,
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
        capabilities = self._check('agents.plans', mutation=True)
        capabilities.require(capability)
        if isinstance(plan, TaskPlan):
            requirements = [task.answer_requirements for task in plan.tasks
                            if isinstance(task, ModelTask) and task.answer_requirements is not None]
            if requirements:
                capabilities.require('agents.output_requirements')
            if any(value.output_schema is not None for value in requirements):
                capabilities.require('agents.output_schema')
        elif plan.answer_requirements is not None:
            capabilities.require('agents.handoffs.output_requirements')
            if plan.answer_requirements.output_schema is not None:
                capabilities.require('agents.output_schema')
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


    def approvals(self, run_id: str) -> tuple[ToolApprovalRecord, ...]:
        path = '/v1/agent-runs/' + address(run_id) + '/approvals'
        self._check('agents.runs').require('agents.approvals')
        request = self.request(run_id)
        row = record(self._client._request('GET', path))
        if row.get('space') != self.expected_space or row.get('run_id') != run_id:
            raise invalid('approval collection identity')
        values = tuple(ToolApprovalRecord.from_json(value, expected_space=self.expected_space, run_id=run_id)
                       for value in items(row.get('items'), 512))
        if len({value.request_id for value in values}) != len(values):
            raise invalid('duplicate approval records')
        pending = [value.call.step_id for value in values if value.revision < 4]
        if len(set(pending)) != len(pending):
            raise invalid('duplicate pending approval steps')
        for value in values:
            value.match(request)
        return values

    def _approval(self, value: ToolApprovalRecord) -> ToolApprovalRecord:
        if (not isinstance(value, ToolApprovalRecord) or value.space != self.expected_space
                or not isinstance(value.call, ApprovalCall)):
            raise invalid('approval identity')
        row = {name: getattr(value, name) for name in ToolApprovalRecord.__dataclass_fields__}
        row['call'] = {name: getattr(value.call, name) for name in ApprovalCall.__dataclass_fields__}
        return ToolApprovalRecord.from_json(row, expected_space=self.expected_space, run_id=value.run_id)

    def decide_tool(self, pending: ToolApprovalRecord, *, decision: str) -> ToolApprovalRecord:
        pending = self._approval(pending)
        if decision not in ('approve', 'deny'):
            raise invalid('approval decision')
        path = '/v1/agent-runs/' + address(pending.run_id) + '/approvals/' + address(pending.request_id) + '/decision'
        self._check('agents.runs', mutation=True).require('agents.approvals')
        current = next((value for value in self.approvals(pending.run_id) if value.request_id == pending.request_id), None)
        if (current is None or not current.same_call(pending)
                or (current.decision is not None and current.decision != decision)):
            raise invalid('approval changed before decision')
        saved = ToolApprovalRecord.from_json(self._client._request('POST', path,
            json={'decision': decision, 'expected_revision': 1}), expected_space=self.expected_space, run_id=pending.run_id)
        if (not saved.same_call(pending) or saved.revision < 2 or saved.decision != decision
                or (current.revision >= 2 and not saved.same_decision(current))):
            raise invalid('approval decision acknowledgement')
        return saved

    def continue_tools(self, run_id: str, *, continuation_id: str,
                       decisions: tuple[ToolApprovalRecord, ...]) -> AgentToolContinuation:
        path = '/v1/agent-runs/' + address(run_id) + '/approval-continuations'
        identifier(continuation_id)
        if not isinstance(decisions, tuple) or not 1 <= len(decisions) <= 32:
            raise invalid('approval selection')
        values = tuple(self._approval(value) for value in decisions)
        if any(value.run_id != run_id or value.revision < 2
               or (value.activation_id is not None and value.activation_id != continuation_id) for value in values):
            raise invalid('approval selection')
        selected = {value.request_id: value for value in values}
        if len(selected) != len(values):
            raise invalid('duplicate approval selection')
        revisions = {key: 2 for key in selected}
        body = bounded_body({'continuation_id': continuation_id, 'decisions': revisions})
        self._check('agents.runs', mutation=True).require('agents.approvals')
        request = self.request(run_id)
        for value in values:
            value.match(request)
        current = {value.request_id: value for value in self.approvals(run_id)}
        for key, value in selected.items():
            latest = current.get(key)
            if (latest is None or not latest.same_decision(value)
                    or (latest.activation_id is not None and latest.activation_id != continuation_id)):
                raise invalid('approval changed before activation')
        prior_group = {value.request_id for value in current.values() if value.activation_id == continuation_id}
        if prior_group and prior_group != set(selected):
            raise invalid('approval activation selection changed')
        row = record(self._client._request('POST', path, json=body))
        status = RunStatus.from_json(row.get('status'), expected_space=self.expected_space, run_id=run_id)
        status.match(request)
        activation = ToolApprovalActivation.from_json(row.get('activation'), expected_space=self.expected_space, run_id=run_id)
        if (activation.activation_id != continuation_id or activation.decisions != revisions
                or activation.decision_digests != {key: value.decision_digest for key, value in selected.items()}):
            raise invalid('approval activation acknowledgement')
        return AgentToolContinuation(status, activation)


    def result(self, run_id: str, *, include_usage: bool = False) -> AgentResult:
        boolean(include_usage)
        path = '/v1/agent-runs/' + address(run_id) + '/result'
        capabilities = self._check('agents.runs')
        if include_usage:
            capabilities.require('agents.usage')
        request = self.request(run_id)
        inputs = self.inputs(run_id) if isinstance(request.plan.plan, TaskPlan) and request.plan.plan.interactive else ()
        # The server's current source verification must be the final remote read.
        if include_usage:
            raw = self._client._request('GET', path, params={'include_usage': 'true'})
        else:
            raw = self._client._request('GET', path)
        return parse_result(raw, request, inputs, include_usage=include_usage)
