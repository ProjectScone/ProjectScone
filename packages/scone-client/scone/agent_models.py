"""Immutable standalone representations of catalog-selected agent workflows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Union

from ._wire import boolean, digest, identifier, integer, invalid, items, names, record, text, timestamp


def _dependencies(values: tuple[str, ...], maximum: int = 31) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise invalid('dependency tuple')
    return names(list(values), maximum)


@dataclass(frozen=True)
class ModelTask:
    task_id: str
    agent_id: str
    model_id: str
    prompt: str
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (self.task_id, self.agent_id, self.model_id):
            identifier(name)
        text(self.prompt, 2000)
        _dependencies(self.depends_on)
        if self.task_id in self.depends_on:
            raise invalid('self dependency')

    def to_json(self) -> Dict[str, object]:
        return {'task_id': self.task_id, 'agent_id': self.agent_id, 'model_id': self.model_id,
                'prompt': self.prompt, 'depends_on': list(self.depends_on)}


@dataclass(frozen=True)
class HumanInput:
    task_id: str
    prompt: str
    depends_on: tuple[str, ...] = ()
    max_response_bytes: int = 4000

    def __post_init__(self) -> None:
        identifier(self.task_id)
        text(self.prompt, 2000)
        _dependencies(self.depends_on)
        integer(self.max_response_bytes, 1, 4000)
        if self.task_id in self.depends_on:
            raise invalid('self dependency')

    def to_json(self) -> Dict[str, object]:
        return {'kind': 'input', 'task_id': self.task_id, 'prompt': self.prompt,
                'depends_on': list(self.depends_on), 'max_response_bytes': self.max_response_bytes}


Task = Union[ModelTask, HumanInput]


@dataclass(frozen=True)
class TaskPlan:
    workflow_id: str
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        identifier(self.workflow_id)
        if not isinstance(self.tasks, tuple) or not 1 <= len(self.tasks) <= 32:
            raise invalid('task graph')
        if any(not isinstance(task, (ModelTask, HumanInput)) for task in self.tasks):
            raise invalid('task')
        known = {task.task_id for task in self.tasks}
        if len(known) != len(self.tasks) or any(set(task.depends_on) - known for task in self.tasks):
            raise invalid('task dependencies')
        done: set[str] = set()
        while len(done) < len(known):
            ready = {task.task_id for task in self.tasks if task.task_id not in done and set(task.depends_on) <= done}
            if not ready:
                raise invalid('dependency cycle')
            done.update(ready)

    @property
    def interactive(self) -> bool:
        return any(isinstance(task, HumanInput) for task in self.tasks)

    def to_json(self) -> Dict[str, object]:
        return {**({'kind': 'interactive'} if self.interactive else {}), 'workflow_id': self.workflow_id,
                'tasks': [task.to_json() for task in self.tasks]}


@dataclass(frozen=True)
class HandoffAgent:
    agent_id: str
    model_id: str
    can_handoff_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.agent_id)
        identifier(self.model_id)
        _dependencies(self.can_handoff_to, 32)

    def to_json(self) -> Dict[str, object]:
        return {'agent_id': self.agent_id, 'model_id': self.model_id, 'can_handoff_to': list(self.can_handoff_to)}


@dataclass(frozen=True)
class HandoffPlan:
    workflow_id: str
    root_agent: str
    agents: tuple[HandoffAgent, ...]
    max_handoffs: int = 3

    def __post_init__(self) -> None:
        identifier(self.workflow_id)
        identifier(self.root_agent)
        integer(self.max_handoffs, 0, 31)
        if not isinstance(self.agents, tuple) or not 1 <= len(self.agents) <= 32:
            raise invalid('handoff agents')
        if any(not isinstance(agent, HandoffAgent) for agent in self.agents):
            raise invalid('handoff agent')
        known = {agent.agent_id for agent in self.agents}
        if (len(known) != len(self.agents) or self.root_agent not in known
                or any(set(agent.can_handoff_to) - known for agent in self.agents)):
            raise invalid('handoff targets')

    def to_json(self) -> Dict[str, object]:
        return {'workflow_id': self.workflow_id, 'root_agent': self.root_agent,
                'agents': [agent.to_json() for agent in self.agents], 'max_handoffs': self.max_handoffs}


Plan = Union[TaskPlan, HandoffPlan]


def parse_plan(value: object) -> Plan:
    row = record(value)
    if 'agents' in row:
        if set(row) - {'workflow_id', 'root_agent', 'agents', 'max_handoffs'}:
            raise invalid('handoff fields')
        agents = []
        for value in items(row.get('agents'), 32):
            agent = record(value)
            if set(agent) - {'agent_id', 'model_id', 'can_handoff_to'}:
                raise invalid('handoff agent fields')
            agents.append(HandoffAgent(identifier(agent.get('agent_id')), identifier(agent.get('model_id')),
                                       names(agent.get('can_handoff_to'))))
        return HandoffPlan(identifier(row.get('workflow_id')), identifier(row.get('root_agent')),
                           tuple(agents), integer(row.get('max_handoffs'), 0, 31))
    if set(row) - {'workflow_id', 'tasks', 'kind'} or ('kind' in row and row['kind'] != 'interactive'):
        raise invalid('task plan fields')
    tasks: list[Task] = []
    for value in items(row.get('tasks'), 32):
        task = record(value)
        task_id, prompt = identifier(task.get('task_id')), text(task.get('prompt'), 2000)
        dependencies = names(task.get('depends_on'), 31)
        if task.get('kind') == 'input':
            if set(task) - {'kind', 'task_id', 'prompt', 'depends_on', 'max_response_bytes'}:
                raise invalid('input fields')
            tasks.append(HumanInput(task_id, prompt, dependencies, integer(task.get('max_response_bytes'), 1, 4000)))
        else:
            if set(task) - {'task_id', 'agent_id', 'model_id', 'prompt', 'depends_on'}:
                raise invalid('model task fields')
            tasks.append(ModelTask(task_id, identifier(task.get('agent_id')), identifier(task.get('model_id')), prompt, dependencies))
    plan = TaskPlan(identifier(row.get('workflow_id')), tuple(tasks))
    if plan.interactive != (row.get('kind') == 'interactive'):
        raise invalid('interactive plan kind')
    return plan


def binding_ids(plan: Plan) -> tuple[str, ...]:
    if isinstance(plan, HandoffPlan):
        return tuple(agent.agent_id for agent in plan.agents)
    return tuple(task.task_id for task in plan.tasks if isinstance(task, ModelTask))


@dataclass(frozen=True)
class SavedPlan:
    space: str
    revision: int
    plan: Plan
    bindings: Mapping[str, str]
    updated_at: str
    configuration_current: Optional[bool]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str) -> SavedPlan:
        row = record(value)
        if row.get('space') != expected_space:
            raise invalid('plan space')
        plan = parse_plan(row.get('plan'))
        bindings = record(row.get('bindings'))
        if set(bindings) != set(binding_ids(plan)):
            raise invalid('model bindings')
        return cls(expected_space, integer(row.get('revision'), 1), plan,
                   MappingProxyType({key: digest(value) for key, value in bindings.items()}),
                   timestamp(row.get('updated_at')),
                   boolean(row['configuration_current']) if 'configuration_current' in row else None)


@dataclass(frozen=True)
class RunRequest:
    space: str
    run_id: str
    question: str
    plan: SavedPlan
    created_at: str
    max_parallel: int

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: str) -> RunRequest:
        row = record(value)
        if row.get('space') != expected_space or row.get('run_id') != identifier(run_id):
            raise invalid('run identity')
        plan = SavedPlan.from_json(row.get('plan'), expected_space=expected_space)
        return cls(expected_space, run_id, text(row.get('question'), 4000), plan,
                   timestamp(row.get('created_at')), integer(row.get('max_parallel', 1), 1,
                                                            1 if isinstance(plan.plan, HandoffPlan) else 8))


@dataclass(frozen=True)
class RunStatus:
    space: str
    run_id: str
    workflow_id: str
    plan_revision: int
    created_at: str
    status: str
    active_local: bool
    completed_steps: tuple[str, ...]
    inflight_steps: tuple[str, ...]
    waiting_steps: tuple[str, ...]
    max_parallel: int
    outcome_unknown: bool
    error_class: Optional[str]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: Optional[str] = None) -> RunStatus:
        row = record(value)
        if row.get('space') != expected_space or (run_id is not None and row.get('run_id') != run_id):
            raise invalid('run status identity')
        state = text(row.get('status'), 64)
        if state not in {'created', 'deadline', 'outcome_unknown', 'retry_not_allowed', 'registered', 'running',
                         'completed', 'failed', 'cancelled', 'sources_invalid', 'verification_unavailable',
                         'unavailable', 'awaiting_input'}:
            raise invalid('run status')
        inflight = identifier(row['inflight']) if row.get('inflight') is not None else None
        inflights = names(row.get('inflight_steps', [inflight] if inflight else []))
        width = integer(row.get('max_parallel', 1), 1, 8)
        if (inflights[0] if inflights else None) != inflight or len(inflights) > width:
            raise invalid('inflight tasks')
        completed, waiting = names(row.get('completed_steps')), names(row.get('waiting_steps', []))
        if set(completed) & set(inflights) or set(waiting) & (set(completed) | set(inflights)):
            raise invalid('overlapping task states')
        return cls(expected_space, identifier(row.get('run_id')), identifier(row.get('workflow_id')),
                   integer(row.get('plan_revision'), 1), timestamp(row.get('created_at')), state,
                   boolean(row.get('active_local')), completed, inflights, waiting, width,
                   boolean(row.get('outcome_unknown')),
                   text(row['error_class'], 128) if row.get('error_class') is not None else None)

    def match(self, request: RunRequest) -> None:
        if (self.space != request.space or self.run_id != request.run_id
                or self.workflow_id != request.plan.plan.workflow_id or self.plan_revision != request.plan.revision
                or self.max_parallel != request.max_parallel
                or datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
                != datetime.fromisoformat(request.created_at.replace('Z', '+00:00'))):
            raise invalid('run request binding')
        plan = request.plan.plan
        if isinstance(plan, HandoffPlan):
            ids = tuple('hop-%02d' % (index + 1) for index in range(plan.max_handoffs + 1))
            if self.completed_steps != ids[:len(self.completed_steps)] or self.waiting_steps:
                raise invalid('handoff progress')
            if self.inflight_steps and self.inflight_steps != ids[len(self.completed_steps):len(self.completed_steps) + 1]:
                raise invalid('handoff inflight')
            return
        ids = tuple(task.task_id for task in plan.tasks)
        human = {task.task_id for task in plan.tasks if isinstance(task, HumanInput)}
        if (set(self.completed_steps + self.inflight_steps + self.waiting_steps) - set(ids)
                or set(self.inflight_steps) & human or set(self.waiting_steps) - human):
            raise invalid('task progress')


@dataclass(frozen=True)
class ModelChoice:
    model_id: str
    label: str
    revision: str


@dataclass(frozen=True)
class AgentChoice:
    agent_id: str
    default_model: str
    models: tuple[ModelChoice, ...]


def parse_catalog(value: object) -> tuple[AgentChoice, ...]:
    catalog = []
    for value in items(record(value).get('agents'), 32):
        row = record(value)
        choices = []
        for model_value in items(row.get('models'), 64):
            model = record(model_value)
            label = text(model.get('label'), 1024, 'model label')
            if len(label) > 256:
                raise invalid('model label')
            choices.append(ModelChoice(identifier(model.get('model_id')), label,
                                       identifier(model.get('revision'))))
        ids = {model.model_id for model in choices}
        default = identifier(row.get('default_model'))
        if len(ids) != len(choices) or default not in ids:
            raise invalid('catalog model choices')
        catalog.append(AgentChoice(identifier(row.get('agent_id')), default, tuple(choices)))
    if len({agent.agent_id for agent in catalog}) != len(catalog):
        raise invalid('catalog agent choices')
    return tuple(catalog)


@dataclass(frozen=True)
class RunPolicy:
    space: str
    max_parallel_tasks: int
    max_active_runs: int

    @classmethod
    def from_json(cls, value: object, *, expected_space: str) -> RunPolicy:
        row = record(value)
        if row.get('space') != expected_space:
            raise invalid('run policy space')
        return cls(expected_space, integer(row.get('max_parallel_tasks'), 1, 8),
                   integer(row.get('max_active_runs'), 1, 32))
