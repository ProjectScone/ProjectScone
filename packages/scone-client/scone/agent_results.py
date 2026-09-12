"""Completed task and handoff outputs tied to immutable execution requests."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Union

from ._wire import digest, identifier, integer, invalid, items, names, record, text
from .agent_evidence import EvidencePacket
from .agent_inputs import InputRecord
from .agent_models import HandoffPlan, HumanInput, ModelTask, RunRequest, TaskPlan


def _output_text(value: object) -> str:
    # Native AgentTaskReceipt permits 64,000 Unicode characters.
    if not isinstance(value, str) or not 1 <= len(value) <= 64000:
        raise invalid('agent output text')
    text('x' + value, 256001, 'agent output text')
    return value


@dataclass(frozen=True)
class ModelOutput:
    task_id: str
    agent_id: str
    model_id: str
    binding: str
    depends_on: tuple[str, ...]
    text: str
    source_status: str
    evidence_ids: tuple[str, ...]
    evidence_packets: tuple[EvidencePacket, ...]
    model_calls: int
    tool_calls: int

    @classmethod
    def from_json(cls, value: object, *, task_id: str, agent_id: str, model_id: str,
                  binding: str, depends_on: tuple[str, ...]) -> ModelOutput:
        row = record(value)
        expected = {'task_id', 'agent_id', 'model_id', 'binding', 'depends_on', 'text',
                    'source_status', 'evidence_ids', 'evidence_packets', 'model_calls', 'tool_calls'}
        if (set(row) != expected or row.get('task_id') != task_id or row.get('agent_id') != agent_id
                or row.get('model_id') != model_id or digest(row.get('binding')) != binding
                or names(row.get('depends_on'), 31) != depends_on):
            raise invalid('agent output binding')
        packets = tuple(EvidencePacket.from_json(raw) for raw in items(row.get('evidence_packets'), 32))
        evidence = tuple(text(value, 64) for value in items(row.get('evidence_ids'), 2048))
        collected = tuple(dict.fromkeys(key for packet in packets for key in packet.evidence_ids))
        if evidence != collected or row.get('source_status') != ('retained' if evidence else 'none'):
            raise invalid('agent output evidence')
        return cls(task_id, agent_id, model_id, binding, depends_on, _output_text(row.get('text')),
                   'retained' if evidence else 'none', evidence, packets,
                   integer(row.get('model_calls'), 1, 17), integer(row.get('tool_calls'), 0, 16))


@dataclass(frozen=True)
class HumanOutput:
    kind: str
    task_id: str
    depends_on: tuple[str, ...]
    text: str
    activation_id: str
    response_digest: str

    @classmethod
    def from_json(cls, value: object, *, task: HumanInput, saved: InputRecord) -> HumanOutput:
        row = record(value)
        if (set(row) != {'kind', 'task_id', 'depends_on', 'text', 'activation_id', 'response_digest'}
                or row.get('kind') != 'human_input' or row.get('task_id') != task.task_id
                or names(row.get('depends_on'), 31) != task.depends_on
                or saved.revision != 3 or saved.activation_id != row.get('activation_id')
                or saved.response != row.get('text')):
            raise invalid('human result binding')
        return cls('human_input', task.task_id, task.depends_on, text(row.get('text'), task.max_response_bytes),
                   identifier(row.get('activation_id')), digest(row.get('response_digest')))


TaskOutput = Union[ModelOutput, HumanOutput]


@dataclass(frozen=True)
class TaskResult:
    space: str
    run_id: str
    status: str
    results: Mapping[str, TaskOutput]
    reused_steps: tuple[str, ...]


@dataclass(frozen=True)
class HandoffHop:
    output: ModelOutput
    handoff_to: Optional[str]


@dataclass(frozen=True)
class HandoffResult:
    space: str
    run_id: str
    status: str
    final: Optional[ModelOutput]
    hops: tuple[HandoffHop, ...]
    reused_hops: tuple[str, ...]


AgentResult = Union[TaskResult, HandoffResult]


def parse_result(value: object, request: RunRequest, inputs: tuple[InputRecord, ...] = ()) -> AgentResult:
    row = record(value)
    if row.get('space') != request.space or row.get('run_id') != request.run_id:
        raise invalid('agent result identity')
    plan = request.plan.plan
    if isinstance(plan, HandoffPlan):
        return _handoff(row, request, plan)
    return _tasks(row, request, plan, inputs)


def _tasks(row: dict[str, object], request: RunRequest, plan: TaskPlan,
           inputs: tuple[InputRecord, ...]) -> TaskResult:
    raw = record(row.get('results'))
    known = {task.task_id for task in plan.tasks}
    reused = names(row.get('reused_steps'))
    if row.get('status') != 'completed' or set(raw) != known or set(reused) - known:
        raise invalid('completed task result')
    saved = {value.task_id: value for value in inputs}
    expected_inputs = {task.task_id for task in plan.tasks if isinstance(task, HumanInput)}
    if len(saved) != len(inputs) or set(saved) != expected_inputs:
        raise invalid('completed human inputs')
    outputs: dict[str, TaskOutput] = {}
    for task in plan.tasks:
        if isinstance(task, HumanInput):
            saved[task.task_id].match(request)
            outputs[task.task_id] = HumanOutput.from_json(raw[task.task_id], task=task, saved=saved[task.task_id])
        elif isinstance(task, ModelTask):
            outputs[task.task_id] = ModelOutput.from_json(raw[task.task_id], task_id=task.task_id,
                agent_id=task.agent_id, model_id=task.model_id, binding=request.plan.bindings[task.task_id],
                depends_on=task.depends_on)
    if sum(len(value.evidence_packets) for value in outputs.values() if isinstance(value, ModelOutput)) > 128:
        raise invalid('agent result packet count')
    return TaskResult(request.space, request.run_id, 'completed', MappingProxyType(outputs), reused)


def _handoff(row: dict[str, object], request: RunRequest, plan: HandoffPlan) -> HandoffResult:
    raw = items(row.get('hops'), plan.max_handoffs + 1)
    if not raw:
        raise invalid('empty handoff result')
    policies = {agent.agent_id: agent for agent in plan.agents}
    current: Optional[str] = plan.root_agent
    hops: list[HandoffHop] = []
    for index, raw_hop in enumerate(raw):
        if current is None:
            raise invalid('handoff after final')
        policy = policies[current]
        hop = record(raw_hop)
        if set(hop) != {'output', 'handoff_to'}:
            raise invalid('handoff receipt fields')
        task_id = 'hop-%02d' % (index + 1)
        dependencies = ('hop-%02d' % index,) if index else ()
        output = ModelOutput.from_json(hop.get('output'), task_id=task_id, agent_id=policy.agent_id,
            model_id=policy.model_id, binding=request.plan.bindings[policy.agent_id], depends_on=dependencies)
        target = identifier(hop['handoff_to']) if hop.get('handoff_to') is not None else None
        if target is not None and target not in policy.can_handoff_to:
            raise invalid('unpermitted handoff target')
        hops.append(HandoffHop(output, target))
        current = target
    ids = {hop.output.task_id for hop in hops}
    reused = names(row.get('reused_hops'))
    if set(reused) - ids or sum(len(hop.output.evidence_packets) for hop in hops) > 128:
        raise invalid('handoff result bounds')
    final = None
    if current is None:
        last = hops[-1].output
        final = ModelOutput.from_json(row.get('final'), task_id=last.task_id, agent_id=last.agent_id,
            model_id=last.model_id, binding=last.binding, depends_on=last.depends_on)
        if final != last or row.get('status') != 'completed':
            raise invalid('handoff final binding')
    elif row.get('status') != 'handoff_limit' or len(hops) != plan.max_handoffs + 1 or row.get('final') is not None:
        raise invalid('handoff limit outcome')
    return HandoffResult(request.space, request.run_id, 'completed' if final is not None else 'handoff_limit',
                         final, tuple(hops), reused)
