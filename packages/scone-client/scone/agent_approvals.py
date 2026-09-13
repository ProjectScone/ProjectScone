"""Detached exact-call records and immutable explicit activation receipts."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, Optional

from ._wire import digest, identifier, integer, invalid, record, text, timestamp
from .agent_models import HandoffPlan, ModelTask, RunRequest, RunStatus, _tool_name


def _arguments(value: object) -> str:
    encoded = text(value, 16000, 'approval arguments')
    try:
        parsed: object = json.loads(encoded)
        record(parsed)
        queue = [(parsed, 0)]
        count = 0
        while queue:
            item, depth = queue.pop()
            count += 1
            if depth > 32 or count > 20000:
                raise invalid('approval argument structure')
            if isinstance(item, dict):
                for key, child in record(item).items():
                    queue.extend(((key, depth + 1), (child, depth + 1)))
            elif isinstance(item, list):
                queue.extend((child, depth + 1) for child in item)
            elif type(item) is int and item.bit_length() > 256:
                raise invalid('approval argument integer')
            elif isinstance(item, float) and not math.isfinite(item):
                raise invalid('approval argument number')
        canonical = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (ValueError, RecursionError, UnicodeError):
        raise invalid('approval arguments') from None
    if canonical != encoded:
        raise invalid('canonical approval arguments')
    return encoded


@dataclass(frozen=True)
class ApprovalCall:
    step_id: str
    selection_id: str
    agent_id: str
    model_id: str
    binding: str
    tool_name: str
    tool_revision: str
    tool_digest: str
    arguments_json: str
    operation_digest: str

    @classmethod
    def from_json(cls, value: object) -> ApprovalCall:
        row = record(value)
        if set(row) != set(cls.__dataclass_fields__):
            raise invalid('approval call fields')
        revision = text(row['tool_revision'], 128, 'tool revision')
        if re.fullmatch(r'[A-Za-z0-9._:-]+', revision) is None:
            raise invalid('tool revision')
        return cls(identifier(row['step_id']), identifier(row['selection_id']), identifier(row['agent_id']),
                   identifier(row['model_id']), digest(row['binding']), _tool_name(row['tool_name']),
                   revision, digest(row['tool_digest']), _arguments(row['arguments_json']),
                   digest(row['operation_digest']))

    def arguments(self) -> dict[str, object]:
        return record(json.loads(self.arguments_json))


@dataclass(frozen=True)
class ToolApprovalRecord:
    space: str
    run_id: str
    request_id: str
    call: ApprovalCall
    revision: int
    created_at: str
    decision: Optional[str]
    decided_by: Optional[str]
    decided_at: Optional[str]
    activation_id: Optional[str]
    activated_at: Optional[str]
    consumed_at: Optional[str]
    decision_digest: Optional[str]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: str) -> ToolApprovalRecord:
        row = record(value)
        if set(row) != set(cls.__dataclass_fields__):
            raise invalid('approval record fields')
        if row['space'] != expected_space or row['run_id'] != identifier(run_id):
            raise invalid('approval identity')
        revision = integer(row['revision'], 1, 4)
        decision = text(row['decision'], 7) if row['decision'] is not None else None
        actor = identifier(row['decided_by']) if row['decided_by'] is not None else None
        decided = timestamp(row['decided_at']) if row['decided_at'] is not None else None
        activation = identifier(row['activation_id']) if row['activation_id'] is not None else None
        activated = timestamp(row['activated_at']) if row['activated_at'] is not None else None
        consumed = timestamp(row['consumed_at']) if row['consumed_at'] is not None else None
        decision_hash = digest(row['decision_digest']) if row['decision_digest'] is not None else None
        if (decision not in (None, 'approve', 'deny')
                or any((value is not None) != (revision >= 2) for value in (decision, actor, decided, decision_hash))
                or any((value is not None) != (revision >= 3) for value in (activation, activated))
                or (consumed is not None) != (revision == 4)):
            raise invalid('approval revision state')
        return cls(expected_space, run_id, digest(row['request_id']), ApprovalCall.from_json(row['call']), revision,
                   timestamp(row['created_at']), decision, actor, decided, activation, activated, consumed, decision_hash)

    def match(self, request: RunRequest) -> None:
        if self.space != request.space or self.run_id != request.run_id:
            raise invalid('approval request binding')
        plan = request.plan.plan
        call = self.call
        if request.plan.bindings.get(call.selection_id) != call.binding:
            raise invalid('approval model binding')
        if isinstance(plan, HandoffPlan):
            hops = tuple('hop-%02d' % (index + 1) for index in range(plan.max_handoffs + 1))
            selected = next((agent for agent in plan.agents if agent.agent_id == call.selection_id), None)
            if (selected is None or call.step_id not in hops
                    or selected.agent_id != call.agent_id or selected.model_id != call.model_id
                    or (call.step_id == 'hop-01' and call.agent_id != plan.root_agent)):
                raise invalid('approval handoff selection')
            edges = {agent.agent_id: agent.can_handoff_to for agent in plan.agents}
            reachable = {plan.root_agent}
            for _ in range(hops.index(call.step_id)):
                reachable = {target for source in reachable for target in edges[source]}
            if call.agent_id not in reachable:
                raise invalid('approval handoff reachability')
        else:
            task = next((task for task in plan.tasks if task.task_id == call.selection_id), None)
            if (not isinstance(task, ModelTask) or call.step_id != task.task_id
                    or call.agent_id != task.agent_id or call.model_id != task.model_id):
                raise invalid('approval task selection')

    def same_call(self, other: ToolApprovalRecord) -> bool:
        return (self.space, self.run_id, self.request_id, self.call, self.created_at) == (
            other.space, other.run_id, other.request_id, other.call, other.created_at)

    def same_decision(self, other: ToolApprovalRecord) -> bool:
        return replace(self, revision=2, activation_id=None, activated_at=None, consumed_at=None) == replace(
            other, revision=2, activation_id=None, activated_at=None, consumed_at=None)


@dataclass(frozen=True)
class ToolApprovalActivation:
    space: str
    run_id: str
    activation_id: str
    decisions: Mapping[str, int]
    decision_digests: Mapping[str, str]
    created_at: str

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, run_id: str) -> ToolApprovalActivation:
        row = record(value)
        if set(row) != set(cls.__dataclass_fields__):
            raise invalid('approval activation fields')
        if row['space'] != expected_space or row['run_id'] != identifier(run_id):
            raise invalid('approval activation identity')
        decisions = {digest(key): integer(value, 2, 2) for key, value in record(row['decisions']).items()}
        digests = {digest(key): digest(value) for key, value in record(row['decision_digests']).items()}
        if not 1 <= len(decisions) <= 32 or decisions.keys() != digests.keys():
            raise invalid('approval activation batch')
        return cls(expected_space, run_id, identifier(row['activation_id']), MappingProxyType(decisions),
                   MappingProxyType(digests), timestamp(row['created_at']))


@dataclass(frozen=True)
class AgentToolContinuation:
    status: RunStatus
    activation: ToolApprovalActivation
