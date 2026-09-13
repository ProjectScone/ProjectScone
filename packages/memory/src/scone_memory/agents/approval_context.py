"""Native approval binding for a host-owned, currently executing workflow step.

The host supplies the actual StepContext under workflow execution ownership and
revalidates preceding source receipts. This binding cannot create ownership or
recover an uncertain workflow attempt. HTTP authorization remains host policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING, cast

from .approval_models import ApprovalCall, ToolApprovalRecord
from .approval_store import AgentApprovalStore, invocation_digest
from .handoff_workflow import AgentHandoffPlan, HandoffReceipt
from .plan_store import _selections
from .run_store import AgentRunRequest
from .task_workflow import AgentTask
from .turn_journal import TurnJournalPaused
from .workflow import JSONValue, StepCheckpoints, StepContext, WorkflowError, _encode, _name

if TYPE_CHECKING:
    from .catalog import BoundAgent
    from .custom_tools import AgentTool
    from ..integrations.scoped_tools import ScopedMemoryTools


class ToolApprovalPaused(TurnJournalPaused):
    """A saved exact call awaits a decision or this continuation's activation."""

    def __init__(self, checkpoint: str, request_id: str) -> None:
        super().__init__(checkpoint)
        self.request_id = request_id


def _digest(value: object) -> str:
    return hashlib.sha256(_encode(cast(JSONValue, value), 4000000)).hexdigest()


@dataclass(frozen=True)
class ApprovalContext:
    store: AgentApprovalStore = field(repr=False)
    context: StepContext = field(repr=False)
    step_id: str
    selection_id: str
    activation_id: str | None = None
    _invocation: str = field(init=False, repr=False)
    _context_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.store, AgentApprovalStore) or not isinstance(self.context, StepContext):
            raise ValueError('approval requires a registered run and active step context')
        _name(self.step_id)
        _name(self.selection_id)
        if self.activation_id is not None:
            _name(self.activation_id)
        if self.context.checkpoints is None:
            raise ValueError('approval requires active step checkpoints')
        request = self._request()
        object.__setattr__(self, '_invocation', invocation_digest(request))
        object.__setattr__(self, '_context_digest', self._context_identity())
        self._selection(request)

    def _context_identity(self) -> str:
        return _digest({'run_id': self.context.run_id, 'space': self.context.space,
            'scope': self.context.scope, 'inputs': self.context.inputs, 'completed': self.context.completed})

    def _request(self) -> AgentRunRequest:
        request = self.store._runs.get(self.context.space, self.context.run_id)
        if request is None:
            raise WorkflowError('run_not_found')
        if request.cancel_requested_at is not None:
            raise WorkflowError('run_cancelled')
        if self.context.inputs != request.question:
            raise WorkflowError('approval_context_mismatch')
        return request

    def _selection(self, request: AgentRunRequest) -> None:
        plan = request.plan.plan
        if isinstance(plan, AgentHandoffPlan):
            policies = {agent.agent_id: agent for agent in plan.agents}
            current: str | None = plan.root_agent
            count = len(self.context.completed)
            if count > plan.max_handoffs or self.step_id != f'hop-{count + 1:02d}':
                raise WorkflowError('approval_selection_mismatch')
            for index in range(count):
                key = f'hop-{index + 1:02d}'
                if key not in self.context.completed or current is None:
                    raise WorkflowError('approval_selection_mismatch')
                receipt = HandoffReceipt.model_validate_json(json.dumps(self.context.completed[key]))
                output = receipt.output
                policy = policies[current]
                if (output.task_id != key or output.agent_id != current or output.model_id != policy.model_id
                        or output.binding != request.plan.bindings[current]
                        or output.depends_on != ((f'hop-{index:02d}',) if index else ())
                        or (receipt.handoff_to is not None and receipt.handoff_to not in policy.can_handoff_to)):
                    raise WorkflowError('approval_selection_mismatch')
                current = receipt.handoff_to
            if current != self.selection_id:
                raise WorkflowError('approval_selection_mismatch')
            return
        task = next((node for node in plan.tasks if node.task_id == self.selection_id), None)
        if (not isinstance(task, AgentTask) or self.step_id != task.task_id
                or self.step_id in self.context.completed
                or not set(task.depends_on) <= self.context.completed.keys()):
            raise WorkflowError('approval_selection_mismatch')

    def check(self, tools: ScopedMemoryTools) -> AgentRunRequest:
        request = self._request()
        memory = tools.journal_binding()
        if (invocation_digest(request) != self._invocation or self._context_identity() != self._context_digest
                or memory['space'] != request.space or memory['scope'] != request.recall_scope().kwargs()
                or memory['excluded_session'] != request.exclude_session_id):
            raise WorkflowError('approval_context_mismatch')
        self._selection(request)
        return request

    def journal_binding(self) -> JSONValue:
        request = self._request()
        plan = request.plan.plan
        completed = self.context.completed
        if not isinstance(plan, AgentHandoffPlan):
            task = next(node for node in plan.tasks if node.task_id == self.selection_id)
            completed = {name: completed[name] for name in task.depends_on}
        # Unrelated DAG siblings can finish while this task awaits approval.
        identity = _digest({'scope': self.context.scope, 'inputs': self.context.inputs, 'completed': completed})
        return {'invocation': self._invocation, 'step': self.step_id,
                'selection': self.selection_id, 'context': identity}

    def bind(self, agent: BoundAgent, tools: ScopedMemoryTools, checkpoints: StepCheckpoints | None) -> BoundApproval:
        request = self.check(tools)
        if (checkpoints is None or checkpoints is not self.context.checkpoints
                or _selections(request.plan.plan).get(self.selection_id) != (agent.definition.agent_id, agent.model_id)
                or request.plan.bindings.get(self.selection_id) != agent.fingerprint):
            raise WorkflowError('approval_selection_mismatch')
        return BoundApproval(self, agent.definition.agent_id, agent.model_id, agent.fingerprint,
                             _digest([tool.info() for tool in agent.tools]))


@dataclass(frozen=True)
class BoundApproval:
    context: ApprovalContext = field(repr=False)
    agent_id: str
    model_id: str
    binding: str
    tools_digest: str

    def check(self, memory: ScopedMemoryTools, tools: tuple[AgentTool, ...]) -> None:
        self.context.check(memory)
        if _digest([tool.info() for tool in tools]) != self.tools_digest:
            raise WorkflowError('approval_tool_mismatch')

    def request(self, tool: AgentTool, arguments_json: str, operation_digest: str) -> ToolApprovalRecord:
        call = ApprovalCall(step_id=self.context.step_id, selection_id=self.context.selection_id,
            agent_id=self.agent_id, model_id=self.model_id, binding=self.binding,
            tool_name=tool.name, tool_revision=tool.revision, tool_digest=_digest(tool.info()),
            arguments_json=arguments_json, operation_digest=operation_digest)
        return self.context.store.request(self.context.context.space, self.context.context.run_id, call)

    def claim(self, record: ToolApprovalRecord) -> ToolApprovalRecord:
        activation = self.context.activation_id
        if activation is None:
            raise WorkflowError('approval_not_activated')
        return self.context.store.claim(record.space, record.run_id, record.request_id,
                                        activation_id=activation, call=record.call)
