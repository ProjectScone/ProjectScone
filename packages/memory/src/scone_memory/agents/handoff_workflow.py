"""Bounded agent delegation with host-fixed models, edges and evidence scope."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

from pydantic import (BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler,
                      field_validator, model_serializer, model_validator)

from .catalog import AgentCatalog, BoundAgent, Identifier
from .handoff_output import final_decision, handoff_requirements
from .task_requirements import TaskAnswerRequirements
from .task_workflow import AgentTaskReceipt, AgentWorkflow, MAX_HANDOFF_BYTES, _json, _verify_agent_evidence
from .workflow import JSONValue, StepContext, WorkflowCompletion, WorkflowResult, WorkflowRunner, WorkflowStatus, WorkflowStep, WorkflowPaused, WorkflowPauseSnapshot
from ..core.validation import check_space
from ..integrations.scoped_tools import ScopedMemoryTools
from ..memory.engine import MemoryEngine
from ..realtime.answer_requirements import AnswerRequirements
from ..retrieval.recall_scope import RecallScope


if TYPE_CHECKING:
    from .history_capture import AgentRunHistory
    from .approval_store import AgentApprovalStore
    from .approval_models import ToolApprovalRecord


class HandoffAgent(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    agent_id: Identifier
    model_id: Identifier | None = None
    can_handoff_to: tuple[Identifier, ...] = Field(default=(), max_length=32)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if len(self.can_handoff_to) != len(set(self.can_handoff_to)):
            raise ValueError('duplicate handoff target')
        return self


class AgentHandoffPlan(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    workflow_id: Identifier
    root_agent: Identifier
    agents: tuple[HandoffAgent, ...] = Field(min_length=1, max_length=32)
    max_handoffs: int = Field(default=3, ge=0, le=31)
    answer_requirements: TaskAnswerRequirements | None = None

    @field_validator('answer_requirements', mode='before')
    @classmethod
    def snapshot_requirements(cls, value: object) -> object:
        return dict(vars(value)) if isinstance(value, AnswerRequirements) else value

    @model_serializer(mode='wrap')
    def serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        if self.answer_requirements is not None:
            if not isinstance(self.answer_requirements, TaskAnswerRequirements):
                raise ValueError('invalid handoff answer requirements')
            TaskAnswerRequirements.model_validate(dict(vars(self.answer_requirements)))
        result = cast(dict[str, object], handler(self))
        if self.answer_requirements is None:
            result.pop('answer_requirements', None)
        return result

    @model_validator(mode='after')
    def valid(self) -> Self:
        ids = {agent.agent_id for agent in self.agents}
        if (len(ids) != len(self.agents) or self.root_agent not in ids
                or any(set(agent.can_handoff_to) - ids for agent in self.agents)):
            raise ValueError('duplicate agent or unknown handoff target/root')
        if self.answer_requirements is not None:
            try:
                final = AnswerRequirements.model_validate(self.answer_requirements.model_dump())
                for agent in self.agents:
                    handoff_requirements(agent.can_handoff_to, final)
            except ImportError:
                raise ValueError('handoff contracts require the structured-output extra') from None
        return self


class HandoffDecision(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    answer: str = Field(min_length=1, max_length=64000)
    handoff_to: Identifier | None

    @model_validator(mode='after')
    def valid(self) -> Self:
        if not self.answer.strip():
            raise ValueError('handoff answer must not be blank')
        return self


class HandoffReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    output: AgentTaskReceipt
    handoff_to: Identifier | None


class HandoffResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    run_id: str
    status: Literal['completed', 'handoff_limit', 'paused']
    final: AgentTaskReceipt | None
    hops: tuple[HandoffReceipt, ...]
    reused_hops: tuple[str, ...]


class AgentHandoffWorkflow:
    """Caller-owned local delegation; unknown model attempts are never replayed.

    Each hop has its own evidence, not evidence inherited from an earlier model.
    Explicit cycles are permitted within max_handoffs + 1 model invocations.
    A budget-exhausted chain has no final answer. Optional output contracts apply
    to terminal answers; intermediate agents may still exchange ordinary notes.
    """
    def __init__(self, path: str | Path, *, key: bytes, catalog: AgentCatalog,
                 plan: AgentHandoffPlan, memory: MemoryEngine, space: str, scope: RecallScope,
                 exclude_session_id: str | None = None, deadline_s: float = 120.0,
                 max_payload_bytes: int = 1000000, approval_store: AgentApprovalStore | None = None,
                 approval_activation: str | None = None, history: AgentRunHistory | None = None) -> None:
        check_space(space)
        if not isinstance(plan, AgentHandoffPlan) or not isinstance(scope, RecallScope):
            raise ValueError('validated handoff plan and recall scope required')
        self._plan = AgentHandoffPlan.model_validate(plan.model_dump())
        self._policies = {agent.agent_id: agent for agent in self._plan.agents}
        self._agents = {name: catalog.bind(name, model_id=agent.model_id)
                        for name, agent in self._policies.items()}
        self._memory, self._space = memory, space
        self._approvals, self._approval_activation = approval_store, approval_activation
        self._history = history
        self._scope = RecallScope.validated(**scope.kwargs())
        self._excluded = exclude_session_id
        self._tools()
        self._final_requirements = (AnswerRequirements.model_validate(self._plan.answer_requirements.model_dump())
                                    if self._plan.answer_requirements is not None else None)
        self._requirements = {name: handoff_requirements(agent.can_handoff_to, self._final_requirements)
                              for name, agent in self._policies.items()}
        signature = hashlib.sha256(_json({'implementation': 'scone-agent-handoff-v1',
            'plan': self._plan.model_dump(),
            'bindings': {name: agent.fingerprint for name, agent in self._agents.items()}}).encode()).hexdigest()
        self._binding_scope = cast(dict[str, JSONValue], json.loads(_json({
            'plan': signature, 'recall': self._scope.as_dict(), 'exclude_session_id': exclude_session_id})))
        from .workflow_approvals import guarded, model_step
        pausable = any(guarded(agent) for agent in self._agents.values())
        self._runner = WorkflowRunner(path, key=key, source_verifier=self._verify,
            deadline=deadline_s, max_payload_bytes=max_payload_bytes, verify_before_step=True,
            completion=WorkflowCompletion(signature, self._finished),
            steps=[model_step(self._hop_id(index), signature, self._hop(index), pausable=pausable)
                   for index in range(self._plan.max_handoffs + 1)])

    @staticmethod
    def _hop_id(index: int) -> str:
        return f'hop-{index + 1:02d}'

    def _tools(self) -> ScopedMemoryTools:
        return ScopedMemoryTools(self._memory, self._space, scope=self._scope,
                                 exclude_session_id=self._excluded)

    def _receipts(self, completed: dict[str, JSONValue]) -> tuple[HandoffReceipt, ...]:
        if len(completed) > self._plan.max_handoffs + 1:
            raise ValueError('handoff budget exceeded')
        receipts: list[HandoffReceipt] = []
        current: str | None = self._plan.root_agent
        for index in range(len(completed)):
            name = self._hop_id(index)
            if name not in completed or current is None:
                raise ValueError('invalid saved handoff prefix')
            receipt = HandoffReceipt.model_validate_json(_json(completed[name]))
            output, agent = receipt.output, self._agents[current]
            dependencies = (self._hop_id(index - 1),) if index else ()
            if (output.task_id != name or output.agent_id != current or output.model_id != agent.model_id
                    or output.binding != agent.fingerprint or output.depends_on != dependencies
                    or (receipt.handoff_to is not None
                        and receipt.handoff_to not in self._policies[current].can_handoff_to)):
                raise ValueError('saved handoff binding changed')
            if (receipt.handoff_to is None and self._final_requirements is not None
                    and not self._final_requirements.accepts(output.text)):
                raise ValueError('saved handoff violates answer requirements')
            receipts.append(receipt)
            current = receipt.handoff_to
        return tuple(receipts)

    async def _verify(self, context: StepContext) -> bool:
        if context.space != self._space or context.scope != self._binding_scope:
            return False
        receipts = self._receipts(context.completed)
        return await _verify_agent_evidence(self._memory, self._space, self._scope,
                                            self._excluded, tuple(hop.output for hop in receipts))

    def _finished(self, context: StepContext) -> bool:
        receipts = self._receipts(context.completed)
        return bool(receipts) and receipts[-1].handoff_to is None

    def _hop(self, index: int) -> Callable[[StepContext], Awaitable[JSONValue | WorkflowPaused]]:
        async def execute(context: StepContext) -> JSONValue | WorkflowPaused:
            receipts = self._receipts(context.completed)
            if len(receipts) != index:
                raise ValueError('handoff prefix does not match next hop')
            current = receipts[-1].handoff_to if receipts else self._plan.root_agent
            if current is None:
                raise ValueError('handoff already finished')
            prior = None
            if receipts:
                output = receipts[-1].output
                prior = _json({'agent_id': output.agent_id, 'model_id': output.model_id,
                    'text': output.text, 'source_status': output.source_status, 'evidence_ids': output.evidence_ids})
                if len(prior.encode()) > MAX_HANDOFF_BYTES:
                    raise ValueError('workflow handoff exceeds its byte budget')
            if not isinstance(context.inputs, str):
                raise ValueError('workflow question required')
            from .workflow_approvals import invoke_agent
            result = await invoke_agent(self._agents[current], AgentWorkflow._question(context.inputs),
                tools=self._tools(), context=context, step_id=self._hop_id(index), selection_id=current,
                store=self._approvals, activation_id=self._approval_activation, history=self._history,
                prior=prior, requirements=self._requirements[current])
            if isinstance(result, WorkflowPaused):
                return result
            if self._final_requirements is None:
                decision = HandoffDecision.model_validate_json(result.output.text)
            else:
                answer, target = final_decision(result.output.text,
                    self._policies[current].can_handoff_to, self._final_requirements)
                decision = HandoffDecision(answer=answer, handoff_to=target)
            if decision.handoff_to is not None and decision.handoff_to not in self._policies[current].can_handoff_to:
                raise ValueError('handoff target is not permitted')
            receipt = AgentTaskReceipt(task_id=self._hop_id(index), agent_id=result.agent_id,
                model_id=result.model_id, binding=result.binding,
                depends_on=(self._hop_id(index - 1),) if index else (), text=decision.answer,
                source_status=cast(Literal['retained', 'none'], result.output.source_status),
                evidence_ids=result.output.evidence_ids, evidence_packets=result.output.evidence_packets,
                model_calls=result.output.model_calls, tool_calls=result.output.tool_calls, usage=result.output.usage)
            return cast(JSONValue, HandoffReceipt(output=receipt, handoff_to=decision.handoff_to).model_dump(mode='json'))
        return execute

    def _result(self, result: WorkflowResult) -> HandoffResult:
        receipts = self._receipts(result.results)
        if result.status == 'paused':
            return HandoffResult(run_id=result.run_id, status='paused', final=None, hops=receipts, reused_hops=result.reused_steps)
        if not receipts:
            raise ValueError('handoff result is empty')
        finished = receipts[-1].handoff_to is None
        if not finished and len(receipts) != self._plan.max_handoffs + 1:
            raise ValueError('handoff result is incomplete')
        return HandoffResult(run_id=result.run_id, status='completed' if finished else 'handoff_limit',
            final=receipts[-1].output if finished else None, hops=receipts, reused_hops=result.reused_steps)

    async def run(self, run_id: str, question: str, *, resume_steps: Sequence[str] | None = None) -> HandoffResult:
        result = await self._runner.run(run_id, space=self._space, scope=self._binding_scope,
                                        inputs=AgentWorkflow._question(question), resume_steps=resume_steps)
        return self._result(result)

    async def inspect_approvals(self, run_id: str, question: str) -> tuple[ToolApprovalRecord, ...]:
        from .workflow_approvals import inspect_workflow_approvals
        def select(snapshot: WorkflowPauseSnapshot) -> tuple[str, BoundAgent]:
            receipts = self._receipts(snapshot.context.completed)
            current = receipts[-1].handoff_to if receipts else self._plan.root_agent
            if current is None or snapshot.step_id != self._hop_id(len(receipts)):
                raise ValueError('invalid pending handoff')
            return current, self._agents[current]
        return await inspect_workflow_approvals(self._runner, self._approvals, run_id=run_id,
            space=self._space, scope=self._binding_scope, inputs=AgentWorkflow._question(question),
            tools=self._tools(), select=select)

    async def read_result(self, run_id: str, question: str) -> HandoffResult | None:
        """Freshly verify stored hops without invoking a model."""
        result = await self._runner.read_result(run_id, space=self._space, scope=self._binding_scope,
                                                inputs=AgentWorkflow._question(question))
        return self._result(result) if result is not None else None

    def progress(self, run_id: str, question: str) -> WorkflowStatus | None:
        """Journal progress only; read_result determines the verified outcome."""
        return self._runner.status(run_id, space=self._space, scope=self._binding_scope,
                                   inputs=AgentWorkflow._question(question))

    def close(self) -> None:
        self._runner.close()
