"""Original sequential agent task graphs with encrypted, evidence-checked receipts.

Task dependencies define which prior outputs reach a model. The host fixes the
catalog, memory space and recall scope; a saved result never grants authority.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
import hashlib
import json
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import AgentCatalog, BoundAgent, Identifier
from .tool_evidence import prepare_tool_evidence
from .workflow import JSONValue, StepContext, WorkflowResult, WorkflowRunner, WorkflowStatus, WorkflowStep
from ..integrations.scoped_tools import ScopedMemoryTools
from ..memory.engine import MemoryEngine
from ..retrieval.recall_scope import RecallScope
from ..core.validation import check_space

MAX_EVIDENCE_PACKETS = 128
MAX_HANDOFF_BYTES = 32000


class AgentTask(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    task_id: Identifier
    agent_id: Identifier
    model_id: Identifier | None = None
    prompt: str = Field(min_length=1, max_length=2000)
    depends_on: tuple[Identifier, ...] = Field(default=(), max_length=31)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if (not self.prompt.strip() or len(self.prompt.encode()) > 2000
                or len(self.depends_on) != len(set(self.depends_on)) or self.task_id in self.depends_on):
            raise ValueError('invalid agent task')
        return self


class AgentTaskPlan(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    workflow_id: Identifier
    tasks: tuple[AgentTask, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode='after')
    def valid(self) -> Self:
        ids = {task.task_id for task in self.tasks}
        if len(ids) != len(self.tasks) or any(set(task.depends_on) - ids for task in self.tasks):
            raise ValueError('duplicate task or unknown dependency')
        self.ordered()
        return self

    def ordered(self) -> tuple[AgentTask, ...]:
        pending = list(self.tasks)
        done: set[str] = set()
        result: list[AgentTask] = []
        while pending:
            ready = [task for task in pending if set(task.depends_on) <= done]
            if not ready:
                raise ValueError('agent task dependencies contain a cycle')
            for task in ready:
                result.append(task)
                done.add(task.task_id)
                pending.remove(task)
        return tuple(result)


class AgentTaskReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    task_id: Identifier
    agent_id: Identifier
    model_id: Identifier
    binding: str = Field(pattern=r'^[a-f0-9]{64}$')
    depends_on: tuple[Identifier, ...]
    text: str = Field(min_length=1, max_length=64000)
    source_status: Literal['retained', 'none']
    evidence_ids: tuple[str, ...] = Field(max_length=2048)
    evidence_packets: tuple[str, ...] = Field(max_length=32)
    model_calls: int = Field(ge=1, le=17)
    tool_calls: int = Field(ge=0, le=16)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


class AgentWorkflow:
    """A caller-owned encrypted workflow with fixed models and declared inputs.

    Execution is sequential in stable dependency order. Completed model results
    can be reused only with the same plan, model configuration, input and scope,
    and after their actual retained evidence is revalidated. Uncertain model
    calls are never automatically retried. This is not distributed execution.
    """
    def __init__(self, path: str | Path, *, key: bytes, catalog: AgentCatalog, plan: AgentTaskPlan,
                 memory: MemoryEngine, space: str, scope: RecallScope,
                 exclude_session_id: str | None = None, deadline_s: float = 120.0,
                 max_payload_bytes: int = 1000000) -> None:
        check_space(space)
        if not isinstance(plan, AgentTaskPlan) or not isinstance(scope, RecallScope):
            raise ValueError('validated agent plan and recall scope required')
        plan = AgentTaskPlan.model_validate(plan.model_dump())
        self._tasks = {task.task_id: task for task in plan.ordered()}
        self._agents = {task.task_id: catalog.bind(task.agent_id, model_id=task.model_id) for task in self._tasks.values()}
        self._memory, self._space = memory, space
        self._scope = RecallScope.validated(**scope.kwargs())
        self._excluded = exclude_session_id
        # Validate the complete tool binding before opening any journal file.
        self._tools()
        signature = hashlib.sha256(_json({'implementation': 'scone-agent-task-v1', 'plan': plan.model_dump(),
            'bindings': {name: agent.fingerprint for name, agent in self._agents.items()}}).encode()).hexdigest()
        self._binding_scope = cast(dict[str, JSONValue], json.loads(_json({
            'plan': signature, 'recall': self._scope.as_dict(), 'exclude_session_id': exclude_session_id})))
        self._runner = WorkflowRunner(path, key=key, source_verifier=self._verify,
            deadline=deadline_s, max_payload_bytes=max_payload_bytes, verify_before_step=True,
            steps=[WorkflowStep(task.task_id, signature, self._step(task, self._agents[task.task_id]))
                   for task in self._tasks.values()])

    def _tools(self) -> ScopedMemoryTools:
        return ScopedMemoryTools(self._memory, self._space, scope=self._scope,
                                 exclude_session_id=self._excluded)

    def _receipts(self, context: StepContext) -> dict[str, AgentTaskReceipt]:
        receipts: dict[str, AgentTaskReceipt] = {}
        for task_id, value in context.completed.items():
            if task_id not in self._tasks:
                raise ValueError('unknown saved task')
            task, agent = self._tasks[task_id], self._agents[task_id]
            receipt = AgentTaskReceipt.model_validate_json(_json(value))
            if (receipt.task_id != task_id or receipt.agent_id != task.agent_id
                    or receipt.model_id != agent.model_id or receipt.binding != agent.fingerprint
                    or receipt.depends_on != task.depends_on or not set(task.depends_on) <= context.completed.keys()):
                raise ValueError('saved task binding changed')
            receipts[task_id] = receipt
        if sum(len(receipt.evidence_packets) for receipt in receipts.values()) > MAX_EVIDENCE_PACKETS:
            raise ValueError('workflow evidence packet limit')
        return receipts

    async def _verify(self, context: StepContext) -> bool:
        if context.space != self._space or context.scope != self._binding_scope:
            return False
        if await self._memory.space_deleted(self._space) is not None:
            return False
        receipts = self._receipts(context)
        revision = await self._memory.documents.revision(self._space)
        validators: list[Callable[[], Awaitable[bool]]] = []
        for receipt in receipts.values():
            ids: list[str] = []
            for raw in receipt.evidence_packets:
                if len(raw.encode()) > 64000:
                    return False
                packet = json.loads(raw)
                if not isinstance(packet, dict):
                    return False
                evidence = await prepare_tool_evidence(self._memory, self._space, self._scope,
                                                       self._excluded, packet, 2.0, raise_unavailable=True)
                if not evidence.evidence_ids:
                    return False
                ids.extend(evidence.evidence_ids)
                validators.append(evidence.validate)
            expected = tuple(dict.fromkeys(ids))
            if receipt.evidence_ids != expected or receipt.source_status != ('retained' if expected else 'none'):
                return False
        for validate in validators:
            if not await validate():
                return False
        return await self._memory.documents.revision(self._space) == revision

    def _step(self, task: AgentTask, agent: BoundAgent) -> Callable[[StepContext], Awaitable[JSONValue]]:
        async def execute(context: StepContext) -> JSONValue:
            receipts = self._receipts(context)
            handoff = None
            if task.depends_on:
                handoff = _json([{'task_id': name, 'agent_id': receipts[name].agent_id,
                                 'text': receipts[name].text, 'source_status': receipts[name].source_status,
                                 'evidence_ids': receipts[name].evidence_ids}
                                for name in task.depends_on])
                if len(handoff.encode()) > MAX_HANDOFF_BYTES:
                    raise ValueError('workflow handoff exceeds its byte budget')
            if not isinstance(context.inputs, str):
                raise ValueError('workflow question required')
            question = task.prompt + '\n\nUser request:\n' + context.inputs
            result = await agent.run(question, tools=self._tools(), context=handoff)
            output = result.output
            receipt = AgentTaskReceipt(task_id=task.task_id, agent_id=result.agent_id, model_id=result.model_id,
                binding=result.binding, depends_on=task.depends_on, text=output.text, source_status=cast(Literal['retained', 'none'], output.source_status),
                evidence_ids=output.evidence_ids, evidence_packets=output.evidence_packets,
                model_calls=output.model_calls, tool_calls=output.tool_calls)
            return cast(JSONValue, receipt.model_dump(mode='json'))
        return execute

    @staticmethod
    def _question(question: str) -> str:
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 4000:
            raise ValueError('workflow question must contain 1..4000 UTF-8 bytes')
        return question

    async def run(self, run_id: str, question: str) -> WorkflowResult:
        return await self._runner.run(run_id, space=self._space, scope=self._binding_scope, inputs=self._question(question))

    def status(self, run_id: str, question: str) -> WorkflowStatus | None:
        return self._runner.status(run_id, space=self._space, scope=self._binding_scope, inputs=self._question(question))

    def close(self) -> None:
        self._runner.close()
