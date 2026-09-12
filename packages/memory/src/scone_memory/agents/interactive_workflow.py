"""Explicitly admitted human-input DAGs with evidence-checked model receipts."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
import hashlib
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from .catalog import AgentCatalog, BoundAgent, Identifier
from .input_store import AgentInputRecord, AgentInputStore, _response_digest
from .interactive_plan import HumanInputTask, InteractiveAgentPlan
from .run_store import AgentRunRequest
from .task_workflow import (AgentTask, AgentTaskReceipt, AgentWorkflow, MAX_HANDOFF_BYTES,
                            _json, _verify_agent_evidence)
from .workflow import (JSONValue, StepContext, WorkflowError, WorkflowInputStep, WorkflowInputValue,
                       WorkflowResult, WorkflowRunner, WorkflowStatus, WorkflowStep)
from ..integrations.scoped_tools import ScopedMemoryTools
from ..memory.engine import MemoryEngine


class HumanInputReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    kind: Literal['human_input']
    task_id: Identifier
    depends_on: tuple[Identifier, ...]
    text: str
    activation_id: Identifier
    response_digest: str


InputReceipt = AgentTaskReceipt | HumanInputReceipt


def _human_receipt(task: HumanInputTask, record: AgentInputRecord) -> HumanInputReceipt:
    if record.response is None or record.activation_id is None:
        raise WorkflowError('input_not_activated')
    return HumanInputReceipt(kind='human_input', task_id=task.task_id, depends_on=task.depends_on,
        text=record.response, activation_id=record.activation_id, response_digest=_response_digest(record))


def _context_text(dependencies: tuple[str, ...], receipts: dict[str, InputReceipt]) -> str:
    values: list[dict[str, object]] = []
    for name in dependencies:
        receipt = receipts[name]
        if isinstance(receipt, HumanInputReceipt):
            values.append({'task_id': name, 'kind': 'human_input', 'text': receipt.text})
        else:
            values.append({'task_id': name, 'agent_id': receipt.agent_id, 'text': receipt.text,
                           'source_status': receipt.source_status, 'evidence_ids': receipt.evidence_ids})
    context = _json(values)
    if len(context.encode()) > MAX_HANDOFF_BYTES:
        raise WorkflowError('input_context_limit')
    return context


class InteractiveAgentWorkflow:
    """Activation is captured at admission; later replies cannot unlock this run."""
    def __init__(self, path: str | Path, *, key: bytes, catalog: AgentCatalog, request: AgentRunRequest,
                 memory: MemoryEngine, inputs: AgentInputStore, activated: Sequence[AgentInputRecord],
                 deadline_s: float = 120.0) -> None:
        plan = request.plan.checked_plan(catalog)
        if not isinstance(plan, InteractiveAgentPlan):
            raise ValueError('interactive plan required')
        self._request, self._memory, self._inputs = request, memory, inputs
        self._tasks = {task.task_id: task for task in plan.ordered()}
        self._agents = {task.task_id: catalog.bind(task.agent_id, model_id=task.model_id)
                        for task in self._tasks.values() if isinstance(task, AgentTask)}
        self._activated: dict[str, AgentInputRecord] = {}
        for record in activated:
            if (record.space != request.space or record.run_id != request.run_id or record.activation_id is None
                    or record.task_id in self._activated or inputs.get(record.space, record.run_id, record.task_id) != record):
                raise WorkflowError('input_activation_conflict')
            self._activated[record.task_id] = AgentInputRecord.model_validate(record.model_dump())
        self._tools()
        signature = hashlib.sha256(_json({'implementation': 'scone-interactive-agent-v1',
            'plan': plan.model_dump(mode='json'),
            'bindings': {name: agent.fingerprint for name, agent in self._agents.items()}}).encode()).hexdigest()
        self._scope = cast(dict[str, JSONValue], {'plan': signature, 'recall': request.scope,
                                                 'exclude_session_id': request.exclude_session_id})
        steps: list[WorkflowStep | WorkflowInputStep] = []
        for task in self._tasks.values():
            if isinstance(task, HumanInputTask):
                steps.append(WorkflowInputStep(task.task_id, signature, self._poll(task)))
            else:
                steps.append(WorkflowStep(task.task_id, signature, self._step(task, self._agents[task.task_id])))
        self._runner = WorkflowRunner(path, key=key, steps=steps, source_verifier=self._verify,
            max_payload_bytes=1000000, deadline=deadline_s, verify_before_step=True,
            max_parallel=request.max_parallel, dependencies={task.task_id: task.depends_on for task in self._tasks.values()})

    def _tools(self) -> ScopedMemoryTools:
        return ScopedMemoryTools(self._memory, self._request.space, scope=self._request.recall_scope(),
                                 exclude_session_id=self._request.exclude_session_id)

    def _receipts(self, completed: dict[str, JSONValue]) -> dict[str, InputReceipt]:
        receipts: dict[str, InputReceipt] = {}
        for name, value in completed.items():
            task = self._tasks.get(name)
            if task is None or not set(task.depends_on) <= completed.keys():
                raise ValueError('unknown input task or incomplete dependencies')
            if isinstance(task, HumanInputTask):
                human = HumanInputReceipt.model_validate_json(_json(value))
                record = self._inputs.get(self._request.space, self._request.run_id, name)
                if record is None or human != _human_receipt(task, record):
                    raise ValueError('input receipt binding changed')
                receipts[name] = human
            else:
                agent = self._agents[name]
                model = AgentTaskReceipt.model_validate_json(_json(value))
                if (model.task_id != name or model.agent_id != task.agent_id or model.model_id != agent.model_id
                        or model.binding != agent.fingerprint or model.depends_on != task.depends_on):
                    raise ValueError('model receipt binding changed')
                receipts[name] = model
        for name, receipt in receipts.items():
            if isinstance(receipt, HumanInputReceipt):
                record = self._inputs.get(self._request.space, self._request.run_id, name)
                if record is None or record.context != _context_text(receipt.depends_on, receipts):
                    raise ValueError('input dependency context changed')
        return receipts

    async def _verify(self, context: StepContext) -> bool:
        if context.space != self._request.space or context.scope != self._scope:
            return False
        try:
            receipts = self._receipts(context.completed)
        except WorkflowError as error:
            if error.code in {'run_store_unavailable', 'run_store_closed'}:
                raise OSError('input storage verification unavailable') from None
            raise
        return await _verify_agent_evidence(self._memory, self._request.space, self._request.recall_scope(),
            self._request.exclude_session_id, tuple(value for value in receipts.values() if isinstance(value, AgentTaskReceipt)))

    def _poll(self, task: HumanInputTask) -> Callable[[StepContext], Awaitable[WorkflowInputValue | None]]:
        async def poll(context: StepContext) -> WorkflowInputValue | None:
            receipts = self._receipts(context.completed)
            record = self._inputs.request(self._request.space, self._request.run_id, task.task_id,
                                          context=_context_text(task.depends_on, receipts))
            selected = self._activated.get(task.task_id)
            if selected is None:
                return None
            if selected != record:
                raise WorkflowError('input_activation_conflict')
            return WorkflowInputValue(cast(JSONValue, _human_receipt(task, record).model_dump(mode='json')))
        return poll

    def _step(self, task: AgentTask, agent: BoundAgent) -> Callable[[StepContext], Awaitable[JSONValue]]:
        async def execute(context: StepContext) -> JSONValue:
            receipts = self._receipts(context.completed)
            handoff = _context_text(task.depends_on, receipts) if task.depends_on else None
            if not isinstance(context.inputs, str):
                raise ValueError('workflow question required')
            result = await agent.run(task.prompt + '\n\nUser request:\n' + context.inputs,
                                     tools=self._tools(), context=handoff)
            output = result.output
            receipt = AgentTaskReceipt(task_id=task.task_id, agent_id=result.agent_id, model_id=result.model_id,
                binding=result.binding, depends_on=task.depends_on, text=output.text,
                source_status=cast(Literal['retained', 'none'], output.source_status),
                evidence_ids=output.evidence_ids, evidence_packets=output.evidence_packets,
                model_calls=output.model_calls, tool_calls=output.tool_calls)
            return cast(JSONValue, receipt.model_dump(mode='json'))
        return execute

    def _bound(self, run_id: str, question: str) -> str:
        if run_id != self._request.run_id or question != self._request.question:
            raise WorkflowError('binding_mismatch')
        return AgentWorkflow._question(question)

    async def inspect_inputs(self, run_id: str, question: str) -> tuple[AgentInputRecord, ...]:
        completed = await self._runner.inspect_completed(run_id, space=self._request.space,
            scope=self._scope, inputs=self._bound(run_id, question))
        receipts = self._receipts(completed)
        records = self._inputs.list(self._request.space, run_id)
        visible = []
        for record in records:
            task = self._tasks[record.task_id]
            if not set(task.depends_on) <= receipts.keys():
                continue
            if record.context != _context_text(task.depends_on, receipts):
                raise WorkflowError('input_key_or_integrity')
            visible.append(record)
        return tuple(visible)

    async def run(self, run_id: str, question: str) -> WorkflowResult:
        return await self._runner.run(run_id, space=self._request.space,
                                     scope=self._scope, inputs=self._bound(run_id, question))

    async def read_result(self, run_id: str, question: str) -> WorkflowResult | None:
        return await self._runner.read_result(run_id, space=self._request.space,
                                             scope=self._scope, inputs=self._bound(run_id, question))

    def status(self, run_id: str, question: str) -> WorkflowStatus | None:
        return self._runner.status(run_id, space=self._request.space,
                                   scope=self._scope, inputs=self._bound(run_id, question))

    def close(self) -> None:
        self._runner.close()
