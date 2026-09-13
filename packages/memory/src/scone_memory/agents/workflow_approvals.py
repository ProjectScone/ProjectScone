"""Shared native workflow wiring for guarded callbacks and inspected proposals."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

from .approval_context import ApprovalContext
from .approval_inspection import inspect_tool_approval
from .approval_models import ToolApprovalRecord
from .approval_store import AgentApprovalStore
from .catalog import AgentResult, BoundAgent
from .turn_journal import TurnJournalPaused
from .workflow import (JSONValue, StepContext, WorkflowError, WorkflowPaused, WorkflowPauseSnapshot,
                       WorkflowPausableStep, WorkflowRunner, WorkflowStep)
from ..integrations.scoped_tools import ScopedMemoryTools
from ..realtime.answer_requirements import AnswerRequirements


def guarded(agent: BoundAgent) -> bool:
    return any(tool.requires_approval for tool in agent.tools)


def model_step(step_id: str, version: str, callback: Callable[[StepContext], Awaitable[JSONValue | WorkflowPaused]],
               *, pausable: bool) -> WorkflowStep | WorkflowPausableStep:
    if pausable:
        return WorkflowPausableStep(step_id, version, callback)
    async def ordinary(context: StepContext) -> JSONValue:
        value = await callback(context)
        if isinstance(value, WorkflowPaused):
            raise WorkflowError('unexpected_agent_pause')
        return value
    return WorkflowStep(step_id, version, ordinary)


async def invoke_agent(agent: BoundAgent, question: str, *, tools: ScopedMemoryTools, context: StepContext,
                       step_id: str, selection_id: str, store: AgentApprovalStore | None,
                       activation_id: str | None, prior: str | None,
                       requirements: AnswerRequirements | None) -> AgentResult | WorkflowPaused:
    if not guarded(agent):
        return await agent.run(question, tools=tools, context=prior, answer_requirements=requirements)
    if store is None:
        raise WorkflowError('approval_store_required')
    approval = ApprovalContext(store, context, step_id=step_id, selection_id=selection_id, activation_id=activation_id)
    try:
        return await agent.run(question, tools=tools, context=prior, answer_requirements=requirements,
                               checkpoints=context.checkpoints, approval=approval)
    except TurnJournalPaused as pause:
        return pause.pause


async def inspect_workflow_approvals(runner: WorkflowRunner, store: AgentApprovalStore | None, *,
    run_id: str, space: str, scope: Mapping[str, JSONValue], inputs: str, tools: ScopedMemoryTools,
    select: Callable[[WorkflowPauseSnapshot], tuple[str, BoundAgent]]) -> tuple[ToolApprovalRecord, ...]:
    if store is None:
        return ()
    try:
        async with asyncio.timeout(30) as timer:
            snapshots = await runner.inspect_pauses(run_id, space=space, scope=scope, inputs=inputs)
            completed = await runner.inspect_completed(run_id, space=space, scope=scope, inputs=inputs)
            by_step = {snapshot.step_id: snapshot for snapshot in snapshots}
            records = store.list(space, run_id)
            visible = []
            verified: set[str] = set()
            for record in records:
                snapshot = by_step.get(record.call.step_id)
                if record.revision == 4:
                    if record.call.step_id in completed or snapshot is not None:
                        visible.append(record)
                    continue
                if snapshot is None:
                    continue
                selection_id, agent = select(snapshot)
                if record.call.selection_id != selection_id or snapshot.step_id in verified:
                    raise WorkflowError('approval_key_or_integrity')
                await inspect_tool_approval(snapshot, record, agent, tools)
                verified.add(snapshot.step_id)
                visible.append(record)
            if verified != by_step.keys():
                raise WorkflowError('approval_not_ready')
            for snapshot in snapshots:
                snapshot.check_current()
            expires = timer.when()
            if timer.expired() or (expires is not None and asyncio.get_running_loop().time() >= expires):
                raise WorkflowError('verification_unavailable')
            active = asyncio.current_task()
            if active is not None and active.cancelling():
                raise asyncio.CancelledError
            return tuple(visible)
    except TimeoutError:
        raise WorkflowError('verification_unavailable') from None
