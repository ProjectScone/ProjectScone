"""Saved replies and explicit continuation enforce the service admission boundary."""
import asyncio

import pytest

from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_run_service import setup


def save_plan(plans, catalog):
    plan = InteractiveAgentPlan(kind='interactive', workflow_id='interactive', tasks=(
        HumanInputTask(kind='input', task_id='choose', prompt='Choose the direction'),
        AgentTask(task_id='independent', agent_id='research', prompt='Independent task'),
        AgentTask(task_id='answer', agent_id='research', prompt='Use the reply', depends_on=('choose',)),))
    plans.save('alpha', plan, catalog=catalog, expected_revision=0)


async def test_save_during_active_sibling_requires_explicit_continuation_after_reopen(setup):
    service, plans, catalog, memory, calls, entered, release, path = setup
    save_plan(plans, catalog)
    await service.start('alpha', 'one', workflow_id='interactive', plan_revision=1, question='Question')
    await entered.wait()
    prompts = await service.inputs('alpha', 'one')
    assert [prompt.task_id for prompt in prompts] == ['choose']
    answered = await service.respond('alpha', 'one', 'choose', response='North', expected_revision=1)
    assert answered.revision == 2
    with pytest.raises(WorkflowError, match='run_busy'):
        await service.continue_run('alpha', 'one', continuation_id='c', responses={'choose': 2})
    assert len(calls) == 1
    release.set()
    status = await service.wait('alpha', 'one')
    assert status.status == 'awaiting_input' and not status.active_local and not status.outcome_unknown
    assert status.waiting_steps == ('choose',)
    await service.aclose()
    reopened = AgentRunService(path / 'runs', key=b'k' * 32, catalog=catalog, plans=plans, memory=memory,
                                scope_for=lambda space: RecallScope.validated())
    try:
        assert (await reopened.inputs('alpha', 'one'))[0].response == 'North'
        # Start retries cannot turn saved replies into implicit continuations.
        assert (await reopened.start('alpha', 'one', workflow_id='interactive', plan_revision=1,
                                     question='Question')).status == 'awaiting_input'
        assert len(calls) == 1
        await reopened.continue_run('alpha', 'one', continuation_id='c', responses={'choose': 2})
        assert (await reopened.wait('alpha', 'one')).status == 'completed'
        assert len(calls) == 2
        assert (await reopened.result('alpha', 'one')).results['choose']['text'] == 'North'
        await reopened.continue_run('alpha', 'one', continuation_id='c', responses={'choose': 2})
        assert len(calls) == 2
    finally:
        await reopened.aclose()


async def test_scope_guard_change_during_verification_cannot_persist_or_activate(setup, monkeypatch):
    service, plans, catalog, _, calls, _, release, _ = setup
    save_plan(plans, catalog)
    release.set()
    await service.start('alpha', 'one', workflow_id='interactive', plan_revision=1, question='Question')
    await service.wait('alpha', 'one')
    def refuse():
        raise WorkflowError('scope_switched')
    with pytest.raises(WorkflowError, match='scope_switched'):
        await service.respond('alpha', 'one', 'choose', response='North', expected_revision=1, admission_guard=refuse)
    assert (await service.inputs('alpha', 'one'))[0].revision == 1
    await service.respond('alpha', 'one', 'choose', response='North', expected_revision=1)
    with pytest.raises(WorkflowError, match='scope_switched'):
        await service.continue_run('alpha', 'one', continuation_id='c', responses={'choose': 2}, admission_guard=refuse)
    assert (await service.inputs('alpha', 'one'))[0].activation_id is None
    assert len(calls) == 1 and not service._owners


async def test_second_service_cannot_continue_an_owned_run(setup):
    service, plans, catalog, memory, calls, entered, release, path = setup
    save_plan(plans, catalog)
    await service.start('alpha', 'one', workflow_id='interactive', plan_revision=1, question='Question')
    await entered.wait()
    other = AgentRunService(path / 'runs', key=b'k' * 32, catalog=catalog, plans=plans, memory=memory,
                            scope_for=lambda space: RecallScope.validated())
    try:
        await other.respond('alpha', 'one', 'choose', response='North', expected_revision=1)
        with pytest.raises(WorkflowError, match='run_busy'):
            await other.continue_run('alpha', 'one', continuation_id='c', responses={'choose': 2})
        assert len(calls) == 1
    finally:
        release.set()
        await other.aclose()
