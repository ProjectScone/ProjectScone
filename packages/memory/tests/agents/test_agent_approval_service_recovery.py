"""Mixed continuations, ownership and recovery must not grant unrelated work."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import call, tool
from .test_evidence_tool_loop import Script
from .test_task_workflow import memory


@pytest.fixture
async def service_host(memory, tmp_path):
    state = SimpleNamespace(effects=[], models={})
    for name in ('first', 'second'):
        state.models[name] = Script(call(call_id=name), ToolStep(content=name + ' finished'))
    registry = AgentCatalog(
        models=[AgentModel(name, name, '1', lambda name=name: state.models[name]) for name in state.models],
        tools=[tool(lambda args, ctx: state.effects.append(args['count']), requires_approval=True)],
        agents=[
            AgentDefinition(
                agent_id=name,
                instructions='Work.',
                models=(name,),
                default_model=name,
                initial_search=False,
                tools=('double_count',),
            )
            for name in state.models
        ],
    )
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    tasks = tuple(AgentTask(task_id=name, agent_id=name, prompt=name) for name in state.models)
    plans.save(
        'alpha', AgentTaskPlan(workflow_id='parallel', tasks=tasks), catalog=registry, expected_revision=0
    )
    plans.save(
        'alpha',
        InteractiveAgentPlan(
            kind='interactive',
            workflow_id='mixed',
            tasks=(
                tasks[0],
                HumanInputTask(kind='input', task_id='human', prompt='Choose'),
                AgentTask(task_id='second', agent_id='second', prompt='Follow', depends_on=('human',)),
            ),
        ),
        catalog=registry,
        expected_revision=0,
    )

    def open_service():
        return AgentRunService(
            tmp_path / 'runs',
            key=b'k' * 32,
            catalog=registry,
            plans=plans,
            memory=memory,
            scope_for=lambda _: RecallScope.validated(),
            max_parallel_tasks=2,
        )

    state.service, state.open = open_service(), open_service

    async def start(workflow='parallel'):
        await state.service.start(
            'alpha', 'one', workflow_id=workflow, plan_revision=1, question='Question', max_parallel=2
        )
        return await state.service.wait('alpha', 'one')

    async def decide(selection='first'):
        record = next(
            record
            for record in await state.service.approvals('alpha', 'one')
            if record.call.selection_id == selection and record.revision == 1
        )
        return await state.service.decide_tool(
            'alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1
        )

    state.start, state.decide = start, decide
    yield state
    await state.service.aclose()
    plans.close()


async def test_subset_tool_activation_does_not_consume_other_ticket(service_host):
    host = service_host
    assert (await host.start()).paused_steps == ('first', 'second')
    first = await host.decide()
    before = host.service._progress(await host.service.request('alpha', 'one')).attempts
    await host.service.continue_tools(
        'alpha', 'one', continuation_id='first-only', decisions={first.request_id: 2}
    )
    status = await host.service.wait('alpha', 'one')
    assert (
        status.status == 'paused'
        and status.completed_steps == ('first',)
        and status.paused_steps == ('second',)
    )
    after = host.service._progress(await host.service.request('alpha', 'one')).attempts
    assert after['second'] == before['second'] == 1 and host.effects == [3]
    second = await host.decide('second')
    await host.service.continue_tools(
        'alpha', 'one', continuation_id='second-only', decisions={second.request_id: 2}
    )
    assert (await host.service.wait('alpha', 'one')).status == 'completed' and host.effects == [3, 3]


async def test_human_continuation_preserves_unrelated_approval_pause(service_host):
    host = service_host
    paused = await host.start('mixed')
    assert paused.paused_steps == ('first',) and paused.waiting_steps == ('human',)
    (human,) = await host.service.inputs('alpha', 'one')
    await host.service.respond('alpha', 'one', 'human', response='Choose second', expected_revision=1)
    await host.service.continue_run('alpha', 'one', continuation_id='human-only', responses={'human': 2})
    status = await host.service.wait('alpha', 'one')
    assert status.paused_steps == ('first', 'second') and status.completed_steps == ('human',)
    progress = host.service._progress(await host.service.request('alpha', 'one'))
    assert progress.attempts['first'] == 1 and host.effects == []


async def test_tool_continuation_cannot_consume_failed_human_admission(service_host, monkeypatch):
    host = service_host
    await host.start('mixed')
    await host.service.respond('alpha', 'one', 'human', response='Choose second', expected_revision=1)
    original = host.service._open

    def fail_after_activation(request, **kwargs):
        if host.service._inputs.get('alpha', 'one', 'human').revision == 3:
            raise WorkflowError('run_store_unavailable')
        return original(request, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(host.service, '_open', fail_after_activation)
        with pytest.raises(WorkflowError):
            await host.service.continue_run(
                'alpha', 'one', continuation_id='human-only', responses={'human': 2}
            )
    first = await host.decide()
    await host.service.continue_tools(
        'alpha', 'one', continuation_id='tool-only', decisions={first.request_id: 2}
    )
    status = await host.service.wait('alpha', 'one')
    assert status.status == 'awaiting_input' and status.completed_steps == ('first',)
    assert status.waiting_steps == ('human',) and host.models['second'].requests == []
    await host.service.continue_run('alpha', 'one', continuation_id='human-only', responses={'human': 2})
    assert (await host.service.wait('alpha', 'one')).paused_steps == ('second',)


async def test_tool_activation_lost_admission_requires_same_explicit_continuation(service_host, monkeypatch):
    host = service_host
    await host.start()
    first = await host.decide()
    original = host.service._open

    def refuse(request, **kwargs):
        if kwargs.get('approval_activation'):
            raise WorkflowError('run_store_unavailable')
        return original(request, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(host.service, '_open', refuse)
        with pytest.raises(WorkflowError):
            await host.service.continue_tools(
                'alpha', 'one', continuation_id='exact', decisions={first.request_id: 2}
            )
    assert host.effects == []
    assert (await host.service.status('alpha', 'one')).status == 'paused'
    assert not (
        await host.service.start(
            'alpha', 'one', workflow_id='parallel', plan_revision=1, question='Question', max_parallel=2
        )
    ).active_local
    with pytest.raises(WorkflowError, match='approval_activation_conflict'):
        await host.service.continue_tools(
            'alpha', 'one', continuation_id='different', decisions={first.request_id: 2}
        )
    await host.service.continue_tools(
        'alpha', 'one', continuation_id='exact', decisions={first.request_id: 2}
    )
    assert (await host.service.wait('alpha', 'one')).completed_steps == ('first',) and host.effects == [3]


async def test_unknown_attempt_is_not_hidden_by_own_admission(service_host):
    host = service_host
    await host.start()
    first = await host.decide()

    async def fail(*args, **kwargs):
        raise RuntimeError('private failure')

    host.models['first'].steps = [fail]
    await host.service.continue_tools(
        'alpha', 'one', continuation_id='first-only', decisions={first.request_id: 2}
    )
    assert (await host.service.wait('alpha', 'one')).outcome_unknown
    second = await host.decide('second')
    with pytest.raises(WorkflowError, match='outcome_unknown'):
        await host.service.continue_tools(
            'alpha', 'one', continuation_id='second-only', decisions={second.request_id: 2}
        )
    assert host.service._approvals.activation('alpha', 'one', 'second-only') is None
