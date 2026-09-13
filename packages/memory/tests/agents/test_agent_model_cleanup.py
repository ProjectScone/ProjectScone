"""Factory-owned model clients close on every invocation exit."""

import asyncio
import traceback

import httpx
import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolStep
from scone_memory.agents.turn_journal import TurnJournalPaused
from .test_agent_turn_recovery import storage
from .test_custom_tools import call, tool
from .test_evidence_tool_loop import binding
from .test_task_workflow import memory


def registry(factory, *, registered=(), initial_search=False):
    return AgentCatalog(
        models=[AgentModel('chosen', 'Chosen', '1', factory)],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='Answer from evidence.',
                models=('chosen',),
                default_model='chosen',
                initial_search=initial_search,
                tools=tuple(item.name for item in registered),
            )
        ],
        tools=registered,
    )


class Model:
    def __init__(self):
        self.closed = 0
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    async def complete(self, messages, tools):
        assert not self.client.is_closed
        return ToolStep(content='Done')

    async def aclose(self):
        self.closed += 1
        await self.client.aclose()


async def test_each_owned_client_closes_once_before_result_publication(memory):
    created = []

    def factory():
        model = Model()
        created.append(model)
        return model

    selected = registry(factory).bind('worker')
    for question in ['One', 'Two']:
        result = await selected.run(question, tools=binding(memory))
        assert result.model_id == 'chosen' and result.output.text == 'Done'
        assert created[-1].closed == 1 and created[-1].client.is_closed
    assert len(created) == 2 and created[0] is not created[1]
    assert await memory.space_deleted('alpha') is None


@pytest.mark.parametrize('invalid', [False, True])
async def test_failure_or_invalid_factory_product_still_closes(memory, invalid):
    model = Model()

    async def fail(*args):
        raise RuntimeError('provider failed')

    model.complete = None if invalid else fail
    with pytest.raises((RuntimeError, ValueError)):
        await registry(lambda: model).bind('worker').run('Question', tools=binding(memory))
    assert model.closed == 1 and model.client.is_closed


async def test_changed_scope_after_factory_creation_still_closes(memory):
    model = Model()
    scoped = binding(memory)
    points, _ = storage()

    def factory():
        scoped._space = 'bravo'
        return model

    with pytest.raises(RuntimeError, match='binding_mismatch'):
        await registry(factory).bind('worker').run('Question', tools=scoped, checkpoints=points)
    assert model.closed == 1 and model.client.is_closed


@pytest.mark.parametrize('cancel_before_close', [True, False])
async def test_repeated_cancellation_joins_cleanup_before_releasing_invocation(memory, cancel_before_close):
    entered = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()

    class Waiting(Model):
        async def complete(self, messages, tools):
            entered.set()
            if cancel_before_close:
                await asyncio.Event().wait()
            return ToolStep(content='Done')

        async def aclose(self):
            closing.set()
            await release.wait()
            await super().aclose()

    model = Waiting()
    task = asyncio.create_task(registry(lambda: model).bind('worker').run('Question', tools=binding(memory)))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel_before_close:
            task.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not model.client.is_closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.closed == 1 and model.client.is_closed
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await model.client.aclose()


@pytest.mark.parametrize('primary_failure', [False, True])
async def test_cleanup_error_is_fixed_and_does_not_replace_primary_failure(memory, primary_failure):
    class Broken(Model):
        async def complete(self, messages, tools):
            if primary_failure:
                raise RuntimeError('original failure')
            return ToolStep(content='Done')

        async def aclose(self):
            await super().aclose()
            raise RuntimeError('PRIVATE provider credential and endpoint')

    model = Broken()
    with pytest.raises(RuntimeError) as caught:
        await registry(lambda: model).bind('worker').run('Question', tools=binding(memory))
    assert model.closed == 1
    rendered = ''.join(traceback.format_exception(caught.value))
    assert 'PRIVATE' not in rendered
    if primary_failure:
        assert 'agent_model_cleanup_failed' in getattr(caught.value, '__notes__', ())
    else:
        assert str(caught.value) == 'agent_model_cleanup_failed'


async def test_pause_closes_client_and_new_activation_reuses_recorded_proposal(memory):
    models = []
    requests = []
    effects = []

    class Pausing(Model):
        async def complete(self, messages, tools):
            requests.append(1)
            return ToolStep(content='Done') if any(row['role'] == 'tool' for row in messages) else call()

    def factory():
        model = Pausing()
        models.append(model)
        return model

    selected = registry(factory, registered=[tool(lambda args, ctx: effects.append(args['count']))]).bind(
        'worker'
    )
    points, _ = storage()
    with pytest.raises(TurnJournalPaused):
        await selected.run('Count', tools=binding(memory), checkpoints=points, max_new_operations=1)
    assert len(requests) == 1 and effects == [] and models[0].closed == 1
    result = await selected.run('Count', tools=binding(memory), checkpoints=points)
    assert result.output.text == 'Done' and effects == [3] and len(requests) == 2
    assert len(models) == 2 and all(model.closed == 1 for model in models)


async def test_cleanup_cannot_publish_evidence_removed_during_close(memory):
    episode = await memory.remember('alpha', 'Juniper is in Oregon.', metadata={'team': 'blue'})

    class Forgetting(Model):
        async def aclose(self):
            await memory.forget('alpha', episode.episode_id)
            await super().aclose()

    model = Forgetting()
    with pytest.raises(RuntimeError, match='evidence'):
        await registry(lambda: model, initial_search=True).bind('worker').run(
            'Juniper', tools=binding(memory)
        )
    assert model.closed == 1


async def test_direct_loop_keeps_caller_owned_client_open(memory):
    model = Model()
    try:
        result = await EvidenceToolLoop(model, binding(memory)).run([{'role': 'user', 'content': 'Question'}])
        assert result.text == 'Done' and model.closed == 0 and not model.client.is_closed
    finally:
        await model.aclose()


async def test_failed_cleanup_after_effect_does_not_replay_after_workflow_reopen(memory, tmp_path):
    from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow
    from scone_memory.agents.workflow import WorkflowError
    from scone_memory.retrieval.recall_scope import RecallScope

    effects = []
    models = []
    requests = []

    class Broken(Model):
        async def complete(self, messages, tools):
            requests.append(1)
            return ToolStep(content='Done') if any(row['role'] == 'tool' for row in messages) else call()

        async def aclose(self):
            await super().aclose()
            raise RuntimeError('PRIVATE close failure')

    def factory():
        model = Broken()
        models.append(model)
        return model

    catalog = registry(factory, registered=[tool(lambda args, ctx: effects.append(args['count']))])
    plan = AgentTaskPlan(
        workflow_id='job', tasks=(AgentTask(task_id='work', agent_id='worker', prompt='Count'),)
    )

    def open_workflow():
        return AgentWorkflow(
            tmp_path / 'workflow',
            key=b'k' * 32,
            catalog=catalog,
            plan=plan,
            memory=memory,
            space='alpha',
            scope=RecallScope.validated(),
        )

    first = open_workflow()
    try:
        with pytest.raises(WorkflowError):
            await first.run('one', 'Count')
        assert first.status('one', 'Count').status != 'completed'
    finally:
        first.close()
    assert len(requests) == 2 and effects == [3] and models[0].closed == 1
    reopened = open_workflow()
    try:
        with pytest.raises(WorkflowError):
            await reopened.run('one', 'Count')
        assert len(models) == 1 and len(requests) == 2 and effects == [3]
    finally:
        reopened.close()


async def test_saved_guarded_run_closes_each_activation_without_reproposing(memory, tmp_path):
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.run_service import AgentRunService
    from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
    from scone_memory.retrieval.recall_scope import RecallScope

    models = []
    requests = []
    effects = []

    class Guarded(Model):
        async def complete(self, messages, tools):
            requests.append(1)
            return ToolStep(content='Done') if any(row['role'] == 'tool' for row in messages) else call()

    def factory():
        model = Guarded()
        models.append(model)
        return model

    catalog = registry(
        factory, registered=[tool(lambda args, ctx: effects.append(args['count']), requires_approval=True)]
    )
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    plans.save(
        'alpha',
        AgentTaskPlan(
            workflow_id='job', tasks=(AgentTask(task_id='work', agent_id='worker', prompt='Count'),)
        ),
        catalog=catalog,
        expected_revision=0,
    )

    def open_service():
        return AgentRunService(
            tmp_path / 'runs',
            key=b'k' * 32,
            catalog=catalog,
            plans=plans,
            memory=memory,
            scope_for=lambda _: RecallScope.validated(),
        )

    service = open_service()
    try:
        await service.start('alpha', 'one', workflow_id='job', plan_revision=1, question='Count')
        assert (await service.wait('alpha', 'one')).status == 'paused'
        assert len(models) == 1 and models[0].closed == 1 and effects == [] and len(requests) == 1
        await service.aclose()
        service = open_service()
        (pending,) = await service.approvals('alpha', 'one')
        await service.decide_tool(
            'alpha', 'one', pending.request_id, decision='approve', actor='owner', expected_revision=1
        )
        assert len(models) == 1 and len(requests) == 1
        await service.continue_tools(
            'alpha', 'one', continuation_id='continue', decisions={pending.request_id: 2}
        )
        assert (await service.wait('alpha', 'one')).status == 'completed'
        assert len(models) == 2 and all(model.closed == 1 for model in models)
        assert len(requests) == 2 and effects == [3]
        await service.aclose()
        service = open_service()
        await service.continue_tools(
            'alpha', 'one', continuation_id='continue', decisions={pending.request_id: 2}
        )
        assert len(models) == 2 and len(requests) == 2 and effects == [3]
    finally:
        await service.aclose()
        plans.close()
