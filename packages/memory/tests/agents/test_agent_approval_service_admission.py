import asyncio
import os
from dataclasses import replace
import pytest
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import call, tool
from .test_task_workflow import memory


@pytest.fixture
async def approval_service(memory, tmp_path):
    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done') if any(row.get('role') == 'tool' for row in messages) else call()

    registry = AgentCatalog(
        models=[AgentModel('local', 'Local', '1', Model)],
        tools=[tool(lambda args, ctx: None, requires_approval=True)],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='Work',
                models=('local',),
                default_model='local',
                initial_search=False,
                tools=('double_count',),
            )
        ],
    )
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    plans.save(
        'alpha',
        AgentTaskPlan(
            workflow_id='job', tasks=(AgentTask(task_id='work', agent_id='worker', prompt='Work'),)
        ),
        catalog=registry,
        expected_revision=0,
    )
    service = AgentRunService(
        tmp_path / 'runs',
        key=b'k' * 32,
        catalog=registry,
        plans=plans,
        memory=memory,
        scope_for=lambda _: RecallScope.validated(),
        max_active=1,
    )
    records = {}
    for run in ['one', 'two']:
        await service.start('alpha', run, workflow_id='job', plan_revision=1, question='Work')
        assert (await service.wait('alpha', run)).status == 'paused'
        (record,) = await service.approvals('alpha', run)
        records[run] = record
        await service.decide_tool(
            'alpha', run, record.request_id, decision='approve', actor='owner', expected_revision=1
        )
    yield service, records
    await service.aclose()
    plans.close()


async def test_pending_tool_admission_reserves_capacity_before_source_reads(approval_service):
    service, records = approval_service
    original = service.approvals
    entered = []
    first = asyncio.Event()
    release = asyncio.Event()

    async def blocked(space, run_id, **kwargs):
        entered.append(run_id)
        first.set()
        await release.wait()
        return await original(space, run_id, **kwargs)

    service.approvals = blocked

    async def admit(run):
        return await service.continue_tools(
            'alpha', run, continuation_id='next', decisions={records[run].request_id: 2}
        )

    one = asyncio.create_task(admit('one'))
    await first.wait()
    two = asyncio.create_task(admit('two'))
    try:
        await asyncio.sleep(0.02)
        assert entered == ['one']
        assert (
            two.done() and isinstance(two.exception(), WorkflowError) and two.exception().code == 'run_busy'
        )
    finally:
        release.set()
        await asyncio.gather(one, two, return_exceptions=True)


async def test_activation_commit_then_constructor_failure_can_retry_without_effect_replay(approval_service):
    service, records = approval_service
    original = service._open
    once = True

    def fail(request, **kwargs):
        nonlocal once
        if kwargs.get('approval_activation') and once:
            once = False
            raise WorkflowError('journal_unavailable')
        return original(request, **kwargs)

    service._open = fail
    decisions = {records['one'].request_id: 2}
    with pytest.raises(WorkflowError, match='journal_unavailable'):
        await service.continue_tools('alpha', 'one', continuation_id='next', decisions=decisions)
    assert service._approvals.get('alpha', 'one', records['one'].request_id).revision == 3
    assert not service._owners and not service._admissions
    admitted = await service.continue_tools('alpha', 'one', continuation_id='next', decisions=decisions)
    assert admitted.activation.activation_id == 'next'
    assert (await service.wait('alpha', 'one')).status == 'completed'
    assert service._approvals.get('alpha', 'one', records['one'].request_id).revision == 4


async def test_failed_admission_closes_owner_even_when_workflow_close_raises(approval_service):
    service, records = approval_service
    original_open = service._open
    original_status = service._status
    original_claim = service._claim
    descriptors = []

    def claim(request):
        descriptor = original_claim(request)
        descriptors.append(descriptor)
        return descriptor

    def opened(request, **kwargs):
        job = original_open(request, **kwargs)
        if kwargs.get('approval_activation'):
            close = job.close

            def failing_close():
                close()
                raise WorkflowError('journal_unavailable')

            job.close = failing_close
        return job

    def status(request, **kwargs):
        if ('alpha', 'one') in service._workflows:
            raise WorkflowError('journal_unavailable')
        return original_status(request, **kwargs)

    service._open = opened
    service._status = status
    service._claim = claim
    try:
        with pytest.raises(WorkflowError):
            await service.continue_tools(
                'alpha', 'one', continuation_id='next', decisions={records['one'].request_id: 2}
            )
        assert not service._admissions and not service._owners
        assert ('alpha', 'one') not in service._workflows
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
    finally:
        service._open = original_open
        service._status = original_status
        service._claim = original_claim
        service._workflows.pop(('alpha', 'one'), None)
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


async def test_listing_registered_unstarted_run_does_not_create_execution_journal(approval_service):
    service, records = approval_service
    saved = service._plans.get('alpha', 'job')
    request = service._runs.register(
        'alpha', 'unstarted', plan=saved, question='Work', scope=RecallScope.validated()
    )
    path = service._path(request)
    assert not path.exists()
    assert await service.approvals('alpha', 'unstarted') == ()
    assert not path.exists()


async def test_pending_admission_has_exclusive_owner_and_cancellation_releases_it(approval_service):
    service, records = approval_service
    other = AgentRunService(
        service._directory,
        key=b'k' * 32,
        catalog=service._catalog,
        plans=service._plans,
        memory=service._memory,
        scope_for=service._scope_for,
        max_active=1,
    )
    original = service.approvals
    entered = asyncio.Event()
    never = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await never.wait()
        return await original(*args, **kwargs)

    service.approvals = blocked
    decisions = {records['one'].request_id: 2}
    pending = asyncio.create_task(
        service.continue_tools('alpha', 'one', continuation_id='next', decisions=decisions)
    )
    try:
        await entered.wait()
        with pytest.raises(WorkflowError, match='run_busy'):
            await other.continue_tools('alpha', 'one', continuation_id='next', decisions=decisions)
        assert other._approvals.activation('alpha', 'one', 'next') is None
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not service._owners and not service._admissions
        service.approvals = original
        await other.continue_tools('alpha', 'one', continuation_id='next', decisions=decisions)
        assert (await other.wait('alpha', 'one')).status == 'completed'
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await other.aclose()


@pytest.mark.parametrize('change', ['scope', 'catalog'])
async def test_binding_change_during_approval_inspection_refuses_activation(approval_service, change):
    service, records = approval_service
    original = service.approvals
    scope = service._scope_for
    model = service._catalog._models['local']

    async def changed(*args, **kwargs):
        result = await original(*args, **kwargs)
        if change == 'scope':
            service._scope_for = lambda _: RecallScope.validated(where={'changed': 'yes'})
        else:
            service._catalog._models['local'] = replace(model, revision='2')
        return result

    service.approvals = changed
    try:
        with pytest.raises(WorkflowError):
            await service.continue_tools(
                'alpha', 'one', continuation_id='next', decisions={records['one'].request_id: 2}
            )
        assert service._approvals.activation('alpha', 'one', 'next') is None
        assert not service._owners and not service._admissions
    finally:
        service._scope_for = scope
        service._catalog._models['local'] = model
        service.approvals = original
