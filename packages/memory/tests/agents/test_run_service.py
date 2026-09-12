"""Local background runs retain ownership, scope and safe recovery semantics."""
import asyncio
import pytest

from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
from scone_memory.agents.catalog import AgentCatalog,AgentModel,AgentDefinition
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask,AgentTaskPlan
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope


@pytest.fixture
async def setup(tmp_path):
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    calls=[];entered=asyncio.Event();release=asyncio.Event()
    class Model:
        async def complete(self,messages,tools):
            calls.append(messages);entered.set();await release.wait();return ToolStep(content='Answer from configured model.')
    catalog=AgentCatalog(models=[AgentModel('local','Local','1',Model)],agents=[AgentDefinition(
        agent_id='research',instructions='Use evidence.',models=('local',),default_model='local',initial_search=False)])
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    plan=AgentTaskPlan(workflow_id='report',tasks=(AgentTask(task_id='find',agent_id='research',prompt='Find evidence.'),))
    plans.save('alpha',plan,catalog=catalog,expected_revision=0)
    service=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
        scope_for=lambda space:RecallScope.validated(),max_active=1)
    yield service,plans,catalog,memory,calls,entered,release,tmp_path
    release.set();await service.aclose();plans.close();await memory.close()


async def test_start_owns_background_work_and_completed_reads_never_execute(setup):
    service,plans,catalog,memory,calls,entered,release,tmp_path=setup
    admitted=await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
    assert admitted.run_id=='one'
    await entered.wait()
    with pytest.raises(WorkflowError,match='not_completed'):await service.result('alpha','one')
    assert await service.status('bravo','one') is None
    with pytest.raises(WorkflowError,match='run_busy'):
        await service.start('alpha','two',workflow_id='report',plan_revision=1,question='Other')
    release.set();await service.wait('alpha','one')
    assert (await service.status('alpha','one')).status=='completed'
    assert (await service.result('alpha','one')).results['find']['model_id']=='local'
    assert len(calls)==1
    await service.aclose()
    reopened=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
        scope_for=lambda space:RecallScope.validated())
    try:
        assert (await reopened.status('alpha','one')).status=='completed'
        assert (await reopened.result('alpha','one')).results['find']['text']=='Answer from configured model.'
        assert len(calls)==1
    finally:await reopened.aclose()


async def test_cancel_is_owned_and_never_replays_unknown_model_call(setup):
    service,_,_,_,calls,entered,_,_=setup
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
    await entered.wait()
    assert await service.cancel('bravo','one') is None
    cancelled=await service.cancel('alpha','one')
    assert cancelled.status=='cancelled' and cancelled.outcome_unknown
    with pytest.raises(WorkflowError,match='outcome_unknown'):
        await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
    assert len(calls)==1


async def test_existing_run_keeps_original_plan_after_saved_plan_is_edited(setup):
    service,plans,catalog,_,calls,_,release,_=setup
    release.set()
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Original question')
    await service.wait('alpha','one')
    saved=plans.get('alpha','report')
    plans.save('alpha',saved.plan,catalog=catalog,expected_revision=1)
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Original question')
    await service.wait('alpha','one')
    assert len(calls)==1
    with pytest.raises(WorkflowError,match='run_request_conflict'):
        await service.start('alpha','one',workflow_id='report',plan_revision=2,question='Different question')


async def test_immediate_cancel_releases_admission_and_never_calls_model(setup):
    service,_,_,_,calls,_,release,_=setup
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
    cancelled=await service.cancel('alpha','one')
    assert cancelled.status=='cancelled' and not cancelled.active_local and not cancelled.outcome_unknown
    assert calls==[]
    release.set()
    await service.start('alpha','two',workflow_id='report',plan_revision=1,question='Next')
    await service.wait('alpha','two')
    assert len(calls)==1


async def test_cancelling_a_waiter_does_not_cancel_owned_execution(setup):
    service,_,_,_,calls,entered,release,_=setup
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
    await entered.wait()
    waiter=asyncio.create_task(service.wait('alpha','one'));await asyncio.sleep(0);waiter.cancel()
    with pytest.raises(asyncio.CancelledError):await waiter
    assert (await service.status('alpha','one')).active_local
    release.set();assert (await service.wait('alpha','one')).status=='completed'
    assert len(calls)==1


async def test_cancelled_before_start_stays_cancelled_after_restart(setup):
    service,plans,catalog,memory,calls,_,_,tmp_path=setup
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
    await service.cancel('alpha','one');await service.aclose()
    reopened=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
        scope_for=lambda space:RecallScope.validated())
    try:
        assert (await reopened.status('alpha','one')).status=='cancelled'
        with pytest.raises(WorkflowError,match='run_cancelled'):
            await reopened.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
        assert calls==[]
    finally:await reopened.aclose()


async def test_cancel_stops_owned_model_even_if_cancellation_marker_is_unavailable(setup,monkeypatch):
    service,_,_,_,calls,entered,_,_=setup
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question');await entered.wait()
    def unavailable(*args,**kwargs):raise WorkflowError('run_store_unavailable')
    monkeypatch.setattr(service._runs,'request_cancel',unavailable)
    with pytest.raises(WorkflowError,match='run_store_unavailable'):await service.cancel('alpha','one')
    status=await service.status('alpha','one')
    assert not status.active_local and status.outcome_unknown
    assert calls and len(calls)==1


async def test_start_cannot_enter_after_shutdown_has_taken_ownership_snapshot(setup,monkeypatch):
    service,_,_,memory,calls,entered,_,_=setup
    cleanup=asyncio.Event();finish_cleanup=asyncio.Event()
    original_execute=service._execute
    async def delayed_cleanup(request,workflow):
        try:await original_execute(request,workflow)
        finally:cleanup.set();await finish_cleanup.wait()
    monkeypatch.setattr(service,'_execute',delayed_cleanup)
    await service.start('alpha','one',workflow_id='report',plan_revision=1,question='First');await entered.wait()
    scope_entered=asyncio.Event();scope_release=asyncio.Event();pending=None
    original_space=memory.space_deleted
    async def paused_space(space):
        if asyncio.current_task() is pending:
            scope_entered.set();await scope_release.wait()
        return await original_space(space)
    monkeypatch.setattr(memory,'space_deleted',paused_space)
    pending=asyncio.create_task(service.start('alpha','two',workflow_id='report',plan_revision=1,question='Second'))
    await scope_entered.wait();closing=asyncio.create_task(service.aclose());await cleanup.wait()
    try:
        scope_release.set()
        with pytest.raises(WorkflowError,match='run_service_closed'):await pending
    finally:
        finish_cleanup.set();await closing
    assert len(calls)==1 and service._tasks=={}


async def test_foreign_service_cannot_acknowledge_prestart_cancellation(setup):
    service,plans,catalog,memory,calls,_,release,tmp_path=setup
    other=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
        scope_for=lambda space:RecallScope.validated())
    try:
        await service.start('alpha','one',workflow_id='report',plan_revision=1,question='Question')
        with pytest.raises(WorkflowError,match='run_not_owned'):await other.cancel('alpha','one')
        release.set();await service.wait('alpha','one')
        assert len(calls)==1
    finally:await other.aclose()
