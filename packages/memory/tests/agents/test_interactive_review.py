import asyncio
import pytest
from .test_run_service import setup
from .test_interactive_service import save_plan
from scone_memory.agents.interactive_workflow import InteractiveAgentWorkflow
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope

async def test_shutdown_releases_pending_continuation_owner(setup, monkeypatch):
    service,plans,catalog,memory,calls,_,release,path=setup
    save_plan(plans,catalog);release.set()
    await service.start('alpha','one',workflow_id='interactive',plan_revision=1,question='Q')
    await service.wait('alpha','one')
    await service.respond('alpha','one','choose',response='North',expected_revision=1)
    entered=asyncio.Event(); finish=asyncio.Event()
    original=InteractiveAgentWorkflow.inspect_inputs
    async def held(self,*args,**kwargs):
        entered.set(); await finish.wait(); return await original(self,*args,**kwargs)
    monkeypatch.setattr(InteractiveAgentWorkflow,'inspect_inputs',held)
    pending=asyncio.create_task(service.continue_run('alpha','one',continuation_id='c',responses={'choose':2}))
    await entered.wait()
    await service.aclose()
    other=AgentRunService(path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,scope_for=lambda space:RecallScope.validated())
    try:
        request=other._runs.get('alpha','one')
        descriptor=other._claim(request)
        import os;os.close(descriptor)
    finally:
        finish.set(); await asyncio.gather(pending,return_exceptions=True);await other.aclose()

async def test_cancel_pending_continuation_releases_lease_without_activation(setup, monkeypatch):
    service,plans,catalog,memory,calls,_,release,path=setup
    save_plan(plans,catalog);release.set()
    await service.start('alpha','one',workflow_id='interactive',plan_revision=1,question='Q');await service.wait('alpha','one')
    await service.respond('alpha','one','choose',response='North',expected_revision=1)
    entered=asyncio.Event()
    original=InteractiveAgentWorkflow.inspect_inputs
    async def held(self,*a,**kw): entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(InteractiveAgentWorkflow,'inspect_inputs',held)
    pending=asyncio.create_task(service.continue_run('alpha','one',continuation_id='c',responses={'choose':2}))
    await entered.wait();pending.cancel()
    with pytest.raises(asyncio.CancelledError):await pending
    request=service._runs.get('alpha','one');descriptor=service._claim(request)
    import os;os.close(descriptor)
    assert service._inputs.get('alpha','one','choose').activation_id is None
    monkeypatch.setattr(InteractiveAgentWorkflow,'inspect_inputs',original)
    await service.continue_run('alpha','one',continuation_id='c',responses={'choose':2})
    assert (await service.wait('alpha','one')).status=='completed'

async def test_activation_durable_before_open_failure_and_retry_reconciles(setup,monkeypatch):
    service,plans,catalog,memory,calls,_,release,path=setup
    save_plan(plans,catalog);release.set()
    await service.start('alpha','one',workflow_id='interactive',plan_revision=1,question='Q');await service.wait('alpha','one')
    await service.respond('alpha','one','choose',response='North',expected_revision=1)
    original=service._open
    def failure(request):
        if service._inputs.get('alpha','one','choose').activation_id:
            raise WorkflowError('injected_unavailable')
        return original(request)
    monkeypatch.setattr(service,'_open',failure)
    with pytest.raises(WorkflowError,match='injected_unavailable'):
        await service.continue_run('alpha','one',continuation_id='c',responses={'choose':2})
    assert service._inputs.get('alpha','one','choose').activation_id=='c'
    assert len(calls)==1
    monkeypatch.setattr(service,'_open',original)
    assert (await service.start('alpha','one',workflow_id='interactive',plan_revision=1,question='Q')).status=='awaiting_input'
    await service.continue_run('alpha','one',continuation_id='c',responses={'choose':2})
    assert (await service.wait('alpha','one')).status=='completed' and len(calls)==2

async def test_real_retained_evidence_blocks_prompt_response_and_continuation(tmp_path):
    from .test_task_workflow import catalog
    from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.interactive_plan import InteractiveAgentPlan,HumanInputTask
    from scone_memory.agents.task_workflow import AgentTask
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    calls=[];agents=catalog(calls,initial_search=True)
    plans=AgentPlanStore(tmp_path/'plans',key=b'k'*32)
    plans.save('alpha',InteractiveAgentPlan(kind='interactive',workflow_id='p',tasks=(
        AgentTask(task_id='a',agent_id='worker',prompt='Find Juniper'),
        HumanInputTask(kind='input',task_id='h',prompt='Choose',depends_on=('a',)),
        AgentTask(task_id='b',agent_id='worker',prompt='Use choice',depends_on=('h',)),)),catalog=agents,expected_revision=0)
    service=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=agents,plans=plans,memory=memory,scope_for=lambda _:RecallScope.validated())
    try:
        source=await memory.remember('alpha','Juniper is in Oregon')
        await service.start('alpha','r',workflow_id='p',plan_revision=1,question='Where is Juniper?');await service.wait('alpha','r')
        prompts=await service.inputs('alpha','r');assert prompts and 'retained' in prompts[0].context
        await service.respond('alpha','r','h',response='Yes',expected_revision=1)
        await memory.forget('alpha',source.episode_id)
        with pytest.raises(WorkflowError,match='sources_invalid'):await service.inputs('alpha','r')
        with pytest.raises(WorkflowError,match='sources_invalid'):await service.respond('alpha','r','h',response='Yes',expected_revision=1)
        with pytest.raises(WorkflowError,match='sources_invalid'):await service.continue_run('alpha','r',continuation_id='c',responses={'h':2})
        assert len(calls)==1 and service._inputs.get('alpha','r','h').activation_id is None
    finally:await service.aclose();plans.close();await memory.close()

async def test_pending_continuation_reserves_admission_capacity(setup,monkeypatch):
    service,plans,catalog,memory,calls,_,release,path=setup
    save_plan(plans,catalog);release.set()
    for run_id in ('one','two'):
        await service.start('alpha',run_id,workflow_id='interactive',plan_revision=1,question='Q');await service.wait('alpha',run_id)
        await service.respond('alpha',run_id,'choose',response='North',expected_revision=1)
    entered=asyncio.Event();finish=asyncio.Event();seen=[]
    original=InteractiveAgentWorkflow.inspect_inputs
    async def held(self,run_id,*args,**kwargs):
        seen.append(run_id);entered.set();await finish.wait();return await original(self,run_id,*args,**kwargs)
    monkeypatch.setattr(InteractiveAgentWorkflow,'inspect_inputs',held)
    first=asyncio.create_task(service.continue_run('alpha','one',continuation_id='c',responses={'choose':2}))
    await entered.wait()
    second=asyncio.create_task(service.continue_run('alpha','two',continuation_id='c',responses={'choose':2}))
    try:
        for _ in range(10):await asyncio.sleep(0)
        assert seen==['one'], f'capacity=1 allowed simultaneous verification for {seen}'
        assert second.done()
        with pytest.raises(WorkflowError,match='run_busy'):await second
    finally:
        first.cancel();second.cancel();finish.set();await asyncio.gather(first,second,return_exceptions=True)


async def test_reopened_continuation_respects_current_host_parallel_ceiling(setup):
    from scone_memory.agents.interactive_plan import InteractiveAgentPlan,HumanInputTask
    from scone_memory.agents.task_workflow import AgentTask
    service,plans,catalog,memory,calls,entered,release,path=setup
    await service.aclose()
    plans.save('alpha',InteractiveAgentPlan(kind='interactive',workflow_id='wide',tasks=(
        HumanInputTask(kind='input',task_id='h',prompt='Choose'),
        AgentTask(task_id='a',agent_id='research',prompt='First',depends_on=('h',)),
        AgentTask(task_id='b',agent_id='research',prompt='Second',depends_on=('h',)),)),catalog=catalog,expected_revision=0)
    wide=AgentRunService(path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,scope_for=lambda _:RecallScope.validated(),max_parallel_tasks=2)
    try:
        await wide.start('alpha','one',workflow_id='wide',plan_revision=1,question='Q',max_parallel=2)
        await wide.wait('alpha','one');await wide.respond('alpha','one','h',response='Yes',expected_revision=1)
    finally:await wide.aclose()
    narrow=AgentRunService(path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,scope_for=lambda _:RecallScope.validated(),max_parallel_tasks=1)
    try:
        try:await narrow.continue_run('alpha','one',continuation_id='c',responses={'h':2})
        except (WorkflowError,ValueError):
            assert narrow._inputs.get('alpha','one','h').activation_id is None
            assert calls == []
            return
        await entered.wait()
        for _ in range(20):await asyncio.sleep(0)
        status=await narrow.status('alpha','one')
        assert len(calls)<=1, f'policy width1 ran {len(calls)} calls; inflight={status.inflight_steps}'
    finally:release.set();await narrow.aclose()


async def test_local_cancel_stops_pending_continuation(setup,monkeypatch):
    service,plans,catalog,memory,calls,_,release,path=setup
    save_plan(plans,catalog);release.set()
    await service.start('alpha','one',workflow_id='interactive',plan_revision=1,question='Q');await service.wait('alpha','one')
    await service.respond('alpha','one','choose',response='North',expected_revision=1)
    entered=asyncio.Event()
    async def held(self,*a,**kw):entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(InteractiveAgentWorkflow,'inspect_inputs',held)
    pending=asyncio.create_task(service.continue_run('alpha','one',continuation_id='c',responses={'choose':2}))
    await entered.wait()
    try:
        status=await service.cancel('alpha','one')
        assert status.status=='cancelled'
        assert pending.done() and service._inputs.get('alpha','one','choose').activation_id is None
        assert len(calls)==1
    finally:pending.cancel();await asyncio.gather(pending,return_exceptions=True)


@pytest.mark.parametrize('operation', ['read_result', 'run', 'inspect_inputs'])
@pytest.mark.parametrize('error_code', ['run_store_unavailable', 'run_store_closed'])
async def test_inbox_read_outage_preserves_completed_receipts(setup,monkeypatch,operation,error_code):
    service,plans,catalog,memory,calls,_,release,path=setup
    save_plan(plans,catalog);release.set()
    await service.start('alpha','one',workflow_id='interactive',plan_revision=1,question='Q');await service.wait('alpha','one')
    await service.respond('alpha','one','choose',response='North',expected_revision=1)
    await service.continue_run('alpha','one',continuation_id='c',responses={'choose':2});await service.wait('alpha','one')
    request=service._runs.get('alpha','one');workflow=service._open(request)
    before=workflow.status('one','Q')
    original=service._inputs.get
    def unavailable(*args,**kwargs):raise WorkflowError(error_code)
    monkeypatch.setattr(service._inputs,'get',unavailable)
    try:
        with pytest.raises(WorkflowError,match='verification_unavailable'):
            await getattr(workflow,operation)('one','Q')
        after=workflow.status('one','Q')
        assert after.completed_steps==before.completed_steps
        assert after.attempts==before.attempts
        monkeypatch.setattr(service._inputs,'get',original)
        assert (await workflow.read_result('one','Q')).status=='completed'
        assert len(calls)==2
    finally:workflow.close()
