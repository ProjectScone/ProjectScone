"""Handoff plans use the same durable, bounded host execution lifecycle."""
import asyncio
import json
import pytest
from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
from scone_memory.agents.catalog import AgentCatalog,AgentDefinition,AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.handoff_workflow import AgentHandoffPlan,HandoffAgent
from scone_memory.agents.plan_store import AgentPlanStore,PlanConflict,PlanConfigurationChanged
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTaskPlan,AgentTask
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope

@pytest.fixture
async def setup(tmp_path):
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    calls=[];entered=asyncio.Event();release=asyncio.Event();replies={'careful':{'answer':'Research','handoff_to':'write'},'fast':{'answer':'Final','handoff_to':None}}
    class Model:
        def __init__(self,name):self.name=name
        async def complete(self,messages,tools):
            calls.append(self.name)
            if self.name=='fast':entered.set();await release.wait()
            return ToolStep(content=json.dumps(replies[self.name]))
    catalog=AgentCatalog(models=[AgentModel(name,name,'1',lambda name=name:Model(name)) for name in replies],agents=[
        AgentDefinition(agent_id=name,instructions='Work.',models=tuple(replies),default_model=model,initial_search=False)
        for name,model in [('research','careful'),('write','fast')]])
    plan=AgentHandoffPlan(workflow_id='report',root_agent='research',agents=(
        HandoffAgent(agent_id='research',can_handoff_to=('write',)),HandoffAgent(agent_id='write')))
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    service=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
        scope_for=lambda space:RecallScope.validated(),max_parallel_tasks=2,max_active=1)
    yield service,plans,catalog,plan,memory,calls,entered,release,replies,tmp_path
    release.set();await service.aclose();plans.close();await memory.close()

async def test_saved_handoff_defaults_bind_explicit_models_and_mix_with_task_plans(setup):
    _,plans,catalog,plan,_,calls,*_=setup
    saved=plans.save('alpha',plan,catalog=catalog,expected_revision=0)
    assert [a.model_id for a in saved.plan.agents]==['careful','fast']
    assert set(saved.bindings)=={'research','write'}
    tasks=AgentTaskPlan(workflow_id='tasks',tasks=(AgentTask(task_id='one',agent_id='write',prompt='Act.'),))
    plans.save('alpha',tasks,catalog=catalog,expected_revision=0)
    assert {type(item.plan) for item in plans.list('alpha').items}=={AgentTaskPlan,AgentHandoffPlan}
    assert plans.get('bravo','report') is None and not calls
    with pytest.raises(PlanConflict):plans.save('alpha',plan,catalog=catalog,expected_revision=0)

async def test_handoff_runs_reopen_and_keep_original_plan_without_model_replay(setup):
    service,plans,catalog,plan,memory,calls,_,release,_,path=setup
    plans.save('alpha',plan,catalog=catalog,expected_revision=0);release.set()
    await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Original')
    await service.wait('alpha','r')
    result=await service.result('alpha','r')
    assert result.status=='completed' and result.final.text=='Final' and len(result.hops)==2
    plans.save('alpha',plan.model_copy(update={'max_handoffs':0}),catalog=catalog,expected_revision=1)
    assert (await service.request('alpha','r')).plan.plan.max_handoffs==3
    await service.aclose()
    reopened=AgentRunService(path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,scope_for=lambda space:RecallScope.validated())
    try:
        assert (await reopened.result('alpha','r')).final.text=='Final'
        assert (await reopened.start('alpha','r',workflow_id='report',plan_revision=1,question='Original')).status=='completed'
        assert calls==['careful','fast']
    finally:await reopened.aclose()

async def test_parallel_handoff_request_rejected_without_registration(setup):
    service,plans,catalog,plan,_,calls,*_=setup
    plans.save('alpha',plan,catalog=catalog,expected_revision=0)
    with pytest.raises(ValueError):await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Question',max_parallel=2)
    assert await service.request('alpha','r') is None and not calls

async def test_handoff_limit_has_no_final_answer_and_does_not_restart(setup):
    service,plans,catalog,plan,_,calls,*_=setup
    plans.save('alpha',plan.model_copy(update={'max_handoffs':0}),catalog=catalog,expected_revision=0)
    await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Question');await service.wait('alpha','r')
    result=await service.result('alpha','r')
    assert result.status=='handoff_limit' and result.final is None and len(result.hops)==1
    await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Question')
    assert calls==['careful']

async def test_cancel_handoff_owns_second_model_and_never_replays_it(setup):
    service,plans,catalog,plan,_,calls,entered,*_=setup
    plans.save('alpha',plan,catalog=catalog,expected_revision=0)
    await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Question');await entered.wait()
    assert (await service.cancel('alpha','r')).outcome_unknown
    with pytest.raises(WorkflowError,match='outcome_unknown'):await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Question')
    assert calls==['careful','fast']


async def test_handoff_result_refuses_changed_host_scope(setup):
    service,plans,catalog,plan,_,calls,_,release,*_=setup
    plans.save('alpha',plan,catalog=catalog,expected_revision=0);release.set()
    await service.start('alpha','r',workflow_id='report',plan_revision=1,question='Question');await service.wait('alpha','r')
    service._scope_for=lambda space:RecallScope.validated(where={'team':'blue'})
    with pytest.raises(WorkflowError,match='run_scope_changed'):await service.result('alpha','r')
    assert calls==['careful','fast']


async def test_saved_handoff_rejects_changed_model_configuration(setup):
    _,plans,catalog,plan,_,calls,*_=setup
    saved=plans.save('alpha',plan,catalog=catalog,expected_revision=0)
    definitions=[catalog.bind(name).definition for name in ('research','write')]
    models=[catalog.bind('research',model_id=name).model for name in ('careful','fast')]
    changed=AgentCatalog(models=[AgentModel(model.model_id,model.label,'changed',model.factory) for model in models],agents=definitions)
    with pytest.raises(PlanConfigurationChanged):saved.checked_plan(changed)
    assert not calls


async def test_http_handoff_save_start_read_and_permissions(setup):
    import httpx
    from scone_memory.api.app import create_app
    service,plans,catalog,plan,memory,calls,_,release,*_=setup
    app=create_app(memory,{'write':'alpha','read':'alpha','other':'bravo'},roles={'write':'write','read':'read'},
        agent_catalog=catalog,agent_plan_store=plans,agent_run_service=service)
    def auth(key='write'):return {'Authorization':'Bearer '+key}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://scone.test') as client:
        assert (await client.get('/v1/capabilities',headers=auth())).json()['features']['agents.handoffs']
        payload={'expected_revision':0,'plan':plan.model_dump(mode='json')}
        assert (await client.put('/v1/agent-plans/report',json=payload,headers=auth('read'))).status_code==403
        saved=await client.put('/v1/agent-plans/report',json=payload,headers=auth())
        assert saved.status_code==200 and saved.json()['plan']['agents'][0]['model_id']=='careful'
        body={'run_id':'r','workflow_id':'report','plan_revision':1,'question':'Original','max_parallel':2}
        assert (await client.post('/v1/agent-runs',json=body,headers=auth())).status_code==422
        assert await service.request('alpha','r') is None
        body['max_parallel']=1;release.set()
        assert (await client.post('/v1/agent-runs',json=body,headers=auth())).status_code==202
        await service.wait('alpha','r')
        result=await client.get('/v1/agent-runs/r/result',headers=auth('read'))
        assert result.status_code==200 and result.json()['final']['text']=='Final'
        assert [hop['output']['model_id'] for hop in result.json()['hops']]==['careful','fast']
        assert result.headers['cache-control']=='no-store'
        assert (await client.get('/v1/agent-runs/r/request',headers=auth('read'))).json()['plan']['plan']['root_agent']=='research'
        assert (await client.get('/v1/agent-runs/r/result',headers=auth('other'))).status_code==404
        assert calls==['careful','fast']
