"""Authenticated run admission, progress, cancellation and verified output."""
import asyncio
import httpx
import pytest
from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
from scone_memory.agents.catalog import AgentCatalog,AgentDefinition,AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask,AgentTaskPlan
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope


def auth(key='writer'):return {'Authorization':'Bearer '+key}
def body(run_id='one'):return {'run_id':run_id,'workflow_id':'report','plan_revision':1,'question':'Question'}


@pytest.fixture
async def setup(tmp_path, request):
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    release=asyncio.Event();entered=asyncio.Event();calls=[]
    class Model:
        async def complete(self,messages,tools):
            entered.set();calls.append(messages);await release.wait();return ToolStep(content='Selected model answer')
    catalog=AgentCatalog(models=[AgentModel('local','Local','1',Model)],agents=[AgentDefinition(
        agent_id='research',instructions='Use evidence.',models=('local',),default_model='local',initial_search=False)])
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    plans.save('alpha',AgentTaskPlan(workflow_id='report',tasks=(AgentTask(task_id='find',agent_id='research',prompt='Find evidence.'),)),catalog=catalog,expected_revision=0)
    service=AgentRunService(tmp_path/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
        scope_for=lambda space:RecallScope.validated(),max_active=1,max_parallel_tasks=getattr(request,'param',1))
    app=create_app(memory,{'writer':'alpha','reader':'alpha','other':'bravo'},roles={'writer':'write','reader':'read'},
        agent_catalog=catalog,agent_plan_store=plans,agent_run_service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://scone.test') as client:
        yield client,app,service,memory,release,entered,calls
    release.set();await service.aclose();plans.close();await memory.close()


async def test_run_admission_reads_and_result_follow_roles_and_space(setup):
    client,_,service,_,release,entered,calls=setup
    caps=(await client.get('/v1/capabilities',headers=auth())).json()['features'];assert caps['agents.runs']
    assert (await client.post('/v1/agent-runs',json=body())).status_code==401
    assert (await client.post('/v1/agent-runs',json=body(),headers=auth('reader'))).status_code==403
    started=await client.post('/v1/agent-runs',json=body(),headers=auth());assert started.status_code==202,started.text
    await entered.wait()
    assert (await client.get('/v1/agent-runs/one',headers=auth('other'))).status_code==404
    assert (await client.get('/v1/agent-runs/one/result',headers=auth('reader'))).status_code==409
    overloaded=await client.post('/v1/agent-runs',json=body('two'),headers=auth());assert overloaded.status_code==429
    release.set();await service.wait('alpha','one')
    status=await client.get('/v1/agent-runs/one',headers=auth('reader'));assert status.json()['status']=='completed'
    result=await client.get('/v1/agent-runs/one/result',headers=auth('reader'))
    assert result.status_code==200 and result.json()['results']['find']['model_id']=='local'
    assert result.headers['cache-control']=='no-store' and len(calls)==1
    assert (await client.get('/v1/agent-runs',headers=auth('other'))).json()['items']==[]


async def test_cancellation_and_unknown_outcome_are_explicit(setup):
    client,_,_,_,_,entered,calls=setup
    await client.post('/v1/agent-runs',json=body(),headers=auth());await entered.wait()
    assert (await client.post('/v1/agent-runs/one/cancel',headers=auth('reader'))).status_code==403
    cancelled=await client.post('/v1/agent-runs/one/cancel',headers=auth())
    assert cancelled.status_code==200 and cancelled.json()['outcome_unknown']
    again=await client.post('/v1/agent-runs',json=body(),headers=auth())
    assert again.status_code==409 and again.json()['code']=='outcome_unknown' and len(calls)==1


async def test_streamed_start_rechecks_authorization_before_admission(setup):
    import json
    client,app,service,_,_,_,calls=setup
    encoded=json.dumps(body()).encode()
    async def stream():
        yield encoded[:10];app.state.keys['writer']='bravo';yield encoded[10:]
    result=await client.post('/v1/agent-runs',content=stream(),headers=auth())
    assert result.status_code in (401,403)
    assert await service.request('alpha','one') is None
    assert await service.request('bravo','one') is None and calls==[]


@pytest.mark.parametrize('change',('rebind','downgrade','revoke'))
async def test_last_admission_guard_catches_key_changes_during_service_check(setup,monkeypatch,change):
    client,app,service,memory,_,_,calls=setup
    original=memory.space_deleted;checks=0
    async def changed(space):
        nonlocal checks
        checks+=1
        result=await original(space)
        if checks==3:
            if change=='rebind':app.state.keys['writer']='bravo'
            elif change=='downgrade':app.state.roles['writer']='read'
            else:app.state.keys.pop('writer')
        return result
    monkeypatch.setattr(memory,'space_deleted',changed)
    response=await client.post('/v1/agent-runs',json=body(),headers=auth())
    assert response.status_code in (401,403),response.text
    assert await service.request('alpha','one') is None and calls==[]


async def test_result_is_not_returned_after_key_scope_changes(setup,monkeypatch):
    client,app,service,_,release,_,_=setup
    release.set();await client.post('/v1/agent-runs',json=body(),headers=auth());await service.wait('alpha','one')
    original=service.result
    async def changed(space,run_id):
        answer=await original(space,run_id);app.state.keys['reader']='bravo';return answer
    monkeypatch.setattr(service,'result',changed)
    response=await client.get('/v1/agent-runs/one/result',headers=auth('reader'))
    assert response.status_code==401 and 'Selected model answer' not in response.text


async def test_run_requests_are_bounded_and_cancel_rejects_ignored_options(setup):
    client,_,service,_,_,entered,calls=setup
    oversized=await client.post('/v1/agent-runs',content=b'x'*8193,headers=auth());assert oversized.status_code==413
    unknown=await client.post('/v1/agent-runs',json={**body(),'provider_url':'PRIVATE'},headers=auth())
    assert unknown.status_code==422 and 'PRIVATE' not in unknown.text
    await client.post('/v1/agent-runs',json=body(),headers=auth());await entered.wait()
    response=await client.post('/v1/agent-runs/one/cancel',json={'force':True},headers=auth())
    assert response.status_code==422
    assert (await service.status('alpha','one')).active_local and len(calls)==1


async def test_lifespan_closes_owned_tasks_but_leaves_caller_memory_open(setup):
    client,app,service,memory,_,entered,_=setup
    async with app.router.lifespan_context(app):
        await client.post('/v1/agent-runs',json=body(),headers=auth());await entered.wait()
    assert service._closed and not service._tasks and not service._owners
    assert await memory.space_deleted('alpha') is None


async def test_result_withholds_evidence_when_host_scope_narrows(tmp_path, monkeypatch):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    scope = RecallScope.validated()
    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(messages)
            return ToolStep(content='Juniper private address is Oregon.')

    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', Model)], agents=[
        AgentDefinition(agent_id='research', instructions='Use evidence.', models=('local',), default_model='local')])
    plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
    plans.save('alpha', AgentTaskPlan(workflow_id='report', tasks=(
        AgentTask(task_id='find', agent_id='research', prompt='Locate Juniper.'),)),
        catalog=catalog, expected_revision=0)
    service = AgentRunService(tmp_path / 'runs', key=b'k' * 32, catalog=catalog,
        plans=plans, memory=memory, scope_for=lambda _: scope)
    app = create_app(memory, {'writer': 'alpha'}, agent_catalog=catalog,
        agent_plan_store=plans, agent_run_service=service)
    try:
        await memory.remember('alpha', 'Juniper private address is Oregon.', metadata={'team': 'private'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
            response = await client.post('/v1/agent-runs', json={**body(), 'question': 'Where is Juniper?'}, headers=auth())
            assert response.status_code == 202
            await service.wait('alpha', 'one')
            original = memory.documents.get_chunks
            changed = False

            async def restrict(*args):
                nonlocal scope, changed
                changed = True
                scope = RecallScope.validated(where={'team': 'public'})
                return await original(*args)

            monkeypatch.setattr(memory.documents, 'get_chunks', restrict)
            response = await client.get('/v1/agent-runs/one/result', headers=auth())
            assert changed
            assert response.status_code == 409
            assert response.json()['code'] == 'run_scope_changed'
            assert 'Juniper private address' not in response.text
            assert len(calls) == 1
    finally:
        await service.aclose()
        plans.close()
        await memory.close()


@pytest.mark.parametrize('setup',[2],indirect=True)
async def test_parallel_selection_is_bounded_and_bound_to_run(setup):
    client,_,service,_,release,_,calls=setup
    rejected=await client.post('/v1/agent-runs',json={**body(),'max_parallel':3},headers=auth())
    assert rejected.status_code==422
    assert await service.request('alpha','one') is None and not calls
    policy=await client.get('/v1/agents/run-policy',headers=auth('reader'))
    assert policy.status_code==200 and policy.json()=={'space':'alpha','max_parallel_tasks':2,'max_active_runs':1}
    caps=(await client.get('/v1/capabilities',headers=auth())).json()['features'];assert caps['agents.parallel']
    response=await client.post('/v1/agent-runs',json={**body(),'max_parallel':2},headers=auth())
    assert response.status_code==202 and response.json()['max_parallel']==2
    release.set();await service.wait('alpha','one')
    snapshot=(await client.get('/v1/agent-runs/one/request',headers=auth())).json()
    assert snapshot['max_parallel']==2 and (await service.request('alpha','one')).max_parallel==2
    changed=await client.post('/v1/agent-runs',json=body(),headers=auth())
    assert changed.status_code==409 and len(calls)==1
    assert (await client.get('/v1/agent-runs/one/result',headers=auth())).status_code==200


async def test_default_run_policy_is_sequential(setup):
    client,_,service,_,_,_,calls=setup
    policy=(await client.get('/v1/agents/run-policy',headers=auth())).json()
    assert policy['max_parallel_tasks']==1
    assert not (await client.get('/v1/capabilities',headers=auth())).json()['features']['agents.parallel']
    rejected=await client.post('/v1/agent-runs',json={**body(),'max_parallel':2},headers=auth())
    assert rejected.status_code==422 and not calls
    assert await service.request('alpha','one') is None
