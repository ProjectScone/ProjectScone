"""Authenticated plan editing binds model choices without running inference."""
import httpx
import pytest

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.api.app import create_app


def forbidden_model():
    raise AssertionError('Plan routes must never invoke models')


def catalog(revision='1'):
    return AgentCatalog(models=[AgentModel('local','Local',revision,forbidden_model)],agents=[
        AgentDefinition(agent_id='research',instructions='Private system instruction',models=('local',),default_model='local')])


def payload(revision=0,model_id=None):
    return {'expected_revision':revision,'plan':{'workflow_id':'research-plan','tasks':[
        {'task_id':'find','agent_id':'research','prompt':'Find evidence','model_id':model_id,'depends_on':[]}]}}


def auth(key='writer'):
    return {'Authorization':'Bearer '+key}


@pytest.fixture
async def setup(tmp_path):
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    keys={'reader':'alpha','writer':'alpha','reviewer':'alpha','other':'bravo'}
    app=create_app(engine,keys,roles={'reader':'read','writer':'write','reviewer':'review'},
                   agent_catalog=catalog(),agent_plan_store=store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://scone.test') as client:
        yield client,engine,store
    store.close();await engine.close()


async def test_catalog_and_plans_follow_key_space_and_roles(setup):
    client,_,_=setup
    assert (await client.get('/v1/agents/catalog')).status_code==401
    choices=await client.get('/v1/agents/catalog',headers=auth('reader'))
    assert choices.status_code==200 and 'Private system instruction' not in choices.text
    caps=(await client.get('/v1/capabilities',headers=auth())).json()['features']
    assert caps['agents.catalog'] and caps['agents.plans'] and not caps.get('agents.runs',False)
    for role in ('reader','reviewer'):
        assert (await client.put('/v1/agent-plans/research-plan',json=payload(),headers=auth(role))).status_code==403
    saved=await client.put('/v1/agent-plans/research-plan',json=payload(),headers=auth())
    assert saved.status_code==200,saved.text
    assert saved.json()['plan']['tasks'][0]['model_id']=='local'
    assert saved.json()['revision']==1 and saved.json()['configuration_current']
    assert (await client.get('/v1/agent-plans/research-plan',headers=auth('other'))).status_code==404
    listed=await client.get('/v1/agent-plans',headers=auth('reader'))
    assert len(listed.json()['items'])==1
    assert (await client.get('/v1/agent-plans',headers=auth('other'))).json()['items']==[]
    stale=await client.put('/v1/agent-plans/research-plan',json=payload(),headers=auth())
    assert stale.status_code==409 and stale.json()['code']=='plan_revision_conflict'


async def test_bad_requests_never_save_or_leak_inputs(setup):
    client,_,store=setup
    bad=[payload(model_id='unknown'),{**payload(),'secret':'PRIVATE'},payload()]
    bad[-1]['plan']['workflow_id']='different'
    for body in bad:
        result=await client.put('/v1/agent-plans/research-plan',json=body,headers=auth())
        assert result.status_code==422,result.text
        assert 'PRIVATE' not in result.text
    cyclic=payload();cyclic['plan']['tasks'][0]['depends_on']=['find']
    assert (await client.put('/v1/agent-plans/research-plan',json=cyclic,headers=auth())).status_code==422
    oversized=await client.put('/v1/agent-plans/research-plan',content=b'x'*128001,headers=auth())
    assert oversized.status_code==413
    assert store.get('alpha','research-plan') is None


async def test_deleted_space_cannot_read_saved_plans(setup):
    client,engine,_=setup
    await client.put('/v1/agent-plans/research-plan',json=payload(),headers=auth())
    await engine.delete_space('alpha')
    assert (await client.get('/v1/agent-plans',headers=auth())).status_code==404


async def test_routes_are_absent_when_not_host_configured(tmp_path):
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine,{'k':'alpha'})),base_url='http://scone.test') as client:
        assert (await client.get('/v1/agents/catalog',headers=auth('k'))).status_code==404
        assert not (await client.get('/v1/capabilities',headers=auth('k'))).json()['features'].get('agents.plans',False)
    await engine.close()


@pytest.mark.parametrize('change',('delete','revoke','rebind','downgrade'))
async def test_streamed_body_cannot_outlive_its_authorization(tmp_path,change):
    import json
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    app=create_app(engine,{'writer':'alpha'},agent_catalog=catalog(),agent_plan_store=store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://scone.test') as client:
        encoded=json.dumps(payload()).encode()
        async def stream():
            yield encoded[:10]
            if change=='delete':await engine.delete_space('alpha')
            elif change=='revoke':app.state.keys.pop('writer')
            elif change=='rebind':app.state.keys['writer']='bravo'
            else:app.state.roles['writer']='read'
            yield encoded[10:]
        result=await client.put('/v1/agent-plans/research-plan',content=stream(),headers=auth())
        assert result.status_code in (401,403,404),result.text
        assert store.get('alpha','research-plan') is None
        assert store.get('bravo','research-plan') is None
    store.close();await engine.close()
