"""Standard local hosts can configure explicit agents without custom Python code."""
import asyncio
import json
from pathlib import Path
import pytest
import httpx
from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings
from scone_memory.runtime.agent_runtime import load_agent_runtime

def document(tmp_path):
    return {'schema_version':1,'state_dir':str(tmp_path/'agents'),'key_env':'SCONE_TEST_AGENT_KEY',
        'models':[{'model_id':'fast','label':'Fast local','revision':'1','base_url':'http://127.0.0.1:18999/v1','model':'local-model'}],
        'agents':[{'agent_id':'research','instructions':'Use evidence.','models':['fast'],'default_model':'fast','initial_search':False}]}

def write(tmp_path,value):
    path=tmp_path/'agents.json';path.write_text(json.dumps(value));path.chmod(0o600);return path

@pytest.fixture
async def memory():
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    yield engine
    await engine.close()

@pytest.mark.parametrize('composed',[False,True])
async def test_launcher_mounts_agent_catalog_without_contacting_model_and_closes_resources(tmp_path,memory,monkeypatch,composed):
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32)
    path=write(tmp_path,document(tmp_path));env={'SCONE_API_KEY':'key','SCONE_AGENTS_CONFIG':str(path)}
    if composed:env['SCONE_CONVERSATIONS_JOURNAL']=str(tmp_path/'sessions.db')
    app=build_app(Settings.from_env(env),memory)
    owner=app.state.agent_runtime
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://scone.test') as client:
        headers={'Authorization':'Bearer key'}
        features=(await client.get('/v1/capabilities',headers=headers)).json()['features']
        assert features['agents.handoffs'] and features['agents.runs']
        public=(await client.get('/v1/agents/catalog',headers=headers)).json()
        assert public['agents'][0]['default_model']=='fast'
        assert '18999' not in json.dumps(public) and 'SCONE_TEST_AGENT_KEY' not in json.dumps(public)
        assert (await client.get('/v1/agent-plans',headers=headers)).json()['items']==[]
    assert owner.service._closed
    with pytest.raises(Exception):owner.plans.list('default')

@pytest.mark.parametrize('change',['public','missing_key','short_key','duplicate_agent','unknown_model','file_mode'])
def test_invalid_configuration_creates_no_state(tmp_path,memory,monkeypatch,change):
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32);value=document(tmp_path)
    if change=='public':value['models'][0]['base_url']='https://example.com/v1'
    if change=='missing_key':monkeypatch.delenv('SCONE_TEST_AGENT_KEY')
    if change=='short_key':monkeypatch.setenv('SCONE_TEST_AGENT_KEY','ab')
    if change=='duplicate_agent':value['agents']*=2
    if change=='unknown_model':value['agents'][0]['models']=['unknown']
    path=write(tmp_path,value)
    if change=='file_mode':path.chmod(0o644)
    with pytest.raises(ValueError):load_agent_runtime(path,memory)
    assert not (tmp_path/'agents').exists()


def test_endpoint_model_and_protocol_changes_invalidate_catalog_bindings(tmp_path,memory,monkeypatch):
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32);value=document(tmp_path);path=write(tmp_path,value)
    owner=load_agent_runtime(path,memory);original=owner.catalog.bind('research').fingerprint;owner.close_idle()
    for field,changed in [('base_url','http://127.0.0.1:19999/v1'),('model','another-model'),('protocol','structured')]:
        updated=document(tmp_path);updated['models'][0][field]=changed;path=write(tmp_path,updated)
        owner=load_agent_runtime(path,memory)
        try:assert owner.catalog.bind('research').fingerprint!=original
        finally:owner.close_idle()


@pytest.fixture
async def local_model():
    calls=[];entered=asyncio.Event();handlers=set()
    async def handle(reader,writer):
        task=asyncio.current_task();handlers.add(task)
        try:
            headers=(await reader.readuntil(b'\r\n\r\n')).decode()
            length=int(next(line.split(':',1)[1] for line in headers.splitlines() if line.lower().startswith('content-length:')))
            body=json.loads(await reader.readexactly(length));calls.append(body)
            if 'Wait until shutdown' in str(body):
                entered.set();await reader.read();return
            decision={'answer':'Configured '+body['model'],'handoff_to':'write' if body['model']=='reasoner' else None}
            if body.get('response_format',{}).get('json_schema',{}).get('name')=='memory_action':
                decision={'action':'answer','answer':decision}
            payload=json.dumps({'choices':[{'message':{'role':'assistant','content':json.dumps(decision)},'finish_reason':'stop'}]}).encode()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload);await writer.drain()
        finally:
            writer.close();await writer.wait_closed();handlers.discard(task)
    server=await asyncio.start_server(handle,'127.0.0.1',0)
    try:yield f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1',calls,entered
    finally:
        server.close();await server.wait_closed()
        for task in tuple(handlers):task.cancel()
        await asyncio.gather(*handlers,return_exceptions=True)


@pytest.mark.parametrize('composed',[False,True])
@pytest.mark.parametrize('protocol',['native','structured'])
async def test_configured_models_run_over_local_http_and_survive_host_restart(tmp_path,memory,monkeypatch,local_model,composed,protocol):
    endpoint,calls,_=local_model;monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32)
    value=document(tmp_path);value['models'][0].update(base_url=endpoint,protocol=protocol)
    value['models'].append({**value['models'][0],'model_id':'careful','model':'reasoner'})
    value['agents']=[{**value['agents'][0],'models':['fast','careful'],'default_model':'careful'},
                     {**value['agents'][0],'agent_id':'write'}]
    env={'SCONE_API_KEY':'key','SCONE_AGENTS_CONFIG':str(write(tmp_path,value))}
    if composed:env['SCONE_CONVERSATIONS_JOURNAL']=str(tmp_path/'sessions.db')
    settings=Settings.from_env(env);app=build_app(settings,memory);headers={'Authorization':'Bearer key'}
    policy={'workflow_id':'report','root_agent':'research','max_handoffs':2,'agents':[
        {'agent_id':'research','model_id':'careful','can_handoff_to':['write']},
        {'agent_id':'write','model_id':'fast','can_handoff_to':[]}]}
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://scone.test',headers=headers) as client:
        assert not calls
        assert (await client.put('/v1/agent-plans/report',json={'expected_revision':0,'plan':policy})).status_code==200
        assert (await client.post('/v1/agent-runs',json={'run_id':'r','workflow_id':'report','plan_revision':1,'question':'Question'})).status_code==202
        await app.state.agent_runtime.service.wait('default','r')
        result=await client.get('/v1/agent-runs/r/result')
        assert result.status_code==200,result.text
        assert result.json()['final']['text']=='Configured local-model'
        assert [call['model'] for call in calls]==['reasoner','local-model']
    reopened=build_app(settings,memory)
    async with reopened.router.lifespan_context(reopened), httpx.AsyncClient(transport=httpx.ASGITransport(app=reopened),base_url='http://scone.test',headers=headers) as client:
        assert (await client.get('/v1/agent-runs/r/result')).json()['final']['text']=='Configured local-model'
        assert len(calls)==2


@pytest.mark.parametrize('composed',[False,True])
async def test_host_shutdown_cancels_agent_calls_before_closing_stores(tmp_path,memory,monkeypatch,local_model,composed):
    endpoint,calls,entered=local_model;monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32)
    value=document(tmp_path);value['models'][0]['base_url']=endpoint
    env={'SCONE_API_KEY':'key','SCONE_AGENTS_CONFIG':str(write(tmp_path,value))}
    if composed:env['SCONE_CONVERSATIONS_JOURNAL']=str(tmp_path/'sessions.db')
    app=build_app(Settings.from_env(env),memory);owner=app.state.agent_runtime
    async with app.router.lifespan_context(app):
        from scone_memory.agents.handoff_workflow import AgentHandoffPlan,HandoffAgent
        from scone_memory.agents.workflow import WorkflowError
        owner.plans.save('default',AgentHandoffPlan(workflow_id='w',root_agent='research',agents=(HandoffAgent(agent_id='research'),)),catalog=owner.catalog,expected_revision=0)
        await owner.service.start('default','r',workflow_id='w',plan_revision=1,question='Wait until shutdown')
        async with asyncio.timeout(3):await entered.wait()
        with pytest.raises(WorkflowError,match='run_busy'):owner.close_idle()
        assert owner.plans.get('default','w') is not None
    assert owner.service._closed and not owner.service._tasks and not owner.service._owners and len(calls)==1


def test_failed_composed_host_construction_releases_agent_state(tmp_path,memory,monkeypatch):
    from scone_memory.runtime import agent_runtime
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32);created=[];original=agent_runtime.load_agent_runtime
    def load(*args):
        owner=original(*args);created.append(owner);return owner
    monkeypatch.setattr(agent_runtime,'load_agent_runtime',load)
    settings=Settings.from_env({'SCONE_API_KEY':'key','SCONE_AGENTS_CONFIG':str(write(tmp_path,document(tmp_path))),
        'SCONE_CONVERSATIONS_JOURNAL':str(tmp_path/'sessions.db'),'SCONE_CONVERSATIONS_PERSONAS':str(tmp_path/'missing.json')})
    with pytest.raises(ValueError,match='REGISTRY'):build_app(settings,memory)
    assert len(created)==1 and created[0].service._closed


@pytest.mark.parametrize('change',['duplicate_key','symlink','hardlink','directory','fifo','oversized','bool_version'])
def test_configuration_file_boundary(tmp_path,memory,monkeypatch,change):
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32);path=write(tmp_path,document(tmp_path))
    if change=='duplicate_key':path.write_text('{"schema_version":1,"schema_version":1}')
    if change=='symlink':other=tmp_path/'other';path.rename(other);path.symlink_to(other)
    if change=='hardlink':(tmp_path/'other').hardlink_to(path)
    if change=='directory':path.unlink();path.mkdir()
    if change=='fifo':
        import os
        path.unlink();os.mkfifo(path,0o600)
    if change=='oversized':path.write_bytes(b' '*1048577)
    if change=='bool_version':value=document(tmp_path);value['schema_version']=True;write(tmp_path,value)
    with pytest.raises(ValueError):load_agent_runtime(path,memory)
    assert not (tmp_path/'agents').exists()


async def test_missing_selected_service_token_never_falls_back_or_contacts_model(tmp_path,memory,monkeypatch,local_model):
    endpoint,calls,_=local_model;monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32);monkeypatch.delenv('SCONE_MISSING_LOCAL_TOKEN',raising=False)
    value=document(tmp_path);value['models'][0].update(base_url=endpoint,api_key_env='SCONE_MISSING_LOCAL_TOKEN')
    owner=load_agent_runtime(write(tmp_path,value),memory)
    try:
        from scone_memory.agents.handoff_workflow import AgentHandoffPlan,HandoffAgent
        owner.plans.save('default',AgentHandoffPlan(workflow_id='w',root_agent='research',agents=(HandoffAgent(agent_id='research'),)),catalog=owner.catalog,expected_revision=0)
        await owner.service.start('default','r',workflow_id='w',plan_revision=1,question='Question')
        result=await owner.service.wait('default','r')
        assert result.status=='failed' and result.outcome_unknown and not calls
    finally:await owner.aclose()


def test_launcher_failure_before_lifespan_closes_agent_runtime(tmp_path,memory,monkeypatch):
    from scone_memory.api import __main__ as launcher
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32)
    settings=Settings.from_env({'SCONE_API_KEY':'key','SCONE_AGENTS_CONFIG':str(write(tmp_path,document(tmp_path)))})
    captured=[];original=launcher.build_app
    async def engine(settings):return memory
    def app(settings,engine):
        result=original(settings,engine);captured.append(result.state.agent_runtime);return result
    def failed_server(*args):raise RuntimeError('server construction failed')
    monkeypatch.setattr(launcher,'build_engine',engine);monkeypatch.setattr(launcher,'build_app',app);monkeypatch.setattr(launcher,'build_server',failed_server)
    with pytest.raises(RuntimeError,match='server construction failed'):launcher.main(settings)
    assert captured[0].service._closed
    with pytest.raises(Exception):captured[0].plans.list('default')


def test_wrong_persisted_key_is_a_sanitized_launcher_refusal(tmp_path,memory,monkeypatch,capsys):
    from scone_memory.api import __main__ as launcher
    monkeypatch.setenv('SCONE_TEST_AGENT_KEY','6b'*32);path=write(tmp_path,document(tmp_path))
    load_agent_runtime(path,memory).close_idle();monkeypatch.setenv('SCONE_TEST_AGENT_KEY','78'*32)
    async def engine(settings):return memory
    monkeypatch.setattr(launcher,'build_engine',engine)
    settings=Settings.from_env({'SCONE_API_KEY':'key','SCONE_AGENTS_CONFIG':str(path)})
    with pytest.raises(SystemExit) as refused:launcher.main(settings)
    assert refused.value.code==2
    error=capsys.readouterr().err
    assert 'refusing to serve:' in error and '787878' not in error and 'Traceback' not in error
