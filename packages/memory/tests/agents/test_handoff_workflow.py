"""Explicit model choices and allowed dynamic agent edges, with durable hops."""
import asyncio
import json
import pytest
from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
from scone_memory.agents.catalog import AgentCatalog,AgentModel,AgentDefinition
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.handoff_workflow import AgentHandoffPlan,HandoffAgent,AgentHandoffWorkflow
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope


@pytest.fixture
async def memory():
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    yield engine
    await engine.close()


def catalog(calls,replies,*,initial_search=False,revision='1'):
    class Model:
        def __init__(self,name):self.name=name
        async def complete(self,messages,tools):
            calls.append((self.name,messages))
            reply=replies[self.name]
            if isinstance(reply,asyncio.Event):await reply.wait();reply={'answer':'late','handoff_to':None}
            return ToolStep(content=reply if isinstance(reply,str) else json.dumps(reply))
    return AgentCatalog(models=[AgentModel(name,name,revision,lambda name=name:Model(name)) for name in replies],agents=[
        AgentDefinition(agent_id='research',instructions='Find evidence.',models=tuple(replies),default_model=next(iter(replies)),initial_search=initial_search),
        AgentDefinition(agent_id='write',instructions='Write an answer.',models=tuple(replies),default_model=next(iter(replies)),initial_search=initial_search)])


def plan(max_handoffs=3):
    return AgentHandoffPlan(workflow_id='report',root_agent='research',max_handoffs=max_handoffs,agents=(
        HandoffAgent(agent_id='research',model_id='careful',can_handoff_to=('write',)),
        HandoffAgent(agent_id='write',model_id='fast',can_handoff_to=('research',))))


def workflow(path,memory,agents,policy=None):
    return AgentHandoffWorkflow(path,key=b'k'*32,catalog=agents,plan=policy or plan(),memory=memory,space='alpha',scope=RecallScope.validated())


async def test_allowed_handoff_uses_fixed_models_and_stops_without_unused_hops(tmp_path,memory):
    calls=[];agents=catalog(calls,{'careful':{'answer':'Research notes','handoff_to':'write'},'fast':{'answer':'Final answer','handoff_to':None}})
    work=workflow(tmp_path/'run',memory,agents)
    try:
        result=await work.run('r','Write a report')
        assert result.status=='completed' and result.final.text=='Final answer'
        assert [hop.output.model_id for hop in result.hops]==['careful','fast']
        assert len(work.progress('r','Write a report').attempts)==2
        assert 'Research notes' in str(calls[1][1])
        assert 'untrusted data' in str(calls[1][1])
        work.close();work=workflow(tmp_path/'run',memory,agents)
        assert (await work.read_result('r','Write a report')).final.text=='Final answer'
        assert len((await work.run('r','Write a report')).reused_hops)==2 and len(calls)==2
    finally:work.close()


@pytest.mark.parametrize('reply',[
    {'answer':'notes','handoff_to':'unregistered'},
    {'answer':'notes','handoff_to':'research'},
    {'answer':'notes','handoff_to':'write','model_id':'other'},
    {'answer':'   ','handoff_to':None},
    '{"answer":"notes","handoff_to":"write","handoff_to":null}',
    'not JSON',
])
async def test_disallowed_or_malformed_decisions_never_invoke_another_agent(tmp_path,memory,reply):
    calls=[];agents=catalog(calls,{'careful':reply,'fast':{'answer':'wrong','handoff_to':None}})
    work=workflow(tmp_path/'run',memory,agents)
    try:
        with pytest.raises(WorkflowError,match='step_failed'):await work.run('r','Question')
        assert len(calls)==1
        with pytest.raises(WorkflowError,match='outcome_unknown'):await work.run('r','Question')
        assert len(calls)==1
    finally:work.close()


async def test_allowed_cycle_stops_at_explicit_handoff_limit(tmp_path,memory):
    calls=[];agents=catalog(calls,{'careful':{'answer':'Research more','handoff_to':'write'},'fast':{'answer':'Need more','handoff_to':'research'}})
    work=workflow(tmp_path/'run',memory,agents,plan(max_handoffs=1))
    try:
        result=await work.run('r','Question')
        assert result.status=='handoff_limit' and result.final is None and len(result.hops)==2
        assert (await work.read_result('r','Question')).status=='handoff_limit' and len(calls)==2
    finally:work.close()


async def test_first_agent_can_finish_without_delegation(tmp_path,memory):
    calls=[];agents=catalog(calls,{'careful':{'answer':'Done','handoff_to':None},'fast':{'answer':'Unused','handoff_to':None}})
    work=workflow(tmp_path/'run',memory,agents)
    try:
        result=await work.run('r','Question')
        assert len(result.hops)==1 and len(calls)==1
        assert work.progress('r','Question').completed_steps==('hop-01',)
    finally:work.close()


@pytest.mark.parametrize('change',['model','revision','targets','budget','question','scope'])
async def test_saved_hops_require_same_plan_models_question_and_scope(tmp_path,memory,change):
    calls=[];replies={'careful':{'answer':'Done','handoff_to':None},'fast':{'answer':'Unused','handoff_to':None}}
    agents=catalog(calls,replies);work=workflow(tmp_path/'run',memory,agents)
    await work.run('r','Question');work.close()
    policy=plan()
    if change=='model':policy=policy.model_copy(update={'agents':(policy.agents[0].model_copy(update={'model_id':'fast'}),policy.agents[1])})
    if change=='revision':agents=catalog(calls,replies,revision='2')
    if change=='targets':policy=policy.model_copy(update={'agents':(policy.agents[0].model_copy(update={'can_handoff_to':()}),policy.agents[1])})
    if change=='budget':policy=plan(max_handoffs=2)
    scope=RecallScope.validated(where={'team':'blue'}) if change=='scope' else RecallScope.validated()
    work=AgentHandoffWorkflow(tmp_path/'run',key=b'k'*32,catalog=agents,plan=policy,memory=memory,space='alpha',scope=scope)
    try:
        with pytest.raises(WorkflowError,match='binding_mismatch'):
            await work.read_result('r','Different' if change=='question' else 'Question')
        assert len(calls)==1
    finally:work.close()


@pytest.mark.parametrize('change',['forget','exclude'])
async def test_handoff_sources_are_revalidated_on_read(tmp_path,memory,change):
    source=await memory.remember('alpha','Juniper is in Oregon')
    fact=await memory.assert_fact('alpha','Juniper','located_in','Oregon',source_episode_id=source.episode_id,quote='Juniper is in Oregon')
    calls=[];agents=catalog(calls,{'careful':{'answer':'Juniper is in Oregon','handoff_to':'write'},'fast':{'answer':'Oregon','handoff_to':None}},initial_search=True)
    work=workflow(tmp_path/'run',memory,agents)
    try:
        result=await work.run('r','Where is Juniper?')
        assert all(hop.output.source_status=='retained' for hop in result.hops)
        if change=='forget':await memory.forget('alpha',source.episode_id)
        else:await memory.exclude('alpha',fact.fact_id,'withdrawn')
        with pytest.raises(WorkflowError,match='sources_invalid'):await work.read_result('r','Where is Juniper?')
        assert len(calls)==2
    finally:work.close()


async def test_verification_outage_preserves_hops_for_read_without_model_replay(tmp_path,memory,monkeypatch):
    await memory.remember('alpha','Juniper is in Oregon')
    calls=[];agents=catalog(calls,{'careful':{'answer':'Oregon','handoff_to':None},'fast':{'answer':'Unused','handoff_to':None}},initial_search=True)
    work=workflow(tmp_path/'run',memory,agents)
    try:
        await work.run('r','Where is Juniper?')
        original=memory.documents.get_chunks
        async def unavailable(*args,**kwargs):raise ConnectionError('temporary storage outage')
        monkeypatch.setattr(memory.documents,'get_chunks',unavailable)
        with pytest.raises(WorkflowError,match='verification_unavailable'):await work.read_result('r','Where is Juniper?')
        assert work.progress('r','Where is Juniper?').completed_steps==('hop-01',)
        monkeypatch.setattr(memory.documents,'get_chunks',original)
        assert (await work.read_result('r','Where is Juniper?')).status=='completed'
        assert len(calls)==1
    finally:work.close()


async def test_cancelled_second_hop_is_never_replayed_after_reopen(tmp_path,memory):
    calls=[];waiting=asyncio.Event()
    agents=catalog(calls,{'careful':{'answer':'Research notes','handoff_to':'write'},'fast':waiting})
    work=workflow(tmp_path/'run',memory,agents)
    task=asyncio.create_task(work.run('r','Question'))
    try:
        async with asyncio.timeout(3):
            while len(calls)<2:await asyncio.sleep(.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert work.progress('r','Question').completed_steps==('hop-01',)
        work.close();work=workflow(tmp_path/'run',memory,agents)
        with pytest.raises(WorkflowError,match='outcome_unknown'):await work.run('r','Question')
        assert len(calls)==2
    finally:
        if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
        work.close()


async def test_handoff_context_budget_stops_before_next_model_and_receipts_are_encrypted(tmp_path,memory):
    calls=[];answer='private research '*2200
    agents=catalog(calls,{'careful':{'answer':answer,'handoff_to':'write'},'fast':{'answer':'Unused','handoff_to':None}})
    work=workflow(tmp_path/'run',memory,agents)
    try:
        with pytest.raises(WorkflowError,match='step_failed'):await work.run('r','Question')
        assert len(calls)==1
        assert all(b'private research' not in path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    finally:work.close()


@pytest.mark.parametrize('change',['unknown_model','unknown_root','unknown_target','duplicate_agent','negative_budget','bool_budget'])
async def test_invalid_plan_rejected_before_journal_creation(tmp_path,memory,change):
    calls=[];agents=catalog(calls,{'careful':{'answer':'Done','handoff_to':None},'fast':{'answer':'Done','handoff_to':None}})
    policy=plan()
    updates={
        'unknown_model':{'agents':(policy.agents[0].model_copy(update={'model_id':'other'}),policy.agents[1])},
        'unknown_root':{'root_agent':'other'},
        'unknown_target':{'agents':(policy.agents[0].model_copy(update={'can_handoff_to':('other',)}),policy.agents[1])},
        'duplicate_agent':{'agents':(policy.agents[0],policy.agents[0])},
        'negative_budget':{'max_handoffs':-1},'bool_budget':{'max_handoffs':True},
    }
    with pytest.raises(ValueError):workflow(tmp_path/'run',memory,agents,policy.model_copy(update=updates[change]))
    assert not list(tmp_path.iterdir()) and not calls
