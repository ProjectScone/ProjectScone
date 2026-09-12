"""Native agent task plans retain model choices and revalidate saved evidence."""
import asyncio
import json

import pytest

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope


@pytest.fixture
async def memory():
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    yield engine
    await engine.close()


class Script:
    def __init__(self,name,calls):self.name,self.calls=name,calls
    async def complete(self,messages,tools):
        self.calls.append((self.name,messages))
        return ToolStep(content=self.name+' result')


def catalog(calls,initial_search=False):
    return AgentCatalog(models=[AgentModel(name,name,'1',lambda name=name:Script(name,calls)) for name in ('small','large')],
        agents=[AgentDefinition(agent_id='worker',instructions='Use authorized evidence.',models=('small','large'),default_model='small',initial_search=initial_search)])


def plan(model='large'):
    return AgentTaskPlan(workflow_id='research',tasks=(
        AgentTask(task_id='research',agent_id='worker',model_id='small',prompt='Find supporting evidence.'),
        AgentTask(task_id='unrelated',agent_id='worker',model_id='small',prompt='Consider another angle.'),
        AgentTask(task_id='answer',agent_id='worker',model_id=model,prompt='Answer from the supplied evidence.',depends_on=('research',)),
    ))


def build(path,memory,calls,*,task_plan=None,initial_search=False,**options):
    return AgentWorkflow(path,key=b'k'*32,catalog=catalog(calls,initial_search),plan=task_plan or plan(),
                         memory=memory,space='alpha',scope=RecallScope.validated(where={'team':'blue'}),**options)


async def test_models_dependencies_saved_outputs_and_reopen(tmp_path,memory):
    calls=[];path=tmp_path/'tasks.db'
    workflow=build(path,memory,calls)
    result=await workflow.run('run-1','What is the decision?')
    workflow.close()
    assert [name for name,_ in calls]==['small','small','large']
    assert result.results['answer']['model_id']=='large'
    last=json.dumps(calls[-1][1]);assert 'research' in last and 'unrelated' not in last
    reopened=build(path,memory,calls)
    saved=await reopened.run('run-1','What is the decision?')
    assert saved.results==result.results and saved.reused_steps==('research','unrelated','answer')
    assert len(calls)==3 and reopened.status('run-1','What is the decision?').status=='completed'
    reopened.close()
    assert all(b'small result' not in file.read_bytes() for file in tmp_path.iterdir())


async def test_changed_selected_model_cannot_reuse_old_run(tmp_path,memory):
    calls=[];path=tmp_path/'tasks.db'
    first=build(path,memory,calls);await first.run('run-1','Question');first.close()
    changed=build(path,memory,calls,task_plan=plan('small'))
    with pytest.raises(WorkflowError,match='binding_mismatch'):await changed.run('run-1','Question')
    changed.close();assert len(calls)==3


@pytest.mark.parametrize('change',['forget','exclude'])
async def test_saved_source_evidence_is_rechecked_after_restart(tmp_path,memory,change):
    source=await memory.remember('alpha','Juniper is in Oregon',metadata={'team':'blue'})
    fact=await memory.assert_fact('alpha','Juniper','located_in','Oregon',source_episode_id=source.episode_id,quote='Juniper is in Oregon')
    calls=[];path=tmp_path/'tasks.db'
    first=build(path,memory,calls,initial_search=True);await first.run('r','Where is Juniper?');first.close()
    if change=='forget':await memory.forget('alpha',source.episode_id)
    else:await memory.exclude('alpha',fact.fact_id,'withdrawn')
    reopened=build(path,memory,calls,initial_search=True)
    with pytest.raises(WorkflowError,match='sources_invalid'):await reopened.run('r','Where is Juniper?')
    assert len(calls)==3
    reopened.close()


async def test_interrupted_model_call_is_not_automatically_replayed(tmp_path,memory):
    started=asyncio.Event();calls=[]
    class Waiting:
        async def complete(self,messages,tools):
            calls.append(1);started.set();await asyncio.Event().wait()
    agents=AgentCatalog(models=[AgentModel('wait','Wait','1',Waiting)],agents=[AgentDefinition(agent_id='a',instructions='Wait.',models=('wait',),default_model='wait',initial_search=False)])
    tasks=AgentTaskPlan(workflow_id='w',tasks=(AgentTask(task_id='one',agent_id='a',prompt='Answer.'),))
    def create():return AgentWorkflow(tmp_path/'tasks.db',key=b'k'*32,catalog=agents,plan=tasks,memory=memory,space='alpha',scope=RecallScope.validated())
    first=create();run=asyncio.create_task(first.run('r','Question'));await started.wait();run.cancel()
    with pytest.raises(asyncio.CancelledError):await run
    first.close();second=create()
    with pytest.raises(WorkflowError,match='outcome_unknown'):await second.run('r','Question')
    second.close();assert calls==[1]


@pytest.mark.parametrize('dependencies',[('missing',),('a',)])
def test_invalid_dependencies_are_rejected(dependencies):
    with pytest.raises(ValueError):AgentTaskPlan(workflow_id='w',tasks=(AgentTask(task_id='a',agent_id='worker',prompt='Act.',depends_on=dependencies),))


def test_cycles_rejected_and_forward_references_ordered():
    with pytest.raises(ValueError):AgentTaskPlan(workflow_id='w',tasks=(AgentTask(task_id='a',agent_id='worker',prompt='A',depends_on=('b',)),AgentTask(task_id='b',agent_id='worker',prompt='B',depends_on=('a',))))
    tasks=AgentTaskPlan(workflow_id='w',tasks=(AgentTask(task_id='b',agent_id='worker',prompt='B',depends_on=('a',)),AgentTask(task_id='a',agent_id='worker',prompt='A')))
    assert [task.task_id for task in tasks.ordered()]==['a','b']


async def test_unknown_model_fails_before_creating_a_journal(tmp_path,memory):
    path=tmp_path/'tasks.db'
    with pytest.raises(ValueError):build(path,memory,[],task_plan=plan('unregistered'))
    assert not path.exists()


async def test_saved_evidence_is_checked_before_each_downstream_model(tmp_path,memory):
    source=await memory.remember('alpha','Juniper is in Oregon',metadata={'team':'blue'})
    calls=[]
    class Withdraw:
        async def complete(self,messages,tools):
            calls.append('withdraw')
            await memory.forget('alpha',source.episode_id)
            return ToolStep(content='No direct evidence used by this unrelated branch.')
    agents=AgentCatalog(models=[AgentModel('read','Read','1',lambda:Script('read',calls)),AgentModel('withdraw','Withdraw','1',Withdraw)],
        agents=[AgentDefinition(agent_id='reader',instructions='Read.',models=('read',),default_model='read'),
                AgentDefinition(agent_id='withdrawer',instructions='Work.',models=('withdraw',),default_model='withdraw',initial_search=False)])
    tasks=AgentTaskPlan(workflow_id='w',tasks=(AgentTask(task_id='read',agent_id='reader',prompt='Find Juniper.'),
        AgentTask(task_id='withdraw',agent_id='withdrawer',prompt='Another task.'),
        AgentTask(task_id='answer',agent_id='reader',prompt='Answer.',depends_on=('read','withdraw'))))
    workflow=AgentWorkflow(tmp_path/'tasks.db',key=b'k'*32,catalog=agents,plan=tasks,memory=memory,space='alpha',scope=RecallScope.validated(where={'team':'blue'}))
    with pytest.raises(WorkflowError):await workflow.run('r','Where is Juniper?')
    assert len(calls)==2
    workflow.close()


async def test_changed_input_or_authorized_scope_refuses_saved_run(tmp_path,memory):
    calls=[];path=tmp_path/'tasks.db'
    first=build(path,memory,calls);await first.run('r','Question');first.close()
    same=build(path,memory,calls)
    with pytest.raises(WorkflowError,match='binding_mismatch'):await same.run('r','Other question')
    same.close()
    changed=AgentWorkflow(path,key=b'k'*32,catalog=catalog(calls),plan=plan(),memory=memory,space='bravo',scope=RecallScope.validated())
    with pytest.raises(WorkflowError,match='binding_mismatch'):await changed.run('r','Question')
    changed.close();assert len(calls)==3


async def test_oversized_handoff_fails_before_downstream_model(tmp_path,memory):
    calls=[]
    class Large:
        async def complete(self,messages,tools):calls.append(1);return ToolStep(content='x'*16000)
    agents=AgentCatalog(models=[AgentModel('large','Large','1',Large)],agents=[AgentDefinition(agent_id='a',instructions='Work.',models=('large',),default_model='large',initial_search=False)])
    tasks=AgentTaskPlan(workflow_id='w',tasks=(AgentTask(task_id='a',agent_id='a',prompt='First.'),AgentTask(task_id='b',agent_id='a',prompt='Second.'),AgentTask(task_id='c',agent_id='a',prompt='Combine.',depends_on=('a','b'))))
    workflow=AgentWorkflow(tmp_path/'tasks.db',key=b'k'*32,catalog=agents,plan=tasks,memory=memory,space='alpha',scope=RecallScope.validated())
    with pytest.raises(WorkflowError,match='step_failed'):await workflow.run('r','Question')
    workflow.close();assert len(calls)==2


async def test_storage_outage_during_saved_evidence_validation_preserves_receipts(tmp_path,memory,monkeypatch):
    await memory.remember('alpha','Juniper is in Oregon',metadata={'team':'blue'})
    calls=[];path=tmp_path/'tasks.db'
    first=build(path,memory,calls,initial_search=True);await first.run('r','Where is Juniper?');first.close()
    replay=build(path,memory,calls,initial_search=True)
    original=memory.documents.get_chunks;reads=0
    async def outage(*args):
        nonlocal reads
        reads+=1
        if reads==4:raise ConnectionError('temporary storage outage')
        return await original(*args)
    monkeypatch.setattr(memory.documents,'get_chunks',outage)
    with pytest.raises(WorkflowError,match='verification_unavailable'):await replay.run('r','Where is Juniper?')
    assert replay.status('r','Where is Juniper?').completed_steps==('research','unrelated','answer')
    monkeypatch.setattr(memory.documents,'get_chunks',original)
    result=await replay.run('r','Where is Juniper?')
    assert result.reused_steps==('research','unrelated','answer') and len(calls)==3
    replay.close()
