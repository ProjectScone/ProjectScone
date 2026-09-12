"""Agent model choices are explicit, immutable and never silently substituted."""
import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


class Model:
    def __init__(self, text, calls):
        self.text, self.calls = text, calls

    async def complete(self, messages, tools):
        self.calls.append(messages)
        return ToolStep(content=self.text)


def catalog(calls):
    return AgentCatalog(
        models=[AgentModel('small', 'Small local model', 'v1', lambda: Model('small answer', calls)),
                AgentModel('large', 'Large local model', 'v2', lambda: Model('large answer', calls))],
        agents=[AgentDefinition(agent_id='research', instructions='Use the available evidence.',
                                models=('small', 'large'), default_model='small', initial_search=False),
                AgentDefinition(agent_id='restricted', instructions='Use the small model.',
                                models=('small',), default_model='small', initial_search=False)],
    )


async def test_user_can_select_model_per_agent_and_result_records_exact_binding(engine):
    calls = []
    agents = catalog(calls)
    tools = ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated())
    selected = agents.bind('research', model_id='large')
    result = await selected.run('Answer my question.', tools=tools)
    assert result.output.text == 'large answer'
    assert result.agent_id == 'research' and result.model_id == 'large'
    assert result.binding == selected.fingerprint
    assert calls[0][0] == {'role': 'system', 'content': 'Use the available evidence.'}
    assert calls[0][-1] == {'role': 'user', 'content': 'Answer my question.'}
    assert agents.bind('research').model_id == 'small'
    assert [model.model_id for model in agents.choices('restricted')] == ['small']


@pytest.mark.parametrize('agent,model', [('missing', None), ('research', 'missing'), ('restricted', 'large')])
def test_unknown_or_disallowed_choices_fail_before_any_model_call(agent, model):
    calls = []
    with pytest.raises(ValueError):
        catalog(calls).bind(agent, model_id=model)
    assert calls == []


async def test_model_failure_does_not_use_another_allowed_model(engine):
    calls = []
    class Broken:
        async def complete(self, messages, tools):
            raise RuntimeError('model unavailable')
    agents = AgentCatalog(models=[AgentModel('broken', 'Broken', '1', Broken),
        AgentModel('other', 'Other', '1', lambda: Model('fallback', calls))],
        agents=[AgentDefinition(agent_id='agent', instructions='Answer.', models=('broken','other'), default_model='broken', initial_search=False)])
    with pytest.raises(RuntimeError):
        await agents.bind('agent').run('Question', tools=ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated()))
    assert calls == []


def test_binding_fingerprint_changes_for_model_revision_and_agent_policy():
    first = catalog([])
    assert first.bind('research').fingerprint != first.bind('research',model_id='large').fingerprint
    definition = AgentDefinition(agent_id='research', instructions='Different instructions.',models=('small',),default_model='small')
    revised = AgentCatalog(models=[AgentModel('small','Small local model','v2',lambda: Model('answer',[]))],agents=[definition])
    assert first.bind('research').fingerprint != revised.bind('research').fingerprint


def test_catalog_rejects_unknown_defaults_duplicate_ids_and_invalid_factories():
    with pytest.raises(ValueError):
        AgentDefinition(agent_id='a',instructions='x',models=('small',),default_model='large')
    definition=AgentDefinition(agent_id='a',instructions='x',models=('small',),default_model='small')
    model=AgentModel('small','Small','1',lambda: Model('answer',[]))
    for models,agents in [([model,model],[definition]),([model],[definition,definition]),([],[definition])]:
        with pytest.raises(ValueError):
            AgentCatalog(models=models,agents=agents)
    with pytest.raises(ValueError):
        AgentModel('bad','Bad','1',None)


async def test_each_run_owns_a_fresh_model_and_copies_input_policy(engine):
    created=[]
    def factory():
        model=Model('answer',[]);created.append(model);return model
    definition=AgentDefinition(agent_id='a',instructions='Original policy.',models=('small',),default_model='small',initial_search=False)
    agents=AgentCatalog(models=[AgentModel('small','Small','1',factory)],agents=[definition])
    selected=agents.bind('a')
    tools=ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated())
    await selected.run('One',tools=tools);await selected.run('Two',tools=tools)
    assert len(created)==2 and created[0] is not created[1]
    assert created[0].calls[0][-1]['content']=='One'
    assert created[1].calls[0][-1]['content']=='Two'


async def test_selected_model_binding_invalidates_saved_workflow_output(tmp_path, engine):
    from scone_memory.agents.workflow import WorkflowError, WorkflowRunner, WorkflowStep
    calls=[]
    agents=catalog(calls)
    tools=ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated())
    path=tmp_path/'agents.db'
    async def verify(context):
        return context.space=='alpha' and context.scope=={'task':'test'}
    def runner(selected):
        async def execute(context):
            result=await selected.run(context.inputs,tools=tools)
            return {'agent_id':result.agent_id,'model_id':result.model_id,'binding':result.binding,'text':result.output.text}
        return WorkflowRunner(path,key=b'k'*32,steps=[WorkflowStep('answer',selected.fingerprint,execute)],source_verifier=verify)
    first=runner(agents.bind('research'))
    result=await first.run('run-1',space='alpha',scope={'task':'test'},inputs='Question')
    first.close()
    reopened=runner(agents.bind('research'))
    saved=await reopened.run('run-1',space='alpha',scope={'task':'test'},inputs='Question')
    assert saved.results==result.results and saved.reused_steps==('answer',) and len(calls)==1
    reopened.close()
    changed=runner(agents.bind('research',model_id='large'))
    with pytest.raises(WorkflowError,match='binding_mismatch'):
        await changed.run('run-1',space='alpha',scope={'task':'test'},inputs='Question')
    assert len(calls)==1
    changed.close()


async def test_selected_agent_uses_actual_scoped_evidence_and_rechecks_it(engine):
    await engine.remember('alpha','Juniper is in Oregon',metadata={'team':'blue'})
    await engine.remember('alpha','Juniper private red secret',metadata={'team':'red'})
    await engine.remember('bravo','Juniper foreign secret',metadata={'team':'blue'})
    calls=[]
    agents=AgentCatalog(models=[AgentModel('chosen','Chosen','1',lambda: Model('From the retained source.',calls))],
        agents=[AgentDefinition(agent_id='a',instructions='Read the evidence.',models=('chosen',),default_model='chosen')])
    result=await agents.bind('a').run('Where is Juniper?',tools=ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated(where={'team':'blue'})))
    import json
    serialized=json.dumps(calls)
    assert 'Juniper is in Oregon' in serialized and 'red secret' not in serialized and 'foreign secret' not in serialized
    assert result.output.source_status=='retained' and await result.output.validate()


def test_copying_bound_agent_revalidates_policy_and_recomputes_identity():
    from dataclasses import replace
    selected=catalog([]).bind('restricted')
    changed=replace(selected,definition=selected.definition.model_copy(update={'instructions':'Changed policy.'}))
    assert changed.fingerprint!=selected.fingerprint
    with pytest.raises(ValueError):
        replace(selected,model=AgentModel('large','Large','1',lambda: Model('answer',[])))
    with pytest.raises(ValueError):
        replace(selected,definition=selected.definition.model_copy(update={'instructions':''}))


async def test_selected_registration_reaches_native_adapter_request(engine):
    import json
    import httpx
    from scone_memory.providers.tool_chat import SelfHostedToolChat
    seen=[]
    def respond(request):
        body=json.loads(request.content)
        seen.append(body)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'Selected reply'}}]})
    transport=httpx.MockTransport(respond)
    agents=AgentCatalog(models=[AgentModel(name,name,'configured-1',lambda name=name: SelfHostedToolChat(
        'http://127.0.0.1:8000/v1',name,transport=transport)) for name in ('model-a','model-b')],
        agents=[AgentDefinition(agent_id='research',instructions='Use exact source evidence.',models=('model-a','model-b'),default_model='model-a',initial_search=False)])
    result=await agents.bind('research',model_id='model-b').run('Question',tools=ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated()))
    assert result.model_id=='model-b' and len(seen)==1 and seen[0]['model']=='model-b'
    assert seen[0]['messages'][0]['content']=='Use exact source evidence.'
