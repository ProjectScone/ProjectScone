"""Direct tool answers retain durable workflow identities and publication gates."""
import json
import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.handoff_workflow import AgentHandoffPlan, AgentHandoffWorkflow, HandoffAgent
from scone_memory.agents.input_store import AgentInputStore
from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.interactive_workflow import InteractiveAgentWorkflow
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_requirements import TaskAnswerRequirements
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow
from scone_memory.agents.usage import ModelTokenUsage
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import agents, call, tool
from .test_evidence_tool_loop import Script
from .test_task_workflow import memory

KEY=b'k'*32
USAGE={'calls':[{'prompt_tokens':7,'completion_tokens':2,'total_tokens':9}]}
REQUIREMENTS=TaskAnswerRequirements(format='json_object',output_schema={
    'type':'object','properties':{'count':{'type':'integer'}},'required':['count'],'additionalProperties':False})


def configured(value,*,direct=True):
    invoked=[]
    def execute(arguments,context):
        invoked.append((arguments['count'],context.space))
        return value
    model=Script(ToolStep(calls=call().calls,
        usage=ModelTokenUsage(prompt_tokens=7,completion_tokens=2,total_tokens=9)))
    return agents(model,[tool(execute,return_direct=direct)]),model,invoked


def task_workflow(path,memory,catalog,requirements=None):
    return AgentWorkflow(path,key=KEY,catalog=catalog,
        plan=AgentTaskPlan(workflow_id='direct-task',tasks=(AgentTask(task_id='answer',agent_id='worker',
            model_id='local',prompt='Count',answer_requirements=requirements),)),
        memory=memory,space='alpha',scope=RecallScope.validated())


async def test_direct_task_receipt_usage_and_reopen_without_another_call(tmp_path,memory):
    catalog,model,invoked=configured({'count':6})
    work=task_workflow(tmp_path/'journal',memory,catalog,REQUIREMENTS)
    try:
        answer=await work.run('one','Question')
        receipt=answer.results['answer']
        assert receipt['text']=='{"count":6}'
        assert receipt['model_calls']==receipt['tool_calls']==1
        assert receipt['usage']==USAGE and receipt['source_status']=='none' and receipt['evidence_ids']==[]
    finally:work.close()
    reopened=task_workflow(tmp_path/'journal',memory,catalog,REQUIREMENTS)
    try:
        assert (await reopened.read_result('one','Question')).results==answer.results
        assert (await reopened.run('one','Question')).reused_steps==('answer',)
        assert len(model.requests)==len(invoked)==1 and invoked==[(3,'alpha')]
    finally:reopened.close()


async def test_changing_direct_policy_refuses_saved_workflow_before_execution(tmp_path,memory):
    catalog,model,invoked=configured('Direct answer')
    work=task_workflow(tmp_path/'journal',memory,catalog)
    try:await work.run('one','Question')
    finally:work.close()
    changed,changed_model,changed_calls=configured('Direct answer',direct=False)
    reopened=task_workflow(tmp_path/'journal',memory,changed)
    try:
        with pytest.raises(WorkflowError,match='binding_mismatch'):await reopened.run('one','Question')
        assert not changed_calls and not changed_model.requests
        assert len(model.requests)==len(invoked)==1
    finally:reopened.close()


async def test_failed_direct_final_contract_is_unknown_and_never_replays_handler(tmp_path,memory):
    catalog,model,invoked=configured({'count':'wrong type'})
    work=task_workflow(tmp_path/'journal',memory,catalog,REQUIREMENTS)
    try:
        with pytest.raises(WorkflowError,match='step_failed'):await work.run('one','Question')
    finally:work.close()
    reopened=task_workflow(tmp_path/'journal',memory,catalog,REQUIREMENTS)
    try:
        with pytest.raises(WorkflowError,match='outcome_unknown'):await reopened.run('one','Question')
        with pytest.raises(WorkflowError,match='not_completed'):await reopened.read_result('one','Question')
        assert len(model.requests)==len(invoked)==1
    finally:reopened.close()


@pytest.mark.parametrize('contract',[False,True])
async def test_direct_handoff_preserves_existing_envelope_at_zero_handoff_budget(tmp_path,memory,contract):
    payload={'answer':{'count':6} if contract else 'Direct answer','handoff_to':None}
    catalog,model,invoked=configured(payload)
    plan=AgentHandoffPlan(workflow_id='direct-handoff',root_agent='worker',max_handoffs=0,
        agents=(HandoffAgent(agent_id='worker',model_id='local'),),answer_requirements=REQUIREMENTS if contract else None)
    def build():
        return AgentHandoffWorkflow(tmp_path/'journal',key=KEY,catalog=catalog,plan=plan,
            memory=memory,space='alpha',scope=RecallScope.validated())
    work=build()
    try:
        result=await work.run('one','Question')
        assert result.status=='completed' and len(result.hops)==1 and result.final is not None
        assert result.hops[0].handoff_to is None and result.final==result.hops[0].output
        assert result.final.text==('{"count":6}' if contract else 'Direct answer')
        assert result.final.model_calls==result.final.tool_calls==1
        assert result.final.usage.model_dump(mode='json')==USAGE
    finally:work.close()
    reopened=build()
    try:
        assert (await reopened.read_result('one','Question')).final==result.final
        assert (await reopened.run('one','Question')).final==result.final
        assert len(model.requests)==len(invoked)==1
    finally:reopened.close()


async def test_plain_direct_output_cannot_bypass_handoff_envelope(tmp_path,memory):
    catalog,model,invoked=configured('Not a handoff envelope')
    plan=AgentHandoffPlan(workflow_id='direct-handoff',root_agent='worker',max_handoffs=0,
        agents=(HandoffAgent(agent_id='worker',model_id='local'),))
    def build():
        return AgentHandoffWorkflow(tmp_path/'journal',key=KEY,catalog=catalog,plan=plan,
            memory=memory,space='alpha',scope=RecallScope.validated())
    work=build()
    try:
        with pytest.raises(WorkflowError,match='step_failed'):await work.run('one','Question')
    finally:work.close()
    reopened=build()
    try:
        with pytest.raises(WorkflowError,match='outcome_unknown'):await reopened.run('one','Question')
        assert len(model.requests)==len(invoked)==1
    finally:reopened.close()


@pytest.mark.parametrize('budget', [0, 1])
async def test_direct_turn_obeys_allowed_handoff_and_selected_next_model(tmp_path, memory, budget):
    invoked = []
    def forward(arguments, context):
        invoked.append(context.space)
        return {'answer': 'Count three items.', 'handoff_to': 'writer'}
    first = Script(call())
    second = Script(ToolStep(content=json.dumps({'answer': {'count': 6}, 'handoff_to': None})))
    catalog = AgentCatalog(models=[
        AgentModel('careful', 'Careful', '1', lambda: first),
        AgentModel('fast', 'Fast', '1', lambda: second),
    ], agents=[
        AgentDefinition(agent_id='worker', instructions='Prepare notes.', models=('careful', 'fast'),
                        default_model='fast', initial_search=False, tools=('double_count',)),
        AgentDefinition(agent_id='writer', instructions='Write the result.', models=('careful', 'fast'),
                        default_model='careful', initial_search=False),
    ], tools=[tool(forward, return_direct=True)])
    plan = AgentHandoffPlan(workflow_id='direct-chain', root_agent='worker', max_handoffs=budget,
        answer_requirements=REQUIREMENTS, agents=(
            HandoffAgent(agent_id='worker', model_id='careful', can_handoff_to=('writer',)),
            HandoffAgent(agent_id='writer', model_id='fast'),
        ))
    def build():
        return AgentHandoffWorkflow(tmp_path/'journal', key=KEY, catalog=catalog, plan=plan,
            memory=memory, space='alpha', scope=RecallScope.validated())
    work = build()
    try:
        result = await work.run('one', 'Count')
        assert result.hops[0].handoff_to == 'writer' and result.hops[0].output.model_id == 'careful'
        assert result.hops[0].output.model_calls == 1
        if budget:
            assert result.status == 'completed' and result.final.text == '{"count":6}'
            assert result.final.model_id == 'fast' and len(second.requests) == 1
            assert 'Count three items.' in json.dumps(second.requests[0][0])
        else:
            assert result.status == 'handoff_limit' and result.final is None and not second.requests
    finally:
        work.close()
    reopened = build()
    try:
        saved = await reopened.run('one', 'Count')
        assert saved.status == result.status and saved.final == result.final
        assert len(first.requests) == 1 and len(second.requests) == budget and invoked == ['alpha']
    finally:
        reopened.close()


async def test_direct_interactive_continuation_reuses_human_activation_and_usage(tmp_path,memory):
    catalog,model,invoked=configured({'count':6})
    plans=AgentPlanStore(tmp_path/'plans',key=KEY)
    runs=AgentRunStore(tmp_path/'runs',key=KEY)
    try:
        plan=InteractiveAgentPlan(kind='interactive',workflow_id='direct-interactive',tasks=(
            HumanInputTask(kind='input',task_id='choose',prompt='Approve count'),
            AgentTask(task_id='answer',agent_id='worker',model_id='local',prompt='Count',depends_on=('choose',),
                      answer_requirements=REQUIREMENTS)))
        saved=plans.save('alpha',plan,catalog=catalog,expected_revision=0)
        request=runs.register('alpha','one',plan=saved,question='Question',scope=RecallScope.validated())
        inbox=AgentInputStore(runs)
        def build():
            return InteractiveAgentWorkflow(tmp_path/'journal',key=KEY,catalog=catalog,request=request,
                memory=memory,inputs=inbox,activated=inbox.activated('alpha','one'))
        work=build()
        try:
            assert (await work.run('one','Question')).status=='awaiting_input'
            assert not model.requests and not invoked
        finally:work.close()
        inbox.respond('alpha','one','choose',response='Approved',expected_revision=1)
        inbox.activate('alpha','one','approved-once',responses={'choose':2})
        work=build()
        try:
            result=await work.run('one','Question')
            assert result.status=='completed'
            assert result.results['answer']['usage']==USAGE
            assert result.results['answer']['model_calls']==1
            human=result.results['choose']
            assert human['kind']=='human_input' and human['activation_id']=='approved-once'
            assert 'usage' not in human
        finally:work.close()
        reopened=build()
        try:
            assert (await reopened.read_result('one','Question')).results==result.results
            assert (await reopened.run('one','Question')).results==result.results
            assert len(model.requests)==len(invoked)==1
            assert 'Approved' in json.dumps(model.requests[0][0])
        finally:reopened.close()
    finally:
        runs.close();plans.close()
