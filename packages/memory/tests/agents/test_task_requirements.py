"""Authored workflow contracts survive persistence and gate model publication."""
import copy
import json

import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.task_requirements import TaskAnswerRequirements
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow
from scone_memory.agents.workflow import StepContext, WorkflowError
from scone_memory.realtime.answer_requirements import AnswerRequirements
from scone_memory.retrieval.recall_scope import RecallScope
from .test_task_workflow import memory


SCHEMA = {'$defs': {'name': {'type': 'string', 'minLength': 1}},
          'properties': {'name': {'$ref': '#/$defs/name'}},
          'required': ['name'], 'additionalProperties': False}


def task(requirements=None):
    return AgentTask(task_id='one', agent_id='worker', model_id='local', prompt='Answer',
                     answer_requirements=requirements)


def create(path, memory, calls, reply, requirements, *, downstream=False, max_parallel=1):
    class Model:
        async def complete(self, messages, tools):
            calls.append(copy.deepcopy(messages))
            return ToolStep(content=reply)
    agents = AgentCatalog(models=[AgentModel('local', 'Local', '1', Model)], agents=[
        AgentDefinition(agent_id='worker', instructions='Work', models=('local',),
                        default_model='local', initial_search=False)])
    tasks = (task(requirements),)
    if downstream:
        tasks += (AgentTask(task_id='two', agent_id='worker', prompt='Summarize', depends_on=('one',)),)
    plan = AgentTaskPlan(workflow_id='w', tasks=tasks)
    return AgentWorkflow(path, key=b'k'*32, catalog=agents, plan=plan, memory=memory,
                         space='alpha', scope=RecallScope.validated(), max_parallel=max_parallel)


def test_omitted_contract_preserves_exact_historical_task_wire_shape():
    assert task().model_dump(mode='json') == {
        'task_id': 'one', 'agent_id': 'worker', 'model_id': 'local',
        'prompt': 'Answer', 'depends_on': []}


def test_authored_local_references_survive_while_execution_compiles_detached_copy():
    schema = copy.deepcopy(SCHEMA)
    contract = TaskAnswerRequirements(format='json_object', output_schema=schema)
    assert contract.model_dump()['output_schema'] == SCHEMA
    schema['required'].clear()
    assert contract.model_dump()['output_schema'] == SCHEMA
    compiled = AnswerRequirements.model_validate(contract.model_dump())
    assert '$defs' not in compiled.output_schema
    assert compiled.output_schema['type'] == 'object'
    assert compiled.accepts('{"name":"Juniper"}') and not compiled.accepts('{"name":2}')
    assert task(contract).model_dump()['answer_requirements']['output_schema'] == SCHEMA


@pytest.mark.parametrize('value', [
    {'max_bytes': True}, {'max_lines': False}, {'max_bytes': 0},
    {'format': 'json'}, {'instructions': '界' * 2667},
    {'format': 'text', 'output_schema': {}},
    {'format': 'json_object', 'output_schema': {'$ref': 'https://invalid.example/secret'}},
    {'format': 'json_object', 'output_schema': {'type': 'array'}},
    {'format': 'json_object', 'output_schema': {'required': 'name'}},
    {'format': 'json_object', 'output_schema': {'unknown': 'secret'}},
])
def test_invalid_contract_is_rejected(value):
    with pytest.raises(ValueError):
        task(value)


def test_bypassed_and_caller_mutated_contracts_are_revalidated():
    with pytest.raises(ValueError):
        task(TaskAnswerRequirements.model_construct(max_bytes=-1))
    contract = TaskAnswerRequirements(format='json_object', output_schema=copy.deepcopy(SCHEMA))
    snapshot = task(contract)
    contract.output_schema['required'].clear()
    assert snapshot.answer_requirements.output_schema == SCHEMA
    assert task(AnswerRequirements(max_lines=1)).answer_requirements.max_lines == 1


@pytest.mark.parametrize('parallel', [1, 2])
@pytest.mark.parametrize('requirements,reply', [
    (TaskAnswerRequirements(format='json_object'), 'PRIVATE invalid output'),
    (TaskAnswerRequirements(format='json_object'), '{"a":1,"a":2}'),
    (TaskAnswerRequirements(max_lines=1), 'line\u2028line'),
    (TaskAnswerRequirements(max_bytes=5), '界界'),
    (TaskAnswerRequirements(format='json_object', output_schema=SCHEMA), '{"name":2}'),
])
async def test_invalid_result_never_reaches_dependent_task_or_saved_result(tmp_path, memory, requirements, reply, parallel):
    calls = []
    workflow = create(tmp_path/'journal', memory, calls, reply, requirements, downstream=True, max_parallel=parallel)
    try:
        with pytest.raises(WorkflowError, match='step_failed') as error:
            await workflow.run('r', 'Question')
        assert reply not in str(error.value)
        assert len(calls) == 1
        assert 'Answer requirements:' in json.dumps(calls)
        assert workflow.status('r', 'Question').completed_steps == ()
        with pytest.raises(WorkflowError, match='not_completed'):
            await workflow.read_result('r', 'Question')
        with pytest.raises(WorkflowError):
            await workflow.run('r', 'Question')
        assert len(calls) == 1
    finally:
        workflow.close()


async def test_schema_receipt_reopens_without_inference_and_contract_edits_cannot_reuse(tmp_path, memory):
    calls = []
    path = tmp_path/'journal'
    requirements = TaskAnswerRequirements(format='json_object', output_schema=SCHEMA)
    first = create(path, memory, calls, '{"name":"Juniper"}', requirements)
    result = await first.run('r', 'Question')
    first.close()
    reopened = create(path, memory, calls, 'Must not execute', requirements)
    try:
        assert (await reopened.read_result('r', 'Question')).results == result.results
        assert (await reopened.run('r', 'Question')).reused_steps == ('one',)
        assert len(calls) == 1
    finally:
        reopened.close()
    changed = create(path, memory, calls, 'Must not execute', TaskAnswerRequirements(max_lines=1))
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await changed.read_result('r', 'Question')
        assert len(calls) == 1
    finally:
        changed.close()


async def test_receipt_verification_checks_contract_even_for_structurally_valid_receipt(tmp_path, memory):
    calls = []
    workflow = create(tmp_path/'journal', memory, calls, '{"name":"Juniper"}',
                      TaskAnswerRequirements(format='json_object', output_schema=SCHEMA))
    try:
        result = await workflow.run('r', 'Question')
        altered = copy.deepcopy(result.results)
        altered['one']['text'] = '{"name":2}'
        context = StepContext(run_id='r', space='alpha', scope=workflow._binding_scope,
                              inputs='Question', completed=altered)
        with pytest.raises(ValueError, match='answer requirements'):
            workflow._receipts(context)
    finally:
        workflow.close()


@pytest.mark.parametrize('valid', [False, True])
async def test_interactive_resume_enforces_model_contract_and_leaves_human_input_plain(tmp_path, memory, valid):
    from scone_memory.agents.input_store import AgentInputStore
    from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
    from scone_memory.agents.interactive_workflow import InteractiveAgentWorkflow
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.run_store import AgentRunStore
    calls = []
    class Model:
        async def complete(self, messages, tools):
            calls.append(copy.deepcopy(messages))
            return ToolStep(content='{"name":"Juniper"}' if valid else '{"name":2}')
    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', Model)], agents=[
        AgentDefinition(agent_id='worker', instructions='Work', models=('local',),
                        default_model='local', initial_search=False)])
    plans = AgentPlanStore(tmp_path/'plans', key=b'k'*32)
    runs = AgentRunStore(tmp_path/'runs', key=b'k'*32)
    inbox = AgentInputStore(runs)
    contract = TaskAnswerRequirements(format='json_object', output_schema=SCHEMA)
    plan = InteractiveAgentPlan(kind='interactive', workflow_id='w', tasks=(
        HumanInputTask(kind='input', task_id='choose', prompt='Choose a name'),
        task(contract).model_copy(update={'depends_on': ('choose',)})))
    saved = plans.save('alpha', plan, catalog=catalog, expected_revision=0)
    request = runs.register('alpha', 'r', plan=saved, question='Question', scope=RecallScope.validated())
    def build():
        return InteractiveAgentWorkflow(tmp_path/'journal', key=b'k'*32, catalog=catalog,
            request=request, memory=memory, inputs=inbox, activated=inbox.activated('alpha', 'r'))
    first = build()
    try:
        assert (await first.run('r', 'Question')).status == 'awaiting_input'
        assert not calls
    finally:
        first.close()
    inbox.respond('alpha', 'r', 'choose', response='Juniper, please', expected_revision=1)
    inbox.activate('alpha', 'r', 'continue', responses={'choose': 2})
    resumed = build()
    try:
        if valid:
            result = await resumed.run('r', 'Question')
            assert result.status == 'completed'
            assert result.results['choose']['text'] == 'Juniper, please'
            assert 'answer_requirements' not in result.results['choose']
            assert (await resumed.read_result('r', 'Question')).results == result.results
            altered = copy.deepcopy(result.results)
            altered['one']['text'] = '{"name":2}'
            with pytest.raises(ValueError, match='answer requirements'):
                resumed._receipts(altered)
        else:
            with pytest.raises(WorkflowError, match='step_failed'):
                await resumed.run('r', 'Question')
            assert resumed.status('r', 'Question').completed_steps == ('choose',)
        assert len(calls) == 1 and 'Answer requirements:' in json.dumps(calls)
    finally:
        resumed.close()
        runs.close()
        plans.close()


def test_bypassed_task_contract_is_refused_without_serializer_input_warnings():
    import warnings
    forged = task().model_copy(update={'answer_requirements': {'max_bytes': False, 'instructions': 'PRIVATE'}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError):
            AgentTaskPlan.model_validate(AgentTaskPlan(workflow_id='w', tasks=(forged,)).model_dump())
    assert caught == []
