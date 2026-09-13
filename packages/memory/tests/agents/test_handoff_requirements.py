"""Final handoff contracts permit ordinary notes and survive durable reuse."""
import copy
import pytest
from scone_memory.agents.handoff_workflow import AgentHandoffPlan
from scone_memory.agents.task_requirements import TaskAnswerRequirements
from scone_memory.agents.workflow import WorkflowError
from scone_memory.realtime.answer_requirements import AnswerRequirements
from .test_handoff_workflow import catalog, memory, plan, workflow
from .test_task_requirements import SCHEMA


def policy(requirements=None, *, max_handoffs=3):
    return AgentHandoffPlan.model_validate({**plan(max_handoffs).model_dump(),
                                          'answer_requirements': requirements})


def test_legacy_wire_omits_absent_final_contract():
    historical = {'workflow_id': 'report', 'root_agent': 'research', 'max_handoffs': 3,
        'agents': [{'agent_id': 'research', 'model_id': 'careful', 'can_handoff_to': ['write']},
                   {'agent_id': 'write', 'model_id': 'fast', 'can_handoff_to': ['research']}]}
    assert plan().model_dump(mode='json') == historical
    assert policy().model_dump(mode='json') == historical


def test_authored_schema_and_requirements_are_snapshotted():
    contract = TaskAnswerRequirements(format='json_object', output_schema=copy.deepcopy(SCHEMA))
    saved = policy(contract)
    contract.output_schema['required'].clear()
    assert saved.answer_requirements.output_schema == SCHEMA
    assert saved.model_dump()['answer_requirements']['output_schema'] == SCHEMA
    assert policy(AnswerRequirements(max_lines=1)).answer_requirements.max_lines == 1


@pytest.mark.parametrize('direct', [False, True])
async def test_final_object_preserves_number_tokens_and_free_text_delegation(tmp_path, memory, direct):
    requirements = {'format': 'json_object', 'max_bytes': 200, 'max_lines': 1,
                    'output_schema': {'properties': {'value': {'type': 'number'}},
                                      'required': ['value'], 'additionalProperties': False}}
    final = '{ "handoff_to": null, "answer": { "value": 0.12345678901234567890123456789e+02 } }'
    calls = []
    agents = catalog(calls, {'careful': final if direct else {'answer': 'Ordinary research notes', 'handoff_to': 'write'}, 'fast': final})
    work = workflow(tmp_path/'run', memory, agents, policy(requirements))
    try:
        result = await work.run('r', 'Question')
        assert result.final.text == '{"value":0.12345678901234567890123456789e+02}'
        assert len(calls) == (1 if direct else 2)
        if not direct:
            assert result.hops[0].output.text == 'Ordinary research notes'
            assert 'Ordinary research notes' in str(calls[1][1])
        assert (await work.read_result('r', 'Question')).final == result.final
        work.close()
        work = workflow(tmp_path/'run', memory, agents, policy(requirements))
        assert len((await work.run('r', 'Question')).reused_hops) == len(calls)
        assert len(calls) == (1 if direct else 2)
    finally:
        work.close()


@pytest.mark.parametrize('requirements,reply', [
    ({'max_bytes': 5}, {'answer': '界界', 'handoff_to': None}),
    ({'max_lines': 1}, {'answer': 'first\u2028second', 'handoff_to': None}),
    ({'format': 'json_object'}, {'answer': '{}', 'handoff_to': None}),
    ({'format': 'json_object'}, {'answer': {'value': 1}, 'handoff_to': 'write'}),
    ({'format': 'json_object'}, '{"answer":{"a":1,"a":2},"handoff_to":null}'),
    ({'format': 'json_object'}, '{"answer":{},"handoff_to":1}'),
    ({'format': 'json_object', 'output_schema': SCHEMA}, {'answer': {'name': 2}, 'handoff_to': None}),
    ({'format': 'json_object', 'max_bytes': 2}, {'answer': {'name': 'Juniper'}, 'handoff_to': None}),
])
async def test_invalid_contract_output_never_persists_or_replays(tmp_path, memory, requirements, reply):
    calls = []
    agents = catalog(calls, {'careful': reply, 'fast': {'answer': {}, 'handoff_to': None}})
    work = workflow(tmp_path/'run', memory, agents, policy(requirements))
    try:
        with pytest.raises(WorkflowError, match='step_failed'):
            await work.run('r', 'Question')
        assert work.progress('r', 'Question').completed_steps == ()
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await work.run('r', 'Question')
        assert len(calls) == 1
    finally:
        work.close()


async def test_text_contract_only_applies_on_completion_and_limit_has_no_final(tmp_path, memory):
    calls = []
    agents = catalog(calls, {'careful': {'answer': 'Notes exceed final budget\nand contain lines', 'handoff_to': 'write'},
                             'fast': {'answer': 'Done', 'handoff_to': None}})
    requirements = {'max_lines': 1, 'max_bytes': 4, 'instructions': 'Give a short final answer'}
    work = workflow(tmp_path/'run', memory, agents, policy(requirements))
    try:
        assert (await work.run('r', 'Question')).final.text == 'Done'
        assert 'Give a short final answer' in str(calls)
    finally:
        work.close()
    work = workflow(tmp_path/'limited', memory, agents, policy(requirements, max_handoffs=0))
    try:
        result = await work.run('r', 'Question')
        assert result.status == 'handoff_limit' and result.final is None
        assert (await work.read_result('r', 'Question')).status == 'handoff_limit'
    finally:
        work.close()


async def test_receipt_contract_revalidation_and_changed_contract_identity(tmp_path, memory):
    calls = []
    agents = catalog(calls, {'careful': {'answer': {'name': 'Juniper'}, 'handoff_to': None}, 'fast': {}})
    work = workflow(tmp_path/'run', memory, agents, policy({'format': 'json_object', 'output_schema': SCHEMA}))
    try:
        result = await work.run('r', 'Question')
        stored = {'hop-01': result.hops[0].model_dump(mode='json')}
        stored['hop-01']['output']['text'] = '{"name":2}'
        with pytest.raises(ValueError, match='answer requirements'):
            work._receipts(stored)
    finally:
        work.close()
    work = workflow(tmp_path/'run', memory, agents, policy({'format': 'json_object'}))
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await work.read_result('r', 'Question')
        assert len(calls) == 1
    finally:
        work.close()


def test_wrapped_schema_budget_is_checked_when_plan_is_authored():
    schema = {'properties': {f'f{i}': {'type': 'string'} for i in range(254)}}
    contract = TaskAnswerRequirements(format='json_object', output_schema=schema)
    with pytest.raises(ValueError):
        policy(contract)


async def test_forged_contract_is_rejected_before_journal(tmp_path, memory):
    calls = []
    agents = catalog(calls, {'careful': {}, 'fast': {}})
    bad = plan().model_copy(update={'answer_requirements': {'max_lines': 0}})
    with pytest.raises(ValueError):
        workflow(tmp_path/'run', memory, agents, bad)
    assert not list(tmp_path.iterdir()) and not calls


@pytest.mark.parametrize('change', ['forget', 'outage'])
async def test_final_contract_keeps_source_verification_and_no_replay(tmp_path, memory, monkeypatch, change):
    episode = await memory.remember('alpha', 'Juniper is in Oregon')
    calls = []
    agents = catalog(calls, {'careful': {'answer': {'name': 'Juniper'}, 'handoff_to': None}, 'fast': {}}, initial_search=True)
    work = workflow(tmp_path/'run', memory, agents, policy({'format': 'json_object', 'output_schema': SCHEMA}))
    try:
        result = await work.run('r', 'Where is Juniper?')
        assert result.final.source_status == 'retained'
        if change == 'forget':
            await memory.forget('alpha', episode.episode_id)
            with pytest.raises(WorkflowError, match='sources_invalid'):
                await work.read_result('r', 'Where is Juniper?')
        else:
            original = memory.documents.get_chunks
            async def unavailable(*args, **kwargs):
                raise ConnectionError('temporary storage outage')
            monkeypatch.setattr(memory.documents, 'get_chunks', unavailable)
            with pytest.raises(WorkflowError, match='verification_unavailable'):
                await work.read_result('r', 'Where is Juniper?')
            monkeypatch.setattr(memory.documents, 'get_chunks', original)
            assert (await work.read_result('r', 'Where is Juniper?')).final == result.final
        assert len(calls) == 1
    finally:
        work.close()


async def test_provider_and_caller_cannot_weaken_workflow_snapshot(tmp_path, memory):
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.evidence_loop import ToolStep
    calls = []
    class Mutating:
        async def complete(self, messages, tools):
            raise AssertionError('requirements-aware path expected')
        async def complete_with_requirements(self, messages, tools, requirements):
            calls.append(requirements)
            requirements.output_schema.clear()
            return ToolStep(content='{"answer":{"name":2},"handoff_to":null}')
    agents = AgentCatalog(models=[AgentModel(name, name, '1', Mutating) for name in ('careful', 'fast')],
        agents=[AgentDefinition(agent_id=name, instructions='Work', models=('careful', 'fast'),
                                default_model='careful', initial_search=False) for name in ('research', 'write')])
    authored = policy({'format': 'json_object', 'output_schema': copy.deepcopy(SCHEMA)})
    work = workflow(tmp_path/'run', memory, agents, authored)
    authored.answer_requirements.output_schema.clear()
    try:
        with pytest.raises(WorkflowError, match='step_failed'):
            await work.run('r', 'Question')
        assert len(calls) == 1 and work.progress('r', 'Question').completed_steps == ()
    finally:
        work.close()
