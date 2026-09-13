"""Usage follows encrypted model receipts through every native workflow kind."""
import json

import httpx
import pytest

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.handoff_workflow import AgentHandoffPlan, HandoffAgent, HandoffResult
from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentTaskReceipt, AgentWorkflow
from scone_memory.agents.usage import ModelTokenUsage, ToolTokenUsage
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope

COUNTS = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
REPORT = {'calls': [COUNTS]}


@pytest.fixture
async def setup(tmp_path, request):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    calls = []
    class Model:
        def __init__(self, name): self.name = name
        async def complete(self, messages, tools):
            calls.append((self.name, messages))
            text = json.dumps({'answer': 'PRIVATE_ANSWER', 'handoff_to': None}) if self.name == 'handoff' else 'PRIVATE_ANSWER'
            return ToolStep(content=text, usage=ModelTokenUsage(**COUNTS))
    catalog = AgentCatalog(models=[AgentModel(name, name, '1', lambda name=name: Model(name)) for name in ('small', 'large', 'handoff')],
        agents=[AgentDefinition(agent_id='worker', instructions='Use evidence.', models=('small', 'large', 'handoff'),
                                default_model='small', initial_search=getattr(request, 'param', False))])
    plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
    def build():
        return AgentRunService(tmp_path / 'runs', key=b'k' * 32, catalog=catalog, plans=plans, memory=memory,
                               scope_for=lambda _: RecallScope.validated(), max_parallel_tasks=2)
    service = build()
    yield service, plans, catalog, memory, calls, build
    await service.aclose(); plans.close(); await memory.close()


def plan(kind):
    if kind == 'handoff':
        return AgentHandoffPlan(workflow_id='w', root_agent='worker', agents=(HandoffAgent(agent_id='worker', model_id='handoff'),))
    tasks = (AgentTask(task_id='first', agent_id='worker', prompt='Find facts.', model_id='small'),
             AgentTask(task_id='last', agent_id='worker', prompt='Explain.', model_id='large',
                       depends_on=('choose',) if kind == 'interactive' else ()))
    if kind == 'interactive':
        return InteractiveAgentPlan(kind='interactive', workflow_id='w', tasks=(
            HumanInputTask(kind='input', task_id='choose', prompt='Choose.'), *tasks))
    return AgentTaskPlan(workflow_id='w', tasks=tasks)


async def run(setup, kind):
    service, plans, catalog, *_ = setup
    plans.save('alpha', plan(kind), catalog=catalog, expected_revision=0)
    await service.start('alpha', 'r', workflow_id='w', plan_revision=1, question='Question',
                        max_parallel=2 if kind == 'parallel' else 1)
    await service.wait('alpha', 'r')
    if kind == 'interactive':
        await service.respond('alpha', 'r', 'choose', response='Proceed.', expected_revision=1)
        await service.continue_run('alpha', 'r', continuation_id='confirm', responses={'choose': 2})
        await service.wait('alpha', 'r')
    return await service.result('alpha', 'r')


def outputs(result):
    if isinstance(result, HandoffResult):
        return [hop.output.model_dump(mode='json') for hop in result.hops]
    return [row for row in result.results.values() if row.get('kind') != 'human_input']


@pytest.mark.parametrize('kind', ['tasks', 'parallel', 'handoff', 'interactive'])
async def test_usage_persists_after_restart_without_model_replay(setup, kind):
    service, _, _, _, calls, build = setup
    result = await run(setup, kind)
    assert all(row['usage'] == REPORT and row['model_calls'] == 1 for row in outputs(result))
    if kind == 'interactive':
        assert 'usage' not in result.results['choose']
    original_calls = len(calls)
    await service.aclose()
    reopened = build()
    try:
        saved = await reopened.result('alpha', 'r')
        assert outputs(saved) == outputs(result) and len(calls) == original_calls
        assert all('prompt_tokens' not in json.dumps(messages) for _, messages in calls)
    finally:
        await reopened.aclose()


@pytest.mark.parametrize('kind', ['tasks', 'parallel', 'handoff', 'interactive'])
async def test_http_usage_requires_opt_in_and_retains_original_default_shape(setup, kind):
    service, plans, catalog, memory, calls, _ = setup
    await run(setup, kind)
    app = create_app(memory, {'reader': 'alpha', 'other': 'bravo'}, roles={'reader': 'read'},
        agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service)
    headers = {'Authorization': 'Bearer reader'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        caps = (await client.get('/v1/capabilities', headers=headers)).json()['features']
        assert caps['agents.usage'] is True
        plain = await client.get('/v1/agent-runs/r/result', headers=headers)
        enriched = await client.get('/v1/agent-runs/r/result?include_usage=true', headers=headers)
        assert plain.status_code == enriched.status_code == 200
        assert '"usage"' not in plain.text
        data = enriched.json()
        if kind == 'handoff':
            assert data['final']['usage'] == REPORT
            data['final'].pop('usage')
            for hop in data['hops']:
                assert hop['output'].pop('usage') == REPORT
        else:
            for row in data['results'].values():
                if row.get('kind') == 'human_input': assert 'usage' not in row
                else: assert row.pop('usage') == REPORT
        assert data == plain.json()
        assert enriched.headers['cache-control'] == 'no-store'
        assert (await client.get('/v1/agent-runs/r/result?include_usage=true', headers={'Authorization': 'Bearer other'})).status_code == 404
        assert (await client.get('/v1/agent-runs/r/result?include_usage=true')).status_code == 401


async def test_legacy_journal_remains_readable_with_explicit_unknown_usage(setup, monkeypatch):
    original = AgentWorkflow._step
    def legacy(self, task, agent):
        execute = original(self, task, agent)
        async def step(context):
            value = await execute(context)
            value.pop('usage', None)
            return value
        return step
    with monkeypatch.context() as old:
        old.setattr(AgentWorkflow, '_step', legacy)
        await run(setup, 'tasks')
    service, plans, catalog, memory, calls, build = setup
    await service.aclose()
    reopened = build()
    try:
        app = create_app(memory, {'reader': 'alpha'}, agent_catalog=catalog, agent_plan_store=plans, agent_run_service=reopened)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test', headers={'Authorization': 'Bearer reader'}) as client:
            answer = await client.get('/v1/agent-runs/r/result?include_usage=true')
            assert answer.status_code == 200
            assert all(row['usage'] is None for row in answer.json()['results'].values())
            assert len(calls) == 2
    finally: await reopened.aclose()


@pytest.mark.parametrize('query', ['include_usage=1', 'include_usage=yes', 'include_usage=', 'include_usage=true&include_usage=false'])
async def test_invalid_usage_selection_is_refused_before_result_read(setup, monkeypatch, query):
    service, plans, catalog, memory, *_ = setup
    async def forbidden(*args): pytest.fail('invalid query reached service')
    monkeypatch.setattr(service, 'result', forbidden)
    app = create_app(memory, {'reader': 'alpha'}, agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        answer = await client.get('/v1/agent-runs/r/result?' + query, headers={'Authorization': 'Bearer reader'})
        assert answer.status_code == 422


def test_saved_usage_report_count_must_match_model_calls():
    receipt = dict(task_id='t', agent_id='a', model_id='m', binding='a'*64, depends_on=(), text='Done.',
                   source_status='none', evidence_ids=(), evidence_packets=(), model_calls=2, tool_calls=0)
    assert AgentTaskReceipt(**receipt).usage is None
    with pytest.raises(ValueError):
        AgentTaskReceipt(**receipt, usage=ToolTokenUsage(calls=(ModelTokenUsage(**COUNTS),)))


@pytest.mark.parametrize('setup', [True], indirect=True)
async def test_usage_does_not_bypass_source_withdrawal(setup):
    service, plans, catalog, memory, calls, _ = setup
    episode = await memory.remember('alpha', 'Juniper is in Oregon.')
    await run(setup, 'tasks')
    await memory.forget('alpha', episode.episode_id)
    app = create_app(memory, {'reader': 'alpha'}, agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        result = await client.get('/v1/agent-runs/r/result?include_usage=true', headers={'Authorization': 'Bearer reader'})
        assert result.status_code == 409 and 'PRIVATE_ANSWER' not in result.text and 'prompt_tokens' not in result.text
    assert len(calls) == 2


async def test_usage_result_rechecks_key_after_service_returns(setup, monkeypatch):
    service, plans, catalog, memory, *_ = setup
    await run(setup, 'tasks')
    app = create_app(memory, {'reader': 'alpha'}, agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service)
    original = service.result
    async def changed(*args):
        answer = await original(*args)
        app.state.keys['reader'] = 'bravo'
        return answer
    monkeypatch.setattr(service, 'result', changed)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        result = await client.get('/v1/agent-runs/r/result?include_usage=true', headers={'Authorization': 'Bearer reader'})
        assert result.status_code == 401 and 'PRIVATE_ANSWER' not in result.text and 'prompt_tokens' not in result.text


def test_http_projection_keeps_nested_receipts_detached():
    from copy import deepcopy
    from scone_memory.api.agent_runs import _result_payload
    from scone_memory.agents.workflow import WorkflowResult
    raw = {'model': {'usage': deepcopy(REPORT), 'evidence_ids': ['chunk:1']},
           'human': {'kind': 'human_input', 'text': 'Original'}}
    answer = WorkflowResult('r', 'completed', raw, ())
    projected = _result_payload(answer, True)
    projected['results']['model']['usage']['calls'][0]['prompt_tokens'] = 99
    projected['results']['model']['evidence_ids'].append('chunk:2')
    projected['results']['human']['text'] = 'Changed'
    assert raw['model']['usage'] == REPORT and raw['model']['evidence_ids'] == ['chunk:1']
    assert raw['human']['text'] == 'Original'
