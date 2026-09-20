"""Real API/store/workflow integration; scripted models are not UI acceptance."""
from contextlib import asynccontextmanager
import json

import httpx
import pytest

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolCall, ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.tool_recipe_store import ToolRecipeStore
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope
from tests.agents.test_evidence_tool_loop import Script
from tests.agents.test_tool_recipes import capability, recipe
from tests.api.test_agent_runs_api import auth


@pytest.fixture
async def host(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    effects = []
    primitive = capability(effects)
    author = Script(ToolStep(calls=(ToolCall(id='propose', name='propose_tool_recipe', arguments={
        'proposal_id': 'quad-v1', 'recipe_json': recipe(primitive).model_dump_json()}),)),
        ToolStep(content='Proposed a tool for human review.'))
    executor = Script()
    async def request_execution():
        packet = json.loads(executor.requests[-1][0][-1]['content'])['result']
        return ToolStep(calls=(ToolCall(id='invoke', name='invoke_tool_recipe', arguments={
            'proposal_id': 'quad-v1', 'tool_revision': packet['tool']['revision'], 'arguments_json': '{"count":3}'}),))
    executor.steps = [ToolStep(calls=(ToolCall(id='inspect', name='inspect_tool_recipe', arguments={'proposal_id': 'quad-v1'}),)),
                      request_execution, ToolStep(content='The execution response has been received.')]

    @asynccontextmanager
    async def open_host():
        recipes = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
        plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
        tools = [recipes.proposal_tool(space='alpha', proposed_by='author-agent', tools=[primitive]),
                 *recipes.execution_tools(space='alpha', tools=[primitive])]
        catalog = AgentCatalog(models=[AgentModel('author-model', 'Selected author model', '1', lambda: author),
                                       AgentModel('executor-model', 'Selected executor model', '1', lambda: executor)],
            tools=tools, agents=[
                AgentDefinition(agent_id='author', instructions='Propose the requested composition.', models=('author-model',),
                    default_model='author-model', initial_search=False, tools=('propose_tool_recipe',)),
                AgentDefinition(agent_id='executor', instructions='Inspect and request the exact reviewed tool version.',
                    models=('executor-model',), default_model='executor-model', initial_search=False,
                    tools=('inspect_tool_recipe', 'invoke_tool_recipe'))])
        service = AgentRunService(tmp_path / 'runs', key=b'k' * 32, catalog=catalog, plans=plans,
                                  memory=memory, scope_for=lambda _: RecallScope.validated())
        app = create_app(memory, {'writer': 'alpha', 'reader': 'alpha', 'reviewer': 'alpha', 'other': 'bravo'},
                         roles={'writer': 'write', 'reader': 'read', 'reviewer': 'review'},
                         agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service, tool_recipe_store=recipes)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
                yield client, service
        finally:
            await service.aclose()
            plans.close()
            recipes.close()
    try:
        yield open_host, effects, author, executor
    finally:
        await memory.close()


async def start(client, service, agent):
    saved = await client.put('/v1/agent-plans/' + agent, headers=auth(), json={'expected_revision': 0,
        'plan': {'workflow_id': agent, 'tasks': [{'task_id': 'work', 'agent_id': agent,
            'model_id': agent + '-model', 'prompt': 'Create or execute quad-v1.', 'depends_on': []}]}})
    assert saved.status_code == 200, saved.text
    admitted = await client.post('/v1/agent-runs', headers=auth(), json={
        'run_id': agent, 'workflow_id': agent, 'plan_revision': 1, 'question': 'Quadruple 3.'})
    assert admitted.status_code == 202, admitted.text
    return await service.wait('alpha', agent)


@pytest.mark.parametrize('decision', ['approve', 'deny', 'revoke'])
async def test_http_version_review_call_review_restart_and_explicit_activation(host, decision):
    open_host, effects, author, executor = host
    async with open_host() as (client, service):
        assert (await start(client, service, 'author')).status == 'completed'
        proposal = (await client.get('/v1/tool-recipes/quad-v1', headers=auth('reader'))).json()
        assert proposal['status'] == 'pending' and not effects
        reviewed = await client.post('/v1/tool-recipes/quad-v1/decision', headers=auth('reviewer'),
            json={'decision': 'approve', 'expected_revision': 1, 'reason': 'Reviewed both steps and inputs.'})
        assert reviewed.status_code == 200 and not effects
        assert (await start(client, service, 'executor')).status == 'paused' and not effects
        records = await client.get('/v1/agent-runs/executor/approvals', headers=auth('reader'))
        (ticket,) = records.json()['items']
        args = json.loads(ticket['call']['arguments_json'])
        assert ticket['call']['tool_name'] == 'invoke_tool_recipe'
        assert args['proposal_id'] == 'quad-v1' and args['arguments_json'] == '{"count":3}'
        assert len(args['tool_revision']) == 64
        reviewed_call = await client.post('/v1/agent-runs/executor/approvals/' + ticket['request_id'] + '/decision',
            headers=auth('reviewer'), json={'decision': 'deny' if decision == 'deny' else 'approve', 'expected_revision': 1})
        assert reviewed_call.status_code == 200 and not effects
        if decision == 'revoke':
            retired = await client.post('/v1/tool-recipes/quad-v1/revoke', headers=auth('reviewer'),
                                       json={'reason': 'Retired before dispatch', 'expected_revision': 2})
            assert retired.status_code == 200
    # Reopen encrypted stores and execution service; neither review grants execution.
    async with open_host() as (client, service):
        status = await client.get('/v1/agent-runs/executor', headers=auth('reader'))
        assert status.json()['status'] == 'paused' and not effects
        assert (await client.get('/v1/tool-recipes/quad-v1', headers=auth('other'))).status_code == 404
        body = {'continuation_id': 'continue-one', 'decisions': {ticket['request_id']: 2}}
        activated = await client.post('/v1/agent-runs/executor/approval-continuations', headers=auth(), json=body)
        assert activated.status_code == 202, activated.text
        status = await service.wait('alpha', 'executor')
        assert effects == ([(3, 'alpha'), (6, 'alpha')] if decision == 'approve' else [])
        if decision == 'revoke':
            assert status.status != 'completed'
            return
        assert status.status == 'completed'
        result = await client.get('/v1/agent-runs/executor/result', headers=auth('reader'))
        assert result.status_code == 200 and result.json()['results']['work']['model_id'] == 'executor-model'
        repeated = await client.post('/v1/agent-runs/executor/approval-continuations', headers=auth(), json=body)
        assert repeated.status_code == 202
        await service.wait('alpha', 'executor')
        assert len(effects) == (2 if decision == 'approve' else 0)
        assert len(author.requests) == 2 and len(executor.requests) == 3
