"""History delivery rechecks authority and sources without executing a run."""

import pytest

from tests.api.test_agent_runs_api import auth, body, setup


async def start(setup):
    client, _, service, _, release, _, _ = setup
    release.set()
    response = await client.post('/v1/agent-runs', json=body(), headers=auth())
    assert response.status_code == 202
    await service.wait('alpha', 'one')


async def test_history_replay_is_authenticated_private_and_nonexecuting(setup):
    client, _, _, _, _, _, calls = setup
    await start(setup)
    path = '/v1/agent-runs/one/history'
    assert (await client.get(path)).status_code == 401
    assert (await client.get(path, headers=auth('other'))).status_code == 404
    first = await client.get(path, params={'limit': '1'}, headers=auth('reader'))
    assert first.status_code == 200
    assert first.headers['cache-control'] == 'no-store'
    page = first.json()
    assert page['space'] == 'alpha' and page['run_id'] == 'one'
    assert page['available'] and page['items'][0]['event']['kind'] == 'collection_started'
    later = await client.get(path, params={'after': page['next_after']}, headers=auth('reader'))
    assert later.status_code == 200
    assert later.json()['items'][-1]['event']['kind'] == 'collection_finished'
    assert 'Selected model answer' not in later.text and 'Question' not in later.text
    tail = await client.get(path, params={'after': later.json()['next_after']}, headers=auth('reader'))
    assert tail.json()['items'] == [] and len(calls) == 1
    caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
    assert caps['agents.history'] is True


@pytest.mark.parametrize(
    'query',
    [
        'limit=0',
        'limit=101',
        'limit=true',
        'limit=01',
        'limit=1&limit=2',
        'after=a&after=b',
        'after=',
        'unexpected=PRIVATE',
        'after=' + 'a' * 4097,
    ],
)
async def test_history_query_rejects_ambiguous_or_unbounded_values(setup, query):
    client = setup[0]
    response = await client.get('/v1/agent-runs/one/history?' + query, headers=auth())
    assert response.status_code == 422
    assert 'PRIVATE' not in response.text


@pytest.mark.parametrize('change', ['revoke', 'rebind'])
async def test_history_rechecks_key_after_source_inspection(setup, monkeypatch, change):
    client, app, service, _, _, _, calls = setup
    await start(setup)
    original = service._open

    def open_changed(request, **kwargs):
        workflow = original(request, **kwargs)
        inspect = workflow.inspect_approvals

        async def changed(*args):
            result = await inspect(*args)
            if change == 'revoke':
                app.state.keys.pop('reader')
            else:
                app.state.keys['reader'] = 'bravo'
            return result

        monkeypatch.setattr(workflow, 'inspect_approvals', changed)
        return workflow

    monkeypatch.setattr(service, '_open', open_changed)
    response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code == 401
    assert 'collection_started' not in response.text and len(calls) == 1


async def test_history_replay_while_model_runs_never_waits_for_or_restarts_it(setup):
    client, _, service, _, _, entered, calls = setup
    assert (await client.post('/v1/agent-runs', json=body(), headers=auth())).status_code == 202
    await entered.wait()
    response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code == 200
    assert response.json()['items'][0]['event']['kind'] == 'collection_started'
    assert (await service.status('alpha', 'one')).active_local and len(calls) == 1


@pytest.fixture
async def evidence_setup(tmp_path):
    import httpx
    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.evidence_loop import ToolStep
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.run_service import AgentRunService
    from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
    from scone_memory.api.app import create_app
    from scone_memory.retrieval.recall_scope import RecallScope

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(messages)
            return ToolStep(content='Juniper private address is Oregon.')

    catalog = AgentCatalog(
        models=[AgentModel('local', 'Local', '1', Model)],
        agents=[
            AgentDefinition(
                agent_id='research', instructions='Use evidence.', models=('local',), default_model='local'
            )
        ],
    )
    plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
    plans.save(
        'alpha',
        AgentTaskPlan(
            workflow_id='report',
            tasks=(AgentTask(task_id='find', agent_id='research', prompt='Locate Juniper.'),),
        ),
        catalog=catalog,
        expected_revision=0,
    )
    service = AgentRunService(
        tmp_path / 'runs',
        key=b'k' * 32,
        catalog=catalog,
        plans=plans,
        memory=memory,
        scope_for=lambda _: RecallScope.validated(),
    )
    app = create_app(
        memory, {'writer': 'alpha'}, agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service
    )
    try:
        source = await memory.remember('alpha', 'Juniper private address is Oregon.')
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://scone.test'
        ) as client:
            response = await client.post(
                '/v1/agent-runs', json={**body(), 'question': 'Where is Juniper?'}, headers=auth()
            )
            assert response.status_code == 202
            await service.wait('alpha', 'one')
            yield client, app, service, memory, source, calls
    finally:
        await service.aclose()
        plans.close()
        await memory.close()


async def test_forgotten_sources_withhold_history_without_mutating_workflow(evidence_setup):
    client, _, service, memory, source, calls = evidence_setup
    assert (await client.get('/v1/agent-runs/one/history', headers=auth())).status_code == 200
    before = await service.status('alpha', 'one')
    await memory.forget('alpha', source.episode_id)
    response = await client.get('/v1/agent-runs/one/history', headers=auth())
    assert response.status_code == 409 and response.json()['code'] == 'sources_invalid'
    assert 'Juniper' not in response.text and 'collection_started' not in response.text
    assert await service.status('alpha', 'one') == before
    assert len(calls) == 1


async def test_history_refuses_scope_change_during_actual_source_verification(evidence_setup, monkeypatch):
    from scone_memory.retrieval.recall_scope import RecallScope

    client, _, service, memory, _, calls = evidence_setup
    original = memory.documents.get_chunks
    checked = []

    async def restrict(*args):
        checked.append(True)
        service._scope_for = lambda _: RecallScope.validated(where={'team': 'public'})
        return await original(*args)

    monkeypatch.setattr(memory.documents, 'get_chunks', restrict)
    response = await client.get('/v1/agent-runs/one/history', headers=auth())
    assert checked and response.status_code == 409
    assert response.json()['code'] == 'run_scope_changed'
    assert 'Juniper' not in response.text and len(calls) == 1


async def test_history_retention_and_purge_are_explicit_and_never_reexecute(setup):
    client, _, service, _, _, _, calls = setup
    service._history._capacity = 2
    await start(setup)
    response = await client.get('/v1/agent-runs/one/history', headers=auth())
    page = response.json()
    assert response.status_code == 200 and page['omitted'] is not None
    assert len(page['items']) == 2 and page['retained_from'] > 1
    saved = await service.request('alpha', 'one')
    service._history.purge(saved)
    unavailable = await client.get('/v1/agent-runs/one/history', headers=auth())
    assert unavailable.json()['available'] is False and unavailable.json()['items'] == []
    stale = await client.get(
        '/v1/agent-runs/one/history', params={'after': page['next_after']}, headers=auth()
    )
    assert stale.status_code == 422 and stale.json()['code'] == 'invalid_history_cursor'
    assert len(calls) == 1
