"""Trap diagnostics survive service restart without exposing tool payloads."""
from dataclasses import replace
import httpx
import pytest
from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolCall, ToolLoopLimits, ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope


async def test_trap_report_replays_after_restart_under_current_authority(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(True)
            return ToolStep(calls=(ToolCall(id='call-' + str(len(calls)), name='search_memory',
                                            arguments={'query': 'PRIVATE-QUERY'}),))

    catalog = AgentCatalog(models=[AgentModel('chosen', 'Chosen', '1', Model)], agents=[AgentDefinition(
        agent_id='worker', instructions='Find evidence.', models=('chosen',), default_model='chosen',
        initial_search=False, limits=ToolLoopLimits(max_repeated_rounds=2))])
    plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
    plans.save('alpha', AgentTaskPlan(workflow_id='report', tasks=(AgentTask(
        task_id='find', agent_id='worker', prompt='Find evidence.'),)), catalog=catalog, expected_revision=0)
    args = dict(key=b'k' * 32, catalog=catalog, plans=plans, memory=memory,
                scope_for=lambda space: RecallScope.validated())
    first = None
    try:
        for restart in range(2):
            service = AgentRunService(tmp_path / 'runs', **args)
            app = create_app(memory, {'reader': 'alpha', 'writer': 'alpha', 'other': 'bravo'},
                roles={'reader': 'read', 'writer': 'write'}, agent_catalog=catalog,
                agent_plan_store=plans, agent_run_service=service)
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
                    headers = {'Authorization': 'Bearer writer'}
                    if restart == 0:
                        response = await client.post('/v1/agent-runs', headers=headers, json={
                            'run_id': 'one', 'workflow_id': 'report', 'plan_revision': 1, 'question': 'PRIVATE-TASK'})
                        assert response.status_code == 202, response.text
                        await service.wait('alpha', 'one')
                    path = '/v1/agent-runs/one/history'
                    response = await client.get(path, headers={'Authorization': 'Bearer reader'})
                    assert response.status_code == 200, response.text
                    events = [entry['event'] for entry in response.json()['items']]
                    reports = [event for event in events if event.get('kind') == 'trap_detected']
                    assert len(reports) == 1, repr(events)
                    assert reports[0]['trap_graph']['path'] == [1, 1]
                    assert reports[0]['model_id'] == 'chosen'
                    assert all('trap_graph' not in event for event in events if event.get('kind') != 'trap_detected')
                    assert 'PRIVATE' not in response.text
                    if first is None:
                        first = response.json()
                    else:
                        assert response.json() == first
                    assert (await client.get(path, headers={'Authorization': 'Bearer other'})).status_code == 404
                    app.state.keys.pop('reader')
                    assert (await client.get(path, headers={'Authorization': 'Bearer reader'})).status_code == 401
                    assert len(calls) == 2
            finally:
                await service.aclose()
    finally:
        plans.close()
        await memory.close()


@pytest.mark.parametrize('change', ['visits', 'edges', 'pattern', 'boolean', 'unknown_node'])
def test_trap_graph_refuses_inconsistent_reports(change):
    from scone_memory.agents.traps import ObservationGraph, validate_trap_graph

    detector = ObservationGraph(2)
    detector.observe(b'private')
    graph = detector.observe(b'private')
    if change == 'visits':
        graph = replace(graph, nodes=(replace(graph.nodes[0], visits=10),))
    elif change == 'edges':
        graph = replace(graph, edges=())
    elif change == 'pattern':
        graph = replace(graph, repetitions=3)
    elif change == 'boolean':
        graph = replace(graph, path=(True, True))
    else:
        graph = replace(graph, path=(1, 2))
    with pytest.raises(ValueError):
        validate_trap_graph(graph)
