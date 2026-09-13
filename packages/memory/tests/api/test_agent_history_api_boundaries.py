import pytest
from tests.api.test_agent_runs_api import setup, auth, body
from tests.api.test_agent_history_api import start
from scone_memory.retrieval.recall_scope import RecallScope


async def test_registered_history_read_does_not_create_execution_journal(setup):
    client, app, service, memory, release, entered, calls = setup
    saved = service._plans.get('alpha', 'report')
    request = service._runs.register(
        'alpha', 'registered', plan=saved, question='Q', scope=RecallScope.validated()
    )
    path = service._path(request)
    assert not path.exists()
    response = await client.get('/v1/agent-runs/registered/history', headers=auth('reader'))
    assert response.status_code == 200 and not response.json()['available']
    assert not path.exists(), 'read-only history GET created an execution journal'
    assert calls == []


async def test_history_without_source_journal_is_not_delivered(setup):
    client, app, service, memory, release, entered, calls = setup
    await start(setup)
    request = service._runs.get('alpha', 'one')
    path = service._path(request)
    path.unlink()
    response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code != 200
    assert not path.exists()
    assert len(calls) == 1


from tests.api.test_agent_history_api import evidence_setup


async def test_source_inspection_is_the_final_await_before_delivery(evidence_setup, monkeypatch):
    client, app, service, memory, source, calls = evidence_setup
    original_open = service._open
    original_space = service._space
    inspected = []
    late_checks = []

    def open_workflow(request, **kwargs):
        workflow = original_open(request, **kwargs)
        inspect = workflow.inspect_approvals

        async def wrapped(*args):
            value = await inspect(*args)
            inspected.append(True)
            return value

        monkeypatch.setattr(workflow, 'inspect_approvals', wrapped)
        return workflow

    async def space(*args):
        if inspected:
            late_checks.append(True)
            await memory.forget('alpha', source.episode_id)
        return await original_space(*args)

    monkeypatch.setattr(service, '_open', open_workflow)
    monkeypatch.setattr(service, '_space', space)
    response = await client.get('/v1/agent-runs/one/history', headers=auth())
    assert inspected
    assert not late_checks and response.status_code == 200
    assert len(calls) == 1


async def test_history_without_workflow_row_is_not_delivered(setup):
    import sqlite3
    from contextlib import closing

    client, app, service, memory, release, entered, calls = setup
    await start(setup)
    request = service._runs.get('alpha', 'one')
    path = service._path(request)
    with closing(sqlite3.connect(path)) as db:
        db.execute('DELETE FROM workflow_runs')
        db.commit()
    response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code != 200
    assert len(calls) == 1


async def test_read_only_history_does_not_create_lock_or_allow_execution(setup):
    from scone_memory.agents.workflow import WorkflowError

    client, app, service, memory, release, entered, calls = setup
    await start(setup)
    request = service._runs.get('alpha', 'one')
    from pathlib import Path

    lock = Path(str(service._path(request)) + '.lock')
    lock.unlink()
    response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code == 200 and not lock.exists()
    workflow = service._open(request, read_only=True)
    try:
        with pytest.raises(WorkflowError, match='read_only_journal'):
            await workflow.run(request.run_id, request.question)
        with pytest.raises(WorkflowError, match='read_only_journal'):
            await workflow.read_result(request.run_id, request.question)
    finally:
        workflow.close()
    assert len(calls) == 1 and not lock.exists()


@pytest.mark.parametrize('mutation', ['unlink', 'purge', 'retention'])
async def test_history_withholds_page_removed_during_verification(setup, monkeypatch, mutation):
    client, app, service, memory, release, entered, calls = setup
    await start(setup)
    original = service._open

    def open_changed(request, **kwargs):
        workflow = original(request, **kwargs)
        inspect = workflow.inspect_approvals

        async def changed(*args):
            result = await inspect(*args)
            if mutation == 'unlink':
                service._path(request).unlink()
            elif mutation == 'purge':
                service._history.purge(request)
            else:
                row = service._history.read(request).items[-2]
                service._history._capacity = 1
                service._history.append(
                    request, step_id=row.step_id, selection_id=row.selection_id, event=row.event
                )
            return result

        monkeypatch.setattr(workflow, 'inspect_approvals', changed)
        return workflow

    monkeypatch.setattr(service, '_open', open_changed)
    response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code != 200
    assert 'collection_started' not in response.text and len(calls) == 1


from tests.agents.test_agent_approval_service_recovery import service_host, memory


@pytest.mark.parametrize('workflow', ['mixed', 'parallel'])
async def test_real_paused_replay_preserves_ownership_and_tickets(service_host, workflow):
    import httpx
    from scone_memory.api.app import create_app

    host = service_host
    await host.start(workflow)
    service = host.service
    app = create_app(
        service._memory,
        {'reader': 'alpha'},
        roles={'reader': 'read'},
        agent_catalog=service._catalog,
        agent_plan_store=service._plans,
        agent_run_service=service,
    )
    request = await service.request('alpha', 'one')
    path = service._path(request)
    before = path.read_bytes()
    records = service._approvals.list('alpha', 'one')
    count = {key: len(value.requests) for key, value in host.models.items()}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://scone.test'
    ) as client:
        response = await client.get('/v1/agent-runs/one/history', headers=auth('reader'))
    assert response.status_code == 200, response.text
    assert any(row['event'].get('terminal_kind') == 'turn_paused' for row in response.json()['items'])
    assert path.read_bytes() == before
    assert service._approvals.list('alpha', 'one') == records
    assert not service._owners and not service._tasks and not service._admissions
    assert count == {key: len(value.requests) for key, value in host.models.items()}
    assert not host.effects


async def test_observed_source_invalid_root_after_verification_refuses_history(evidence_setup, monkeypatch):
    from scone_memory.agents.workflow import WorkflowError

    client, app, service, memory, source, calls = evidence_setup
    original_open = service._open
    changed = []

    def open_reader(request, **kwargs):
        workflow = original_open(request, **kwargs)
        inspect = workflow.inspect_approvals

        async def wrapped(*args):
            result = await inspect(*args)
            await memory.forget('alpha', source.episode_id)
            writer = original_open(request)
            try:
                with pytest.raises(WorkflowError, match='sources_invalid'):
                    await writer.read_result(request.run_id, request.question)
            finally:
                writer.close()
            changed.append(True)
            return result

        monkeypatch.setattr(workflow, 'inspect_approvals', wrapped)
        return workflow

    monkeypatch.setattr(service, '_open', open_reader)
    response = await client.get('/v1/agent-runs/one/history', headers=auth())
    assert changed
    assert response.status_code == 409 and response.json()['code'] == 'sources_invalid'
