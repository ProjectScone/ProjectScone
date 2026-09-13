"""Owned observation records coverage without becoming execution authority."""

import asyncio

import pytest

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.event_history import AgentEventHistoryStore
from scone_memory.agents.history_capture import AgentRunHistory
from scone_memory.agents.history_models import AgentCollectionEvent
from scone_memory.agents.progress import AgentProgressEvent, AgentProgressGap
from scone_memory.agents.workflow import WorkflowError
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from tests.agents.test_event_history import run_request
from tests.agents.test_run_service import setup


def selected(factory):
    return AgentCatalog(
        models=[AgentModel('local', 'Local', '1', factory)],
        agents=[
            AgentDefinition(
                agent_id='research',
                instructions='Use authorized evidence.',
                models=('local',),
                default_model='local',
            )
        ],
    ).bind('research')


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        yield engine
    finally:
        await engine.close()


def stored(store, request):
    return store.read(request, limit=100).items


async def test_collection_records_real_invocation_and_explicit_buffer_loss(tmp_path, run_request, memory):
    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(True)
            return ToolStep(content='Done')

    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    history = AgentRunHistory(store, run_request, max_events=1)
    try:
        async with history.capture(step_id='find', selection_id='find') as events:
            result = await selected(Model).run(
                'Q', tools=ScopedMemoryTools(memory, 'alpha', scope=run_request.recall_scope()), events=events
            )
        rows = stored(store, run_request)
        assert result.output.text == 'Done' and calls == [True]
        assert rows[0].event.kind == 'collection_started'
        end = rows[-1].event
        assert isinstance(end, AgentCollectionEvent) and end.kind == 'collection_finished'
        assert end.terminal_kind == 'turn_completed' and end.lost_events > 0
        assert end.observed_events + end.lost_events == end.last_sequence
        assert len({row.collection_id for row in rows}) == 1
        assert any(isinstance(row.event, AgentProgressGap) for row in rows)
    finally:
        store.close()


async def test_storage_failure_marks_partial_collection_without_repeating_model(
    tmp_path, run_request, memory, monkeypatch
):
    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(True)
            return ToolStep(content='Done')

    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    original = store._append

    def fail(*args, **kwargs):
        if isinstance(kwargs['event'], AgentProgressEvent):
            raise WorkflowError('history_store_unavailable')
        return original(*args, **kwargs)

    monkeypatch.setattr(store, '_append', fail)
    try:
        async with AgentRunHistory(store, run_request).capture(step_id='find', selection_id='find') as stream:
            result = await selected(Model).run(
                'Q', tools=ScopedMemoryTools(memory, 'alpha', scope=run_request.recall_scope()), events=stream
            )
        end = stored(store, run_request)[-1].event
        assert result.output.text == 'Done' and calls == [True]
        assert end.kind == 'collection_failed' and end.error == 'history_unavailable'
        assert end.terminal_kind == 'turn_completed'
    finally:
        store.close()


async def test_purge_during_collection_cannot_repopulate_history(tmp_path, run_request, memory):
    entered = asyncio.Event()
    release = asyncio.Event()

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            await release.wait()
            return ToolStep(content='Done')

    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)

    async def run():
        async with AgentRunHistory(store, run_request).capture(step_id='find', selection_id='find') as stream:
            return await selected(Model).run(
                'Q', tools=ScopedMemoryTools(memory, 'alpha', scope=run_request.recall_scope()), events=stream
            )

    task = asyncio.create_task(run())
    try:
        await entered.wait()
        store.purge(run_request)
        release.set()
        assert (await task).output.text == 'Done'
        assert store.read(run_request).available is False
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        store.close()


async def test_body_failure_without_invocation_records_incomplete_collection(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        with pytest.raises(ValueError, match='host failure'):
            async with AgentRunHistory(store, run_request).capture(step_id='find', selection_id='find'):
                raise ValueError('host failure')
        end = stored(store, run_request)[-1].event
        assert end.kind == 'collection_failed' and end.error == 'collection_interrupted'
        assert end.terminal_kind is None and end.last_sequence == 0
    finally:
        store.close()


async def test_service_collects_chosen_model_and_history_reads_do_not_execute(setup):
    service, plans, catalog, memory, calls, entered, release, path = setup
    await service.start('alpha', 'observed-run', workflow_id='report', plan_revision=1, question='Question')
    await entered.wait()
    running = await service.history('alpha', 'observed-run')
    assert running.available and running.items[0].event.kind == 'collection_started'
    release.set()
    await service.wait('alpha', 'observed-run')
    finished = await service.history('alpha', 'observed-run')
    assert finished.items[-1].event.kind == 'collection_finished'
    assert {row.event.model_id for row in finished.items if isinstance(row.event, AgentProgressEvent)} == {
        'local'
    }
    count = len(calls)
    assert await service.history('alpha', 'observed-run') == finished
    assert len(calls) == count


async def test_service_history_survives_restart_and_refuses_changed_scope(setup):
    from scone_memory.agents.run_service import AgentRunService
    from scone_memory.retrieval.recall_scope import RecallScope

    service, plans, catalog, memory, calls, entered, release, path = setup
    release.set()
    await service.start('alpha', 'restart', workflow_id='report', plan_revision=1, question='Question')
    await service.wait('alpha', 'restart')
    before = await service.history('alpha', 'restart')
    await service.aclose()
    reopened = AgentRunService(
        path / 'runs',
        key=b'k' * 32,
        catalog=catalog,
        plans=plans,
        memory=memory,
        scope_for=lambda _: RecallScope.validated(),
    )
    try:
        assert await reopened.history('alpha', 'restart') == before
        assert len(calls) == 1
        reopened._scope_for = lambda _: RecallScope.validated(where={'team': 'changed'})
        with pytest.raises(WorkflowError, match='run_scope_changed'):
            await reopened.history('alpha', 'restart')
    finally:
        await reopened.aclose()


async def test_repeated_cancel_waits_for_model_cleanup_before_collection_finishes(
    tmp_path, run_request, memory
):
    entered = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closing.set()
            await release.wait()

    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)

    async def run():
        async with AgentRunHistory(store, run_request).capture(step_id='find', selection_id='find') as stream:
            await selected(Model).run(
                'Q', tools=ScopedMemoryTools(memory, 'alpha', scope=run_request.recall_scope()), events=stream
            )

    task = asyncio.create_task(run())
    try:
        await entered.wait()
        task.cancel()
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not any(
            isinstance(row.event, AgentCollectionEvent) and row.event.kind == 'collection_finished'
            for row in stored(store, run_request)
        )
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        end = stored(store, run_request)[-1].event
        assert end.kind == 'collection_finished' and end.terminal_kind == 'turn_cancelled'
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        store.close()


async def test_parallel_service_tasks_have_distinct_collections(setup):
    from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan

    service, plans, catalog, memory, calls, entered, release, path = setup
    service._parallel = 2
    plan = AgentTaskPlan(
        workflow_id='parallel',
        tasks=tuple(AgentTask(task_id=name, agent_id='research', prompt=name) for name in ('left', 'right')),
    )
    plans.save('alpha', plan, catalog=catalog, expected_revision=0)
    release.set()
    await service.start(
        'alpha', 'parallel-run', workflow_id='parallel', plan_revision=1, question='Q', max_parallel=2
    )
    await service.wait('alpha', 'parallel-run')
    rows = (await service.history('alpha', 'parallel-run', limit=100)).items
    ends = [
        row
        for row in rows
        if isinstance(row.event, AgentCollectionEvent) and row.event.kind == 'collection_finished'
    ]
    assert len(calls) == 2 and {row.step_id for row in ends} == {'left', 'right'}
    assert len({row.collection_id for row in ends}) == 2


async def test_service_history_refuses_deleted_space_and_guard(setup):
    service, plans, catalog, memory, calls, entered, release, path = setup
    release.set()
    await service.start('alpha', 'guarded', workflow_id='report', plan_revision=1, question='Q')
    await service.wait('alpha', 'guarded')

    def refuse():
        raise WorkflowError('scope_switched')

    with pytest.raises(WorkflowError, match='scope_switched'):
        await service.history('alpha', 'guarded', admission_guard=refuse)
    await memory.delete_space('alpha')
    with pytest.raises(WorkflowError, match='space_deleted'):
        await service.history('alpha', 'guarded')
    assert len(calls) == 1


async def test_process_interruption_leaves_unfinished_collection_in_encrypted_history(tmp_path, run_request):
    import subprocess
    import sys

    script = """
import asyncio,os,sys
from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
from scone_memory.agents.catalog import AgentCatalog,AgentDefinition,AgentModel
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.event_history import AgentEventHistoryStore
from scone_memory.agents.history_capture import AgentRunHistory
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
class Model:
    async def complete(self,messages,tools):os._exit(0)
async def main():
    runs=AgentRunStore(sys.argv[1],key=b'k'*32)
    request=runs.get('alpha','run_request-1')
    store=AgentEventHistoryStore(sys.argv[2],key=b'k'*32)
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    catalog=AgentCatalog(models=[AgentModel('local','Local','1',Model)],agents=[AgentDefinition(agent_id='research',instructions='Use authorized evidence.',models=('local',),default_model='local')])
    async with AgentRunHistory(store,request).capture(step_id='find',selection_id='find') as stream:
        await catalog.bind('research').run('Q',tools=ScopedMemoryTools(memory,'alpha',scope=request.recall_scope()),events=stream)
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path / 'runs.db'), str(tmp_path / 'history.db')],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        rows = stored(store, run_request)
        assert rows[0].event.kind == 'collection_started'
        assert any(
            isinstance(row.event, AgentProgressEvent) and row.event.kind == 'operation_started'
            for row in rows
        )
        assert not any(
            isinstance(row.event, AgentCollectionEvent) and row.event.kind != 'collection_started'
            for row in rows
        )
        assert not any(
            isinstance(row.event, AgentProgressEvent) and row.event.kind == 'turn_completed' for row in rows
        )
    finally:
        store.close()


async def test_collector_refuses_changed_workflow_binding_before_model(tmp_path, run_request, memory):
    from scone_memory.agents.workflow import StepContext
    from scone_memory.agents.workflow_approvals import invoke_agent

    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(True)
            return ToolStep(content='Done')

    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    context = StepContext(
        run_request.run_id,
        'alpha',
        {'plan': 'changed', 'recall': run_request.scope, 'exclude_session_id': None},
        run_request.question,
        {},
    )
    try:
        with pytest.raises(WorkflowError, match='history_invocation_mismatch'):
            await invoke_agent(
                selected(Model),
                'Q',
                tools=ScopedMemoryTools(memory, 'alpha', scope=run_request.recall_scope()),
                context=context,
                step_id='find',
                selection_id='find',
                store=None,
                activation_id=None,
                prior=None,
                requirements=None,
                history=AgentRunHistory(store, run_request),
            )
        assert calls == [] and not store.read(run_request).available
    finally:
        store.close()
