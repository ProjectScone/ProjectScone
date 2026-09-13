import asyncio
import pytest
from scone_memory.agents.event_history import AgentEventHistoryStore
from scone_memory.agents.history_capture import AgentRunHistory, _Collection
from scone_memory.agents.history_models import AgentCollectionEvent
from scone_memory.agents.progress import AgentProgressEvent
from scone_memory.agents.workflow import WorkflowError
from tests.agents.test_event_history import run_request, events
from tests.agents.test_history_collection import memory, selected, stored
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.integrations.scoped_tools import ScopedMemoryTools


async def run_model(store, request, memory):
    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    async with AgentRunHistory(store, request).capture(step_id='find', selection_id='find') as stream:
        return await selected(Model).run(
            'Q', tools=ScopedMemoryTools(memory, 'alpha', scope=request.recall_scope()), events=stream
        )


@pytest.mark.parametrize('purge', [False, True])
async def test_first_commit_lost_ack_never_retries(tmp_path, run_request, memory, monkeypatch, purge):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    original = store._append
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(kwargs['event'])
        original(*args, **kwargs)
        if purge:
            store.purge(run_request)
        raise OSError('PRIVATE-credential')

    monkeypatch.setattr(store, '_append', uncertain)
    try:
        assert (await run_model(store, run_request, memory)).output.text == 'Done'
        assert len(calls) == 1
        rows = stored(store, run_request)
        assert len(rows) == (0 if purge else 1)
        assert not rows or rows[0].event.kind == 'collection_started'
    finally:
        store.close()


async def test_purge_and_new_generation_remain_separate(tmp_path, run_request, monkeypatch):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        async with AgentRunHistory(store, run_request).capture(step_id='find', selection_id='find') as stream:
            emitter = stream._begin('research', 'local', run_request.plan.bindings['find'])
            await asyncio.sleep(0)
            store.purge(run_request)
            replacement = (await events())[0]
            store.append(run_request, step_id='find', selection_id='find', event=replacement)
            emitter.finish('turn_completed')
        rows = stored(store, run_request)
        assert len(rows) == 1 and rows[0].event == replacement
    finally:
        store.close()


async def test_repeated_cancel_joins_owned_reader(tmp_path, run_request, monkeypatch):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    original = _Collection.consume

    async def slow(self, stream):
        entered.set()
        await release.wait()
        await original(self, stream)
        finished.set()

    monkeypatch.setattr(_Collection, 'consume', slow)

    async def run():
        async with AgentRunHistory(store, run_request).capture(step_id='find', selection_id='find') as stream:
            emitter = stream._begin('research', 'local', run_request.plan.bindings['find'])
            emitter.finish('turn_completed')

    task = asyncio.create_task(run())
    try:
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert stored(store, run_request)[-1].event.kind == 'collection_finished'
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        store.close()


async def test_storage_private_exception_is_not_logged(tmp_path, run_request, memory, monkeypatch):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    original = store._append
    loop = asyncio.get_running_loop()
    observed = []
    handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: observed.append(context))

    def fail(*args, **kwargs):
        if isinstance(kwargs['event'], AgentProgressEvent):
            raise RuntimeError('PRIVATE-credential')
        return original(*args, **kwargs)

    monkeypatch.setattr(store, '_append', fail)
    try:
        assert (await run_model(store, run_request, memory)).output.text == 'Done'
        await asyncio.sleep(0)
        assert not observed
        end = stored(store, run_request)[-1].event
        assert end.kind == 'collection_failed' and end.error == 'history_unavailable'
        assert 'PRIVATE-credential' not in repr(stored(store, run_request))
    finally:
        loop.set_exception_handler(handler)
        store.close()


from tests.agents.test_run_service import setup


async def test_late_history_marker_cannot_publish_after_workflow_deadline(setup, monkeypatch):
    import time

    service, plans, catalog, memory, calls, entered, release, path = setup
    service._deadline = 0.03
    original = service._history._append

    def slow(*args, **kwargs):
        if (
            isinstance(kwargs['event'], AgentCollectionEvent)
            and kwargs['event'].kind == 'collection_finished'
        ):
            time.sleep(0.07)
        return original(*args, **kwargs)

    monkeypatch.setattr(service._history, '_append', slow)
    release.set()
    await service.start('alpha', 'late-marker', workflow_id='report', plan_revision=1, question='Q')
    status = await service.wait('alpha', 'late-marker')
    assert status.status == 'deadline' and status.outcome_unknown
    assert not status.completed_steps and len(calls) == 1
    with pytest.raises(WorkflowError):
        await service.result('alpha', 'late-marker')


async def test_source_revalidation_after_reader_drain(setup, monkeypatch):
    service, plans, catalog, memory, calls, entered, release, path = setup
    original = _Collection.consume

    async def drain_then_delete(self, stream):
        await original(self, stream)
        await memory.delete_space('alpha')

    monkeypatch.setattr(_Collection, 'consume', drain_then_delete)
    release.set()
    await service.start('alpha', 'deleted-tail', workflow_id='report', plan_revision=1, question='Q')
    tasks = tuple(service._tasks.values())
    await asyncio.gather(*tasks)
    request = service._runs.get('alpha', 'deleted-tail')
    workflow = service._open(request)
    try:
        progress = workflow.status(request.run_id, request.question)
        assert progress.status == 'sources_invalid' and not progress.completed_steps
    finally:
        workflow.close()
    assert len(calls) == 1


async def test_cancellation_during_final_marker_cannot_commit_success(setup, monkeypatch):
    service, plans, catalog, memory, calls, entered, release, path = setup
    original = service._history._append

    def cancel(*args, **kwargs):
        if (
            isinstance(kwargs['event'], AgentCollectionEvent)
            and kwargs['event'].kind == 'collection_finished'
        ):
            asyncio.current_task().cancel()
        return original(*args, **kwargs)

    monkeypatch.setattr(service._history, '_append', cancel)
    release.set()
    await service.start('alpha', 'cancel-marker', workflow_id='report', plan_revision=1, question='Q')
    try:
        await service.wait('alpha', 'cancel-marker')
    except asyncio.CancelledError:
        pass
    status = await service.status('alpha', 'cancel-marker')
    assert status.status in ('cancelled', 'failed') and status.outcome_unknown
    assert status.error_class == 'CancelledError'
    assert not status.completed_steps and len(calls) == 1
    with pytest.raises(WorkflowError):
        await service.result('alpha', 'cancel-marker')


async def test_late_collection_start_does_not_dispatch_model(setup, monkeypatch):
    import time

    service, plans, catalog, memory, calls, entered, release, path = setup
    service._deadline = 0.03
    original = service._history._append

    def slow(*args, **kwargs):
        if isinstance(kwargs['event'], AgentCollectionEvent) and kwargs['event'].kind == 'collection_started':
            time.sleep(0.07)
        return original(*args, **kwargs)

    monkeypatch.setattr(service._history, '_append', slow)
    release.set()
    await service.start('alpha', 'late-start', workflow_id='report', plan_revision=1, question='Q')
    status = await service.wait('alpha', 'late-start')
    assert status.status == 'deadline'
    assert calls == []


async def test_cancelled_service_shutdown_closes_history_after_reader_join(setup, monkeypatch):
    service, plans, catalog, memory, calls, entered, release, path = setup
    drained = asyncio.Event()
    finish = asyncio.Event()
    original = _Collection.consume

    async def stalled(self, stream):
        await original(self, stream)
        drained.set()
        await finish.wait()

    monkeypatch.setattr(_Collection, 'consume', stalled)
    await service.start('alpha', 'closing', workflow_id='report', plan_revision=1, question='Q')
    await entered.wait()
    closing = asyncio.create_task(service.aclose())
    try:
        await drained.wait()
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        finish.set()
        try:
            await closing
        except asyncio.CancelledError:
            pass
        assert not service._tasks and not service._owners
        assert service._closed
        assert service._history._storage._closed
        assert service._runs._storage._closed
    finally:
        finish.set()
        await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.parametrize(
    'case',
    [
        test_late_history_marker_cannot_publish_after_workflow_deadline,
        test_cancellation_during_final_marker_cannot_commit_success,
        test_late_collection_start_does_not_dispatch_model,
    ],
    ids=['deadline-tail', 'cancel-tail', 'deadline-start'],
)
async def test_parallel_deadline_and_cancel_fences(setup, monkeypatch, case):
    service = setup[0]
    service._parallel = 2
    original = service.start

    async def parallel(*args, **kwargs):
        kwargs['max_parallel'] = 2
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, 'start', parallel)
    await case(setup, monkeypatch)


@pytest.mark.parametrize(
    'changes',
    [
        {'collection_id': 'b' * 32},
        {'occurred_at': '2026-09-13T00:00:00'},
        {'last_sequence': True},
        {'observed_events': 1},
        {'terminal_kind': 'turn_completed'},
        {'kind': 'collection_finished'},
        {'kind': 'collection_failed'},
        {'error': 'PRIVATE-credential'},
    ],
)
async def test_invalid_collection_metadata_is_refused_without_mutation(tmp_path, run_request, changes):
    from dataclasses import replace

    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    marker = AgentCollectionEvent('collection_started', 'a' * 32, '2026-09-13T00:00:00+00:00')
    try:
        with pytest.raises(WorkflowError) as error:
            store._append(
                run_request,
                step_id='find',
                selection_id='find',
                event=replace(marker, **changes),
                collection_id=marker.collection_id,
            )
        assert 'PRIVATE-credential' not in str(error.value)
        assert store.read(run_request).available is False
    finally:
        store.close()
