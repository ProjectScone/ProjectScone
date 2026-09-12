"""Configured background sync: real sources, explicit recovery and ownership."""
import asyncio

import pytest

from scone_memory.agents.workflow import WorkflowError
from scone_memory.ingestion.directory_service import DirectoryCollection, DirectorySyncService
from tests.ingestion.test_directory_sync import env, runner


def service(env, *, sync=None, **options):
    memory, _, parent = env
    collection = DirectoryCollection('notes', 'Team notes', sync or runner(env), allow_delete_missing=True)
    return DirectorySyncService(parent / 'service', key=b'd' * 32, memory=memory,
                                collections=(collection,), **options)


async def settled(host, run_id='scan'):
    async with asyncio.timeout(5):
        while True:
            status = await host.status('alpha', run_id)
            if not status.active_local:
                return status
            await asyncio.sleep(.001)


async def test_real_lifecycle_and_reopen_do_not_replay(env):
    memory, root, _ = env
    sync = runner(env)
    (root / 'note.txt').write_text('First observatory schedule')
    host = service(env, sync=sync)
    try:
        catalog = await host.catalog('alpha')
        assert catalog[0].collection_id == 'notes' and catalog[0].allow_delete_missing
        assert str(root) not in repr(catalog)
        assert await host.catalog('bravo') == ()
        for run_id, text, outcome in [('add', None, 'added'), ('edit', 'Revised schedule', 'updated'),
                                     ('same', None, 'unchanged'), ('delete', '', 'deleted')]:
            if text == '':
                (root / 'note.txt').unlink()
            elif text is not None:
                (root / 'note.txt').write_text(text)
            await host.start('alpha', run_id, collection_id='notes', delete_missing=True)
            status = await settled(host, run_id)
            assert status.record.status == 'completed'
            assert (await host.result('alpha', run_id)).items[0].source.status == outcome
        calls = sync.parser.calls
        duplicate = await host.start('alpha', 'delete', collection_id='notes', delete_missing=True)
        assert duplicate.record.attempt == 1
    finally:
        await host.aclose()
    host = service(env, sync=sync)
    try:
        assert (await host.status('alpha', 'delete')).record.status == 'completed'
        assert len((await host.list('alpha')).items) == 4
        assert (await host.result('alpha', 'add')).items[0].source.status == 'added'
        assert sync.parser.calls == calls
        assert await host.status('bravo', 'add') is None
    finally:
        await host.aclose()
    # Closing the service does not close its host engine.
    assert await memory.space_deleted('alpha') is None


async def test_other_owner_is_active_and_cannot_be_cancelled_or_duplicated(env):
    _, root, _ = env
    (root / 'note.txt').write_text('A gated source')
    sync = runner(env)
    entered, release = asyncio.Event(), asyncio.Event()
    parse = sync.parser.parse
    async def gated(*args):
        entered.set()
        await release.wait()
        return await parse(*args)
    sync.parser.parse = gated
    first, second = service(env, sync=sync), service(env)
    try:
        await first.start('alpha', 'scan', collection_id='notes')
        await asyncio.wait_for(entered.wait(), 3)
        observed = await second.status('alpha', 'scan')
        assert observed.active_elsewhere and observed.status == 'running'
        assert not observed.active_local and not observed.outcome_unknown
        assert (await second.start('alpha', 'scan', collection_id='notes')).active_elsewhere
        with pytest.raises(WorkflowError, match='sync_owned_elsewhere'):
            await second.cancel('alpha', 'scan', expected_revision=observed.record.revision)
        with pytest.raises(WorkflowError, match='sync_busy'):
            await second.start('alpha', 'another', collection_id='notes')
        assert await second.request('alpha', 'another') is None
        with pytest.raises(WorkflowError, match='sync_busy'):
            await first.start('alpha', 'another', collection_id='notes')
        release.set()
        assert (await settled(first)).status == 'completed'
    finally:
        release.set()
        await first.aclose()
        await second.aclose()


async def test_cancel_shutdown_and_explicit_resume_preserve_intent(env):
    _, root, _ = env
    (root / 'note.txt').write_text('Recoverable source')
    sync = runner(env)
    entered = asyncio.Event()
    parse = sync.parser.parse
    async def gated(*args):
        entered.set()
        await asyncio.Event().wait()
    sync.parser.parse = gated
    host = service(env, sync=sync)
    await host.start('alpha', 'scan', collection_id='notes')
    await asyncio.wait_for(entered.wait(), 3)
    before = await host.status('alpha', 'scan')
    cancelled = await host.cancel('alpha', 'scan', expected_revision=before.record.revision)
    assert cancelled.record.cancel_requested_at is not None
    await host.aclose()
    sync.parser.parse = parse
    host = service(env, sync=sync)
    try:
        before = await host.status('alpha', 'scan')
        assert before.status == 'cancelled' and before.outcome_unknown
        repeat = await host.start('alpha', 'scan', collection_id='notes')
        assert repeat.record == before.record and not repeat.active_local
        await host.resume('alpha', 'scan', expected_revision=before.record.revision)
        assert (await settled(host)).record.attempt == 2
        assert (await host.result('alpha', 'scan')).items[0].source.status == 'added'
    finally:
        await host.aclose()


async def test_configuration_policy_and_admission_guard(env):
    memory, _, parent = env
    sync = runner(env)
    restricted = DirectoryCollection('notes', 'Notes', sync)
    host = DirectorySyncService(parent / 'service', key=b'd' * 32, memory=memory, collections=(restricted,))
    try:
        with pytest.raises(WorkflowError, match='sync_delete_forbidden'):
            await host.start('alpha', 'scan', collection_id='notes', delete_missing=True)
        with pytest.raises(WorkflowError, match='sync_collection_not_found'):
            await host.start('bravo', 'scan', collection_id='notes')
        def revoked():
            raise WorkflowError('credential_revoked')
        with pytest.raises(WorkflowError, match='credential_revoked'):
            await host.start('alpha', 'scan', collection_id='notes', admission_guard=revoked)
        assert await host.request('alpha', 'scan') is None
        with pytest.raises(ValueError, match='duplicate'):
            DirectorySyncService(parent / 'duplicate', key=b'd' * 32, memory=memory,
                collections=(restricted, DirectoryCollection('alias', 'Alias', runner(env))))
    finally:
        await host.aclose()


async def test_changed_configuration_refuses_resume_but_history_stays_readable(env):
    _, root, _ = env
    (root / 'note.txt').write_text('A source')
    host = service(env)
    await host.start('alpha', 'scan', collection_id='notes')
    # Immediate close exercises cancellation before a worker's first instruction.
    await host.aclose()
    host = service(env, sync=runner(env, parser_revision='v2'))
    try:
        record = await host.request('alpha', 'scan')
        assert record is not None and record.attempt == 1
        with pytest.raises(WorkflowError, match='sync_configuration_changed'):
            await host.resume('alpha', 'scan', expected_revision=record.revision)
    finally:
        await host.aclose()


async def test_interrupted_disk_record_is_passive_until_explicit_resume(env):
    from scone_memory.ingestion.directory_runs import DirectoryRunStore
    from scone_memory.ingestion.directory_run_types import SyncRunSpec
    _, root, parent = env
    (root / 'note.txt').write_text('Resume a journal after owner exit')
    sync = runner(env)
    host = service(env, sync=sync)
    config = (await host.catalog('alpha'))[0].configuration
    await host.aclose()
    registry = DirectoryRunStore(parent / 'service' / 'runs.sqlite', key=b'd' * 32)
    saved = registry.register('alpha', 'scan', SyncRunSpec(collection_id='notes', configuration=config))
    registry.start_attempt('alpha', 'scan', expected_revision=saved.revision)
    registry.close()
    host = service(env, sync=sync)
    try:
        record = await host.status('alpha', 'scan')
        assert record.status == 'interrupted' and record.outcome_unknown
        assert sync.parser.calls == 0
        await host.resume('alpha', 'scan', expected_revision=record.record.revision)
        assert (await settled(host)).status == 'completed'
        assert sync.parser.calls == 1
    finally:
        await host.aclose()


async def test_shutdown_before_worker_start_is_reported_as_interruption(env):
    host = service(env)
    await host.start('alpha', 'scan', collection_id='notes')
    await host.aclose()
    host = service(env)
    try:
        status = await host.status('alpha', 'scan')
        assert status.status == 'interrupted' and status.outcome_unknown
        assert status.record.error_code == 'sync_interrupted'
    finally:
        await host.aclose()


async def test_deadline_is_durable_and_never_automatically_retried(env):
    _, root, _ = env
    (root / 'note.txt').write_text('Provider deadline source')
    sync = runner(env)
    async def gated(*args):
        await asyncio.Event().wait()
    sync.parser.parse = gated
    host = service(env, sync=sync, deadline_s=.02)
    try:
        await host.start('alpha', 'scan', collection_id='notes')
        status = await settled(host)
        assert status.status == 'failed' and status.record.error_code == 'sync_deadline'
        assert status.record.attempt == 1 and status.outcome_unknown
        with pytest.raises(WorkflowError, match='sync_result_unavailable'):
            await host.result('alpha', 'scan')
        assert (await host.start('alpha', 'scan', collection_id='notes')).record.attempt == 1
    finally:
        await host.aclose()


async def test_deleted_space_revokes_catalog_admission_and_history(env):
    memory, _, _ = env
    host = service(env)
    try:
        await host.start('alpha', 'scan', collection_id='notes')
        await settled(host)
        await memory.delete_space('alpha')
        for operation in (lambda: host.catalog('alpha'), lambda: host.list('alpha'),
                          lambda: host.request('alpha', 'scan'), lambda: host.status('alpha', 'scan'),
                          lambda: host.result('alpha', 'scan'),
                          lambda: host.start('alpha', 'new', collection_id='notes')):
            with pytest.raises(WorkflowError, match='space_deleted'):
                await operation()
    finally:
        await host.aclose()


async def test_local_capacity_is_shared_across_different_collections(env):
    memory, root, parent = env
    other_root = parent / 'other-root'
    other_root.mkdir()
    (root / 'note.txt').write_text('Hold one source')
    sync = runner(env)
    entered, release = asyncio.Event(), asyncio.Event()
    parse = sync.parser.parse
    async def gated(*args):
        entered.set()
        await release.wait()
        return await parse(*args)
    sync.parser.parse = gated
    other = runner((memory, other_root, parent), journal=parent / 'other.journal')
    host = DirectorySyncService(parent / 'service', key=b'd' * 32, memory=memory, max_active=1,
        collections=(DirectoryCollection('one', 'One', sync), DirectoryCollection('two', 'Two', other)))
    try:
        await host.start('alpha', 'scan', collection_id='one')
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(WorkflowError, match='sync_busy'):
            await host.start('alpha', 'second', collection_id='two')
        assert await host.request('alpha', 'second') is None
        release.set()
        await settled(host)
        await host.start('alpha', 'second', collection_id='two')
        assert (await settled(host, 'second')).status == 'completed'
    finally:
        release.set()
        await host.aclose()


async def test_cancel_storage_failure_still_stops_owned_worker(env, monkeypatch):
    _, root, _ = env
    (root / 'note.txt').write_text('Cancellation must reach worker')
    sync = runner(env)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    parse = sync.parser.parse
    async def gated(*args):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return await parse(*args)
    sync.parser.parse = gated
    host = service(env, sync=sync)
    try:
        await host.start('alpha', 'scan', collection_id='notes')
        await asyncio.wait_for(entered.wait(), 3)
        current = await host.status('alpha', 'scan')
        with pytest.raises(WorkflowError, match='sync_request_conflict'):
            await host.cancel('alpha', 'scan', expected_revision=current.record.revision - 1)
        assert not cancelled.is_set()
        def broken(*args, **kwargs):
            raise OSError('unavailable cancellation storage')
        monkeypatch.setattr(host._runs, 'request_cancel', broken)
        with pytest.raises(OSError):
            await host.cancel('alpha', 'scan', expected_revision=current.record.revision)
        await asyncio.sleep(0)
        assert cancelled.is_set()
        assert (await settled(host)).status == 'interrupted'
    finally:
        release.set()
        await host.aclose()


async def test_shutdown_caller_cancellation_does_not_interrupt_worker_drain(env):
    _, root, _ = env
    (root / 'note.txt').write_text('Drain provider cleanup')
    sync = runner(env)
    entered, draining, release, drained = (asyncio.Event() for _ in range(4))
    async def gated(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            await release.wait()
            drained.set()
            raise
    sync.parser.parse = gated
    host = service(env, sync=sync)
    await host.start('alpha', 'scan', collection_id='notes')
    await asyncio.wait_for(entered.wait(), 3)
    closing = asyncio.create_task(host.aclose())
    try:
        await asyncio.wait_for(draining.wait(), 3)
        closing.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done() and not host._closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert drained.is_set() and host._closed and host._runs._storage._closed
        assert not host._tasks and not host._owners
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        await host.aclose()


@pytest.mark.parametrize('change', ['space', 'memory', 'parser'])
async def test_host_context_changes_are_refused_before_admission(env, change):
    sync = runner(env)
    host = service(env, sync=sync)
    original = getattr(sync, change)
    try:
        setattr(sync, change, 'bravo' if change == 'space' else object())
        with pytest.raises(WorkflowError, match='sync_configuration_changed'):
            await host.start('alpha', 'scan', collection_id='notes')
        assert await host.request('alpha', 'scan') is None
    finally:
        setattr(sync, change, original)
        await host.aclose()


async def test_registration_race_does_not_report_its_own_admission_lock_as_another_owner(env, monkeypatch):
    from scone_memory.ingestion.directory_runs import DirectoryRunStore
    _, _, parent = env
    host = service(env)
    other = DirectoryRunStore(parent / 'service' / 'runs.sqlite', key=b'd' * 32)
    register = host._runs.register
    def raced(space, run_id, spec):
        prior = other.register(space, run_id, spec)
        other.start_attempt(space, run_id, expected_revision=prior.revision)
        return register(space, run_id, spec)
    monkeypatch.setattr(host._runs, 'register', raced)
    try:
        status = await host.start('alpha', 'scan', collection_id='notes')
        assert status.status == 'interrupted' and status.outcome_unknown
        assert not status.active_elsewhere and not status.active_local
        assert status.record.attempt == 1
    finally:
        other.close()
        await host.aclose()


async def test_admitted_execution_cannot_be_redirected_to_another_space(env):
    memory, root, _ = env
    (root / 'note.txt').write_text('Keep this source in its admitted space')
    sync = runner(env)
    entered, release = asyncio.Event(), asyncio.Event()
    parse = sync.parser.parse
    async def gated(*args):
        entered.set()
        await release.wait()
        return await parse(*args)
    sync.parser.parse = gated
    host = service(env, sync=sync)
    try:
        await host.start('alpha', 'scan', collection_id='notes')
        await asyncio.wait_for(entered.wait(), 3)
        sync.space = 'bravo'
        release.set()
        status = await settled(host)
        assert (await memory.documents.counts('bravo')).episodes == 0
        assert (await memory.documents.counts('alpha')).episodes == 1
        assert status.status == 'failed' and status.record.error_code == 'sync_configuration_changed'
        with pytest.raises(WorkflowError, match='sync_result_unavailable'):
            await host.result('alpha', 'scan')
        sync.space = 'alpha'
        await host.resume('alpha', 'scan', expected_revision=status.record.revision)
        assert (await settled(host)).status == 'completed'
        assert (await host.result('alpha', 'scan')).items[0].source.status == 'unchanged'
    finally:
        sync.space = 'alpha'
        release.set()
        await host.aclose()
