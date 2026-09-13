"""Independent recovery regressions promoted from boundary review."""

import asyncio
import pytest
from scone_memory.agents.turn_journal import ToolTurnJournal, TurnJournalError
from scone_memory.agents.workflow import StepCheckpoints


def storage():
    records = {}

    def put(key, value):
        records[key] = value

    return StepCheckpoints(records.get, put), records


@pytest.mark.parametrize('left,right', [(True, 1), (False, 0), (1, 1.0)])
async def test_binding_typed_identity(left, right):
    points, _ = storage()
    ToolTurnJournal(points, binding={'selected': left}).pause()
    with pytest.raises(TurnJournalError, match='binding_mismatch'):
        ToolTurnJournal(points, binding={'selected': right})


async def test_changed_operation_after_await_does_not_mutate_identity():
    points, _ = storage()
    request = {'arguments': {'number': 1}}

    async def operation():
        request['arguments']['number'] = 2
        await asyncio.sleep(0)
        return {'saved': True}

    first = ToolTurnJournal(points, binding={})
    await first.execute('custom', request, operation)
    first.pause()
    second = ToolTurnJournal(points, binding={})

    async def forbidden():
        pytest.fail('completed operation dispatched')

    assert await second.execute('custom', {'arguments': {'number': 1}}, forbidden) == {'saved': True}


async def test_post_completion_write_failure_replays_without_effect():
    points, records = storage()
    fail = False
    writes = 0

    def put(key, value):
        nonlocal writes
        points.put(key, value)
        if fail:
            writes += 1
            if writes == 2:
                raise OSError('private late acknowledgement failure')

    calls = []

    async def operation():
        calls.append(1)
        return {'value': 'saved'}

    first = ToolTurnJournal(StepCheckpoints(points.get, put), binding={})
    fail = True
    with pytest.raises(TurnJournalError):
        await first.execute('custom', {}, operation)
    assert calls == [1]
    second = ToolTurnJournal(points, binding={})
    assert await second.execute('custom', {}, operation) == {'value': 'saved'}
    assert calls == [1]


async def test_cancel_swallow_does_not_complete_operation():
    points, _ = storage()
    started = asyncio.Event()
    calls = []

    async def operation():
        calls.append(1)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return {'unsafe': 'late'}

    first = ToolTurnJournal(points, binding={})
    task = asyncio.create_task(first.execute('custom', {}, operation))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(TurnJournalError, match='outcome_unknown'):
        ToolTurnJournal(points, binding={})
    assert calls == [1]


async def test_concurrent_operation_never_dispatches_second():
    points, _ = storage()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def first_op():
        calls.append('first')
        started.set()
        await release.wait()
        return 'first'

    async def second_op():
        calls.append('second')
        return 'second'

    journal = ToolTurnJournal(points, binding={})
    running = asyncio.create_task(journal.execute('custom', {'id': 1}, first_op))
    await started.wait()
    try:
        with pytest.raises(TurnJournalError):
            await journal.execute('custom', {'id': 2}, second_op)
    finally:
        release.set()
        await running
    assert calls == ['first']


@pytest.mark.parametrize('stage', ['read', 'write'])
async def test_actual_sqlite_storage_errors_are_normalized(stage):
    import sqlite3

    secret = 'private sql / filesystem detail'

    def get(key):
        if stage == 'read':
            raise sqlite3.OperationalError(secret)
        return None

    def put(key, value):
        raise sqlite3.OperationalError(secret)

    with pytest.raises(TurnJournalError, match='journal_unavailable'):
        journal = ToolTurnJournal(StepCheckpoints(get, put), binding={})
        journal.pause()


async def test_huge_timeout_is_controlled_refusal():
    points, _ = storage()
    with pytest.raises(TurnJournalError, match='invalid_journal_configuration'):
        ToolTurnJournal(points, binding={}, timeout_s=10**10000)


async def test_persisted_excessive_result_depth_refused_at_open():
    import json

    points, records = storage()

    async def op():
        return 'valid'

    first = ToolTurnJournal(points, binding={})
    await first.execute('model', {}, op)
    key = next(iter(records))
    state = json.loads(records[key])
    nested = 'private deep result'
    for _ in range(40):
        nested = [nested]
    state['events'][0]['result'] = nested
    records[key] = json.dumps(state).encode()
    with pytest.raises(TurnJournalError, match='journal_integrity'):
        ToolTurnJournal(points, binding={})


async def test_cached_result_decode_must_obey_active_deadline(monkeypatch):
    import time
    import scone_memory.agents.turn_journal as module

    points, _ = storage()

    async def op():
        return {'cached': 'yes'}

    first = ToolTurnJournal(points, binding={}, timeout_s=20.0)
    await first.execute('model', {}, op)
    first.pause()
    second = ToolTurnJournal(points, binding={}, timeout_s=20.0)
    original = module._bytes

    def slow(value, maximum):
        result = original(value, maximum)
        if value == {'cached': 'yes'}:
            second._base_elapsed = 21.0
        return result

    monkeypatch.setattr(module, '_bytes', slow)
    with pytest.raises(TurnJournalError, match='deadline'):
        await second.execute('model', {}, op)
