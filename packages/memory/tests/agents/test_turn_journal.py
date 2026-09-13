"""Durable operation identity and non-replay boundaries for native agent turns."""
import asyncio

import pytest

from scone_memory.agents.turn_journal import ToolTurnJournal, TurnJournalError
from scone_memory.agents.workflow import StepCheckpoints


def checkpoints():
    records = {}
    def put(key, value):
        records[key] = value
    return StepCheckpoints(records.get, put), records


async def test_completed_operations_replay_detached_without_dispatch():
    storage, records = checkpoints()
    calls = []
    response = {'content': 'original', 'calls': []}
    async def model():
        calls.append(1)
        return response
    first = ToolTurnJournal(storage, binding={'model': 'chosen-v1'})
    result = await first.execute('model', {'messages': ['question']}, model)
    result['content'] = 'caller mutation'
    response['content'] = 'provider mutation'
    first.pause()
    second = ToolTurnJournal(storage, binding={'model': 'chosen-v1'})
    assert await second.execute('model', {'messages': ['question']}, model) == {'content': 'original', 'calls': []}
    second.finish()
    assert calls == [1] and len(records) == 1


@pytest.mark.parametrize('change', ['binding', 'request', 'kind'])
async def test_changed_operation_never_dispatches(change):
    storage, _ = checkpoints()
    async def operation():
        return 'saved'
    first = ToolTurnJournal(storage, binding={'revision': 1})
    await first.execute('model', {'prompt': 'original'}, operation)
    first.pause()
    with pytest.raises(TurnJournalError, match='binding_mismatch|operation_mismatch'):
        second = ToolTurnJournal(storage, binding={'revision': 2 if change == 'binding' else 1})
        await second.execute('custom' if change == 'kind' else 'model',
            {'prompt': 'changed' if change == 'request' else 'original'}, operation)


@pytest.mark.parametrize('cancel', [False, True])
async def test_started_operation_blocks_replay_before_any_callback(cancel):
    storage, _ = checkpoints()
    calls = []
    async def operation():
        calls.append(1)
        if cancel:
            raise asyncio.CancelledError()
        raise ValueError('private operation exception')
    first = ToolTurnJournal(storage, binding={})
    with pytest.raises((TurnJournalError, asyncio.CancelledError)):
        await first.execute('custom', {'arguments': {}}, operation)
    with pytest.raises(TurnJournalError, match='outcome_unknown'):
        ToolTurnJournal(storage, binding={})
    assert calls == [1]


async def test_finished_journal_refuses_extra_operation_and_unread_suffix():
    storage, _ = checkpoints()
    async def operation():
        return 'saved'
    first = ToolTurnJournal(storage, binding={})
    await first.execute('model', {}, operation)
    first.finish()
    second = ToolTurnJournal(storage, binding={})
    with pytest.raises(TurnJournalError, match='unread_operations'):
        second.finish()
    assert await second.execute('model', {}, operation) == 'saved'
    with pytest.raises(TurnJournalError, match='turn_finished'):
        await second.execute('custom', {}, operation)


async def test_stale_instance_cannot_overwrite_new_operation():
    storage, _ = checkpoints()
    first = ToolTurnJournal(storage, binding={})
    second = ToolTurnJournal(storage, binding={})
    async def operation():
        return None
    await first.execute('model', {}, operation)
    with pytest.raises(TurnJournalError, match='journal_changed'):
        await second.execute('model', {}, operation)


async def test_failed_start_write_does_not_dispatch():
    storage, records = checkpoints()
    fail = False
    def put(key, value):
        if fail:
            raise OSError('private storage fault')
        storage.put(key, value)
    calls = []
    async def operation():
        calls.append(1)
        return None
    journal = ToolTurnJournal(StepCheckpoints(storage.get, put), binding={})
    fail = True
    with pytest.raises(TurnJournalError, match='journal_unavailable'):
        await journal.execute('model', {}, operation)
    assert not calls
    assert ToolTurnJournal(storage, binding={})


@pytest.mark.parametrize('field,value', [('max_operations', True), ('max_operations', 0), ('max_operations', 65),
    ('max_bytes', 1023), ('max_bytes', 9*1024*1024), ('max_new_operations', False), ('max_new_operations', 0)])
async def test_configuration_bounds_are_strict(field, value):
    storage, _ = checkpoints()
    with pytest.raises(TurnJournalError, match='invalid_journal_configuration'):
        ToolTurnJournal(storage, binding={}, **{field: value})


async def test_scheduling_yield_has_no_started_operation_and_can_resume():
    from scone_memory.agents.turn_journal import TurnJournalPaused
    import json
    storage, records = checkpoints()
    calls = []
    async def operation():
        calls.append(1)
        return 'done'
    first = ToolTurnJournal(storage, binding={}, max_new_operations=1)
    await first.execute('model', {}, operation)
    with pytest.raises(TurnJournalPaused):
        await first.execute('custom', {}, operation)
    state = json.loads(next(iter(records.values())))
    assert len(state['events']) == 1 and state['events'][0]['status'] == 'completed'
    second = ToolTurnJournal(storage, binding={}, max_new_operations=1)
    await second.execute('model', {}, operation)
    await second.execute('custom', {}, operation)
    second.finish()
    assert calls == [1, 1]


async def test_total_operation_cap_and_result_byte_cap_remain_effective():
    storage, _ = checkpoints()
    calls = []
    async def operation():
        calls.append(1)
        return 'done'
    journal = ToolTurnJournal(storage, binding={}, max_operations=1)
    await journal.execute('model', {}, operation)
    with pytest.raises(TurnJournalError, match='operation_limit'):
        await journal.execute('custom', {}, operation)
    assert calls == [1]
    fresh, _ = checkpoints()
    journal = ToolTurnJournal(fresh, binding={}, max_bytes=1024)
    async def large():
        return 'x' * 1025
    with pytest.raises(TurnJournalError):
        await journal.execute('custom', {}, large)
    with pytest.raises(TurnJournalError, match='outcome_unknown'):
        ToolTurnJournal(fresh, binding={}, max_bytes=1024)


async def test_expired_checkpoint_handle_cannot_replay_cached_operation():
    from scone_memory.agents.workflow import WorkflowError
    original, _ = checkpoints()
    active = True
    def get(key):
        if not active:
            raise WorkflowError('checkpoint_inactive')
        return original.get(key)
    journal = ToolTurnJournal(StepCheckpoints(get, original.put), binding={})
    async def operation():
        return 'done'
    await journal.execute('model', {}, operation)
    active = False
    with pytest.raises(TurnJournalError, match='journal_unavailable'):
        journal.pause()
