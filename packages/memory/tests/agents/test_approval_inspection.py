"""Only the exact next saved call can be presented as an actionable approval."""
from dataclasses import replace

import pytest

from scone_memory.agents.approval_inspection import inspect_tool_approval
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.workflow import WorkflowError, WorkflowPausableStep, WorkflowRunner
from .test_custom_tools import call
from .test_evidence_tool_loop import search
from .test_task_workflow import memory
from .test_tool_approval_recovery import host


def reader(host, tmp_path):
    return WorkflowRunner(tmp_path/'workflow', key=b'k'*32,
        steps=[WorkflowPausableStep('count', '1', host.execute)], source_verifier=host.valid)


async def test_inspection_matches_second_call_in_same_saved_model_batch(host, tmp_path):
    host.model.steps = [ToolStep(calls=(*call().calls, *call(call_id='second').calls)), ToolStep(content='Done.')]
    await host.run()
    first = host.activate()
    await host.run()
    pending = next(record for record in host.store.list('alpha', 'one') if record.revision == 1)
    job = reader(host, tmp_path)
    try:
        snapshot, = await job.inspect_pauses('one', space='alpha', scope={}, inputs='Count')
        before = (tmp_path/'workflow').read_bytes()
        await inspect_tool_approval(snapshot, pending, host.agent, host.scoped)
        assert len(host.model.requests) == 1 and host.effects == [3]
        assert (tmp_path/'workflow').read_bytes() == before
        with pytest.raises(WorkflowError):
            await inspect_tool_approval(snapshot, first, host.agent, host.scoped)
    finally:
        job.close()


@pytest.mark.parametrize('change', ['arguments', 'operation', 'tool', 'step', 'binding'])
async def test_changed_call_cannot_be_presented_as_the_pending_proposal(host, tmp_path, change):
    await host.run()
    pending, = host.store.list('alpha', 'one')
    updates = {'arguments': {'arguments_json': '{"count":4}'}, 'operation': {'operation_digest': 'f'*64},
               'tool': {'tool_digest': 'f'*64}, 'step': {'step_id': 'other'}, 'binding': {'binding': 'f'*64}}
    pending = pending.model_copy(update={'call': pending.call.model_copy(update=updates[change])})
    job = reader(host, tmp_path)
    try:
        snapshot, = await job.inspect_pauses('one', space='alpha', scope={}, inputs='Count')
        with pytest.raises(WorkflowError):
            await inspect_tool_approval(snapshot, pending, host.agent, host.scoped)
        assert len(host.model.requests) == 1 and host.effects == []
    finally:
        job.close()


async def test_inspection_restores_sources_without_search_and_refuses_revocation(host, memory, tmp_path):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    host.model.steps = [search(), call()]
    await host.run()
    pending, = host.store.list('alpha', 'one')
    async def forbidden(*args, **kwargs):
        pytest.fail('inspection performed a new search')
    host.scoped.run = forbidden
    job = reader(host, tmp_path)
    try:
        snapshot, = await job.inspect_pauses('one', space='alpha', scope={}, inputs='Count')
        await inspect_tool_approval(snapshot, pending, host.agent, host.scoped)
        await memory.documents.delete_episode('alpha', episode.episode_id)
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await inspect_tool_approval(snapshot, pending, host.agent, host.scoped)
        assert host.effects == [] and len(host.model.requests) == 2
    finally:
        job.close()


async def test_consumed_pause_during_source_restore_cannot_acknowledge_old_request(host, memory, tmp_path):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    host.model.steps = [search(), call(), ToolStep(content='Done.')]
    await host.run()
    pending = host.activate()
    job = reader(host, tmp_path)
    try:
        snapshot, = await job.inspect_pauses('one', space='alpha', scope={}, inputs='Count')
        restore = host.scoped.restore
        changed = False
        async def advance(*args, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                await host.run()
            return await restore(*args, **kwargs)
        host.scoped.restore = advance
        with pytest.raises(WorkflowError, match='pause_snapshot_changed'):
            await inspect_tool_approval(snapshot, pending, host.agent, host.scoped)
        assert host.effects == [3]
    finally:
        job.close()
