"""Interrupted steps retain encrypted work without issuing a completed receipt."""
import asyncio
import sqlite3

import pytest

from scone_memory.agents import WorkflowError, WorkflowStep
from .test_agent_workflows import runner


async def test_checkpoint_survives_restart_and_is_scoped_to_its_step(tmp_path):
    path = tmp_path / 'work.db'
    handles = []
    async def first(context):
        handles.append(context.checkpoints)
        saved = context.checkpoints.get('batch-1')
        if saved is None:
            context.checkpoints.put('batch-1', b'private completed embedding')
            raise asyncio.CancelledError()
        assert saved == b'private completed embedding'
        return 'first done'
    async def second(context):
        assert context.checkpoints.get('batch-1') is None
        return 'second done'
    steps = [WorkflowStep('first', '1', first, idempotent=True, retryable=True),
             WorkflowStep('second', '1', second)]
    job = runner(path, steps)
    with pytest.raises(asyncio.CancelledError):
        await job.run('run', space='alpha', scope={}, inputs='source')
    assert job.status('run', space='alpha', scope={}, inputs='source').checkpoint_count == 1
    with pytest.raises(WorkflowError, match='checkpoint_inactive'):
        handles[0].put('late', b'late output')
    job.close()
    assert b'private completed embedding' not in path.read_bytes()
    job = runner(path, steps)
    try:
        result = await job.run('run', space='alpha', scope={}, inputs='source')
        assert result.results == {'first': 'first done', 'second': 'second done'}
        assert job.status('run', space='alpha', scope={}, inputs='source').checkpoint_count == 0
    finally:
        job.close()


async def test_invalid_sources_clear_only_their_run_checkpoints(tmp_path):
    path = tmp_path / 'work.db'
    valid = True
    async def verify(context):
        return valid
    async def work(context):
        context.checkpoints.put('batch', b'private vector')
        raise asyncio.CancelledError()
    job = runner(path, [WorkflowStep('index', '1', work, idempotent=True, retryable=True)],
                 source_verifier=verify)
    try:
        for run_id in ['first', 'second']:
            with pytest.raises(asyncio.CancelledError):
                await job.run(run_id, space='s', scope={}, inputs=None)
        valid = False
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await job.run('first', space='s', scope={}, inputs=None)
        assert job.status('first', space='s', scope={}, inputs=None).checkpoint_count == 0
        assert job.status('second', space='s', scope={}, inputs=None).checkpoint_count == 1
    finally:
        job.close()


async def test_checkpoint_integrity_and_invocation_binding(tmp_path):
    path = tmp_path / 'work.db'
    async def work(context):
        saved = context.checkpoints.get('batch')
        if saved is None:
            context.checkpoints.put('batch', b'first vector')
            raise asyncio.CancelledError()
        return None
    steps = [WorkflowStep('index', '1', work, idempotent=True, retryable=True)]
    job = runner(path, steps)
    with pytest.raises(asyncio.CancelledError):
        await job.run('r', space='s', scope={}, inputs='source')
    with pytest.raises(WorkflowError, match='binding_mismatch'):
        await job.run('r', space='other', scope={}, inputs='source')
    job.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE workflow_runs SET payload=? WHERE token LIKE '%:checkpoint:%'", (b'corrupted',))
    job = runner(path, steps)
    try:
        with pytest.raises(WorkflowError, match='step_failed'):
            await job.run('r', space='s', scope={}, inputs='source')
        state = job.status('r', space='s', scope={}, inputs='source')
        assert state.completed_steps == () and state.checkpoint_count == 1
    finally:
        job.close()
