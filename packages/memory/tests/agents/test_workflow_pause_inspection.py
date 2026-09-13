"""Inspection authenticates paused bytes without granting a checkpoint lease."""
import asyncio
import hashlib

import pytest

from scone_memory.agents.workflow import WorkflowError, WorkflowPaused, WorkflowPausableStep
from .test_workflow_pauses import runner


@pytest.mark.parametrize('parallel', [False, True])
async def test_inspect_pauses_is_read_only_and_detects_later_consumption(tmp_path, parallel):
    effects = []
    resume = False
    async def execute(context):
        effects.append(1)
        context.checkpoints.put('proposal', b'private exact proposal')
        return 'done' if resume else WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)], parallel=parallel)
    try:
        await job.run('one', space='alpha', scope={'host': 'policy'}, inputs='question')
        before = (tmp_path/'journal').read_bytes()
        paused, = await job.inspect_pauses('one', space='alpha', scope={'host': 'policy'}, inputs='question')
        assert paused.step_id == 'action' and paused.checkpoint == 'proposal'
        assert paused.payload == b'private exact proposal'
        assert paused.context.inputs == 'question' and paused.context.checkpoints is None
        paused.check_current()
        assert before == (tmp_path/'journal').read_bytes() and effects == [1]
        resume = True
        await job.run('one', space='alpha', scope={'host': 'policy'}, inputs='question')
        with pytest.raises(WorkflowError, match='pause_snapshot_changed'):
            paused.check_current()
    finally:
        job.close()


async def test_changed_checkpoint_cannot_be_inspected_under_an_old_ticket(tmp_path):
    async def execute(context):
        context.checkpoints.put('proposal', b'original')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)])
    try:
        await job.run('one', space='alpha', scope={}, inputs=None)
        token, = job._db.execute("SELECT token FROM workflow_runs WHERE token LIKE '%:checkpoint:%'").fetchone()
        job._db.execute('UPDATE workflow_runs SET payload=? WHERE token=?', (job._seal(token, b'changed'), token))
        with pytest.raises(WorkflowError, match='pause_checkpoint_changed'):
            await job.inspect_pauses('one', space='alpha', scope={}, inputs=None)
    finally:
        job.close()


@pytest.mark.parametrize('failure', ['revoked', 'outage', 'changed'])
async def test_failed_or_changed_verification_never_returns_actionable_snapshot(tmp_path, failure):
    inspecting = False
    job = None
    async def verify(context):
        if not inspecting:
            return True
        if failure == 'revoked':
            return False
        if failure == 'outage':
            raise OSError('private storage details')
        token, payload = job._db.execute("SELECT token,payload FROM workflow_runs WHERE token NOT LIKE '%:checkpoint:%'").fetchone()
        job._db.execute('UPDATE workflow_runs SET payload=? WHERE token=?', (job._seal(token, job._unseal(token, payload)), token))
        return True
    async def execute(context):
        context.checkpoints.put('proposal', b'original')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)], verifier=verify)
    try:
        await job.run('one', space='alpha', scope={}, inputs=None)
        before = job.status('one', space='alpha', scope={}, inputs=None)
        inspecting = True
        with pytest.raises(WorkflowError) as caught:
            await job.inspect_pauses('one', space='alpha', scope={}, inputs=None)
        assert caught.value.code == {'revoked': 'sources_invalid', 'outage': 'verification_unavailable', 'changed': 'pause_snapshot_changed'}[failure]
        assert job.status('one', space='alpha', scope={}, inputs=None) == before
    finally:
        job.close()
