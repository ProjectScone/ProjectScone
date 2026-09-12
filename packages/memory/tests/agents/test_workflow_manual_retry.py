"""Manual retries retain a bounded attempt budget across journal reopen."""
import pytest

from scone_memory.agents import WorkflowError, WorkflowStep
from .test_agent_workflows import runner


async def test_manual_retry_does_one_attempt_and_preserves_checkpoint_on_reopen(tmp_path):
    calls = []
    async def work(context):
        calls.append(context.checkpoints.get('work'))
        context.checkpoints.put('work', b'saved')
        if len(calls) == 1:
            raise RuntimeError('temporary failure')
        return 'done'
    steps = [WorkflowStep('index', '1', work, idempotent=True, retryable=True)]
    job = runner(tmp_path / 'job.db', steps, automatic_retries=False, max_retries=2)
    with pytest.raises(WorkflowError, match='step_failed'):
        await job.run('run', space='s', scope={}, inputs=None)
    assert calls == [None]
    job.close()
    job = runner(tmp_path / 'job.db', steps, automatic_retries=False, max_retries=2)
    try:
        result = await job.run('run', space='s', scope={}, inputs=None)
        assert result.results == {'index': 'done'} and calls == [None, b'saved']
    finally:
        job.close()


async def test_manual_retry_budget_cannot_be_reset_by_reopening(tmp_path):
    calls = 0
    async def work(context):
        nonlocal calls
        calls += 1
        raise RuntimeError('failure')
    steps = [WorkflowStep('index', '1', work, idempotent=True, retryable=True)]
    for expected in range(1, 5):
        job = runner(tmp_path / 'job.db', steps, automatic_retries=False, max_retries=2)
        try:
            with pytest.raises(WorkflowError, match='step_failed' if expected <= 3 else 'retries_exhausted'):
                await job.run('run', space='s', scope={}, inputs=None)
            assert calls == min(expected, 3)
        finally:
            job.close()


async def test_manual_retry_policy_is_bound_to_the_journal(tmp_path):
    async def work(context):
        raise RuntimeError('failure')
    steps = [WorkflowStep('index', '1', work, idempotent=True, retryable=True)]
    job = runner(tmp_path / 'job.db', steps, automatic_retries=False)
    with pytest.raises(WorkflowError, match='step_failed'):
        await job.run('run', space='s', scope={}, inputs=None)
    job.close()
    job = runner(tmp_path / 'job.db', steps)
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await job.run('run', space='s', scope={}, inputs=None)
    finally:
        job.close()
