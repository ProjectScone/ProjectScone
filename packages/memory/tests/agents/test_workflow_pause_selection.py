"""A continuation selects its pauses without consuming other waiting actions."""
import pytest

from scone_memory.agents.workflow import WorkflowError, WorkflowPaused, WorkflowPausableStep, WorkflowRunner
from .test_workflow_pauses import runner


@pytest.mark.parametrize('parallel', [False, True])
async def test_empty_selection_preserves_pause_attempts_and_checkpoint(tmp_path, parallel):
    calls = []
    async def execute(context):
        calls.append(1)
        context.checkpoints.put('proposal', b'exact action')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute, max_resumes=1)], parallel=parallel)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        for _ in range(4):
            assert (await job.run('one', space='alpha', scope={}, inputs=None, resume_steps=())).status == 'paused'
        assert calls == [1]
        assert job.status('one', space='alpha', scope={}, inputs=None).attempts == {'action': 1}
    finally:
        job.close()


async def test_selected_dag_pause_finishes_while_other_ticket_stays_untouched(tmp_path):
    calls = {'first': 0, 'second': 0}
    def execute(name):
        async def step(context):
            calls[name] += 1
            if calls[name] == 1:
                context.checkpoints.put('proposal', name.encode())
                return WorkflowPaused('proposal')
            return name
        return step
    job = runner(tmp_path/'journal', [WorkflowPausableStep(name, '1', execute(name)) for name in calls], parallel=True)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        result = await job.run('one', space='alpha', scope={}, inputs=None, resume_steps=('second',))
        assert result.status == 'paused' and result.results == {'second': 'second'}
        assert calls == {'first': 1, 'second': 2}
        status = job.status('one', space='alpha', scope={}, inputs=None)
        assert status.paused_steps == ('first',) and status.attempts == {'first': 1, 'second': 2}
        assert (await job.run('one', space='alpha', scope={}, inputs=None, resume_steps=('first',))).status == 'completed'
        assert calls == {'first': 2, 'second': 2}
    finally:
        job.close()


@pytest.mark.parametrize('selection', [('unknown',), ('action', 'action'), 'action', (True,)])
async def test_invalid_pause_selection_never_enters_callback(tmp_path, selection):
    async def execute(context):
        pytest.fail('invalid selection executed')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)])
    try:
        with pytest.raises(WorkflowError, match='invalid_resume_selection'):
            await job.run('one', space='alpha', scope={}, inputs=None, resume_steps=selection)
        assert job.status('one', space='alpha', scope={}, inputs=None) is None
    finally:
        job.close()
