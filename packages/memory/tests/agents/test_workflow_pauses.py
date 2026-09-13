"""Only an acknowledged checkpoint-backed pause permits step resumption."""
import asyncio

import pytest

from scone_memory.agents.workflow import (
    WorkflowError, WorkflowPaused, WorkflowPausableStep, WorkflowRunner, WorkflowStep,
)


async def valid(context):
    return True


def runner(path, steps, *, parallel=False, verifier=valid, **options):
    return WorkflowRunner(path, key=b'k' * 32, steps=steps, source_verifier=verifier,
        dependencies={step.step_id: () for step in steps} if parallel else None,
        max_parallel=2 if parallel else 1, **options)


@pytest.mark.parametrize('parallel', [False, True])
async def test_pause_reopen_reuses_exact_checkpoint_and_revokes_old_lease(tmp_path, parallel):
    calls, leases = [], []
    approved = False
    async def execute(context):
        calls.append(1)
        leases.append(context.checkpoints)
        if context.checkpoints.get('proposal') is None:
            context.checkpoints.put('proposal', b'private exact proposal')
        if not approved:
            return WorkflowPaused('proposal')
        assert context.checkpoints.get('proposal') == b'private exact proposal'
        return {'executed': True}
    steps = [WorkflowPausableStep('action', '1', execute)]
    job = runner(tmp_path/'journal', steps, parallel=parallel)
    try:
        result = await job.run('one', space='alpha', scope={}, inputs='question')
        assert result.status == 'paused' and result.results == {}
        status = job.status('one', space='alpha', scope={}, inputs='question')
        assert status.paused_steps == ('action',) and status.inflight_steps == ()
        assert status.waiting_steps == () and status.checkpoint_count == 1
        assert await job.inspect_completed('one', space='alpha', scope={}, inputs='question') == {}
        with pytest.raises(WorkflowError, match='not_completed'):
            await job.read_result('one', space='alpha', scope={}, inputs='question')
        with pytest.raises(WorkflowError, match='checkpoint_inactive'):
            leases[0].put('proposal', b'changed after return')
    finally:
        job.close()
    assert b'private exact proposal' not in (tmp_path/'journal').read_bytes()
    approved = True
    reopened = runner(tmp_path/'journal', steps, parallel=parallel)
    try:
        result = await reopened.run('one', space='alpha', scope={}, inputs='question')
        assert result.status == 'completed' and result.results == {'action': {'executed': True}}
        assert calls == [1, 1]
        status = reopened.status('one', space='alpha', scope={}, inputs='question')
        assert status.paused_steps == () and status.checkpoint_count == 0
        assert (await reopened.run('one', space='alpha', scope={}, inputs='question')).reused_steps == ('action',)
        assert calls == [1, 1]
    finally:
        reopened.close()


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('receipt', [None, b''])
async def test_missing_or_empty_checkpoint_cannot_authorize_a_pause(tmp_path, parallel, receipt):
    invoked = []
    async def execute(context):
        invoked.append(1)
        if receipt is not None:
            context.checkpoints.put('proposal', receipt)
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)], parallel=parallel)
    try:
        with pytest.raises(WorkflowError):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert job.status('one', space='alpha', scope={}, inputs=None).paused_steps == ()
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert invoked == [1]
    finally:
        job.close()


@pytest.mark.parametrize('parallel', [False, True])
async def test_interrupted_active_pausable_step_is_never_replayed(tmp_path, parallel):
    invoked = []
    async def execute(context):
        invoked.append(1)
        context.checkpoints.put('proposal', b'checkpoint is not a pause ticket')
        raise asyncio.CancelledError()
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)], parallel=parallel)
    try:
        with pytest.raises((asyncio.CancelledError, WorkflowError)):
            await job.run('one', space='alpha', scope={}, inputs=None)
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert invoked == [1]
    finally:
        job.close()


@pytest.mark.parametrize('parallel', [False, True])
async def test_resumption_budget_does_not_busy_loop_and_preserves_pending_ticket(tmp_path, parallel):
    invoked = []
    async def execute(context):
        invoked.append(1)
        context.checkpoints.put('proposal', b'pending')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute, max_resumes=1)], parallel=parallel)
    try:
        for _ in range(2):
            assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        with pytest.raises(WorkflowError, match='resumes_exhausted'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert len(invoked) == 2
        assert job.status('one', space='alpha', scope={}, inputs=None).paused_steps == ('action',)
    finally:
        job.close()


async def test_ordinary_step_cannot_mint_pause_ticket(tmp_path):
    async def execute(context):
        context.checkpoints.put('proposal', b'ordinary callback')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowStep('action', '1', execute)])
    try:
        with pytest.raises(WorkflowError):
            await job.run('one', space='alpha', scope={}, inputs=None)
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await job.run('one', space='alpha', scope={}, inputs=None)
    finally:
        job.close()


@pytest.mark.parametrize('width', [1, 2])
async def test_pause_blocks_dependents_but_finishes_independent_steps(tmp_path, width):
    calls = []
    approved = False
    async def execute(context):
        calls.append('pause')
        if approved:
            return 'accepted'
        context.checkpoints.put('proposal', b'exact proposal')
        return WorkflowPaused('proposal')
    async def sibling(context):
        calls.append('sibling')
        return 'independent'
    async def dependent(context):
        calls.append('dependent')
        assert context.completed['action'] == 'accepted'
        return 'after acceptance'
    job = WorkflowRunner(tmp_path/'journal', key=b'k'*32, source_verifier=valid,
        steps=[WorkflowPausableStep('action', '1', execute), WorkflowStep('child', '1', dependent),
               WorkflowStep('sibling', '1', sibling)],
        dependencies={'action': (), 'child': ('action',), 'sibling': ()}, max_parallel=width)
    try:
        first = await job.run('one', space='alpha', scope={}, inputs=None)
        assert first.status == 'paused' and first.results == {'sibling': 'independent'}
        assert sorted(calls) == ['pause', 'sibling']
        assert await job.inspect_completed('one', space='alpha', scope={}, inputs=None) == first.results
        approved = True
        second = await job.run('one', space='alpha', scope={}, inputs=None)
        assert second.status == 'completed'
        assert second.reused_steps == ('sibling',)
        assert calls.count('pause') == 2 and calls.count('sibling') == 1 and calls[-1] == 'dependent'
    finally:
        job.close()


async def test_pause_and_human_input_remain_distinct_and_poll_once(tmp_path):
    from scone_memory.agents.workflow import WorkflowInputStep, WorkflowInputValue
    answer = None
    counts = {'pause': 0, 'input': 0}
    async def execute(context):
        counts['pause'] += 1
        context.checkpoints.put('proposal', b'proposal')
        return WorkflowPaused('proposal')
    async def poll(context):
        counts['input'] += 1
        return WorkflowInputValue(answer) if answer is not None else None
    job = WorkflowRunner(tmp_path/'journal', key=b'k'*32, source_verifier=valid,
        steps=[WorkflowPausableStep('action', '1', execute), WorkflowInputStep('human', '1', poll)],
        dependencies={'action': (), 'human': ()}, max_parallel=2)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        status = job.status('one', space='alpha', scope={}, inputs=None)
        assert status.paused_steps == ('action',) and status.waiting_steps == ('human',)
        assert counts == {'pause': 1, 'input': 1}
        answer = 'response'
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).results == {'human': 'response'}
        status = job.status('one', space='alpha', scope={}, inputs=None)
        assert status.paused_steps == ('action',) and status.waiting_steps == ()
        assert counts == {'pause': 2, 'input': 2}
    finally:
        job.close()


@pytest.mark.parametrize('parallel', [False, True])
async def test_verification_outage_preserves_ticket_and_revocation_invalidates_it(tmp_path, parallel):
    mode = 'valid'
    calls = []
    async def verify(context):
        if mode == 'outage':
            raise OSError('private storage failure')
        return mode == 'valid'
    async def execute(context):
        calls.append(1)
        context.checkpoints.put('proposal', b'original')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)], parallel=parallel, verifier=verify)
    try:
        await job.run('one', space='alpha', scope={}, inputs=None)
        mode = 'outage'
        with pytest.raises(WorkflowError, match='verification_unavailable'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        status = job.status('one', space='alpha', scope={}, inputs=None)
        assert status.paused_steps == ('action',) and status.attempts == {'action': 1}
        mode = 'valid'
        await job.run('one', space='alpha', scope={}, inputs=None)
        assert len(calls) == 2
        mode = 'revoked'
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        status = job.status('one', space='alpha', scope={}, inputs=None)
        assert status.paused_steps == () and status.checkpoint_count == 0
        mode = 'valid'
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert len(calls) == 2
    finally:
        job.close()


@pytest.mark.parametrize('parallel', [False, True])
async def test_replaced_checkpoint_fails_before_resume(tmp_path, parallel):
    calls = []
    async def execute(context):
        calls.append(1)
        context.checkpoints.put('proposal', b'original')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)], parallel=parallel)
    try:
        await job.run('one', space='alpha', scope={}, inputs=None)
        token = job._db.execute("SELECT token FROM workflow_runs WHERE token LIKE '%:checkpoint:%'").fetchone()[0]
        job._db.execute('UPDATE workflow_runs SET payload=? WHERE token=?', (job._seal(token, b'replaced'), token))
        with pytest.raises(WorkflowError, match='pause_checkpoint_changed'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert calls == [1]
    finally:
        job.close()


@pytest.mark.parametrize('change', ['scope', 'inputs', 'space', 'version', 'max_resumes', 'kind'])
async def test_pause_is_bound_to_invocation_and_step_definition(tmp_path, change):
    async def execute(context):
        context.checkpoints.put('proposal', b'original')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)])
    await job.run('one', space='alpha', scope={}, inputs=None)
    job.close()
    step = WorkflowStep('action', '1', execute) if change == 'kind' else WorkflowPausableStep(
        'action', '2' if change == 'version' else '1', execute, max_resumes=1 if change == 'max_resumes' else 32)
    job = runner(tmp_path/'journal', [step])
    args = {'space': 'beta' if change == 'space' else 'alpha',
            'scope': {'user': 'other'} if change == 'scope' else {},
            'inputs': 'changed' if change == 'inputs' else None}
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await job.run('one', **args)
    finally:
        job.close()


@pytest.mark.parametrize('count', [True, False, 0, -1, 129, 1.5, '1', None])
def test_resume_limit_is_strict_and_validated_before_opening_journal(tmp_path, count):
    async def execute(context):
        return None
    with pytest.raises(WorkflowError):
        runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute, max_resumes=count)])
    assert not (tmp_path/'journal').exists()


async def test_completion_policy_cannot_discard_a_paused_step(tmp_path):
    from scone_memory.agents.workflow import WorkflowCompletion
    stop = False
    async def execute(context):
        context.checkpoints.put('proposal', b'pending')
        return WorkflowPaused('proposal')
    job = runner(tmp_path/'journal', [WorkflowPausableStep('action', '1', execute)],
        completion=WorkflowCompletion('1', lambda context: stop))
    try:
        await job.run('one', space='alpha', scope={}, inputs=None)
        stop = True
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert job.status('one', space='alpha', scope={}, inputs=None).paused_steps == ('action',)
        with pytest.raises(WorkflowError, match='not_completed'):
            await job.read_result('one', space='alpha', scope={}, inputs=None)
    finally:
        job.close()


async def test_independent_pauses_resume_once_per_explicit_run(tmp_path):
    calls = []
    finish = False
    def step(name):
        async def execute(context):
            calls.append(name)
            if finish:
                return name
            context.checkpoints.put('proposal', name.encode())
            return WorkflowPaused('proposal')
        return WorkflowPausableStep(name, '1', execute)
    job = runner(tmp_path/'journal', [step('first'), step('second')], parallel=True)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        assert calls == ['first', 'second']
        assert job.status('one', space='alpha', scope={}, inputs=None).paused_steps == ('first', 'second')
        finish = True
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).results == {'first': 'first', 'second': 'second'}
        assert calls == ['first', 'second', 'first', 'second']
    finally:
        job.close()
