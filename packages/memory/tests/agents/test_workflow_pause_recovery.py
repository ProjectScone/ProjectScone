"""Adversarial journal, cancellation and deadline recovery regressions."""

import asyncio
import hashlib
import json
import sqlite3
import pytest
from scone_memory.agents.workflow import (
    WorkflowRunner,
    WorkflowStep,
    WorkflowPausableStep,
    WorkflowPaused,
    WorkflowError,
)


async def valid(context):
    return True


def make(path, steps, dag=False, **kw):
    return WorkflowRunner(
        path,
        key=b'k' * 32,
        steps=steps,
        source_verifier=valid,
        dependencies={s.step_id: [] for s in steps} if dag else None,
        max_parallel=2 if dag else 1,
        **kw
    )


async def run(job):
    return await job.run('r', space='alpha', scope={}, inputs=None)


def status(job):
    return job.status('r', space='alpha', scope={}, inputs=None)


@pytest.mark.parametrize('dag', [False, True])
async def test_consumption_write_failure_keeps_ticket(tmp_path, dag):
    calls = []

    async def step(ctx):
        calls.append(1)
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    try:
        await run(job)
        original = job._save
        injected = []

        def fail(token, state):
            if (
                state.get('status') == 'running'
                and state['attempts'].get('a') == 2
                and not state.get('pauses')
                and not injected
            ):
                injected.append(1)
                raise sqlite3.OperationalError('private storage fault')
            original(token, state)

        job._save = fail
        with pytest.raises(WorkflowError):
            await run(job)
        job._save = original
        assert injected and calls == [1]
        assert status(job).paused_steps == ('a',)
        assert status(job).attempts == {'a': 1}
        assert (await run(job)).status == 'paused'
        assert calls == [1, 1]
    finally:
        job.close()


@pytest.mark.parametrize('dag', [False, True])
async def test_commit_then_cancel_is_unknown(tmp_path, dag):
    calls = []

    async def step(ctx):
        calls.append(1)
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    try:
        await run(job)
        original = job._save
        injected = []

        def fail(token, state):
            original(token, state)
            if (
                state.get('status') == 'running'
                and state['attempts'].get('a') == 2
                and not state.get('pauses')
                and not injected
            ):
                injected.append(1)
                raise asyncio.CancelledError()

        job._save = fail
        with pytest.raises(asyncio.CancelledError):
            await run(job)
        job._save = original
        assert calls == [1] and injected
        assert status(job).paused_steps == ()
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await run(job)
        assert calls == [1]
    finally:
        job.close()


@pytest.mark.parametrize('dag', [False, True])
async def test_cancel_swallow_cannot_mint_ticket(tmp_path, dag):
    started = asyncio.Event()

    async def step(ctx):
        ctx.checkpoints.put('p', b'bound')
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    try:
        task = asyncio.create_task(run(job))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert status(job).paused_steps == ()
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await run(job)
    finally:
        job.close()


async def test_paused_sibling_does_not_authorize_failed_sibling_replay(tmp_path):
    calls = []

    async def paused(ctx):
        calls.append('paused')
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    async def failed(ctx):
        calls.append('failed')
        await asyncio.sleep(0.02)
        raise ValueError('private')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', paused), WorkflowStep('b', 'v', failed)], True)
    try:
        with pytest.raises(WorkflowError):
            await run(job)
        assert status(job).paused_steps == ('a',)
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await run(job)
        assert calls == ['paused', 'failed']
    finally:
        job.close()


@pytest.mark.parametrize(
    'mutation', ['bool_attempt', 'overlap', 'wrong_kind', 'bad_hash', 'missing_dependency']
)
async def test_malformed_tickets_refused_on_all_reads(tmp_path, mutation):
    async def done(ctx):
        return 'done'

    async def paused(ctx):
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowStep('b', 'v', done), WorkflowPausableStep('a', 'v', paused)])
    try:
        await run(job)
        token, payload = job._db.execute(
            "SELECT token,payload FROM workflow_runs WHERE token NOT LIKE '%:checkpoint:%'"
        ).fetchone()
        state = json.loads(job._unseal(token, payload))
        if mutation == 'bool_attempt':
            state['attempts']['a'] = True
        elif mutation == 'overlap':
            state['results']['a'] = 'forged'
        elif mutation == 'wrong_kind':
            state['pauses']['b'] = state['pauses'].pop('a')
        elif mutation == 'bad_hash':
            state['pauses']['a']['digest'] = 'Z' * 64
        else:
            del state['results']['b']
        job._save(token, state)
        with pytest.raises(WorkflowError, match='journal_key_or_integrity'):
            status(job)
        with pytest.raises(WorkflowError, match='journal_key_or_integrity'):
            await job.inspect_completed('r', space='alpha', scope={}, inputs=None)
        with pytest.raises(WorkflowError, match='journal_key_or_integrity'):
            await job.read_result('r', space='alpha', scope={}, inputs=None)
        with pytest.raises(WorkflowError, match='journal_key_or_integrity'):
            await run(job)
    finally:
        job.close()


def test_legacy_revision_unchanged(tmp_path):
    async def done(ctx):
        return 'done'

    job = make(tmp_path / 'j', [WorkflowStep('a', 'v', done)])
    try:
        legacy = [{'id': 'a', 'version': 'v', 'idempotent': False, 'retryable': False}]
        assert (
            job._revision
            == hashlib.sha256(
                json.dumps(legacy, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
            ).hexdigest()
        )
    finally:
        job.close()


@pytest.mark.parametrize('dag', [False, True])
async def test_expired_callback_cannot_mint_pause(tmp_path, dag):

    calls = []

    async def step(ctx):
        calls.append(1)
        ctx.checkpoints.put('p', b'bound')
        job._pause_expires_at = asyncio.get_running_loop().time() - 1
        return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    try:
        with pytest.raises(WorkflowError, match='deadline'):
            await run(job)
        assert status(job).paused_steps == ()
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await run(job)
        assert calls == [1]
    finally:
        job.close()


@pytest.mark.parametrize('dag', [False, True])
async def test_callback_workflow_error_private_message_not_exposed(tmp_path, dag):
    secret = 'private-token-value-must-not-escape'

    async def step(ctx):
        raise WorkflowError(secret)

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    try:
        with pytest.raises(WorkflowError) as caught:
            await run(job)
        assert secret not in str(caught.value)
        assert secret not in repr(status(job))
    finally:
        job.close()


@pytest.mark.parametrize('dag', [False, True])
async def test_cancel_swallowing_verifier_cannot_dispatch_resume(tmp_path, dag):
    calls = []
    entered = asyncio.Event()
    block = False

    async def verifier(ctx):
        nonlocal block
        if block:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                block = False
                return True
        return True

    async def step(ctx):
        calls.append(1)
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    job._verifier = verifier
    try:
        await run(job)
        block = True
        task = asyncio.create_task(run(job))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.2)
        assert calls == [1]
    finally:
        job.close()


@pytest.mark.parametrize('dag', [False, True])
async def test_pause_write_past_deadline_does_not_return_success(tmp_path, dag):

    async def step(ctx):
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    job = make(tmp_path / 'j', [WorkflowPausableStep('a', 'v', step)], dag)
    original = job._save
    delayed = []

    def slow(token, state):
        original(token, state)
        if state.get('pauses') and not delayed:
            delayed.append(1)
            job._pause_expires_at = asyncio.get_running_loop().time() - 1

    job._save = slow
    try:
        with pytest.raises(WorkflowError, match='deadline'):
            await run(job)
        assert delayed
        assert status(job).paused_steps == ('a',)
        assert status(job).status == 'deadline'
    finally:
        job.close()


async def test_consumption_cancel_and_drained_sibling_cannot_restore_ticket(tmp_path):
    from scone_memory.agents.workflow import WorkflowInputStep, WorkflowInputValue

    ready = False
    calls = []
    sibling_started = asyncio.Event()

    async def human(ctx):
        return WorkflowInputValue('ready') if ready else None

    async def sibling(ctx):
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return 'completed sibling'

    async def paused(ctx):
        calls.append(1)
        ctx.checkpoints.put('p', b'bound')
        return WorkflowPaused('p')

    steps = [
        WorkflowInputStep('h', 'v', human),
        WorkflowStep('b', 'v', sibling),
        WorkflowPausableStep('a', 'v', paused),
    ]

    async def verify(ctx):
        if ready and 'h' in ctx.completed and not sibling_started.is_set():
            await asyncio.sleep(0)
        return True

    job = WorkflowRunner(
        tmp_path / 'j',
        key=b'k' * 32,
        steps=steps,
        source_verifier=verify,
        dependencies={'h': [], 'b': ['h'], 'a': []},
        max_parallel=2,
    )
    try:
        assert (await run(job)).status == 'paused'
        ready = True
        original = job._save
        injected = []

        def fail(token, state):
            original(token, state)
            if (
                state.get('status') == 'running'
                and state['attempts'].get('a') == 2
                and not state.get('pauses')
                and not injected
            ):
                injected.append(1)
                raise asyncio.CancelledError()

        job._save = fail
        with pytest.raises(asyncio.CancelledError):
            await run(job)
        job._save = original
        assert sibling_started.is_set() and injected and calls == [1]
        # A callback swallowing cancellation cannot authorize a completed receipt.
        # Its uncertain attempt still must not restore the consumed pause ticket.
        assert 'b' not in status(job).completed_steps
        assert 'b' in status(job).inflight_steps
        assert status(job).paused_steps == ()
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await run(job)
    finally:
        job.close()
