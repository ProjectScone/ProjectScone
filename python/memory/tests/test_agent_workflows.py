"""Offline durable workflows use real encrypted SQLite checkpoints."""
import asyncio
import os
import sqlite3

import pytest

from scone_memory.agents import WorkflowError, WorkflowRunner, WorkflowStep

KEY = b'k' * 32

async def valid(context):
    return True


def runner(path, steps, **kwargs):
    return WorkflowRunner(path, key=KEY, steps=steps, source_verifier=kwargs.pop('source_verifier', valid), **kwargs)


async def test_completed_results_survive_close_reopen_without_reexecution(tmp_path):
    path = tmp_path / 'workflow.db'
    calls = []
    async def retrieve(context):
        calls.append(context.inputs)
        return {'text': 'private source passage', 'chunk_id': 7}
    steps = [WorkflowStep('retrieve', '1', retrieve)]
    first = runner(path, steps)
    result = await first.run('run-1', space='alpha', scope={'owner': 'alice'}, inputs={'query': 'secret query'})
    first.close()
    second = runner(path, steps)
    resumed = await second.run('run-1', space='alpha', scope={'owner': 'alice'}, inputs={'query': 'secret query'})
    second.close()
    assert resumed.results == result.results
    assert resumed.reused_steps == ('retrieve',)
    assert len(calls) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    for file in tmp_path.iterdir():
        assert b'private source passage' not in file.read_bytes()
        assert b'secret query' not in file.read_bytes()


@pytest.mark.parametrize('change', ['inputs', 'scope', 'space', 'version', 'order'])
async def test_resume_refuses_changed_binding(tmp_path, change):
    async def work(context): return 'ok'
    steps = [WorkflowStep('a', '1', work), WorkflowStep('b', '1', work)]
    path = tmp_path / 'w.db'
    a = runner(path, steps)
    await a.run('r', space='alpha', scope={'owner': 'a'}, inputs='query')
    a.close()
    if change == 'version': steps[0] = WorkflowStep('a', '2', work)
    if change == 'order': steps.reverse()
    b = runner(path, steps)
    with pytest.raises(WorkflowError, match='binding_mismatch'):
        await b.run('r', space='beta' if change == 'space' else 'alpha', scope={'owner': 'b' if change == 'scope' else 'a'}, inputs='changed' if change == 'inputs' else 'query')
    b.close()


async def test_retry_and_partial_resume_have_lifetime_attempt_bound(tmp_path):
    calls = []
    async def first(context):
        calls.append('first')
        return 'first result'
    async def second(context):
        calls.append('second')
        if calls.count('second') < 2: raise OSError('private credentials')
        assert context.completed == {'first': 'first result'}
        return 'second result'
    r = runner(tmp_path / 'w.db', [WorkflowStep('first', '1', first), WorkflowStep('second', '1', second, idempotent=True, retryable=True)])
    result = await r.run('r', space='s', scope={}, inputs=None)
    assert result.results == {'first': 'first result', 'second': 'second result'}
    assert calls == ['first', 'second', 'second']
    r.close()


@pytest.mark.parametrize('idempotent,retryable', [(False, False), (True, False)])
async def test_unapproved_retry_is_never_replayed(tmp_path, idempotent, retryable):
    calls = []
    async def fails(context):
        calls.append(1)
        raise RuntimeError('secret failure detail')
    r = runner(tmp_path / 'w.db', [WorkflowStep('work', '1', fails, idempotent=idempotent, retryable=retryable)])
    for _ in range(2):
        with pytest.raises(WorkflowError) as failure:
            await r.run('r', space='s', scope={}, inputs=None)
        assert 'secret' not in str(failure.value)
    assert calls == [1]
    r.close()


async def test_cancellation_marks_nonidempotent_outcome_unknown(tmp_path):
    started = asyncio.Event()
    async def work(context):
        started.set()
        await asyncio.Event().wait()
    path = tmp_path / 'w.db'
    r = runner(path, [WorkflowStep('write', '1', work)])
    task = asyncio.create_task(r.run('r', space='s', scope={}, inputs=None))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    r.close()
    reopened = runner(path, [WorkflowStep('write', '1', work)])
    with pytest.raises(WorkflowError, match='outcome_unknown'):
        await reopened.run('r', space='s', scope={}, inputs=None)
    reopened.close()


async def test_concurrent_connections_do_not_duplicate_execution(tmp_path):
    started, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def work(context):
        calls.append(1)
        started.set()
        await finish.wait()
        return 'done'
    path = tmp_path / 'w.db'
    a, b = runner(path, [WorkflowStep('work', '1', work)]), runner(path, [WorkflowStep('work', '1', work)])
    task = asyncio.create_task(a.run('r', space='s', scope={}, inputs=None))
    await started.wait()
    with pytest.raises(WorkflowError, match='busy'):
        await b.run('r', space='s', scope={}, inputs=None)
    finish.set()
    await task
    assert (await b.run('r', space='s', scope={}, inputs=None)).reused_steps == ('work',)
    assert calls == [1]
    a.close(); b.close()


async def test_verifier_rechecked_before_resume_and_final_return(tmp_path):
    retained = True
    checks = []
    async def verify(context):
        checks.append(tuple(context.completed))
        return retained
    async def work(context): return {'source': 'private retained text'}
    r = runner(tmp_path / 'w.db', [WorkflowStep('read', '1', work)], source_verifier=verify)
    await r.run('r', space='s', scope={}, inputs=None)
    assert checks == [(), ('read',)]
    retained = False
    with pytest.raises(WorkflowError, match='sources_invalid'):
        await r.run('r', space='s', scope={}, inputs=None)
    assert len(checks) == 3
    r.close()


async def test_sources_changed_during_step_never_return_results(tmp_path):
    retained = True
    async def verify(context): return retained
    async def work(context):
        nonlocal retained
        retained = False
        return 'stale'
    r = runner(tmp_path / 'w.db', [WorkflowStep('read', '1', work)], source_verifier=verify)
    with pytest.raises(WorkflowError, match='sources_invalid'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()


async def test_payload_and_deadline_are_bounded(tmp_path):
    async def huge(context): return 'x' * 5000
    r = runner(tmp_path / 'w.db', [WorkflowStep('big', '1', huge)], max_payload_bytes=1024)
    with pytest.raises(WorkflowError, match='payload_limit'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()
    async def slow(context): await asyncio.Event().wait()
    r = runner(tmp_path / 'slow.db', [WorkflowStep('slow', '1', slow)], deadline=.01)
    with pytest.raises(WorkflowError, match='deadline'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()


async def test_wrong_key_and_foreign_database_are_refused(tmp_path):
    async def work(context): return 'secret'
    path = tmp_path / 'w.db'
    r = runner(path, [WorkflowStep('a', '1', work)])
    await r.run('r', space='s', scope={}, inputs=None)
    r.close()
    with pytest.raises(WorkflowError, match='journal_key_or_integrity'):
        WorkflowRunner(path, key=b'z'*32, steps=[WorkflowStep('a', '1', work)], source_verifier=valid)
    foreign = tmp_path / 'foreign.db'
    sqlite3.connect(foreign).execute('CREATE TABLE unrelated (id INT)').connection.close()
    os.chmod(foreign, 0o600)
    with pytest.raises(WorkflowError, match='foreign_journal'):
        runner(foreign, [WorkflowStep('a', '1', work)])


async def test_callback_workflowerror_cannot_publish_private_message(tmp_path):
    async def work(context): raise WorkflowError('private-provider-secret')
    r = runner(tmp_path / 'w.db', [WorkflowStep('a', '1', work)])
    with pytest.raises(WorkflowError, match='^step_failed$'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()


async def test_cancelled_idempotent_step_resumes_completed_prefix_once(tmp_path):
    started = asyncio.Event()
    calls = []
    async def first(context): calls.append('first'); return 'prefix'
    async def second(context):
        calls.append('second')
        if calls.count('second') == 1:
            started.set()
            await asyncio.Event().wait()
        return context.completed['first']
    path = tmp_path / 'w.db'
    steps = [WorkflowStep('first', '1', first), WorkflowStep('second', '1', second, idempotent=True, retryable=True)]
    r = runner(path, steps)
    task = asyncio.create_task(r.run('r', space='s', scope={}, inputs=None))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    r.close()
    resumed = runner(path, steps)
    result = await resumed.run('r', space='s', scope={}, inputs=None)
    assert result.reused_steps == ('first',)
    assert result.results == {'first': 'prefix', 'second': 'prefix'}
    assert calls == ['first', 'second', 'second']
    resumed.close()


async def test_retry_exhaustion_survives_reopen(tmp_path):
    calls = []
    async def fails(context): calls.append(1); raise OSError('secret')
    path = tmp_path / 'w.db'
    steps = [WorkflowStep('a', '1', fails, idempotent=True, retryable=True)]
    r = runner(path, steps, max_retries=2)
    with pytest.raises(WorkflowError, match='step_failed'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()
    r = runner(path, steps, max_retries=2)
    with pytest.raises(WorkflowError, match='retries_exhausted'):
        await r.run('r', space='s', scope={}, inputs=None)
    assert len(calls) == 3
    r.close()


async def test_callback_mutations_do_not_change_binding_or_downstream_inputs(tmp_path):
    async def mutate(context):
        context.scope['owner'] = 'other'
        context.inputs['query'] = 'changed'
        return {'value': 1}
    async def next_step(context):
        assert context.scope == {'owner': 'alice'}
        assert context.inputs == {'query': 'original'}
        context.completed['a']['value'] = 99
        return 'done'
    r = runner(tmp_path / 'w.db', [WorkflowStep('a', '1', mutate), WorkflowStep('b', '1', next_step)])
    result = await r.run('r', space='s', scope={'owner': 'alice'}, inputs={'query': 'original'})
    assert result.results['a'] == {'value': 1}
    r.close()


@pytest.mark.parametrize('options', [
    {'key': b'short'}, {'max_payload_bytes': True}, {'max_payload_bytes': 1000001},
    {'max_retries': True}, {'max_retries': 4}, {'deadline': True}, {'deadline': float('nan')},
    {'deadline': 301},
])
def test_invalid_configuration_refused_before_creating_files(tmp_path, options):
    async def work(context): return None
    kwargs = {'key': KEY, **options}
    with pytest.raises(WorkflowError):
        WorkflowRunner(tmp_path / 'w.db', steps=[WorkflowStep('a', '1', work)], source_verifier=valid, **kwargs)
    assert not list(tmp_path.iterdir())


async def test_crash_inflight_never_replays_nonidempotent_step(tmp_path):
    import subprocess
    import sys
    path = tmp_path / 'crash.db'
    script = '''
import asyncio, os, sys
from scone_memory.agents import WorkflowRunner, WorkflowStep
async def valid(context): return True
async def crash(context): os._exit(23)
r = WorkflowRunner(sys.argv[1], key=b'k'*32, steps=[WorkflowStep('write','1',crash)], source_verifier=valid)
asyncio.run(r.run('r', space='s', scope={}, inputs=None))
'''
    process = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, timeout=10)
    assert process.returncode == 23
    async def forbidden(context): pytest.fail('crashed side effect was replayed')
    reopened = runner(path, [WorkflowStep('write', '1', forbidden)])
    with pytest.raises(WorkflowError, match='outcome_unknown'):
        await reopened.run('r', space='s', scope={}, inputs=None)
    reopened.close()


async def test_non_json_and_aggregate_payload_limits(tmp_path):
    async def work(context): return {'a': 'x' * 700}
    r = runner(tmp_path / 'aggregate.db', [WorkflowStep('a', '1', work), WorkflowStep('b', '1', work)], max_payload_bytes=1024)
    with pytest.raises(WorkflowError, match='payload_limit'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()
    r = runner(tmp_path / 'json.db', [WorkflowStep('a', '1', work)])
    for value in [float('nan'), {'not_json'}, '\ud800']:
        with pytest.raises(WorkflowError, match='invalid_payload'):
            await r.run('r', space='s', scope={}, inputs=value)
    r.close()


def test_duplicate_steps_and_insecure_journal_are_rejected(tmp_path):
    async def work(context): return None
    with pytest.raises(WorkflowError, match='invalid_steps'):
        runner(tmp_path / 'absent.db', [WorkflowStep('a', '1', work)] * 2)
    path = tmp_path / 'open.db'
    path.touch(mode=0o644)
    with pytest.raises(WorkflowError, match='private_file_required'):
        runner(path, [WorkflowStep('a', '1', work)])
    private = tmp_path / 'private.db'
    private.touch(mode=0o600)
    link = tmp_path / 'linked.db'
    link.symlink_to(private)
    with pytest.raises(WorkflowError, match='journal_unavailable'):
        runner(link, [WorkflowStep('a', '1', work)])


async def test_source_verifier_exceptions_do_not_leak_or_execute_steps(tmp_path):
    async def verify(context): raise RuntimeError('private source or credentials')
    async def forbidden(context): pytest.fail('step ran after verifier failure')
    r = runner(tmp_path / 'w.db', [WorkflowStep('a', '1', forbidden)], source_verifier=verify)
    with pytest.raises(WorkflowError, match='^sources_invalid$'):
        await r.run('r', space='s', scope={}, inputs=None)
    r.close()


async def test_resume_receipt_preserves_workflow_order_after_json_checkpoint(tmp_path):
    async def work(context): return 'ok'
    path = tmp_path / 'w.db'
    steps = [WorkflowStep('retrieve', '1', work), WorkflowStep('compose', '1', work)]
    r = runner(path, steps)
    await r.run('r', space='s', scope={}, inputs=None)
    r.close()
    r = runner(path, steps)
    assert (await r.run('r', space='s', scope={}, inputs=None)).reused_steps == ('retrieve', 'compose')
    r.close()


@pytest.mark.parametrize('steps', [None, ['not a step'], [], [False]])
def test_invalid_step_registry_has_safe_boundary_error(tmp_path, steps):
    with pytest.raises(WorkflowError, match='invalid_steps'):
        runner(tmp_path / 'w.db', steps)


async def test_non_mapping_scope_is_rejected(tmp_path):
    async def work(context): return 'ok'
    r = runner(tmp_path / 'w.db', [WorkflowStep('a', '1', work)])
    with pytest.raises(WorkflowError, match='invalid_payload'):
        await r.run('r', space='s', scope=None, inputs=None)
    r.close()
