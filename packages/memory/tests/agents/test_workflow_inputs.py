"""Input requests pause dependencies without becoming uncertain model attempts."""
import asyncio
import sqlite3
import pytest
from scone_memory.agents import workflow as module
from scone_memory.agents.workflow import WorkflowError, WorkflowRunner, WorkflowStep

PARAMS = dict(run_id='r', space='alpha', scope={}, inputs='question')


async def valid(context):
    return True


def input_types():
    assert hasattr(module, 'WorkflowInputStep'), 'input scheduler nodes are missing'
    assert hasattr(module, 'WorkflowInputValue'), 'activated input values are missing'
    return module.WorkflowInputStep, module.WorkflowInputValue


def make(path, steps, dependencies, **options):
    return WorkflowRunner(path, key=b'k' * 32, steps=steps, dependencies=dependencies,
                          source_verifier=options.pop('source_verifier', valid), **options)


@pytest.mark.parametrize('width', [1, 2])
async def test_pending_input_does_not_starve_sibling_and_reopen_reuses_it(tmp_path, width):
    Input, Value = input_types(); activated = None; calls = []; polls = []
    async def poll(context): polls.append(context.run_id); return activated
    async def sibling(context): calls.append('b'); return {'text': ['retained']}
    async def dependent(context): calls.append('c'); assert context.completed['h'] is None; return 'done'
    steps = [Input('h', '1', poll), WorkflowStep('b', '1', sibling), WorkflowStep('c', '1', dependent)]
    deps = {'h': (), 'b': (), 'c': ('h',)}
    path = tmp_path / 'journal'
    work = make(path, steps, deps, max_parallel=width)
    try:
        for iteration in range(2):
            result = await work.run(**PARAMS)
            assert result.status == 'awaiting_input' and result.results == {'b': {'text': ['retained']}}
            state = work.status(**PARAMS)
            assert state.waiting_steps == ('h',) and state.attempts == {'b': 1} and not state.inflight_steps
            assert calls == ['b'] and len(polls) == iteration + 1
            work.close(); work = make(path, steps, deps, max_parallel=width)
        activated = Value(None)
        result = await work.run(**PARAMS)
        assert result.status == 'completed' and result.reused_steps == ('b',)
        assert calls == ['b', 'c'] and work.status(**PARAMS).attempts == {'b': 1, 'c': 1}
        assert work.status(**PARAMS).waiting_steps == ()
        assert (await work.read_result(**PARAMS)).results == result.results
    finally:
        work.close()


async def test_pending_input_is_visible_while_model_runs_and_inspection_is_detached(tmp_path):
    Input, _ = input_types(); entered = asyncio.Event(); release = asyncio.Event(); polls = []
    async def seed(context): return {'proof': ['saved']}
    async def sibling(context): entered.set(); await release.wait(); return 'b'
    async def poll(context): polls.append(1); return None
    steps = [WorkflowStep('a', '1', seed), WorkflowStep('b', '1', sibling), Input('h', '1', poll)]
    deps = {'a': (), 'b': ('a',), 'h': ('a',)}
    work = make(tmp_path/'journal', steps, deps, max_parallel=1)
    reader = make(tmp_path/'journal', steps, deps, max_parallel=1)
    task = asyncio.create_task(work.run(**PARAMS))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(20):
            if work.status(**PARAMS).waiting_steps: break
            await asyncio.sleep(0)
        state = work.status(**PARAMS)
        assert state.status == 'running' and state.waiting_steps == ('h',) and state.inflight_steps == ('b',)
        before = reader._db.execute('SELECT token,payload FROM workflow_runs ORDER BY token').fetchall()
        result = await reader.inspect_completed(**PARAMS)
        result['a']['proof'].append('mutated')
        assert (await work.inspect_completed(**PARAMS)) == {'a': {'proof': ['saved']}}
        assert reader._db.execute('SELECT token,payload FROM workflow_runs ORDER BY token').fetchall() == before
        assert polls == [1]
        release.set(); assert (await task).status == 'awaiting_input'
    finally:
        release.set(); task.cancel(); await asyncio.gather(task, return_exceptions=True); reader.close(); work.close()


@pytest.mark.parametrize('failure,code', [(False, 'sources_invalid'), (FileNotFoundError(), 'sources_invalid'),
                                        (ConnectionError('private'), 'verification_unavailable'),
                                        (sqlite3.OperationalError('private'), 'verification_unavailable')])
async def test_inspection_is_bound_and_never_mutates_on_verification_failure(tmp_path, failure, code):
    calls = []
    async def step(context): calls.append(1); return {'proof': ['saved']}
    state = True
    async def verifier(context):
        if isinstance(state, BaseException): raise state
        return state
    work = make(tmp_path/'journal', [WorkflowStep('a', '1', step)], {'a': ()}, source_verifier=verifier)
    try:
        assert hasattr(work, 'inspect_completed'), 'non-executing snapshot inspection is missing'
        assert await work.inspect_completed(**PARAMS) == {}
        await work.run(**PARAMS)
        before = work._db.execute('SELECT token,payload FROM workflow_runs ORDER BY token').fetchall()
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await work.inspect_completed(**{**PARAMS, 'inputs': 'changed'})
        state = failure
        with pytest.raises(WorkflowError, match=code): await work.inspect_completed(**PARAMS)
        assert before == work._db.execute('SELECT token,payload FROM workflow_runs ORDER BY token').fetchall()
        state = True
        assert await work.inspect_completed(**PARAMS) == {'a': {'proof': ['saved']}} and calls == [1]
    finally: work.close()


@pytest.mark.parametrize('failure,code', [(ConnectionError('private'), 'verification_unavailable'),
                                        (RuntimeError('private'), 'input_poll_failed')])
async def test_poll_failure_does_not_burn_input_or_new_model_attempt(tmp_path, failure, code):
    Input, Value = input_types(); broken = True; calls = []
    async def before(context): calls.append('a'); return 'saved'
    async def poll(context):
        if broken: raise failure
        return Value('reply')
    async def after(context): calls.append('c'); return 'done'
    work = make(tmp_path/'journal', [WorkflowStep('a','1',before),Input('h','1',poll),WorkflowStep('c','1',after)],
                {'a':(), 'h':('a',), 'c':('h',)})
    try:
        with pytest.raises(WorkflowError, match=code): await work.run(**PARAMS)
        assert work.status(**PARAMS).attempts == {'a':1}
        assert await work.inspect_completed(**PARAMS) == {'a':'saved'}
        broken = False
        assert (await work.run(**PARAMS)).status == 'completed' and calls == ['a','c']
    finally: work.close()


@pytest.mark.parametrize('reply', ['not wrapped', {'not': 'wrapped'}])
async def test_poll_rejects_unwrapped_values_without_marking_attempt(tmp_path, reply):
    Input, _ = input_types()
    async def poll(context): return reply
    work = make(tmp_path/'journal', [Input('h','1',poll)], {'h':()})
    try:
        with pytest.raises(WorkflowError, match='invalid_input_value'): await work.run(**PARAMS)
        assert work.status(**PARAMS).attempts == {}
    finally: work.close()


async def test_activated_input_respects_aggregate_budget(tmp_path):
    Input, Value = input_types()
    async def poll(context): return Value('x'*300)
    work = make(tmp_path/'journal', [Input('a','1',poll), Input('b','1',poll)], {'a':(), 'b':('a',)}, max_payload_bytes=512)
    try:
        with pytest.raises(WorkflowError, match='payload_limit'): await work.run(**PARAMS)
        assert work.status(**PARAMS).attempts == {} and work.status(**PARAMS).completed_steps == ('a',)
    finally: work.close()


def test_input_callback_validation_and_explicit_dependency_requirement(tmp_path):
    Input, _ = input_types()
    async def poll(context): return None
    with pytest.raises(WorkflowError, match='input_dependencies_required'):
        WorkflowRunner(tmp_path/'missing', key=b'k'*32, steps=[Input('h','1',poll)], source_verifier=valid)
    for callback in [lambda context: None, None]:
        with pytest.raises(WorkflowError, match='invalid_input_step'):
            make(tmp_path/'invalid', [Input('h','1',callback)], {'h':()})
    assert not (tmp_path/'invalid').exists() and not (tmp_path/'missing').exists()


def test_regular_step_revision_hashes_stay_byte_compatible(tmp_path):
    async def step(context): return 'x'
    cases = [({}, 'f02109ce3481d63c19f097c5268db4507d10aa86c645a32325562a37521eb8a7'),
             ({'dependencies':{'a':(),'b':('a',)},'max_parallel':2}, '5e4ec2f261b36c33af63d1a55f26315cd061191ae76898498773d6b86d59fd65'),
             ({'verify_before_step':True}, '4eb292c4b5e3a6f3a99070af679aac9b379c5234c4c0bae6de432e03cbac99cc')]
    for index, (options, expected) in enumerate(cases):
        work = WorkflowRunner(tmp_path/str(index), key=b'k'*32, steps=[WorkflowStep('a','1',step),WorkflowStep('b','2',step)], source_verifier=valid, **options)
        try: assert work._revision == expected
        finally: work.close()


async def test_cancel_with_waiting_input_drains_sibling_and_blocks_unknown_replay(tmp_path):
    Input, _ = input_types(); entered = asyncio.Event(); stopped = asyncio.Event(); polls = []
    async def poll(context): polls.append(1); return None
    async def sibling(context):
        entered.set()
        try: await asyncio.Event().wait()
        finally: stopped.set()
    work = make(tmp_path/'journal', [Input('h','1',poll),WorkflowStep('b','1',sibling)], {'h':(),'b':()})
    task = asyncio.create_task(work.run(**PARAMS))
    try:
        await asyncio.wait_for(entered.wait(), 2); task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert stopped.is_set() and work.status(**PARAMS).attempts == {'b':1}
        assert work.status(**PARAMS).waiting_steps == ('h',)
        with pytest.raises(WorkflowError, match='outcome_unknown'): await work.run(**PARAMS)
        assert polls == [1]
    finally: task.cancel(); await asyncio.gather(task, return_exceptions=True); work.close()


async def test_poll_outage_preserves_sibling_completed_during_the_poll(tmp_path):
    Input, Value = input_types(); release = asyncio.Event(); done = asyncio.Event(); broken = True; calls = []
    async def sibling(context):
        calls.append('b'); await release.wait(); done.set(); return {'saved':True}
    async def poll(context):
        if broken:
            release.set(); await done.wait(); raise sqlite3.OperationalError('private detail')
        return Value('reply')
    work = make(tmp_path/'journal', [WorkflowStep('b','1',sibling),Input('h','1',poll)], {'h':(),'b':()})
    try:
        with pytest.raises(WorkflowError, match='verification_unavailable'): await work.run(**PARAMS)
        state = work.status(**PARAMS)
        assert state.completed_steps == ('b',) and state.attempts == {'b':1} and state.inflight is None
        broken = False
        assert (await work.run(**PARAMS)).results == {'b':{'saved':True},'h':'reply'}
        assert calls == ['b']
    finally: work.close()


async def test_source_deleted_during_poll_prevents_input_consumption_and_dependents(tmp_path):
    Input, Value = input_types(); retained = True; calls = []
    async def verifier(context): return retained
    async def poll(context):
        nonlocal retained
        retained = False
        return Value('untrusted after deletion')
    async def dependent(context): calls.append(1); return 'never'
    work = make(tmp_path/'journal', [Input('h','1',poll),WorkflowStep('c','1',dependent)],
                {'h':(),'c':('h',)}, source_verifier=verifier)
    try:
        with pytest.raises(WorkflowError, match='sources_invalid'): await work.run(**PARAMS)
        assert work.status(**PARAMS).completed_steps == () and work.status(**PARAMS).attempts == {} and calls == []
    finally: work.close()


async def test_inspection_cancellation_never_cancels_active_work_or_changes_journal(tmp_path):
    Input, _ = input_types(); running = asyncio.Event(); release = asyncio.Event(); checking = asyncio.Event()
    async def poll(context): return None
    async def sibling(context): running.set(); await release.wait(); return 'saved'
    async def slow_verifier(context): checking.set(); await asyncio.Event().wait()
    steps = [Input('h','1',poll),WorkflowStep('b','1',sibling)]; deps = {'h':(),'b':()}
    work = make(tmp_path/'journal', steps, deps)
    reader = make(tmp_path/'journal', steps, deps, source_verifier=slow_verifier)
    task = asyncio.create_task(work.run(**PARAMS)); inspection = None
    try:
        await asyncio.wait_for(running.wait(), 2)
        before = reader._db.execute('SELECT token,payload FROM workflow_runs ORDER BY token').fetchall()
        inspection = asyncio.create_task(reader.inspect_completed(**PARAMS))
        await asyncio.wait_for(checking.wait(), 2); inspection.cancel()
        with pytest.raises(asyncio.CancelledError): await inspection
        assert not task.done() and before == reader._db.execute('SELECT token,payload FROM workflow_runs ORDER BY token').fetchall()
        release.set(); assert (await task).results == {'b':'saved'}
    finally:
        release.set(); task.cancel()
        if inspection is not None: inspection.cancel(); await asyncio.gather(inspection, return_exceptions=True)
        await asyncio.gather(task, return_exceptions=True); reader.close(); work.close()


async def test_payload_refusal_has_failed_status_without_uncertain_input_attempt(tmp_path):
    Input, Value = input_types()
    async def poll(context): return Value('x'*300)
    work = make(tmp_path/'journal', [Input('a','1',poll),Input('b','1',poll)], {'a':(),'b':('a',)}, max_payload_bytes=512)
    try:
        with pytest.raises(WorkflowError, match='payload_limit'): await work.run(**PARAMS)
        assert work.status(**PARAMS).status == 'failed'
        assert work.status(**PARAMS).attempts == {}
    finally: work.close()


async def test_input_versions_bind_reopened_journal_and_unordered_dependencies_work(tmp_path):
    Input, Value = input_types()
    async def poll(context): return Value('answer')
    async def dependent(context): return context.completed['h']
    deps = {'c':('h',),'h':()}; path = tmp_path/'journal'
    work = make(path, [WorkflowStep('c','1',dependent),Input('h','1',poll)], deps)
    try: assert (await work.run(**PARAMS)).results == {'h':'answer','c':'answer'}
    finally: work.close()
    work = make(path, [WorkflowStep('c','1',dependent),Input('h','2',poll)], deps)
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'): await work.inspect_completed(**PARAMS)
        with pytest.raises(WorkflowError, match='binding_mismatch'): await work.run(**PARAMS)
    finally: work.close()


@pytest.mark.parametrize('committed', [False, True])
async def test_input_publication_storage_failure_reconciles_without_model_replay(tmp_path, monkeypatch, committed):
    Input, Value = input_types(); polls = []; calls = []
    async def poll(context): polls.append(1); return Value({'reply':'accepted'})
    async def dependent(context): calls.append(1); return 'done'
    steps = [Input('h','1',poll), WorkflowStep('c','1',dependent)]; deps = {'h':(),'c':('h',)}
    path = tmp_path/'journal'; work = make(path, steps, deps); save = work._save; broken = True
    def interrupted(token, state):
        nonlocal broken
        if broken and 'h' in state['results']:
            broken = False
            if committed: save(token, state)
            raise sqlite3.OperationalError('private simulated storage failure')
        save(token, state)
    monkeypatch.setattr(work, '_save', interrupted)
    try:
        with pytest.raises(WorkflowError, match='journal_unavailable'): await work.run(**PARAMS)
        assert calls == [] and work.status(**PARAMS).attempts == {}
        work.close(); work = make(path, steps, deps)
        assert (await work.run(**PARAMS)).results == {'h':{'reply':'accepted'}, 'c':'done'}
        assert len(polls) == (1 if committed else 2) and calls == [1]
    finally: work.close()


async def test_inspection_does_not_publish_snapshot_invalidated_by_active_owner(tmp_path):
    Input, _ = input_types(); active = asyncio.Event(); release = asyncio.Event(); checked = asyncio.Event(); publish = asyncio.Event()
    retained = True
    async def verifier(context): return retained
    async def inspection_verifier(context): checked.set(); await publish.wait(); return True
    async def seed(context): return 'retained source'
    async def model(context): active.set(); await release.wait(); return 'later'
    async def poll(context): return None
    steps = [WorkflowStep('a','1',seed),WorkflowStep('b','1',model),Input('h','1',poll)]
    deps = {'a':(),'b':('a',),'h':('a',)}
    work = make(tmp_path/'journal', steps, deps, source_verifier=verifier)
    reader = make(tmp_path/'journal', steps, deps, source_verifier=inspection_verifier)
    task = asyncio.create_task(work.run(**PARAMS)); inspection = None
    try:
        await asyncio.wait_for(active.wait(),2)
        inspection = asyncio.create_task(reader.inspect_completed(**PARAMS)); await asyncio.wait_for(checked.wait(),2)
        retained = False; release.set()
        with pytest.raises(WorkflowError, match='sources_invalid'): await task
        publish.set()
        with pytest.raises(WorkflowError, match='sources_invalid'): await inspection
        assert work.status(**PARAMS).completed_steps == ()
    finally:
        release.set(); publish.set(); task.cancel(); await asyncio.gather(task,return_exceptions=True)
        if inspection is not None: inspection.cancel(); await asyncio.gather(inspection,return_exceptions=True)
        reader.close(); work.close()
