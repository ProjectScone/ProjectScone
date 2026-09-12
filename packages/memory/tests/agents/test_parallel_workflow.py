"""Durable dependency scheduling under concurrent completion and failure."""
import asyncio
import pytest
from scone_memory.agents.workflow import WorkflowError, WorkflowRunner, WorkflowStep


async def valid(context):
    return True


def runner(path, steps, dependencies, **options):
    return WorkflowRunner(path, key=b'k' * 32, steps=steps, source_verifier=valid,
        dependencies=dependencies, max_parallel=2, verify_before_step=True, **options)


async def test_bounded_overlap_fanin_and_independent_checkpoint_leases(tmp_path):
    entered = asyncio.Event(); release = asyncio.Event(); active = 0; peak = 0
    started = []; leases = []
    def branch(name):
        async def execute(context):
            nonlocal active, peak
            active += 1; peak = max(peak, active); started.append(name)
            context.checkpoints.put('same', name.encode()); leases.append(context.checkpoints)
            if len(started) == 2: entered.set()
            try:
                await release.wait()
                assert context.checkpoints.get('same') == name.encode()
                return name
            finally: active -= 1
        return execute
    async def join(context):
        assert set(context.completed) == {'a', 'b', 'c'}
        assert active == 0
        return 'joined'
    steps = [WorkflowStep(name, '1', branch(name)) for name in ['a', 'b', 'c']]
    steps.append(WorkflowStep('join', '1', join))
    work = runner(tmp_path/'run', steps, {'a': (), 'b': (), 'c': (), 'join': ('a', 'b', 'c')})
    task = asyncio.create_task(work.run('r', space='alpha', scope={}, inputs='question'))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert started == ['a', 'b']
        status = work.status('r', space='alpha', scope={}, inputs='question')
        assert status.inflight_steps == ('a', 'b')
        release.set(); result = await task
        assert peak == 2 and result.results['join'] == 'joined'
        for lease in leases:
            with pytest.raises(WorkflowError, match='checkpoint_inactive'): lease.get('same')
        again = await work.run('r', space='alpha', scope={}, inputs='question')
        assert again.reused_steps == ('a', 'b', 'c', 'join') and started == ['a', 'b', 'c']
    finally:
        release.set(); task.cancel(); await asyncio.gather(task, return_exceptions=True); work.close()


async def test_failure_preserves_committed_branch_and_refuses_unknown_replay(tmp_path):
    released = asyncio.Event(); entered = asyncio.Event(); calls = []
    async def first(context): calls.append('first'); return 'retained'
    async def uncertain(context):
        calls.append('uncertain'); entered.set(); await released.wait(); raise RuntimeError('PRIVATE failure')
    async def blocked(context): calls.append('blocked'); return 'must not run'
    steps = [WorkflowStep('first','1',first),WorkflowStep('uncertain','1',uncertain),WorkflowStep('blocked','1',blocked)]
    deps = {'first': (), 'uncertain': (), 'blocked': ('first','uncertain')}
    work = runner(tmp_path/'run',steps,deps)
    task = asyncio.create_task(work.run('r',space='alpha',scope={},inputs='q'))
    try:
        await entered.wait()
        for _ in range(30):
            if work.status('r',space='alpha',scope={},inputs='q').completed_steps:break
            await asyncio.sleep(0)
        assert work.status('r',space='alpha',scope={},inputs='q').completed_steps == ('first',)
        released.set()
        with pytest.raises(WorkflowError,match='step_failed'):await task
        assert work.status('r',space='alpha',scope={},inputs='q').completed_steps == ('first',)
        work.close();work=runner(tmp_path/'run',steps,deps)
        with pytest.raises(WorkflowError,match='outcome_unknown'):await work.run('r',space='alpha',scope={},inputs='q')
        assert calls == ['first','uncertain']
    finally:
        released.set();task.cancel();await asyncio.gather(task,return_exceptions=True);work.close()


@pytest.mark.parametrize('deadline', [False, True])
async def test_cancel_or_deadline_drains_every_worker(tmp_path, deadline):
    entered=asyncio.Event();started=[];stopped=[]
    def step(name):
        async def execute(context):
            started.append(name)
            if len(started)==2:entered.set()
            try:await asyncio.Event().wait()
            finally:stopped.append(name)
        return execute
    work=runner(tmp_path/'run',[WorkflowStep(name,'1',step(name)) for name in ['a','b']],{'a':(),'b':()},deadline=0.2 if deadline else 10)
    task=asyncio.create_task(work.run('r',space='alpha',scope={},inputs='q'))
    try:
        await asyncio.wait_for(entered.wait(),2)
        if not deadline:task.cancel()
        with pytest.raises(WorkflowError if deadline else asyncio.CancelledError):await task
        assert set(stopped)=={'a','b'}
        with pytest.raises(WorkflowError,match='outcome_unknown'):await work.run('r',space='alpha',scope={},inputs='q')
        assert len(started)==2
    finally:
        task.cancel();await asyncio.gather(task,return_exceptions=True);work.close()


async def test_schedule_changes_refuse_old_receipts_and_validate_before_files(tmp_path):
    async def step(context):return 'result'
    steps=[WorkflowStep('a','1',step),WorkflowStep('b','1',step)]
    work=runner(tmp_path/'run',steps,{'a':(),'b':()})
    await work.run('r',space='alpha',scope={},inputs='q');work.close()
    changed=runner(tmp_path/'run',steps,{'a':(),'b':('a',)})
    try:
        with pytest.raises(WorkflowError,match='binding_mismatch'):await changed.run('r',space='alpha',scope={},inputs='q')
    finally:changed.close()
    for deps in [{'a':('b',),'b':('a',)},{'a':(),'b':('missing',)},{'a':()}]:
        path=tmp_path/'invalid'
        with pytest.raises(WorkflowError):runner(path,steps,deps)
        assert not path.exists()


@pytest.mark.parametrize('names', [('good','bad'), ('bad','good')])
async def test_simultaneous_success_is_committed_even_if_failure_is_first(tmp_path,names):
    barrier=asyncio.Event();entered=[]
    def callback(name):
        async def execute(context):
            entered.append(name)
            if len(entered)==2:barrier.set()
            await barrier.wait()
            if name=='bad':raise ValueError('private callback detail')
            return 'keep this receipt'
        return execute
    work=runner(tmp_path/'run',[WorkflowStep(name,'1',callback(name)) for name in names],{name:() for name in names})
    try:
        with pytest.raises(WorkflowError,match='step_failed'):await work.run('r',space='alpha',scope={},inputs='q')
        assert work.status('r',space='alpha',scope={},inputs='q').completed_steps==('good',)
    finally:work.close()


async def test_invalidated_sources_cannot_be_republished_by_cancellation_cleanup(tmp_path):
    alive=asyncio.Event();stopped=asyncio.Event();seen=[]
    async def good(context):return 'source-backed'
    async def slow(context):
        alive.set()
        try:await asyncio.Event().wait()
        except asyncio.CancelledError:return 'late output must be discarded'
        finally:stopped.set()
    async def blocked(context):seen.append('blocked');return 'not reached'
    async def verify(context):return not bool(context.completed)
    work=WorkflowRunner(tmp_path/'run',key=b'k'*32,
        steps=[WorkflowStep('good','1',good),WorkflowStep('slow','1',slow),WorkflowStep('blocked','1',blocked)],
        source_verifier=verify,dependencies={'good':(),'slow':(),'blocked':('good',)},max_parallel=2)
    try:
        with pytest.raises(WorkflowError,match='sources_invalid'):await work.run('r',space='alpha',scope={},inputs='q')
        assert alive.is_set() and stopped.is_set() and not seen
        status=work.status('r',space='alpha',scope={},inputs='q')
        assert status.status=='sources_invalid' and not status.completed_steps
    finally:work.close()


async def test_worker_result_is_copied_before_other_task_can_mutate_it(tmp_path):
    result={'text':'original'}
    async def first(context):return result
    async def mutate(context):result['text']='changed';return 'other'
    work=runner(tmp_path/'run',[WorkflowStep('first','1',first),WorkflowStep('mutate','1',mutate)],{'first':(),'mutate':()})
    try:
        answer=await work.run('r',space='alpha',scope={},inputs='q')
        assert answer.results['first']=={'text':'original'}
    finally:work.close()


async def test_second_cancellation_during_drain_keeps_journal_owned(tmp_path):
    entered=asyncio.Event();draining=asyncio.Event();release=asyncio.Event();count=0
    async def step(context):
        nonlocal count
        count+=1
        if count==2:entered.set()
        try:await asyncio.Event().wait()
        finally:draining.set();await release.wait()
    steps=[WorkflowStep(name,'1',step) for name in ['a','b']]
    work=runner(tmp_path/'run',steps,{'a':(),'b':()})
    other=runner(tmp_path/'run',steps,{'a':(),'b':()})
    task=asyncio.create_task(work.run('r',space='alpha',scope={},inputs='q'))
    try:
        await entered.wait();task.cancel();await draining.wait();task.cancel();await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(WorkflowError,match='busy'):await other.run('r',space='alpha',scope={},inputs='q')
        release.set()
        with pytest.raises(asyncio.CancelledError):await task
        with pytest.raises(WorkflowError,match='outcome_unknown'):await other.run('r',space='alpha',scope={},inputs='q')
    finally:
        release.set();task.cancel();await asyncio.gather(task,return_exceptions=True);work.close();other.close()


@pytest.mark.parametrize('fault', ['storage','budget'])
async def test_publication_failure_drains_active_sibling(tmp_path,monkeypatch,fault):
    import sqlite3
    stopped=asyncio.Event();alive=asyncio.Event()
    async def fast(context):await alive.wait();return 'x'*600 if fault=='budget' else 'good'
    async def slow(context):
        alive.set()
        try:await asyncio.Event().wait()
        finally:stopped.set()
    work=runner(tmp_path/'run',[WorkflowStep('fast','1',fast),WorkflowStep('slow','1',slow)],{'fast':(),'slow':()},max_payload_bytes=512)
    original=work._save
    def fail_save(token,state):
        if state['results']:raise sqlite3.OperationalError('private storage path')
        return original(token,state)
    if fault=='storage':monkeypatch.setattr(work,'_save',fail_save)
    try:
        with pytest.raises(WorkflowError,match='journal_unavailable' if fault=='storage' else 'payload_limit'):
            await work.run('r',space='alpha',scope={},inputs='q')
        assert stopped.is_set()
        monkeypatch.setattr(work,'_save',original)
        with pytest.raises(WorkflowError,match='outcome_unknown'):await work.run('r',space='alpha',scope={},inputs='q')
    finally:work.close()


async def test_native_selected_agents_overlap_and_join_only_declared_outputs(tmp_path):
    from scone_memory import MemoryEngine,InMemoryDocumentStore,InMemoryVectorIndex,HashEmbedder
    from scone_memory.agents.catalog import AgentCatalog,AgentModel,AgentDefinition
    from scone_memory.agents.evidence_loop import ToolStep
    from scone_memory.agents.task_workflow import AgentWorkflow,AgentTask,AgentTaskPlan
    from scone_memory.retrieval.recall_scope import RecallScope
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    entered=asyncio.Event();release=asyncio.Event();calls=[]
    class Model:
        def __init__(self,name):self.name=name
        async def complete(self,messages,tools):
            calls.append((self.name,messages))
            if self.name!='join':
                if len(calls)==2:entered.set()
                await release.wait()
            return ToolStep(content=self.name+' answer')
    catalog=AgentCatalog(models=[AgentModel(name,name,'1',lambda name=name:Model(name)) for name in ['left','right','join']],agents=[
        AgentDefinition(agent_id='research',instructions='Use evidence.',models=('left','right','join'),default_model='left',initial_search=False)])
    plan=AgentTaskPlan(workflow_id='report',tasks=tuple(AgentTask(task_id=name,agent_id='research',model_id=name,prompt=name,depends_on=('left','right') if name=='join' else ()) for name in ['left','right','join']))
    work=AgentWorkflow(tmp_path/'native',key=b'k'*32,catalog=catalog,plan=plan,memory=memory,space='alpha',scope=RecallScope.validated(),max_parallel=2)
    task=asyncio.create_task(work.run('r','Question'))
    try:
        await asyncio.wait_for(entered.wait(),2);assert len(calls)==2
        release.set();result=await task
        assert set(result.results)=={'left','right','join'}
        joined=str(calls[-1][1]);assert 'left answer' in joined and 'right answer' in joined
        assert result.results['left']['model_id']=='left'
        assert len((await work.read_result('r','Question')).results)==3 and len(calls)==3
        work.close()
        work=AgentWorkflow(tmp_path/'native',key=b'k'*32,catalog=catalog,plan=plan,memory=memory,space='alpha',scope=RecallScope.validated(),max_parallel=2)
        assert len((await work.run('r','Question')).reused_steps)==3 and len(calls)==3
    finally:
        release.set();task.cancel();await asyncio.gather(task,return_exceptions=True);work.close();await memory.close()


async def test_admission_verification_outage_preserves_finished_receipt(tmp_path):
    finished=asyncio.Event();checks=0;calls=[];outage=True
    async def first(context):calls.append('first');finished.set();return 'already complete'
    async def second(context):calls.append('second');return 'second'
    async def verify(context):
        nonlocal checks
        checks+=1
        if checks==3 and outage:
            await finished.wait();await asyncio.sleep(0);raise ConnectionError('temporary outage')
        return True
    work=WorkflowRunner(tmp_path/'run',key=b'k'*32,steps=[WorkflowStep('first','1',first),WorkflowStep('second','1',second)],
        source_verifier=verify,dependencies={'first':(),'second':()},max_parallel=2)
    try:
        with pytest.raises(WorkflowError,match='verification_unavailable'):await work.run('r',space='alpha',scope={},inputs='q')
        assert work.status('r',space='alpha',scope={},inputs='q').completed_steps==('first',)
        outage=False
        result=await work.run('r',space='alpha',scope={},inputs='q')
        assert result.reused_steps==('first',) and calls==['first','second']
    finally:work.close()


async def test_source_invalidation_revokes_running_checkpoint_leases(tmp_path):
    refused=[]
    async def first(context):return 'source-backed'
    async def second(context):
        context.checkpoints.put('earlier',b'work')
        try:await asyncio.Event().wait()
        except asyncio.CancelledError:
            try:context.checkpoints.put('late',b'invalidated source work')
            except WorkflowError as error:refused.append(error.code)
            raise
    async def blocked(context):raise AssertionError('must not run')
    async def verify(context):return not bool(context.completed)
    work=WorkflowRunner(tmp_path/'run',key=b'k'*32,steps=[WorkflowStep('first','1',first),WorkflowStep('second','1',second),WorkflowStep('blocked','1',blocked)],
        source_verifier=verify,dependencies={'first':(),'second':(),'blocked':('first',)},max_parallel=2)
    try:
        with pytest.raises(WorkflowError,match='sources_invalid'):await work.run('r',space='alpha',scope={},inputs='q')
        assert refused==['checkpoint_inactive']
        status=work.status('r',space='alpha',scope={},inputs='q')
        assert status.checkpoint_count==0 and not status.completed_steps
    finally:work.close()
