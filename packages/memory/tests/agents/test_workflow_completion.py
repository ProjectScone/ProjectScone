"""Bounded sequential workflows can finish on a verified result prefix."""
import pytest
from scone_memory.agents.workflow import WorkflowCompletion,WorkflowRunner,WorkflowStep,WorkflowError


async def valid(context):return True


def make(path,calls,*,version='1',predicate=None,verify=valid):
    async def first(context):calls.append('first');return {'done':True,'answer':'finished'}
    async def second(context):calls.append('second');return 'should not run'
    return WorkflowRunner(path,key=b'k'*32,steps=[WorkflowStep('first','1',first),WorkflowStep('second','1',second)],
        source_verifier=verify,completion=WorkflowCompletion(version,predicate or (lambda context: bool(context.completed))))


async def test_completed_prefix_reopens_and_reads_without_remaining_callbacks(tmp_path):
    calls=[];work=make(tmp_path/'run',calls)
    try:
        result=await work.run('r',space='alpha',scope={},inputs='q')
        assert result.results=={'first':{'done':True,'answer':'finished'}} and calls==['first']
        assert work.status('r',space='alpha',scope={},inputs='q').status=='completed'
        work.close();work=make(tmp_path/'run',calls)
        assert (await work.read_result('r',space='alpha',scope={},inputs='q')).reused_steps==('first',)
        assert (await work.run('r',space='alpha',scope={},inputs='q')).reused_steps==('first',)
        assert calls==['first']
    finally:work.close()


async def test_changed_condition_revision_refuses_old_prefix(tmp_path):
    calls=[];work=make(tmp_path/'run',calls)
    await work.run('r',space='alpha',scope={},inputs='q');work.close()
    work=make(tmp_path/'run',calls,version='2',predicate=lambda context:False)
    try:
        with pytest.raises(WorkflowError,match='binding_mismatch'):await work.run('r',space='alpha',scope={},inputs='q')
        assert calls==['first']
    finally:work.close()


async def test_prefix_still_requires_fresh_evidence(tmp_path):
    calls=[];retained=True
    async def verify(context):return retained
    work=make(tmp_path/'run',calls,verify=verify)
    try:
        await work.run('r',space='alpha',scope={},inputs='q');retained=False
        with pytest.raises(WorkflowError,match='sources_invalid'):await work.read_result('r',space='alpha',scope={},inputs='q')
        assert not work.status('r',space='alpha',scope={},inputs='q').completed_steps and calls==['first']
    finally:work.close()


@pytest.mark.parametrize('value',[None,1,'yes'])
async def test_completion_requires_a_real_boolean(tmp_path,value):
    calls=[];work=make(tmp_path/'run',calls,predicate=lambda context:value)
    try:
        with pytest.raises(WorkflowError,match='completion_failed'):await work.run('r',space='alpha',scope={},inputs='q')
        assert calls==[]
    finally:work.close()


def test_async_completion_and_parallel_completion_are_rejected_before_files(tmp_path):
    async def step(context):return 'x'
    async def condition(context):return True
    path=tmp_path/'invalid'
    with pytest.raises(WorkflowError):WorkflowCompletion('1',condition)
    assert not path.exists()
    with pytest.raises(WorkflowError):WorkflowRunner(path,key=b'k'*32,steps=[WorkflowStep('one','1',step)],
        source_verifier=valid,dependencies={'one':()},max_parallel=2,completion=WorkflowCompletion('1',lambda context:False))
    assert not path.exists()


async def test_completed_prefix_survives_temporary_verification_outage(tmp_path):
    calls=[];outage=False
    async def verify(context):
        if outage:raise ConnectionError('temporary')
        return True
    work=make(tmp_path/'run',calls,verify=verify)
    try:
        await work.run('r',space='alpha',scope={},inputs='q');outage=True
        with pytest.raises(WorkflowError,match='verification_unavailable'):await work.read_result('r',space='alpha',scope={},inputs='q')
        outage=False
        assert (await work.read_result('r',space='alpha',scope={},inputs='q')).reused_steps==('first',)
        assert calls==['first']
    finally:work.close()


async def test_completion_cannot_hide_an_unresolved_attempt(tmp_path):
    done=False;calls=[]
    async def first(context):calls.append('first');return 'ok'
    async def second(context):calls.append('second');raise RuntimeError('unknown side effect')
    work=WorkflowRunner(tmp_path/'run',key=b'k'*32,steps=[WorkflowStep('first','1',first),WorkflowStep('second','1',second)],
        source_verifier=valid,completion=WorkflowCompletion('1',lambda context:done))
    try:
        with pytest.raises(WorkflowError,match='step_failed'):await work.run('r',space='alpha',scope={},inputs='q')
        done=True
        with pytest.raises(WorkflowError,match='outcome_unknown'):await work.run('r',space='alpha',scope={},inputs='q')
        assert calls==['first','second']
    finally:work.close()
