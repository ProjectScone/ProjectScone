import asyncio
import time
from dataclasses import replace
import pytest
from .test_tool_approval_recovery import host
from .test_task_workflow import memory
from .test_approval_inspection import reader
from .test_custom_tools import call
from scone_memory.agents.approval_inspection import inspect_tool_approval
from scone_memory.agents.turn_journal import TurnJournalInspection
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.workflow import WorkflowError,WorkflowPausableStep,WorkflowRunner

async def test_later_unexecuted_call_is_not_current_pending_call(host,tmp_path):
    later=call({'count':4},call_id='second').calls[0]
    host.model.steps=[ToolStep(calls=(*call().calls,later))]
    await host.run()
    record,=host.store.list('alpha','one')
    job=reader(host,tmp_path)
    try:
        snapshot,=await job.inspect_pauses('one',space='alpha',scope={},inputs='Count')
        changed=record.model_copy(update={'call':record.call.model_copy(update={
            'arguments_json':'{"count":4}',
            'operation_digest':TurnJournalInspection.read(snapshot.payload).pending_identity('custom',later.model_dump(mode='json'))})})
        with pytest.raises(WorkflowError):await inspect_tool_approval(snapshot,changed,host.agent,host.scoped)
    finally:job.close()

@pytest.mark.parametrize('swallow',[False,True])
async def test_pause_inspection_cannot_succeed_after_verifier_deadline(host,tmp_path,swallow):
    await host.run()
    async def slow(ctx):
        if swallow:
            try:await asyncio.sleep(.05)
            except asyncio.CancelledError:pass
        else:time.sleep(.03)
        return True
    job=WorkflowRunner(tmp_path/'workflow',key=b'k'*32,steps=[WorkflowPausableStep('count','1',host.execute)],
        source_verifier=slow,deadline=.01)
    try:
        with pytest.raises(WorkflowError):await job.inspect_pauses('one',space='alpha',scope={},inputs='Count')
    finally:job.close()

from .test_evidence_tool_loop import search
import scone_memory.agents.approval_inspection as inspection
from types import SimpleNamespace

@pytest.mark.parametrize('swallow',[False,True])
async def test_approval_restore_must_not_ack_after_its_timeout(host,memory,tmp_path,monkeypatch,swallow):
    await memory.remember('alpha','Juniper uses Polaris.',metadata={'team':'blue'})
    host.model.steps=[search(),call()]
    await host.run()
    record,=host.store.list('alpha','one')
    job=reader(host,tmp_path)
    try:
        snapshot,=await job.inspect_pauses('one',space='alpha',scope={},inputs='Count')
        original_restore=host.scoped.restore
        async def slow_restore(*args,**kwargs):
            evidence=await original_restore(*args,**kwargs)
            async def late_validate():
                valid=await evidence.validate()
                if swallow:
                    try:await asyncio.sleep(.05)
                    except asyncio.CancelledError:pass
                else:time.sleep(.03)
                return valid
            return replace(evidence,_validator=late_validate)
        host.scoped.restore=slow_restore
        timeout=asyncio.timeout
        monkeypatch.setattr(inspection,'asyncio',SimpleNamespace(timeout=lambda seconds:timeout(.01), current_task=asyncio.current_task,
            get_running_loop=asyncio.get_running_loop, CancelledError=asyncio.CancelledError))
        with pytest.raises(WorkflowError):await inspect_tool_approval(snapshot,record,host.agent,host.scoped)
    finally:job.close()
