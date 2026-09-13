"""Regression cases found during independent custom-tool review."""
import asyncio,time
import pytest
from scone_memory.agents.custom_tools import ToolContext
from scone_memory.retrieval.recall_scope import RecallScope
from scone_memory.agents.evidence_loop import ToolStep
from .test_custom_tools import tool,agents,call,memory
from .test_evidence_tool_loop import Script,binding

def test_read_only_schema_aliases_are_bounded_before_expansion():
    from types import MappingProxyType
    visits = []
    class CountedList(list):
        def __iter__(self):
            for value in super().__iter__():
                visits.append(1)
                yield value
    aliases = CountedList([0] * 10)
    for _ in range(3):
        aliases = CountedList([aliases] * 10)
    with pytest.raises(ValueError):
        tool(lambda args, ctx: None,
             parameters=MappingProxyType({'type': 'object', 'examples': aliases}))
    assert len(visits) <= 4096

async def test_bound_schema_cannot_change_without_binding_change(memory):
    invoked=[]
    model=Script(call({'count':'not-an-integer'}),ToolStep(content='Done'))
    bound=agents(model,[tool(lambda args,ctx:invoked.append(args))]).bind('worker')
    original=bound.fingerprint
    try:bound.tools[0].parameters.clear()
    except (TypeError,AttributeError):return
    try:
        await bound.run('Question',tools=binding(memory))
    except (ValueError,RuntimeError):pass
    print('binding unchanged',bound.fingerprint==original,'invoked',invoked)
    assert not invoked,'bound schema weakened under unchanged fingerprint'

async def test_already_expired_context_never_invokes_handler():
    invoked=[]
    async def execute(args,ctx):invoked.append(1);return 'done'
    registration=tool(execute)
    context=ToolContext('alpha',RecallScope.validated(),None,asyncio.get_running_loop().time()-1)
    try:await registration.invoke({'count':3},context)
    except (RuntimeError,asyncio.CancelledError):pass
    print('expired direct invocation',invoked)
    assert not invoked,'expired invocation called handler'

async def test_validation_crossing_deadline_does_not_execute_side_effect(memory,monkeypatch):
    import scone_memory.agents.custom_tools as module
    from scone_memory.agents.evidence_loop import EvidenceToolLoop,ToolLoopLimits
    real=module.accepts_schema
    invoked=[]
    async def execute(args,ctx):invoked.append(1);return 'done'
    def slow(*args):time.sleep(.03);return real(*args)
    monkeypatch.setattr(module,'accepts_schema',slow)
    model=Script(call(),ToolStep(content='Must not happen'))
    work=EvidenceToolLoop(model,binding(memory),custom_tools=[tool(execute)],limits=ToolLoopLimits(timeout_s=.01))
    try:await work.run([{'role':'user','content':'Question'}])
    except (RuntimeError,TimeoutError):pass
    print('expired after validation',invoked)
    assert not invoked,'validation crossed deadline but handler still invoked'

async def test_already_oversized_transcript_never_invokes_handler(memory):
    from scone_memory.agents.evidence_loop import EvidenceToolLoop,ToolLoopLimits
    invoked=[]
    async def execute(args,ctx):invoked.append(1);return 'done'
    registration=tool(execute,parameters={'type':'object','properties':{'text':{'type':'string'}},'required':['text']})
    model=Script(call({'text':'x'*2000}),ToolStep(content='Must not happen'))
    work=EvidenceToolLoop(model,binding(memory),custom_tools=[registration],limits=ToolLoopLimits(max_transcript_bytes=1024))
    try:await work.run([{'role':'user','content':'Question'}])
    except RuntimeError:pass
    print('over transcript before handler',invoked)
    assert not invoked,'already oversized transcript dispatched application effect'

async def test_cancelled_sync_factory_closes_late_coroutine_result(recwarn):
    import threading,gc
    started=threading.Event();release=threading.Event();finished=threading.Event()
    async def deferred():return 'done'
    def execute(args,ctx):
        started.set();release.wait(2)
        result=deferred();finished.set();return result
    registration=tool(execute)
    context=ToolContext('alpha',RecallScope.validated(),None,asyncio.get_running_loop().time()+5)
    pending=asyncio.create_task(registration.invoke({'count':3},context))
    try:
        assert await asyncio.to_thread(started.wait,1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):await pending
    finally:
        release.set();await asyncio.to_thread(finished.wait,1)
    await asyncio.sleep(.01);gc.collect()
    notices=[str(w.message) for w in recwarn if 'was never awaited' in str(w.message)]
    print('late coroutine warnings',notices)
    assert not notices

async def test_exhausted_tool_output_budget_never_dispatches_next_handler(memory):
    import json
    from scone_memory.agents.evidence_loop import EvidenceToolLoop,ToolLoopLimits
    invoked=[]
    overhead=len(json.dumps({'ok':True,'status':'prepared','verified_accuracy':False,'source_status':'unverified','result':''},separators=(',',':')).encode())
    async def execute(args,ctx):
        invoked.append(args['count'])
        return 'x'*(512-overhead) if args['count']==1 else None
    model=Script(call({'count':1},call_id='first'),call({'count':2},call_id='second'),ToolStep(content='Done'))
    work=EvidenceToolLoop(model,binding(memory),custom_tools=[tool(execute)],limits=ToolLoopLimits(max_tool_bytes=512))
    try:await work.run([{'role':'user','content':'Question'}])
    except RuntimeError:pass
    print('exhausted tool output invoked',invoked)
    assert invoked==[1],'second effect ran with zero bytes left for any result'
