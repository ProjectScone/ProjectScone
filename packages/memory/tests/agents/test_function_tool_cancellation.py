"""Argument conversion cannot admit a function after its turn has ended."""
import asyncio
import threading
import time
import pytest
from scone_memory.agents.function_tools import function_tool
from scone_memory.agents.function_types import ParameterType
from scone_memory.agents.custom_tools import ToolContext
from scone_memory.retrieval.recall_scope import RecallScope

@pytest.mark.parametrize('cancel',[False,True])
async def test_pending_decode_must_not_admit_effect_after_turn_ends(monkeypatch,cancel):
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    original=ParameterType.decode;invoked=[]
    def decode(self,value):
        entered.set();release.wait(2)
        try:return original(self,value)
        finally:finished.set()
    monkeypatch.setattr(ParameterType,'decode',decode)
    def effect(value:int)->object:
        invoked.append(value);return value
    registration=function_tool(effect,revision='1',description='Effect')
    context=ToolContext('alpha',RecallScope.validated(),None,time.monotonic()+(.5 if not cancel else 5))
    pending=asyncio.create_task(registration.invoke({'value':1},context))
    try:
        assert await asyncio.to_thread(entered.wait,1)
        if cancel:pending.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):await pending
    finally:
        release.set()
        if not pending.done():
            pending.cancel();await asyncio.gather(pending,return_exceptions=True)
    assert await asyncio.to_thread(finished.wait,1)
    await asyncio.sleep(.01)
    assert not invoked

async def test_cancelled_dispatch_marker_does_not_cancel_another_invocation(monkeypatch):
    from scone_memory.agents.custom_tools import _dispatch_abort
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    original=ParameterType.decode;invoked=[]
    def decode(self,value):
        if value==1:
            entered.set();release.wait(3)
            try:return original(self,value)
            finally:finished.set()
        return original(self,value)
    monkeypatch.setattr(ParameterType,'decode',decode)
    def effect(value:int)->object:
        invoked.append(value);return value
    registration=function_tool(effect,revision='1',description='Effect')
    context=ToolContext('alpha',RecallScope.validated(),None,time.monotonic()+5)
    first=asyncio.create_task(registration.invoke({'value':1},context))
    try:
        assert await asyncio.to_thread(entered.wait,1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):await first
        assert _dispatch_abort.get() is None
        await registration.invoke({'value':2},context)
    finally:
        release.set()
        if not first.done():
            first.cancel();await asyncio.gather(first,return_exceptions=True)
    assert await asyncio.to_thread(finished.wait,1)
    await asyncio.sleep(.01)
    assert invoked==[2]
    assert _dispatch_abort.get() is None
