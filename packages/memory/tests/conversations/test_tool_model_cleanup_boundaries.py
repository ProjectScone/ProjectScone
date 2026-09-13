"""Independent conversation cleanup ownership and deadline regressions."""

import asyncio
import time
import pytest
from ..agents.test_task_workflow import memory
from scone_memory.realtime.text import TextConversation
from scone_memory.agents.evidence_loop import ToolStep


async def test_outer_turn_deadline_fences_blocking_cleanup(memory):
    closed = []
    observed = []

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

        async def aclose(self):
            time.sleep(0.08)
            closed.append(True)

    conversation = TextConversation(
        memory, 'alpha', 'outer-deadline', tool_model_factory=Model, turn_timeout=0.05
    )

    async def observe(text):
        observed.append(text)

    try:
        with pytest.raises(TimeoutError):
            await conversation.reply('Q', on_text=observe)
        assert closed == [True] and observed == []
        episodes = await memory.episodes('alpha', {'session_id': 'outer-deadline'})
        assert [e.metadata['role'] for e in episodes] == ['user']
    finally:
        await conversation.close()


async def test_close_joins_owned_cleanup_even_if_close_caller_cancelled_twice(memory):
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = []
    observed = []

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

        async def aclose(self):
            closing.set()
            await release.wait()
            closed.append(True)

    conversation = TextConversation(memory, 'alpha', 'close-join', tool_model_factory=Model)

    async def observe(text):
        observed.append(text)

    reply = asyncio.create_task(conversation.reply('Q', on_text=observe))
    shutdown = None
    try:
        await asyncio.wait_for(closing.wait(), 1)
        shutdown = asyncio.create_task(conversation.close())
        await asyncio.sleep(0)
        shutdown.cancel()
        await asyncio.sleep(0)
        shutdown.cancel()
        await asyncio.sleep(0)
        assert not shutdown.done() and not reply.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        with pytest.raises(asyncio.CancelledError):
            await reply
        assert closed == [True] and observed == [] and conversation.closed
        episodes = await memory.episodes('alpha', {'session_id': 'close-join'})
        assert [e.metadata['role'] for e in episodes] == ['user']
    finally:
        release.set()
        await asyncio.gather(reply, *([shutdown] if shutdown else []), return_exceptions=True)
        await conversation.close()


async def test_scoped_tool_construction_failure_still_disposes_factory_model(memory, monkeypatch):
    import scone_memory.realtime.text as module

    closed = []

    class Model:
        async def complete(self, messages, tools):
            raise AssertionError('must not execute')

        async def aclose(self):
            closed.append(True)

    def refuse(*args, **kwargs):
        raise RuntimeError('binding unavailable')

    monkeypatch.setattr(module, 'ScopedMemoryTools', refuse)
    conversation = TextConversation(memory, 'alpha', 'binding-failure', tool_model_factory=Model)
    try:
        with pytest.raises(RuntimeError):
            await conversation.reply('Q')
        assert closed == [True] and conversation.closed
    finally:
        await conversation.close()


async def test_callback_observes_already_closed_model_and_source_revocation_blocks_capture(memory):
    source = await memory.remember('alpha', 'Juniper uses Polaris.')
    closed = []
    observed = []

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Juniper uses Polaris.')

        async def aclose(self):
            closed.append(True)

    conversation = TextConversation(
        memory, 'alpha', 'callback-source', tool_model_factory=Model, tool_initial_search=True
    )

    async def observe(text):
        assert closed == [True]
        observed.append(text)
        await memory.forget('alpha', source.episode_id)

    try:
        with pytest.raises(RuntimeError, match='evidence'):
            await conversation.reply('Juniper', on_text=observe)
        assert observed == ['Juniper uses Polaris.']
        episodes = await memory.episodes('alpha', {'session_id': 'callback-source'})
        assert [e.metadata['role'] for e in episodes] == ['user']
    finally:
        await conversation.close()


async def test_cleanup_self_close_is_refused_without_deadlock_or_publication(memory):
    observed = []
    entered = []

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

        async def aclose(self):
            entered.append(True)
            await conversation.close()

    conversation = TextConversation(memory, 'alpha', 'self-close', tool_model_factory=Model)

    async def observe(text):
        observed.append(text)

    try:
        with pytest.raises(RuntimeError, match='agent_model_cleanup_failed'):
            await asyncio.wait_for(conversation.reply('Q', on_text=observe), 1)
        assert entered == [True] and observed == [] and conversation.closed
    finally:
        await conversation.close()
