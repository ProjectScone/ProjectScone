"""Tool conversation factories release clients before public text or capture."""

import asyncio

import pytest

from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.realtime.text import TextConversation
from ..agents.test_agent_model_cleanup import Model
from ..agents.test_task_workflow import memory


async def test_tool_conversation_closes_each_fresh_model_before_callback(memory):
    models = []
    observed = []

    def factory():
        model = Model()
        models.append(model)
        return model

    conversation = TextConversation(memory, 'alpha', 'conversation', tool_model_factory=factory)

    async def observe(text):
        assert models[-1].closed == 1 and models[-1].client.is_closed
        observed.append(text)

    try:
        await conversation.reply('First', on_text=observe)
        await conversation.reply('Second', on_text=observe)
        assert observed == ['Done', 'Done'] and len(models) == 2
        assert all(model.closed == 1 for model in models)
        assert len(await memory.episodes('alpha', {'session_id': 'conversation'})) == 4
    finally:
        await conversation.close()


@pytest.mark.parametrize('failure', ['model', 'invalid', 'cleanup'])
async def test_failure_closes_client_and_withholds_publication(memory, failure):
    class Broken(Model):
        async def complete(self, messages, tools):
            if failure == 'model':
                raise RuntimeError('private provider failure')
            return ToolStep(content='Done')

        async def aclose(self):
            await super().aclose()
            if failure == 'cleanup':
                raise RuntimeError('private cleanup failure')

    model = Broken()
    if failure == 'invalid':
        model.complete = None
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, 'alpha', 'conversation', tool_model_factory=lambda: model)
    try:
        with pytest.raises(RuntimeError):
            await conversation.reply('Question', on_text=observe)
        assert model.closed == 1 and model.client.is_closed and not observed
        assert [
            row.metadata['role'] for row in await memory.episodes('alpha', {'session_id': 'conversation'})
        ] == ['user']
    finally:
        await conversation.close()


async def test_conversation_shutdown_joins_cleanup_before_returning(memory):
    closing = asyncio.Event()
    release = asyncio.Event()
    observed = []

    class Waiting(Model):
        async def aclose(self):
            closing.set()
            await release.wait()
            await super().aclose()

    model = Waiting()

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, 'alpha', 'conversation', tool_model_factory=lambda: model)
    running = asyncio.create_task(conversation.reply('Question', on_text=observe))
    stopping = None
    try:
        await asyncio.wait_for(closing.wait(), 1)
        stopping = asyncio.create_task(conversation.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not stopping.done() and not running.done() and not observed
        release.set()
        await stopping
        with pytest.raises(asyncio.CancelledError):
            await running
        assert model.closed == 1 and model.client.is_closed and not observed
        assert [
            row.metadata['role'] for row in await memory.episodes('alpha', {'session_id': 'conversation'})
        ] == ['user']
    finally:
        release.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, *([stopping] if stopping else []), return_exceptions=True)
        await conversation.close()
        await model.client.aclose()


async def test_forgetting_during_cleanup_refuses_before_callback_and_capture(memory):
    source = await memory.remember('alpha', 'Juniper is in Oregon.')

    class Forgetting(Model):
        async def aclose(self):
            await memory.forget('alpha', source.episode_id)
            await super().aclose()

    model = Forgetting()
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(
        memory, 'alpha', 'conversation', tool_model_factory=lambda: model, tool_initial_search=True
    )
    try:
        with pytest.raises(RuntimeError, match='evidence'):
            await conversation.reply('Juniper', on_text=observe)
        assert model.closed == 1 and not observed
        assert [
            row.metadata['role'] for row in await memory.episodes('alpha', {'session_id': 'conversation'})
        ] == ['user']
    finally:
        await conversation.close()
