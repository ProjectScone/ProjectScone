"""A service owns indexing work; retained messages never await an embedder."""
import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.deferred_capture import DeferredTextCapture
from scone_memory.realtime.text import TextConversation
from .test_deferred_capture import ControlledEmbedder, message
from .test_text_conversation import ScriptedModel


async def test_service_retains_messages_while_embedding_is_blocked():
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    service = DeferredTextCapture(memory)
    service.start()
    try:
        embedder.release.clear()
        first = await asyncio.wait_for(service.capture('alpha', message()), 1)
        await asyncio.wait_for(embedder.entered.wait(), 1)
        second = await asyncio.wait_for(service.capture('alpha', message('two')), 1)
        assert first.indexing == second.indexing == 'pending'
        assert first.episode_id != second.episode_id
        assert len(await memory.documents.inflight()) == 2
        embedder.release.set()
        await asyncio.wait_for(service.wait_idle(), 1)
        assert not await memory.documents.inflight()
    finally:
        await service.aclose()
        await memory.close()


async def test_queue_overflow_keeps_durable_recovery_intent_and_close_joins_worker():
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    service = DeferredTextCapture(memory, max_pending=1)
    service.start()
    try:
        embedder.release.clear()
        await service.capture('alpha', message())
        await asyncio.wait_for(embedder.entered.wait(), 1)
        overflow = await asyncio.wait_for(service.capture('alpha', message('two')), 1)
        assert overflow.indexing == 'pending'
        await asyncio.wait_for(service.aclose(), 1)
        assert len(await memory.documents.inflight()) == 2
        with pytest.raises(RuntimeError, match='closed'):
            await service.capture('alpha', message('three'))
        embedder.release.set()
        assert (await memory.recover()).completed == 2
    finally:
        await service.aclose()
        await memory.close()


async def test_failed_indexing_retries_without_recapturing_and_stops_at_limit():
    class BrokenEmbedder(HashEmbedder):
        calls = 0

        async def embed(self, texts):
            self.calls += 1
            raise RuntimeError('synthetic embedding failure')

    embedder = BrokenEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    service = DeferredTextCapture(memory, max_attempts=2, retry_delay=0)
    service.start()
    try:
        captured = await service.capture('alpha', message())
        await asyncio.wait_for(service.wait_idle(), 1)
        assert embedder.calls == 2
        assert (await memory.documents.get_episode('alpha', captured.episode_id)).content == message().content
        assert len(await memory.documents.inflight()) == 1
        assert (await memory.documents.counts('alpha')).episodes == 1
    finally:
        await service.aclose()
        await memory.close()


async def test_native_reply_finishes_before_document_embeddings_and_reports_pending():
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    service = DeferredTextCapture(memory)
    service.start()
    conversation = TextConversation(memory, 'alpha', 'fast', ScriptedModel,
                                    deferred_capture=service, recall_timeout=0.01)
    try:
        embedder.release.clear()
        result = await asyncio.wait_for(conversation.reply('Hello'), 1)
        assert result['text'] == 'Scripted reply.'
        assert result['capture_indexing'] == {'mode': 'deferred', 'user': 'pending', 'assistant': 'pending'}
        for role in ('user', 'assistant'):
            assert await memory.documents.get_episode('alpha', result[role + '_episode_id'])
    finally:
        await conversation.close()
        await service.aclose()
        await memory.close()


async def test_deferred_capture_requires_same_engine_and_rejects_tool_mode():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    other = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    service = DeferredTextCapture(memory)
    try:
        with pytest.raises(ValueError, match='same memory'):
            TextConversation(other, 'alpha', 'wrong', ScriptedModel, deferred_capture=service)
        with pytest.raises(ValueError, match='tool'):
            TextConversation(memory, 'alpha', 'tools', deferred_capture=service, tool_model_factory=lambda: object())
        with pytest.raises(ValueError, match='128000'):
            TextConversation(memory, 'alpha', 'oversized', ScriptedModel, deferred_capture=service,
                             max_reply_bytes=140000, max_history_bytes=200000)
    finally:
        await service.aclose()
        await memory.close()
        await other.close()


def test_worker_limits_are_explicit_and_bounded():
    memory = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
    for options in ({'max_pending': 0}, {'max_pending': True}, {'max_attempts': 0},
                    {'max_attempts': 101}, {'retry_delay': float('nan')}, {'retry_delay': -1}):
        with pytest.raises(ValueError):
            DeferredTextCapture(memory, **options)


async def test_cancelling_close_still_joins_capture_and_its_worker(monkeypatch):
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    service = DeferredTextCapture(memory)
    service.start()
    entered, release = asyncio.Event(), asyncio.Event()
    original = memory.capture_text

    async def held_capture(space, record):
        result = await original(space, record)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(memory, 'capture_text', held_capture)
    capture = asyncio.create_task(service.capture('alpha', message()))
    try:
        embedder.release.clear()
        await entered.wait()
        closing = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done(), 'close must retain ownership while a capture settles'
        release.set()
        await capture
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 1)
        assert not any(task.get_name() == 'scone-text-indexing' and not task.done()
                       for task in asyncio.all_tasks())
        assert await memory.documents.inflight()
    finally:
        release.set()
        await asyncio.gather(capture, return_exceptions=True)
        await service.aclose()
        await memory.close()


async def test_same_engine_recovery_waits_for_active_indexing_and_repairs_cancellation():
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    captured = await memory.capture_text('alpha', message())
    embedder.release.clear()
    indexing = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
    recovery = None
    try:
        await embedder.entered.wait()
        recovery = asyncio.create_task(memory.recover())
        await asyncio.sleep(0)
        assert embedder.calls == 1, 'recovery must not race a worker writing the same vectors'
        indexing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await indexing
        embedder.release.set()
        assert (await asyncio.wait_for(recovery, 1)).completed == 1
        assert not await memory.documents.inflight()
        assert await memory.vectors.ids('alpha')
    finally:
        embedder.release.set()
        await asyncio.gather(indexing, *([recovery] if recovery is not None else []), return_exceptions=True)
        await memory.close()
