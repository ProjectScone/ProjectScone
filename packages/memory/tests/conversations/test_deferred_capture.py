"""Durable messages must not await vectors, or resurrect deleted sources."""
import asyncio
from dataclasses import replace

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.errors import InvalidInput


class ControlledEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def embed(self, texts):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return await super().embed(texts)


def message(key='one'):
    return Record('Lyra launch owner is Leo.', kind='conversation', source='session',
        metadata={'session_id': 'session', 'turn_id': key, 'role': 'user'}, dedup_key=key)


async def test_capture_is_durable_and_text_searchable_without_embedding(tmp_path):
    embedder = ControlledEmbedder()
    path = tmp_path / 'memory.db'
    memory = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        assert embedder.calls == 0
        assert captured.indexing == 'pending'
        assert (await memory.documents.get_episode('alpha', captured.episode_id)).content == message().content
        found = await memory.recall('alpha', 'Lyra launch', lanes=('text',))
        assert found.items[0].episode_id == captured.episode_id
        assert not (await memory.recall('beta', 'Lyra launch', lanes=('text',))).items
        assert await memory.documents.inflight()
    finally:
        await memory.close()
    reopened = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), ControlledEmbedder()).open()
    try:
        assert not await reopened.documents.inflight()
        assert (await reopened.recall('alpha', 'Lyra launch', lanes=('vector',))).items
    finally:
        await reopened.close()


async def test_capture_deduplicates_and_indexes_only_once():
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    try:
        first = await memory.capture_text('alpha', message())
        second = await memory.capture_text('alpha', message())
        assert second.episode_id == first.episode_id
        assert await memory.index_captured_text('alpha', first.episode_id) is True
        assert await memory.index_captured_text('alpha', first.episode_id) is True
        assert embedder.calls == 1
        assert (await memory.capture_text('alpha', message())).indexing == 'ready'
    finally:
        await memory.close()


async def test_forget_during_embedding_does_not_recreate_vectors():
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        embedder.release.clear()
        task = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
        await embedder.entered.wait()
        await memory.forget('alpha', captured.episode_id)
        embedder.release.set()
        assert await task is False
        assert await vectors.ids('alpha') == []
        assert not await memory.documents.inflight()
    finally:
        embedder.release.set()
        await memory.close()


async def test_cancelled_indexing_keeps_the_recovery_intent():
    embedder = ControlledEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        embedder.release.clear()
        task = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
        await embedder.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await memory.documents.inflight()
        embedder.release.set()
        assert (await memory.recover()).completed == 1
        assert not await memory.documents.inflight()
    finally:
        embedder.release.set()
        await memory.close()


@pytest.mark.parametrize('failure', [OSError, asyncio.CancelledError])
async def test_lost_clear_acknowledgment_never_erases_vectors_and_recovery_intent(failure):
    class LostAcknowledgment(InMemoryDocumentStore):
        fail = False

        async def clear_inflight(self, space, digest):
            await super().clear_inflight(space, digest)
            if self.fail:
                self.fail = False
                raise failure('lost acknowledgment')

    store, vectors = LostAcknowledgment(), InMemoryVectorIndex()
    memory = await MemoryEngine(store, vectors, ControlledEmbedder()).open()
    try:
        captured = await memory.capture_text('alpha', message())
        store.fail = True
        with pytest.raises(failure):
            await memory.index_captured_text('alpha', captured.episode_id)
        assert await vectors.ids('alpha') or await store.inflight()
        assert await memory.index_captured_text('alpha', captured.episode_id)
        assert await vectors.ids('alpha')
    finally:
        await memory.close()


async def test_recovery_uses_the_same_forget_safe_indexing_path():
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        embedder.release.clear()
        repair = asyncio.create_task(memory.recover())
        await embedder.entered.wait()
        await memory.forget('alpha', captured.episode_id)
        embedder.release.set()
        report = await repair
        assert report.completed == 0
        assert report.forgotten == 1
        assert await vectors.ids('alpha') == []
    finally:
        embedder.release.set()
        await memory.close()


async def test_replaced_key_keeps_the_new_sources_recovery_intent():
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        embedder.release.clear()
        indexing = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
        await embedder.entered.wait()
        await memory.forget('alpha', captured.episode_id)
        newer = await memory.capture_text('alpha', replace(message(), content='Lyra launch owner is Mira.'))
        embedder.release.set()
        assert await indexing is False
        assert await memory.documents.inflight()
        assert await vectors.ids('alpha') == []
        assert await memory.index_captured_text('alpha', newer.episode_id)
        assert await vectors.ids('alpha')
    finally:
        embedder.release.set()
        await memory.close()


async def test_expiry_while_embedding_cannot_publish_vectors():
    now = ['2026-10-03T10:00:00Z']
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder, clock=lambda: now[0]).open()
    try:
        captured = await memory.capture_text('alpha', replace(message(), forget_after='2026-10-03T10:01:00Z'))
        embedder.release.clear()
        indexing = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
        await embedder.entered.wait()
        now[0] = '2026-10-03T10:02:00Z'
        embedder.release.set()
        assert await indexing is False
        assert await vectors.ids('alpha') == []
    finally:
        embedder.release.set()
        await memory.close()


async def test_capture_duplicate_refuses_different_tags_or_metadata():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), ControlledEmbedder()).open()
    try:
        original = replace(message(), tags=('private',))
        captured = await memory.capture_text('alpha', original)
        for changed in [replace(original, tags=('public',)), replace(original, metadata={'session_id': 'other'})]:
            with pytest.raises(InvalidInput):
                await memory.capture_text('alpha', changed)
        assert (await memory.capture_text('alpha', original)).episode_id == captured.episode_id
        assert (await memory.capture_text('alpha', replace(original, dedup_key='two'))).episode_id != captured.episode_id
    finally:
        await memory.close()


async def test_wrong_space_episode_is_refused_before_embedding():
    class WrongSpace(InMemoryDocumentStore):
        foreign = None

        async def get_episode(self, space, episode_id):
            return self.foreign or await super().get_episode(space, episode_id)

    store, embedder, vectors = WrongSpace(), ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(store, vectors, embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        store.foreign = await store.get_episode('alpha', captured.episode_id)
        with pytest.raises(InvalidInput):
            await memory.index_captured_text('beta', captured.episode_id)
        assert embedder.calls == 0
        assert await vectors.ids('beta') == []
    finally:
        await memory.close()


@pytest.mark.parametrize('patch', [{'space': 'beta'}, {'episode_id': 999}, {'text': 'invented text'},
                                  {'start': 1}, {'ordinal': 2}])
async def test_wrong_chunk_identity_or_content_cannot_be_indexed(patch):
    class WrongChunks(InMemoryDocumentStore):
        corrupt = False

        async def chunks_of(self, space, episode_id):
            chunks = await super().chunks_of(space, episode_id)
            return [chunk.model_copy(update=patch) for chunk in chunks] if self.corrupt else chunks

    store, embedder, vectors = WrongChunks(), ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(store, vectors, embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        store.corrupt = True
        with pytest.raises(InvalidInput):
            await memory.index_captured_text('alpha', captured.episode_id)
        assert embedder.calls == 0
        assert await vectors.ids('alpha') == []
        assert await store.inflight()
    finally:
        await memory.close()


async def test_marker_alone_does_not_exempt_ordinary_ingestion_from_forget_guard():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), ControlledEmbedder()).open()
    try:
        added = await memory.remember('alpha', 'Ordinary pending text.', kind='conversation',
                                      metadata={'capture_indexing': 'deferred'})
        episode = await memory.documents.get_episode('alpha', added.episode_id)
        await memory.documents.mark_inflight('alpha', episode.content_hash)
        with pytest.raises(InvalidInput, match='indexing is unfinished'):
            await memory.forget('alpha', added.episode_id)
    finally:
        await memory.close()


async def test_recovery_rechunks_an_interrupted_capture_without_changing_source():
    class InterruptedChunks(InMemoryDocumentStore):
        fail = False

        async def insert_chunks(self, chunks):
            if self.fail:
                self.fail = False
                raise OSError('before chunks committed')
            return await super().insert_chunks(chunks)

    store = InterruptedChunks()
    memory = await MemoryEngine(store, InMemoryVectorIndex(), ControlledEmbedder()).open()
    try:
        store.fail = True
        with pytest.raises(OSError):
            await memory.capture_text('alpha', message())
        report = await memory.recover()
        assert report.completed == 1
        assert report.rechunked == 1
        assert not await store.inflight()
        assert (await memory.recall('alpha', 'Lyra launch', lanes=('vector',))).items[0].text == message().content
    finally:
        await memory.close()


@pytest.mark.parametrize('cancel', [False, True])
async def test_deletion_or_cancellation_during_vector_write_leaves_no_orphan_vectors(cancel):
    class HeldWrite(InMemoryVectorIndex):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert(self, points):
            await super().upsert(points)
            self.entered.set()
            await self.release.wait()

    vectors = HeldWrite()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, ControlledEmbedder()).open()
    try:
        captured = await memory.capture_text('alpha', message())
        indexing = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
        await vectors.entered.wait()
        if cancel:
            indexing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await indexing
            assert await memory.documents.inflight()
        else:
            await memory.forget('alpha', captured.episode_id)
            vectors.release.set()
            assert await indexing is False
        assert await vectors.ids('alpha') == []
    finally:
        vectors.release.set()
        await memory.close()


async def test_changed_source_or_chunks_while_embedding_are_not_published():
    for change in ('metadata', 'chunk'):
        store, embedder, vectors = InMemoryDocumentStore(), ControlledEmbedder(), InMemoryVectorIndex()
        memory = await MemoryEngine(store, vectors, embedder).open()
        try:
            captured = await memory.capture_text('alpha', message())
            embedder.release.clear()
            indexing = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
            await embedder.entered.wait()
            if change == 'metadata':
                # A frozen model can still contain a mutable metadata mapping.
                store._episodes[captured.episode_id].metadata['role'] = 'assistant'
            else:
                chunk = (await store.chunks_of('alpha', captured.episode_id))[0]
                store._chunks[chunk.chunk_id] = chunk.model_copy(update={'text': 'changed'})
            embedder.release.set()
            assert await indexing is False
            assert await vectors.ids('alpha') == []
            assert await store.inflight()
        finally:
            embedder.release.set()
            await memory.close()


async def test_recovery_never_embeds_a_source_whose_expiry_passed():
    now = ['2026-10-03T10:00:00Z']
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder, clock=lambda: now[0]).open()
    try:
        await memory.capture_text('alpha', replace(message(), forget_after='2026-10-03T10:01:00Z'))
        now[0] = '2026-10-03T10:02:00Z'
        report = await memory.recover()
        assert report.completed == 0
        assert report.forgotten == 1
        assert embedder.calls == 0
        assert await vectors.ids('alpha') == []
    finally:
        await memory.close()


async def test_deleted_space_while_embedding_cannot_regain_vectors():
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder).open()
    try:
        captured = await memory.capture_text('alpha', message())
        embedder.release.clear()
        indexing = asyncio.create_task(memory.index_captured_text('alpha', captured.episode_id))
        await embedder.entered.wait()
        await memory.delete_space('alpha')
        embedder.release.set()
        assert await indexing is False
        assert await vectors.ids('alpha') == []
        assert not await memory.documents.inflight()
    finally:
        embedder.release.set()
        await memory.close()


async def test_ordinary_ingestion_cannot_forge_the_native_deferred_contract():
    embedder, vectors = ControlledEmbedder(), InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, embedder).open()
    try:
        native_metadata = {'capture_indexing': 'deferred', 'capture_schema': '1',
                           'capture_key': 'one', 'chunking': 'length'}
        with pytest.raises(InvalidInput, match='native deferred capture'):
            await memory.remember_many('alpha', [replace(message(), metadata=native_metadata)])
        assert embedder.calls == 0
        assert (await memory.documents.counts('alpha')).episodes == 0
        assert not await memory.documents.inflight()
        assert not await vectors.ids('alpha')
    finally:
        await memory.close()


async def test_old_index_acknowledgment_cannot_clear_a_recaptured_keys_intent():
    class HeldClear(InMemoryDocumentStore):
        def __init__(self):
            super().__init__()
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.hold = True

        async def clear_inflight(self, space, digest):
            if self.hold:
                self.hold = False
                self.entered.set()
                await self.release.wait()
            await super().clear_inflight(space, digest)

    store, vectors = HeldClear(), InMemoryVectorIndex()
    memory = await MemoryEngine(store, vectors, ControlledEmbedder()).open()
    indexing = forgetting = recapture = None
    try:
        first = await memory.capture_text('alpha', message())
        indexing = asyncio.create_task(memory.index_captured_text('alpha', first.episode_id))
        await store.entered.wait()
        # Retirement can remove the source while the old clear owns its short
        # write section. Both retirement's clear and a new capture must wait.
        forgetting = asyncio.create_task(memory.forget('alpha', first.episode_id))
        for _ in range(20):
            if await store.get_episode('alpha', first.episode_id) is None:
                break
            await asyncio.sleep(0)
        assert await store.get_episode('alpha', first.episode_id) is None
        recapture = asyncio.create_task(memory.capture_text('alpha', replace(message(), content='New owner is Mira.')))
        await asyncio.sleep(0)
        assert not recapture.done(), 'capture must not publish an intent while an old clear is pending'
        store.release.set()
        await indexing
        await forgetting
        newer = await recapture
        assert newer.episode_id != first.episode_id and newer.indexing == 'pending'
        assert await store.inflight()
        assert await memory.index_captured_text('alpha', newer.episode_id)
        assert await vectors.ids('alpha')
    finally:
        store.release.set()
        await asyncio.gather(*(task for task in (indexing, forgetting, recapture) if task is not None), return_exceptions=True)
        await memory.close()


async def test_retirement_acknowledgment_cannot_clear_a_recaptured_keys_intent():
    class HeldRetirementClear(InMemoryDocumentStore):
        def __init__(self):
            super().__init__()
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.hold = True

        async def clear_inflight(self, space, digest):
            if self.hold:
                self.hold = False
                self.entered.set()
                await self.release.wait()
            await super().clear_inflight(space, digest)

    store = HeldRetirementClear()
    memory = await MemoryEngine(store, InMemoryVectorIndex(), ControlledEmbedder()).open()
    forgetting = recapture = None
    try:
        first = await memory.capture_text('alpha', message())
        forgetting = asyncio.create_task(memory.forget('alpha', first.episode_id))
        await store.entered.wait()
        recapture = asyncio.create_task(memory.capture_text('alpha', replace(message(), content='New owner is Mira.')))
        await asyncio.sleep(0)
        assert not recapture.done()
        store.release.set()
        await forgetting
        newer = await recapture
        assert newer.indexing == 'pending' and await store.inflight()
        assert await memory.index_captured_text('alpha', newer.episode_id)
        assert not await store.inflight()
    finally:
        store.release.set()
        await asyncio.gather(*(task for task in (forgetting, recapture) if task is not None), return_exceptions=True)
        await memory.close()
