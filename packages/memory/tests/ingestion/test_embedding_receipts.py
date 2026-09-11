"""A searchable receipt requires one valid embedding for each requested chunk."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.ports import NewEpisode
from scone_memory.ingestion.batch import EMBED_BATCH, recover
from scone_memory.ingestion.records import Record
from .test_ingestion_component import STAMP, runtime


class MalformedEmbedder(HashEmbedder):
    def __init__(self, damage, *, fail_call=1):
        super().__init__()
        self.damage = damage
        self.fail_call = fail_call
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        values = await super().embed(texts)
        if self.calls != self.fail_call:
            return values
        if self.damage == 'missing':
            return values[:-1]
        if self.damage == 'extra':
            return values + [values[0]]
        if self.damage == 'dimension':
            values[-1] = values[-1][:-1]
        elif self.damage == 'nan':
            values[-1][0] = float('nan')
        elif self.damage == 'boolean':
            values[-1][0] = True
        elif self.damage == 'non_numeric':
            values[-1][0] = '1.0'
        elif self.damage == 'not_vectors':
            return None
        return values


class ObservedStore(InMemoryDocumentStore):
    def __init__(self):
        super().__init__()
        self.insert_calls = 0

    async def insert_episode(self, episode):
        self.insert_calls += 1
        return await super().insert_episode(episode)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'dimension', 'nan', 'boolean', 'non_numeric', 'not_vectors'])
async def test_malformed_embeddings_cannot_issue_searchable_receipts_or_start_writes(damage):
    documents = ObservedStore()
    memory = await MemoryEngine(documents, InMemoryVectorIndex(), MalformedEmbedder(damage)).open()
    try:
        with pytest.raises(ValueError) as error:
            await memory.ingest_batch('alpha', [Record('First source'), Record('Second source')], request_id='request')
        assert documents.insert_calls == 0
        assert 'embedding' in str(error.value)
        assert (await documents.counts('alpha')).episodes == 0
        assert await documents.inflight() == []
        assert await memory.jobs('alpha') == []
        assert await documents.revision('alpha') == 0
    finally:
        await memory.close()


@pytest.mark.parametrize('damage', ['missing', 'extra'])
async def test_each_embedding_call_is_checked_before_batch_boundaries_can_shift_vectors(damage):
    documents = ObservedStore()
    embedder = MalformedEmbedder(damage, fail_call=2)
    memory = await MemoryEngine(documents, InMemoryVectorIndex(), embedder).open()
    try:
        records = [Record(f'Unique source {i}') for i in range(EMBED_BATCH + 2)]
        with pytest.raises(ValueError, match='embedding'):
            await memory.remember_many('alpha', records)
        assert embedder.calls == 2 and documents.insert_calls == 0
    finally:
        await memory.close()


@pytest.mark.parametrize('damage', ['missing', 'extra', 'dimension', 'nan'])
async def test_recovery_keeps_retry_marker_until_all_chunk_vectors_are_valid(damage):
    documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    embedder = MalformedEmbedder(damage)
    await vectors.ensure(embedder.dim)
    episode = await documents.insert_episode(NewEpisode(space='alpha', kind='note',
        content='Café recovery ' * 20, content_hash='interrupted', created_at=STAMP, ingested_at=STAMP))
    await documents.mark_inflight('alpha', 'interrupted')
    events = []
    with pytest.raises(ValueError, match='embedding'):
        await recover(runtime(documents, vectors, embedder, events))
    assert await documents.inflight() == [('alpha', 'interrupted')]
    assert events == []
    assert await vectors.ids('alpha') == []
    report = await recover(runtime(documents, vectors, HashEmbedder(), events))
    assert report.completed == 1 and await documents.inflight() == []
    chunks = await documents.chunks_of('alpha', episode.episode_id)
    assert len(chunks) > 1
    assert await vectors.ids('alpha') == sorted(chunk.chunk_id for chunk in chunks)


async def test_provider_response_buffer_reuse_cannot_rewrite_earlier_chunk_vectors():
    class ReusedBuffers:
        id, dim = 'fixture-buffers', 2
        def __init__(self):
            self.buffers = [[0., 0.] for _ in range(EMBED_BATCH)]
            self.calls = 0
        async def embed(self, texts):
            self.calls += 1
            for row in self.buffers:
                row[:] = [1., 0.] if self.calls == 1 else [0., 1.]
            return self.buffers[:len(texts)]
    vectors = InMemoryVectorIndex()
    memory = await MemoryEngine(InMemoryDocumentStore(), vectors, ReusedBuffers()).open()
    try:
        added = await memory.remember_many('alpha', [Record(f'Unique source {i}') for i in range(EMBED_BATCH + 1)])
        first, = await memory.documents.chunks_of('alpha', added[0].episode_id)
        last, = await memory.documents.chunks_of('alpha', added[-1].episode_id)
        scores = dict(await vectors.search('alpha', [1., 0.], EMBED_BATCH + 1))
        assert scores[first.chunk_id] == 1. and scores[last.chunk_id] == 0.
        assert len(await vectors.ids('alpha')) == EMBED_BATCH + 1
    finally:
        await memory.close()
