"""Batch writes and recovery run against ports without an engine instance."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex
from scone_memory.core.ports import NewEpisode

STAMP = '2026-09-08T00:00:00.000Z'


def runtime(documents, vectors, embedder, events):
    from scone_memory.ingestion.batch import IngestionRuntime
    async def emit(space, kind, payload):
        events.append((space, kind, payload))
        return None
    return IngestionRuntime(documents, vectors, embedder, lambda: STAMP, 80,
        lambda episode, text: text, emit)


async def test_component_deduplicates_and_preserves_original_utf8_chunks():
    from scone_memory.ingestion.batch import remember_many
    from scone_memory.ingestion.records import Record
    documents, vectors, embedder = InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    await vectors.ensure(embedder.dim)
    content = 'Café calibration uses Polaris. ' * 8
    added = await remember_many(runtime(documents, vectors, embedder, []), 'alpha',
        [Record(content), Record('Separate record'), Record(content)])
    assert [a.deduplicated for a in added] == [False, False, True]
    assert added[0].episode_id == added[2].episode_id
    chunks = await documents.chunks_of('alpha', added[0].episode_id)
    assert ''.join(c.text for c in chunks) == content
    assert all(content.encode()[c.start:c.end].decode() == c.text for c in chunks)
    assert await documents.revision('alpha') == 1
    assert await documents.inflight() == []


async def test_component_rolls_back_partial_vector_failure(monkeypatch):
    from scone_memory.ingestion.batch import remember_many
    from scone_memory.ingestion.records import Record
    documents, vectors, embedder = InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    await vectors.ensure(embedder.dim)
    original = vectors.upsert
    calls = 0
    async def upsert(points):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError('second write fails')
        await original(points)
    monkeypatch.setattr(vectors, 'upsert', upsert)
    with pytest.raises(ConnectionError):
        await remember_many(runtime(documents, vectors, embedder, []), 'alpha', [Record('first'), Record('second')])
    counts = await documents.counts('alpha')
    assert counts.episodes == counts.chunks == 0
    assert await documents.inflight() == []
    assert await documents.revision('alpha') == 0
    added = await remember_many(runtime(documents, vectors, embedder, []), 'alpha', [Record('first'), Record('second')])
    assert all(not item.deduplicated for item in added)


async def test_component_recovers_missing_chunks_and_drops_orphan_marks():
    from scone_memory.ingestion.batch import recover
    documents, vectors, embedder = InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    await vectors.ensure(embedder.dim)
    episode = await documents.insert_episode(NewEpisode(space='alpha', kind='note', content='Café Polaris ' * 12,
        content_hash='kept', created_at=STAMP, ingested_at=STAMP))
    await documents.mark_inflight('alpha', 'kept')
    await documents.mark_inflight('alpha', 'missing')
    events = []
    report = await recover(runtime(documents, vectors, embedder, events))
    assert (report.completed, report.rechunked, report.forgotten) == (1, 1, 1)
    chunks = await documents.chunks_of('alpha', episode.episode_id)
    assert ''.join(c.text for c in chunks) == episode.content
    assert await documents.inflight() == []
    assert events == [('alpha', 'recover', {'marks':2, 'completed':1, 'rechunked':1, 'forgotten':1})]
