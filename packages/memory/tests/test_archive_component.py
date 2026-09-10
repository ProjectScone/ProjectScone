"""Archive transfer over storage and ingestion ports without an engine instance."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex
from scone_memory.core.ports import NewTombstone
from scone_memory.ingestion.batch import remember_many
from scone_memory.ingestion.records import content_hash
from test_ingestion_component import STAMP, runtime as ingestion_runtime


async def archive(documents):
    from scone_memory.memory.archive import ArchiveRuntime
    vectors, embedder = InMemoryVectorIndex(), HashEmbedder()
    await vectors.ensure(embedder.dim)
    ingestion = ingestion_runtime(documents, vectors, embedder, [])
    async def ingest(space, records):
        return await remember_many(ingestion, space, records)
    return ArchiveRuntime(documents, lambda:STAMP, ingest)


async def test_archive_component_rebuilds_chunks_and_remaps_fact_and_link_sources():
    from scone_memory.memory.archive import export_records, import_records
    documents = InMemoryDocumentStore()
    runtime = await archive(documents)
    text = 'Café calibration uses Polaris. ' * 8
    records = [
        {'type':'episode', 'space':'source', 'episode_id':90, 'content':text,
         'content_hash':content_hash('source', text)},
        {'type':'fact', 'fact_id':100, 'subject':'juniper', 'predicate':'uses', 'object':'Polaris',
         'valid_from':STAMP, 'source_episode_id':90, 'quote':'Café calibration'},
        {'type':'fact', 'fact_id':200, 'subject':'juniper', 'predicate':'uses', 'object':'Polaris',
         'valid_from':STAMP, 'source_episode_id':90, 'quote':'uses Polaris'},
        {'type':'fact_link', 'from_fact':100, 'to_fact':200, 'kind':'supports',
         'source_episode_id':90, 'quote':'uses Polaris'},
    ]
    first = await import_records(runtime, 'target', records)
    assert (first.episodes, first.facts, first.links) == (1, 2, 1)
    [episode] = await documents.recent_episodes('target', 10)
    assert episode.episode_id != 90 and episode.content_hash == content_hash('target', text)
    chunks = await documents.chunks_of('target', episode.episode_id)
    assert len(chunks) > 1 and ''.join(row.text for row in chunks) == text
    dumped = [row async for row in export_records(documents, 'target')]
    facts = [row for row in dumped if row['type'] == 'fact']
    [link] = [row for row in dumped if row['type'] == 'fact_link']
    assert {link['from_fact'], link['to_fact']} == {row['fact_id'] for row in facts}
    assert link['source_episode_id'] == episode.episode_id and link['created_at'] == STAMP
    assert {row['source_episode_id'] for row in facts} == {episode.episode_id}
    revision = await documents.revision('target')
    again = await import_records(runtime, 'target', records)
    assert (again.episodes, again.deduplicated, again.facts, again.facts_skipped,
            again.links, again.links_skipped) == (0, 1, 0, 2, 0, 1)
    assert await documents.revision('target') == revision


async def test_archive_component_respects_tombstone_until_explicit_resurrection():
    from scone_memory.memory.archive import import_records
    documents = InMemoryDocumentStore()
    runtime = await archive(documents)
    await documents.record_tombstone(NewTombstone('target', 90,
        content_hash('target', 'forgotten text'), STAMP))
    records = [{'type':'episode', 'space':'source', 'episode_id':90, 'content':'forgotten text',
                'content_hash':content_hash('source', 'forgotten text')}]
    result = await import_records(runtime, 'target', records)
    assert result.tombstoned == 1 and result.episodes == 0
    assert (await documents.counts('target')).episodes == 0
    restored = await import_records(runtime, 'target', records, resurrect=True)
    assert restored.tombstoned == 0 and restored.episodes == 1


async def test_engine_keeps_space_guards_and_bound_ingestion_callback(monkeypatch):
    from scone_memory import MemoryEngine
    from scone_memory.core.errors import NotFound, InvalidInput
    from scone_memory.memory.archive import ImportSummary
    from scone_memory.memory.engine import ImportSummary as LegacySummary
    assert LegacySummary is ImportSummary
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    calls = []
    async def ingest(space, records):
        calls.append((space, records))
        return []
    monkeypatch.setattr(engine, 'remember_many', ingest)
    try:
        await engine.import_records('alpha', [])
        assert calls == [('alpha', [])]
        with pytest.raises(InvalidInput):
            await engine.import_records('invalid space', [])
        with pytest.raises(InvalidInput):
            _ = [row async for row in engine.export('invalid space')]
        await engine.delete_space('alpha')
        with pytest.raises(NotFound, match='was deleted'):
            await engine.import_records('alpha', [])
        assert len(calls) == 1
    finally:
        await engine.close()
