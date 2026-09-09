"""Import preserves distinct evidence and link targets without undoing exclusion."""
from dataclasses import replace

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.ports import NewFact, NewFactLink


@pytest.mark.parametrize('difference', ['source', 'quote', 'origin'])
async def test_import_keeps_distinct_fact_provenance_and_repeated_import_is_idempotent(engine, difference):
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        first = await source.remember('origin', 'Lab note: Juniper uses Polaris for calibration.')
        second = await source.remember('origin', 'Run report: Juniper uses Polaris for calibration.')
        base = NewFact(space='origin', subject='juniper', predicate='uses', object='Polaris',
            valid_from='2026-01-01T00:00:00Z', status='proposed', origin='extracted',
            source_episode_id=first.episode_id, quote='Juniper uses Polaris')
        changes = {
            'source': {'source_episode_id': second.episode_id},
            'quote': {'quote': 'Juniper uses Polaris for calibration'},
            'origin': {'origin': 'inferred'},
        }
        left = await source.documents.insert_fact(base)
        right = await source.documents.insert_fact(replace(base, **changes[difference]))
        await source.documents.insert_fact_link(NewFactLink(space='origin', from_fact=left.fact_id,
            to_fact=right.fact_id, kind='supports', created_at='2026-01-01T00:00:00Z',
            source_episode_id=second.episode_id, quote='Juniper uses Polaris'))
        dump = [record async for record in source.export('origin')]
    finally:
        await source.close()

    await engine.remember('target', 'Unrelated retained source shifts target IDs.')
    first_import = await engine.import_records('target', dump)
    assert first_import.facts == 2, 'different provenance must not collapse into one fact'
    assert first_import.links == 1 and first_import.links_skipped == 0
    facts = await engine.facts('target', status='proposed', include_excluded=True)
    assert len(facts) == 2
    stored = {(f.origin, f.quote, f.excluded_reason,
        (await engine.episode('target', f.source_episode_id)).content) for f in facts}
    expected = {
        (record['origin'], record['quote'], record['excluded_reason'], next(
            episode['content'] for episode in dump if episode['type'] == 'episode'
            and episode['episode_id'] == record['source_episode_id']))
        for record in dump if record['type'] == 'fact'
    }
    assert stored == expected
    [link] = await engine.fact_links('target', facts[0].fact_id)
    assert {link.from_fact, link.to_fact} == {fact.fact_id for fact in facts}
    assert (await engine.episode('target', link.source_episode_id)).content.startswith('Run report:')
    repeated = await engine.import_records('target', dump)
    assert (repeated.facts, repeated.facts_skipped, repeated.links, repeated.links_skipped) == (0, 2, 0, 1)


async def test_reimport_does_not_create_an_unexcluded_copy_of_a_locally_excluded_fact(engine):
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        episode = await source.remember('origin', 'Juniper uses Polaris for calibration.')
        await source.assert_fact('origin', 'juniper', 'uses', 'Polaris',
            source_episode_id=episode.episode_id, quote='Juniper uses Polaris')
        dump = [record async for record in source.export('origin')]
    finally:
        await source.close()
    await engine.import_records('target', dump)
    [fact] = await engine.facts('target')
    await engine.exclude('target', fact.fact_id, 'Source disputed locally')
    repeated = await engine.import_records('target', dump)
    assert repeated.facts == 0 and repeated.facts_skipped == 1
    assert await engine.facts('target') == []
    [retained] = await engine.facts('target', include_excluded=True)
    assert retained.fact_id == fact.fact_id and retained.excluded_reason == 'Source disputed locally'
