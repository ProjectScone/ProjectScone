"""Relationship invariants exercised through storage and host callbacks."""
import asyncio
from functools import partial

import pytest

from scone_memory.core.errors import InvalidInput, NotFound


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
async def test_relationships_work_without_an_engine_and_preserve_source_events(backend, tmp_path):
    from scone_memory import InMemoryDocumentStore
    from scone_memory.backends.sqlite import SqliteDocumentStore
    from scone_memory.core.ports import NewEpisode
    from scone_memory.memory import fact_placement, fact_relationships

    documents = InMemoryDocumentStore() if backend == 'memory' else SqliteDocumentStore(tmp_path / 'links.db')
    events = []
    stamp = '2026-01-01T00:00:00.000Z'

    async def emit(space, kind, payload):
        events.append((space, kind, payload))

    async def living(space):
        pass

    placement = fact_placement.FactPlacementRuntime(documents, lambda: stamp, emit,
        partial(fact_placement.place, documents), partial(fact_placement.truncate, documents))

    async def link(*args, **kwargs):
        return await fact_relationships.link_facts(runtime, *args, **kwargs)

    runtime = fact_relationships.FactRelationshipsRuntime(documents, lambda: stamp, emit, living,
        partial(fact_placement.assert_placed, placement), link, partial(fact_relationships.depends_on, documents))
    try:
        premise = await fact_relationships.assert_fact(runtime, 'alpha', 'sensor', 'uses', 'Polaris')
        claim = await fact_relationships.assert_fact(runtime, 'alpha', 'forecast', 'uses', 'sensor',
                                                     derived_from=[premise.fact_id])
        assert claim.origin == 'inferred'
        stored = await documents.fact_links('alpha', claim.fact_id)
        assert [(item.from_fact, item.to_fact, item.kind) for item in stored] == [
            (claim.fact_id, premise.fact_id, 'derived_from')]
        source = await documents.insert_episode(NewEpisode('alpha', 'note', 'Forecast confirms calibration.',
                                                          'source', stamp, stamp))
        revision, event_count = await documents.revision('alpha'), len(events)
        with pytest.raises(InvalidInput, match='not a substring'):
            await link('alpha', claim.fact_id, premise.fact_id, 'supports',
                       source_episode_id=source.episode_id, quote='Fabricated observation')
        assert await documents.revision('alpha') == revision and len(events) == event_count
        grounded = await link('alpha', claim.fact_id, premise.fact_id, 'supports',
                              source_episode_id=source.episode_id, quote='confirms calibration')
        assert grounded.quote == 'confirms calibration' and grounded.created_at == stamp
        assert await documents.revision('alpha') == revision + 1
        assert events[-1] == ('alpha', 'fact_link', {'link_id': grounded.link_id, 'from_fact': claim.fact_id,
            'to_fact': premise.fact_id, 'kind': 'supports', 'source_episode_id': source.episode_id})
        event_count = len(events)
        repeated = await link('alpha', claim.fact_id, premise.fact_id, 'supports')
        assert repeated == grounded and len(events) == event_count
        assert await documents.revision('alpha') == revision + 1
    finally:
        if isinstance(documents, SqliteDocumentStore):
            await documents.close()


async def test_component_preserves_dependency_direction_and_idempotency(engine):
    from scone_memory.memory import fact_relationships

    first = await engine.assert_fact('alpha', 'sensor', 'uses', 'Polaris')
    second = await engine.assert_fact('alpha', 'calibration', 'uses', 'sensor')
    third = await engine.assert_fact('alpha', 'forecast', 'uses', 'calibration')
    runtime = engine._relationships_runtime()
    link = await fact_relationships.link_facts(runtime, 'alpha', third.fact_id, second.fact_id, 'derived_from')
    await fact_relationships.link_facts(runtime, 'alpha', second.fact_id, first.fact_id, 'extends')
    revision = await engine.revision('alpha')
    repeated = await fact_relationships.link_facts(runtime, 'alpha', third.fact_id, second.fact_id, 'derived_from')
    assert repeated == link
    assert await engine.revision('alpha') == revision
    assert await fact_relationships.depends_on(engine.documents, 'alpha', third.fact_id, first.fact_id)
    assert not await fact_relationships.depends_on(engine.documents, 'alpha', first.fact_id, third.fact_id)
    with pytest.raises(InvalidInput, match='dependency cycle'):
        await fact_relationships.link_facts(runtime, 'alpha', first.fact_id, third.fact_id, 'extends')
    assert await engine.revision('alpha') == revision
    # A non-dependency may point back without creating a dependency cycle.
    await fact_relationships.link_facts(runtime, 'alpha', first.fact_id, third.fact_id, 'supports')
    assert not await fact_relationships.depends_on(engine.documents, 'alpha', first.fact_id, third.fact_id)


async def test_assertion_checks_every_premise_before_writing(engine):
    from scone_memory.memory import fact_relationships

    held = await engine.assert_fact('alpha', 'sensor', 'uses', 'Polaris')
    proposed = await engine.assert_fact('alpha', 'forecast', 'uses', 'sensor', proposed=True)
    revision = await engine.revision('alpha')
    with pytest.raises(InvalidInput, match='only a held or closed fact'):
        await fact_relationships.assert_fact(engine._relationships_runtime(), 'alpha', 'result', 'is', 'ready',
                                             derived_from=[held.fact_id, proposed.fact_id])
    assert await engine.revision('alpha') == revision
    assert len(await engine.documents.list_facts('alpha', True)) == 2
    assert await engine.fact_links('alpha', held.fact_id) == []
    with pytest.raises(NotFound):
        await fact_relationships.assert_fact(engine._relationships_runtime(), 'beta', 'result', 'is', 'ready',
                                             derived_from=[held.fact_id])
    assert await engine.documents.list_facts('beta', True) == []


async def test_dependency_cancellation_cannot_store_link(engine, monkeypatch):
    first = await engine.assert_fact('alpha', 'sensor', 'uses', 'Polaris')
    second = await engine.assert_fact('alpha', 'forecast', 'uses', 'sensor')
    revision = await engine.revision('alpha')

    async def cancel(*args):
        raise asyncio.CancelledError()

    monkeypatch.setattr(engine, '_depends_on', cancel)
    with pytest.raises(asyncio.CancelledError):
        await engine.link_facts('alpha', second.fact_id, first.fact_id, 'derived_from')
    assert await engine.revision('alpha') == revision
    assert await engine.fact_links('alpha', first.fact_id) == []
