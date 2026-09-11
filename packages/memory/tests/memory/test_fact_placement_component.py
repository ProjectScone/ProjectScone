"""Temporal placement through explicit storage and event callbacks."""
import asyncio
from functools import partial
from itertools import permutations

import pytest

from scone_memory import InMemoryDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import NewEpisode
from scone_memory.core.timeutil import parse_rfc3339

STAMP = '2026-01-01T00:00:00.000Z'


def runtime(documents, events):
    from scone_memory.memory.fact_placement import FactPlacementRuntime, place, truncate
    async def emit(space, kind, payload):
        events.append((space, kind, payload))
    return FactPlacementRuntime(documents, lambda: STAMP, emit,
        partial(place, documents), partial(truncate, documents))


@pytest.mark.parametrize('arrival', list(permutations(range(3))))
async def test_component_partitions_history_independent_of_arrival_order(arrival):
    from scone_memory.memory.fact_placement import assert_placed, _covers
    documents, events = InMemoryDocumentStore(), []
    context = runtime(documents, events)
    history = [('Vega', '2022-01-01'), ('Polaris', '2023-01-01'), ('Sirius', '2024-01-01')]
    for index in arrival:
        value, start = history[index]
        await assert_placed(context, 'alpha', 'Juniper', 'uses', value, valid_from=start)
    facts = await documents.facts_for('alpha', 'juniper', 'uses')
    for expected, date in history:
        assert [fact.object for fact in facts if _covers(fact, parse_rfc3339(date + 'T00:00:00.000Z'))] == [expected]
    revision = await documents.revision('alpha')
    repeated = await assert_placed(context, 'alpha', 'Juniper', 'uses', 'Polaris', valid_from='2023-06-01')
    assert repeated.object == 'Polaris' and len(await documents.list_facts('alpha', True)) == 3
    assert await documents.revision('alpha') == revision
    assert events[-1][2]['outcome'] == 'restated'


async def test_component_validates_source_before_proposing_and_never_closes_held_fact():
    from scone_memory.memory.fact_placement import assert_placed
    documents, events = InMemoryDocumentStore(), []
    context = runtime(documents, events)
    held = await assert_placed(context, 'alpha', 'juniper', 'uses', 'Vega')
    episode = await documents.insert_episode(NewEpisode('alpha', 'note', 'Juniper uses Polaris.', 'source', STAMP, STAMP))
    revision = await documents.revision('alpha')
    with pytest.raises(InvalidInput, match='not a substring'):
        await assert_placed(context, 'alpha', 'juniper', 'uses', 'Polaris', proposed=True,
            source_episode_id=episode.episode_id, quote='Not in this source')
    assert await documents.revision('alpha') == revision and len(events) == 1
    proposal = await assert_placed(context, 'alpha', 'juniper', 'uses', 'Polaris', proposed=True,
        source_episode_id=episode.episode_id, quote='uses Polaris')
    assert proposal.status == 'proposed' and proposal.grounded
    assert (await documents.get_fact('alpha', held.fact_id)).status == 'active'
    assert events[-1][2]['outcome'] == 'proposed' and events[-1][2]['superseded'] == []


async def test_engine_retains_bound_placement_and_truncation_callbacks(engine, monkeypatch):
    from scone_memory.memory.fact_placement import Placement
    from scone_memory.memory.engine import _Placement
    assert _Placement is Placement
    calls = []
    original_place, original_truncate = engine._place, engine._truncate
    async def place(*args, **kwargs):
        calls.append('place')
        return await original_place(*args, **kwargs)
    async def truncate(*args, **kwargs):
        calls.append('truncate')
        await original_truncate(*args, **kwargs)
    monkeypatch.setattr(engine, '_place', place)
    monkeypatch.setattr(engine, '_truncate', truncate)
    await engine.assert_fact('alpha', 'juniper', 'uses', 'Vega')
    assert calls == ['place', 'truncate']


async def test_component_propagates_source_read_cancellation(monkeypatch):
    from scone_memory.memory.fact_placement import assert_placed
    documents, events = InMemoryDocumentStore(), []
    async def cancel(*args):
        raise asyncio.CancelledError()
    monkeypatch.setattr(documents, 'get_episode', cancel)
    with pytest.raises(asyncio.CancelledError):
        await assert_placed(runtime(documents, events), 'alpha', 'juniper', 'uses', 'Vega',
            source_episode_id=1, quote='Vega')
    assert await documents.list_facts('alpha', True) == [] and events == []


_RETURN = [('Acme', '2020-01-01'), ('Globex', '2021-01-01'), ('Acme', '2023-01-01')]
_LOST = ("A restatement is returned unchanged and its own start is kept nowhere, so a backfill that "
         "later cuts the covering fact short erases the reaffirmed stretch. Needs affirmation times "
         "stored per fact (a schema change on every backend); an owner decision, not yet made.")


@pytest.mark.parametrize('arrival', [
    pytest.param(order, marks=pytest.mark.xfail(strict=True, reason=_LOST)) if order == (0, 2, 1) else order
    for order in permutations(range(3))])
async def test_a_value_that_returns_holds_again_whatever_the_arrival_order(arrival):
    """Acme from 2020, Globex from 2021, Acme again from 2023. Told Acme,
    Acme again, then the late Globex, the second Acme is folded into the
    first as a restatement, and the Globex backfill then cuts it short at
    2021: the ledger says Globex in 2024. Every other order is right."""
    from scone_memory.memory.fact_placement import assert_placed, _covers
    documents = InMemoryDocumentStore()
    context = runtime(documents, [])
    for index in arrival:
        value, start = _RETURN[index]
        await assert_placed(context, 'alpha', 'alice', 'works_at', value, valid_from=start)
    facts = await documents.facts_for('alpha', 'alice', 'works_at')
    for moment, expected in (('2020-06-01', 'Acme'), ('2022-06-01', 'Globex'), ('2024-06-01', 'Acme')):
        assert [fact.object for fact in facts if _covers(fact, parse_rfc3339(moment + 'T00:00:00.000Z'))] == [expected]
