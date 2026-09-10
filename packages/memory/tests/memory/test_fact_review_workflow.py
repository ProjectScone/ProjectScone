"""Review ordering and failure boundaries survive engine decomposition."""
import asyncio
from functools import partial

import pytest

from scone_memory.core.errors import InvalidInput


async def test_batch_deduplicates_ids_but_preserves_historical_ties_and_result_order(engine, monkeypatch):
    first = await engine.assert_fact('alpha', 'juniper', 'uses', 'Vega', proposed=True)
    second = await engine.assert_fact('alpha', 'juniper', 'uses', 'Polaris', proposed=True)
    third = await engine.assert_fact('alpha', 'juniper', 'uses', 'Sirius', proposed=True)
    calls = []
    original = engine._decide_one

    async def decide_one(space, decision, fact_id, reason, actor):
        calls.append((fact_id, actor))
        return await original(space, decision, fact_id, reason, actor)

    monkeypatch.setattr(engine, '_decide_one', decide_one)
    revision = await engine.revision('alpha')
    result = await engine.decide('alpha', 'approve', [third.fact_id, first.fact_id, third.fact_id, second.fact_id], actor='reviewer')
    assert calls == [(fact.fact_id, 'reviewer') for fact in (first, second, third)]
    assert [item.fact_id for item in result.results] == [third.fact_id, first.fact_id, second.fact_id]
    assert result.applied == 3 and result.revision == revision + 3


async def test_batch_cancellation_keeps_completed_decisions_and_propagates(engine, monkeypatch):
    first = await engine.assert_fact('alpha', 'juniper', 'uses', 'Vega', proposed=True)
    second = await engine.assert_fact('alpha', 'juniper', 'uses', 'Polaris', proposed=True)
    calls = []
    original = engine.approve

    async def approve(space, fact_id, *, actor=None):
        calls.append((fact_id, actor))
        if fact_id == second.fact_id:
            raise asyncio.CancelledError()
        return await original(space, fact_id, actor=actor)

    monkeypatch.setattr(engine, 'approve', approve)
    revision = await engine.revision('alpha')
    with pytest.raises(asyncio.CancelledError):
        await engine.decide('alpha', 'approve', [second.fact_id, first.fact_id], actor='reviewer')
    assert calls == [(first.fact_id, 'reviewer'), (second.fact_id, 'reviewer')]
    assert (await engine.fact('alpha', first.fact_id)).status == 'active'
    assert (await engine.fact('alpha', second.fact_id)).status == 'proposed'
    assert await engine.revision('alpha') == revision + 1


async def test_review_event_failure_does_not_claim_to_roll_back_storage(engine, monkeypatch):
    fact = await engine.assert_fact('alpha', 'juniper', 'uses', 'Vega', proposed=True)
    revision = await engine.revision('alpha')
    observed = []

    async def fail_emit(space, kind, payload):
        observed.append((kind, payload, await engine.revision(space)))
        raise RuntimeError('event store unavailable')

    monkeypatch.setattr(engine, '_emit', fail_emit)
    with pytest.raises(RuntimeError, match='event store unavailable'):
        await engine.decline('alpha', fact.fact_id, '  unsupported  ', actor='reviewer')
    saved = await engine.fact('alpha', fact.fact_id)
    assert saved.status == 'declined' and saved.closed_reason == 'unsupported'
    assert observed == [('fact_review', {'fact_id': fact.fact_id, 'decision': 'declined', 'actor': 'reviewer'}, revision + 1)]


async def test_include_and_close_noops_do_not_advance_revision_or_emit(engine, monkeypatch):
    fact = await engine.assert_fact('alpha', 'juniper', 'uses', 'Vega')
    events = []

    async def emit(space, kind, payload):
        events.append((kind, payload))

    monkeypatch.setattr(engine, '_emit', emit)
    revision = await engine.revision('alpha')
    assert await engine.include('alpha', fact.fact_id) == fact
    assert await engine.revision('alpha') == revision and events == []
    excluded = await engine.exclude('alpha', fact.fact_id, '  private  ', actor='reviewer')
    assert excluded.excluded_reason == 'private' and excluded.valid_until is None
    await engine.include('alpha', fact.fact_id, actor='reviewer')
    closed = await engine.close_fact('alpha', fact.fact_id, '  finished  ', actor='reviewer')
    assert closed.closed_reason == 'finished'
    assert await engine.close_fact('alpha', fact.fact_id, 'another reason') == closed
    assert await engine.revision('alpha') == revision + 3
    assert [(kind, payload.get('action'), payload['actor']) for kind, payload in events] == [
        ('fact_exclude', 'exclude', 'reviewer'), ('fact_exclude', 'include', 'reviewer'),
        ('fact_close', None, 'reviewer'),
    ]


@pytest.mark.parametrize('decision', ['decline', 'exclude'])
async def test_invalid_batch_reason_is_rejected_before_reading_revision(engine, monkeypatch, decision):
    async def forbidden_read(space):
        raise AssertionError('invalid batch touched storage')

    monkeypatch.setattr(engine.documents, 'revision', forbidden_read)
    with pytest.raises(InvalidInput, match='reason must be'):
        await engine.decide('alpha', decision, [999], reason=' ')


async def test_review_component_runs_without_an_engine():
    from scone_memory import InMemoryDocumentStore
    from scone_memory.core.ports import NewFact
    from scone_memory.memory import fact_review as review
    from scone_memory.memory.fact_placement import place, truncate

    documents, events = InMemoryDocumentStore(), []
    stamp = '2026-01-01T00:00:00.000Z'

    async def emit(space, kind, payload):
        events.append((kind, payload))

    async def proposed(space, fact_id):
        return await review.proposed(runtime, space, fact_id)

    async def approve(space, fact_id, actor=None):
        return await review.approve(runtime, space, fact_id, actor=actor)

    async def decline(space, fact_id, reason, actor=None):
        return await review.decline(runtime, space, fact_id, reason, actor=actor)

    async def exclude(space, fact_id, reason, actor=None):
        return await review.exclude(runtime, space, fact_id, reason, actor=actor)

    async def decide_one(space, decision, fact_id, reason, actor):
        return await review.decide_one(runtime, space, decision, fact_id, reason, actor)

    runtime = review.FactReviewRuntime(documents, lambda: stamp, emit,
        partial(place, documents), partial(truncate, documents), proposed,
        approve, decline, exclude, decide_one)
    first = await documents.insert_fact(NewFact(space='alpha', subject='juniper', predicate='uses',
        object='Vega', valid_from=stamp, status='proposed'))
    duplicate = await documents.insert_fact(NewFact(space='alpha', subject='juniper', predicate='uses',
        object='Vega', valid_from=stamp, status='proposed'))
    result = await review.decide(runtime, 'alpha', 'approve', [duplicate.fact_id, first.fact_id], actor='reviewer')
    assert [item.outcome for item in result.results] == ['duplicate_of', 'approved']
    assert result.results[0].held_fact_id == first.fact_id
    assert (await documents.get_fact('alpha', duplicate.fact_id)).status == 'declined'
    assert [payload['decision'] for _, payload in events] == ['approved', 'duplicate']
    assert all(payload['actor'] == 'reviewer' for _, payload in events)
