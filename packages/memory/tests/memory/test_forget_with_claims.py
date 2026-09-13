"""Forgetting a source together with the claims that rest on it.

`forget` removes the episode, its chunks and vectors, and leaves the
claims that cited it standing: a source being gone is a fact about the
evidence, and the receipt names them. That stays the default. But a
claim extracted from a source is that source's content in another
shape, and G12 says recovery cannot resurrect deleted memory. So forget
can also take the claims with it -- not erasing them (forgetting differs
from physical erasure) but excluding them from recall with a reason that
names the episode, reversibly, and only when no retained source still
supports them. The receipt says which claims went which way, and why.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.observability.events import InMemoryEventLog

NOW = "2026-09-13T00:00:00Z"
LATER = "2026-09-14T00:00:00Z"


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(), clock=lambda: NOW).open()
    yield engine
    await engine.close()


async def sourced_claim(memory, text="Alice works at Acme"):
    source = await memory.remember("alpha", text)
    fact = await memory.assert_fact("alpha", "Alice", "works_at", "Acme", source_episode_id=source.episode_id, quote=text)
    return source, fact


async def held(memory, fact_id):
    return await memory.documents.get_fact("alpha", fact_id)


async def test_by_default_the_claims_stand_and_the_receipt_says_so(memory):
    source, fact = await sourced_claim(memory)
    receipt = await memory.forget("alpha", source.episode_id)
    assert receipt.claims_policy == "keep" and receipt.facts_citing == [fact.fact_id]
    assert receipt.claims_excluded == [] and receipt.claims_kept_other_support == []
    kept = await held(memory, fact.fact_id)
    assert kept.status == "active" and not kept.excluded
    assert [f.fact_id for f in await memory.facts("alpha")] == [fact.fact_id]


async def test_exclude_takes_an_unsupported_claim_out_of_recall_naming_the_episode(memory):
    source, fact = await sourced_claim(memory)
    receipt = await memory.forget("alpha", source.episode_id, with_claims="exclude")
    assert receipt.claims_policy == "exclude" and receipt.claims_excluded == [fact.fact_id]
    assert receipt.facts_citing == [fact.fact_id], "the receipt still names every claim that cited the source"
    gone = await held(memory, fact.fact_id)
    assert gone.excluded and str(source.episode_id) in gone.excluded_reason
    assert gone.status == "active", "excluded, not closed: its interval and history are untouched"
    assert [f.fact_id for f in await memory.facts("alpha")] == []
    assert [f.fact_id for f in await memory.facts("alpha", include_excluded=True)] == [fact.fact_id]
    found = await memory.recall("alpha", "where does Alice work", limit=5)
    assert found.items == [] and found.facts == []


async def test_exclusion_is_reversible(memory):
    source, fact = await sourced_claim(memory)
    await memory.forget("alpha", source.episode_id, with_claims="exclude")
    back = await memory.include("alpha", fact.fact_id)
    assert not back.excluded and [f.fact_id for f in await memory.facts("alpha")] == [fact.fact_id]


async def test_a_claim_another_retained_source_affirms_is_kept_and_listed(memory):
    first, fact = await sourced_claim(memory)
    second = await memory.remember("alpha", "Alice is still at Acme")
    again = await memory.assert_fact("alpha", "Alice", "works_at", "Acme", valid_from=LATER,
                                     source_episode_id=second.episode_id, quote="Alice is still at Acme")
    assert again.fact_id == fact.fact_id, "a restatement affirms the fact rather than duplicating it"
    receipt = await memory.forget("alpha", first.episode_id, with_claims="exclude")
    assert receipt.claims_kept_other_support == [fact.fact_id] and receipt.claims_excluded == []
    assert not (await held(memory, fact.fact_id)).excluded
    # The last retained support goes: the claim cites the first episode, and
    # this one only affirmed it, but with both gone nothing holds it up.
    receipt = await memory.forget("alpha", second.episode_id, with_claims="exclude")
    assert receipt.affirmations_citing and receipt.facts_citing == []
    assert receipt.claims_excluded == [fact.fact_id]
    assert (await held(memory, fact.fact_id)).excluded


async def test_an_already_excluded_claim_keeps_its_own_reason(memory):
    source, fact = await sourced_claim(memory)
    await memory.exclude("alpha", fact.fact_id, "a person said so")
    receipt = await memory.forget("alpha", source.episode_id, with_claims="exclude")
    assert receipt.claims_already_excluded == [fact.fact_id] and receipt.claims_excluded == []
    assert (await held(memory, fact.fact_id)).excluded_reason == "a person said so"


async def test_a_bad_policy_is_refused_before_anything_is_removed(memory):
    source, fact = await sourced_claim(memory)
    with pytest.raises(InvalidInput):
        await memory.forget("alpha", source.episode_id, with_claims="erase")
    assert await memory.documents.get_episode("alpha", source.episode_id) is not None
    assert not (await held(memory, fact.fact_id)).excluded


async def test_nothing_is_excluded_until_the_source_is_gone_and_a_retry_finishes_the_job(memory, monkeypatch):
    """An interruption after the deed and before the exclusion leaves the
    claims standing, which is the documented default, never a half state
    where claims are excluded for a source that is still there."""
    source, fact = await sourced_claim(memory)
    original = memory.documents.record_tombstone

    async def interrupt(*args, **kwargs):
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(memory.documents, "record_tombstone", interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await memory.forget("alpha", source.episode_id, with_claims="exclude")
    monkeypatch.setattr(memory.documents, "record_tombstone", original)
    assert not (await held(memory, fact.fact_id)).excluded
    receipt = await memory.forget("alpha", source.episode_id, with_claims="exclude")
    assert receipt.claims_excluded == [fact.fact_id] and receipt.forgotten_at is not None


async def test_each_exclusion_is_an_event_with_the_actor_named(memory):
    source, fact = await sourced_claim(memory)
    await memory.forget("alpha", source.episode_id, with_claims="exclude")
    excluded = await memory.events.query("alpha", kind="fact_exclude")
    assert len(excluded) == 1 and excluded[0].payload["actor"] == "forget" and excluded[0].payload["fact_id"] == fact.fact_id


async def test_a_claim_stated_on_its_own_authority_is_kept_when_its_only_affirmation_goes(memory):
    fact = await memory.assert_fact("alpha", "Alice", "works_at", "Acme")
    assert fact.source_episode_id is None
    source = await memory.remember("alpha", "Alice works at Acme")
    again = await memory.assert_fact("alpha", "Alice", "works_at", "Acme", valid_from=LATER,
                                     source_episode_id=source.episode_id, quote="Alice works at Acme")
    assert again.fact_id == fact.fact_id
    receipt = await memory.forget("alpha", source.episode_id, with_claims="exclude")
    assert receipt.affirmations_citing and receipt.claims_kept_other_support == [fact.fact_id]
    assert not (await held(memory, fact.fact_id)).excluded


async def test_a_proposed_claim_is_left_for_review_and_listed(memory):
    """A proposed claim never held and cannot be excluded; forgetting its
    source leaves it in the review queue, and the receipt says so."""
    source = await memory.remember("alpha", "Alice may work at Acme")
    proposed = await memory.assert_fact("alpha", "Alice", "works_at", "Acme", source_episode_id=source.episode_id,
                                        quote="Alice may work at Acme", proposed=True)
    assert proposed.status == "proposed"
    receipt = await memory.forget("alpha", source.episode_id, with_claims="exclude")
    assert receipt.claims_kept_not_in_ledger == [proposed.fact_id] and receipt.claims_excluded == []
    still = await held(memory, proposed.fact_id)
    assert still.status == "proposed" and not still.excluded
