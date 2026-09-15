"""A merge decision is about names, never a claim about the world.

It is kept in the ledger so it has a time and a history, but nothing that
reads claims as things said may read it: recall, the profile, derivation
and the role index retrieval walks all leave it out. And only the merge
operations change it: the ordinary close and reopen of a fact refuse, so a
writer cannot unmerge without the review role, and a reopened decision
cannot hold beside the one that replaced it.
"""
from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import Fact
from scone_memory.entities.merges import SAME_ENTITY, entity_merges
from scone_memory.entities.read import read_ledger
from scone_memory.entities.roles import role_index
from scone_memory.memory.engine import derivation_groups
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def merged_engine(documents: InMemoryDocumentStore | None = None, clock: Clock | None = None) -> MemoryEngine:
    engine = await MemoryEngine(documents or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=clock or Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "dr. alice chen", "leads", "Robotics Lab", valid_from=DAY)
    await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="one person")
    return engine


async def test_recall_and_the_profile_never_return_a_merge_decision():
    engine = await merged_engine()
    found = await engine.recall("alpha", "is dr. alice chen the same entity as alice chen", limit=10)
    assert found.facts and all(fact.predicate != SAME_ENTITY for fact in found.facts)
    profile = await engine.profile("alpha", limit=20)
    assert profile.static_facts and all(fact.predicate != SAME_ENTITY for fact in profile.static_facts)


async def test_derivation_and_the_role_index_leave_the_decision_out():
    engine = await merged_engine()
    groups = derivation_groups(await engine.facts("alpha"))
    assert all(fact.predicate != SAME_ENTITY for group in groups for fact in group)
    roles = role_index(await read_ledger(engine, "alpha"))
    decision = next(fact for fact in await engine.facts("alpha") if fact.predicate == SAME_ENTITY)
    assert decision.fact_id not in {fact_id for ids in (*roles.subjects.values(), *roles.objects.values())
                                    for fact_id in ids}


async def test_the_ordinary_close_and_reopen_refuse_a_merge_decision():
    clock = Clock("2025-06-01T00:00:00.000Z")
    engine = await merged_engine(clock=clock)
    decision = next(fact for fact in await engine.facts("alpha") if fact.predicate == SAME_ENTITY)
    with pytest.raises(InvalidInput, match="unmerge_entities"):
        await engine.close_fact("alpha", decision.fact_id, "not the same")
    clock.now = "2025-06-02T00:00:00.000Z"
    await engine.unmerge_entities("alpha", "dr. alice chen", reason="two people")
    await engine.merge_entities("alpha", "dr. alice chen", "robotics lab", reason="a test")
    with pytest.raises(InvalidInput, match="merge_entities"):
        await engine.reopen("alpha", decision.fact_id, "bring it back")


def test_a_decision_replaced_while_both_hold_is_reported_as_replaced():
    def merge(number: int, into: str) -> Fact:
        return Fact(fact_id=number, space="alpha", subject="bob", predicate=SAME_ENTITY, object=into,
                    valid_from=DAY)

    merges = entity_merges([merge(1, "Robert Smith"), merge(2, "Bob Jones")])
    assert [(item.fact_id, item.outcome, item.into_key) for item in merges.decisions] == [
        (1, "replaced", "robert smith"), (2, "applied", "bob jones")]
    assert merges.canonical("bob") == "bob jones"


class _Yielding(InMemoryDocumentStore):
    """A store whose reads give other tasks a turn, as a networked one does."""

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        await asyncio.sleep(0)
        return await super().facts_for(space, subject, predicate)


async def test_two_merges_racing_in_opposite_directions_cannot_both_be_recorded():
    engine = await MemoryEngine(_Yielding(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "a. chen", "knows", "Bob Stone", valid_from=DAY)
    await engine.assert_fact("alpha", "alice chen", "knows", "Cara Ruiz", valid_from=DAY)
    outcomes = await asyncio.gather(
        engine.merge_entities("alpha", "a. chen", "alice chen", reason="one way"),
        engine.merge_entities("alpha", "alice chen", "a. chen", reason="the other way"), return_exceptions=True)
    assert sorted(type(outcome).__name__ for outcome in outcomes) == ["Fact", "InvalidInput"]
    assert len([fact for fact in await engine.facts("alpha") if fact.predicate == SAME_ENTITY]) == 1
