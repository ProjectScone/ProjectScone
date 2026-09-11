"""Multi-hop reaches facts whose object is the entity a seed is about.

The ledger indexes facts by subject, so traversal could only follow a fact's
object to the facts about that thing. An entity role index, built from the
ledger read the projection cache holds, also answers "which facts point at
this entity", so a seed about Lisbon reaches the company based there and the
person who works there. The index is only a candidate generator: each
candidate is re-read and must pass the same join rule as a forward step,
so an index that lags the ledger may miss a fact but never invents one.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.models import RecallResult
from scone_memory.core.ports import NewFact
from scone_memory.entities.roles import role_index
from scone_memory.entities.read import read_ledger
from scone_memory.retrieval.multihop import MultiHopLimits, expand_multihop

STAMP = "2025-01-01T00:00:00Z"


async def fact(engine, subject, predicate, obj, *, space="alpha"):
    quote = f"{subject} {predicate} {obj}."
    source = await engine.remember(space, quote, created_at=STAMP)
    return await engine.documents.insert_fact(NewFact(space=space, subject=subject, predicate=predicate, object=obj,
                                                      valid_from=STAMP, source_episode_id=source.episode_id,
                                                      quote=quote))


@pytest.fixture
async def ledger():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    facts = {
        "summit": await fact(engine, "lisbon", "hosts", "the Web Summit"),
        "acme": await fact(engine, "acme robotics", "based_in", "Lisbon"),
        "alice": await fact(engine, "alice chen", "works_at", "Acme Robotics"),
        "bob": await fact(engine, "bob stone", "lives_in", "Lisbon"),
    }
    yield engine, facts
    await engine.close()


async def roles_of(engine):
    return role_index(await read_ledger(engine, "alpha"))


async def test_a_seed_reaches_the_facts_that_point_at_its_subject(ledger):
    engine, facts = ledger
    roles = await roles_of(engine)
    result = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["summit"]]), roles=roles)
    reached = {fact.fact_id for fact in result.facts}
    assert {facts["acme"].fact_id, facts["alice"].fact_id, facts["bob"].fact_id} <= reached
    alice = next(path for path in result.paths if path.fact_ids[-1] == facts["alice"].fact_id)
    assert alice.fact_ids == [facts["summit"].fact_id, facts["acme"].fact_id, facts["alice"].fact_id]
    assert alice.directions == ["reverse", "reverse"]
    edge = next(edge for edge in result.edges if edge.to_fact == facts["summit"].fact_id)
    assert edge.kind == "subject_object" and edge.from_fact in (facts["acme"].fact_id, facts["bob"].fact_id)


async def test_without_an_index_traversal_is_unchanged(ledger):
    engine, facts = ledger
    plain = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["summit"]]))
    assert [fact.fact_id for fact in plain.facts] == [facts["summit"].fact_id]
    asked = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["summit"]]),
                                  reverse_joins=True)
    assert "entity_roles_unavailable" in asked.coverage.reasons


async def test_an_index_that_lags_the_ledger_misses_but_never_invents(ledger):
    engine, facts = ledger
    moved = await fact(engine, "dana", "lives_in", "Lisbon")
    roles = await roles_of(engine)
    await engine.documents.update_fact(facts["bob"].model_copy(update={"excluded_reason": "private"}))
    await engine.documents.update_fact(moved.model_copy(update={"object": "Porto"}))
    await engine.documents.bump_revision("alpha")
    late = await fact(engine, "carol", "moved_to", "Lisbon")
    result = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["summit"]]), roles=roles)
    reached = {fact.fact_id for fact in result.facts}
    assert facts["bob"].fact_id not in reached and late.fact_id not in reached and moved.fact_id not in reached
    assert facts["acme"].fact_id in reached and "entity_roles_stale" in result.coverage.reasons


async def test_a_hub_is_expanded_as_a_seed_but_not_walked_through(ledger):
    engine, facts = ledger
    for number in range(6):
        await fact(engine, f"visitor {number}", "visited", "Acme Robotics")
    roles = await roles_of(engine)
    from_hub = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["acme"]]),
                                     roles=roles, limits=MultiHopLimits(hub_degree=3))
    assert len({fact.fact_id for fact in from_hub.facts}) >= 7
    through = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["summit"]]),
                                    roles=roles, limits=MultiHopLimits(hub_degree=3))
    assert facts["acme"].fact_id in {fact.fact_id for fact in through.facts}
    assert not any(fact.subject.startswith("visitor") for fact in through.facts)
    assert "hub_skipped" in through.coverage.reasons


async def test_the_newest_facts_pointing_at_an_entity_come_first(ledger):
    engine, facts = ledger
    extra = [await fact(engine, f"team {number}", "based_in", "Lisbon") for number in range(4)]
    roles = await roles_of(engine)
    result = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts["summit"]]),
                                   roles=roles, limits=MultiHopLimits(per_node_limit=2, max_hops=1))
    # The walker's window is per_node_limit plus one sentinel, as for its other steps.
    reached = [fact.fact_id for fact in result.facts[1:]]
    assert reached == [extra[3].fact_id, extra[2].fact_id, extra[1].fact_id]
    assert "candidate_window" in result.coverage.reasons


async def test_engine_roles_come_from_the_held_ledger_and_never_build(ledger):
    from scone_memory.entities.read import load_projection

    engine, facts = ledger
    assert engine.entities.roles("alpha") is None
    await load_projection(engine, "alpha", mode="current")
    roles = engine.entities.roles("alpha")
    assert roles is not None and roles.revision == await engine.revision("alpha")
    assert facts["acme"].fact_id in roles.facts_touching("lisbon", role="object", limit=10)
