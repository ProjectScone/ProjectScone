"""Recording, refusing and undoing a merge through the engine.

A merge is written only by ``merge_entities``, which a person calls with a
reason; no assertion, proposal or extraction may state the reserved
predicate, so no document and no model can merge two entities. Undoing it
closes the decision: the names part from that moment, and a view of an
earlier moment still shows them joined.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput, NotFound
from scone_memory.entities.duplicates import likely_duplicates
from scone_memory.entities.ids import key_id
from scone_memory.entities.merges import SAME_ENTITY
from scone_memory.entities.read import load_projection
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def engine_with(*triples: tuple[str, str, str], clock: Clock | None = None) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=clock or Clock("2025-06-01T00:00:00.000Z"), events=InMemoryEventLog()).open()
    for subject, predicate, obj in triples:
        await engine.assert_fact("alpha", subject, predicate, obj, valid_from=DAY)
    return engine


def chen() -> tuple[tuple[str, str, str], ...]:
    return (("alice chen", "works_at", "Acme Robotics"), ("dr. alice chen", "leads", "Robotics Lab"),
            ("bob stone", "reports_to", "Dr. Alice Chen"))


async def keys(engine: MemoryEngine, **view: object) -> set[str]:
    projection, _ = await load_projection(engine, "alpha", **view)  # type: ignore[arg-type]
    return {entity.key for entity in projection.entities}


async def test_a_merge_is_recorded_with_its_reason_and_the_graph_honours_it():
    engine = await engine_with(*chen())
    decision = await engine.merge_entities("alpha", "Dr. Alice Chen", "Alice Chen", reason="one person", actor="mark")
    assert (decision.subject, decision.predicate, decision.object, decision.status) == (
        "dr. alice chen", SAME_ENTITY, "Alice Chen", "active")
    assert "dr. alice chen" not in await keys(engine, mode="current")
    [event] = await engine.events.query("alpha", kind="entity_merge", limit=5)
    assert event.payload == {"fact_id": decision.fact_id, "alias": "dr. alice chen", "into": "alice chen",
                             "reason": "one person", "actor": "mark"}


async def test_no_assertion_may_state_the_reserved_predicate():
    engine = await engine_with(*chen())
    for proposed in (False, True):
        with pytest.raises(InvalidInput, match="merge_entities"):
            await engine.assert_fact("alpha", "dr. alice chen", "Scone:Same  Entity", "alice chen", proposed=proposed)
    assert "dr. alice chen" in await keys(engine, mode="all")


async def test_a_merge_that_cannot_name_one_thing_or_names_itself_is_refused():
    engine = await engine_with(*chen())
    for alias, into in (("Alice  CHEN", "alice chen"), ("she", "alice chen"), ("alice chen", '"the lead engineer"')):
        with pytest.raises(InvalidInput):
            await engine.merge_entities("alpha", alias, into, reason="checked")
    with pytest.raises(InvalidInput, match="reason"):
        await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="  ")
    assert await engine.facts("alpha", include_closed=True, status=None) and not [
        fact for fact in await engine.facts("alpha", include_closed=True) if fact.predicate == SAME_ENTITY]


async def test_a_merge_that_would_loop_back_is_refused_before_it_is_written():
    engine = await engine_with(*chen(), ("a. chen", "knows", "Bob Stone"))
    await engine.merge_entities("alpha", "a. chen", "dr. alice chen", reason="initial")
    await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="title")
    with pytest.raises(InvalidInput, match="cycle"):
        await engine.merge_entities("alpha", "alice chen", "A. Chen", reason="backwards")
    assert await keys(engine, mode="current") >= {"alice chen"} and "a. chen" not in await keys(engine, mode="current")


async def test_unmerging_parts_the_names_from_then_and_an_earlier_view_still_sees_them_joined():
    clock = Clock("2025-06-01T00:00:00.000Z")
    engine = await engine_with(*chen(), clock=clock)
    await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="one person")
    clock.now = "2025-07-01T00:00:00.000Z"
    closed = await engine.unmerge_entities("alpha", "Dr. Alice Chen", reason="two people after all", actor="mark")
    assert (closed.status, closed.valid_until, closed.closed_reason) == (
        "closed", "2025-07-01T00:00:00.000Z", "two people after all")
    assert "dr. alice chen" in await keys(engine, mode="current")
    assert "dr. alice chen" not in await keys(engine, mode="current", as_of="2025-06-15T00:00:00Z")
    assert "dr. alice chen" in await keys(engine, mode="history")
    with pytest.raises(NotFound):
        await engine.unmerge_entities("alpha", "dr. alice chen", reason="again")


async def test_pointing_an_alias_elsewhere_closes_the_decision_it_replaces():
    clock = Clock("2025-06-01T00:00:00.000Z")
    engine = await engine_with(*chen(), ("alice mei chen", "knows", "Bob Stone"), clock=clock)
    first = await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="guess")
    clock.now = "2025-06-02T00:00:00.000Z"
    await engine.merge_entities("alpha", "dr. alice chen", "alice mei chen", reason="checked the badge")
    assert (await engine.fact("alpha", first.fact_id)).status == "closed"
    projection, _ = await load_projection(engine, "alpha", mode="current")
    [merge] = projection.merges
    assert (merge.alias_key, merge.into_key) == ("dr. alice chen", "alice mei chen")


async def test_a_merged_pair_is_no_longer_suggested_as_a_duplicate():
    engine = await engine_with(*chen())
    before = await likely_duplicates(engine, "alpha")
    assert any({pair["a"]["key"], pair["b"]["key"]} == {"alice chen", "dr. alice chen"} for pair in before.pairs)
    await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="one person")
    after = await likely_duplicates(engine, "alpha")
    assert not any("dr. alice chen" in (pair["a"]["key"], pair["b"]["key"]) for pair in after.pairs)



async def test_a_proposal_about_an_alias_is_shown_under_the_entity_it_was_merged_into():
    engine = await engine_with(*chen())
    await engine.merge_entities("alpha", "dr. alice chen", "alice chen", reason="one person")
    await engine.assert_fact("alpha", "Dr. Alice Chen", "mentors", "Cara Ruiz", proposed=True, origin="extracted")
    projection, _ = await load_projection(engine, "alpha", mode="proposed")
    [role] = projection.roles
    assert role.predicate == "mentors" and role.subject_id == key_id("alpha", "alice chen")


async def test_a_chain_too_long_to_check_for_a_loop_is_refused(monkeypatch):
    from scone_memory.memory import entity_merging

    engine = await engine_with(("n0", "knows", "n1"), ("n1", "knows", "n2"), ("n2", "knows", "n3"))
    await engine.merge_entities("alpha", "n1", "n2", reason="chain")
    await engine.merge_entities("alpha", "n2", "n3", reason="chain")
    monkeypatch.setattr(entity_merging, "MAX_MERGE_CHAIN", 2)
    with pytest.raises(InvalidInput, match="more than 2 merges"):
        await engine.merge_entities("alpha", "n0", "n1", reason="too deep to check")
    monkeypatch.setattr(entity_merging, "MAX_MERGE_CHAIN", 3)
    assert (await engine.merge_entities("alpha", "n0", "n1", reason="checked")).predicate == SAME_ENTITY
