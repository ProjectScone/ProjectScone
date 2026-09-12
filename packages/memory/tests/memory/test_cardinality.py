"""How many values a predicate holds at once, as its owner configures it.

The ledger keeps one value per subject and predicate at a time: a new
object closes the one before it, so "alice lives in Porto" ends "alice
lives in Lisbon". That is right for many predicates and wrong for others
("alice knows Bob" and "alice knows Carol" both hold). Which is which is
the owner's decision, stated per predicate: nothing is inferred from a
predicate's name, and a predicate nobody configured keeps one value at a
time, as it always has.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.testing import Clock

SPACE = "alpha"


async def engine(**options) -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z"), events=InMemoryEventLog(), **options).open()


async def held(memory: MemoryEngine, subject: str, predicate: str) -> list[tuple[str, str, str | None]]:
    facts = await memory.documents.facts_for(SPACE, subject, predicate)
    return sorted((fact.object, fact.valid_from[:10], fact.valid_until and fact.valid_until[:10])
                  for fact in facts if fact.in_ledger)


async def test_a_predicate_nobody_configured_keeps_one_value_at_a_time():
    memory = await engine()
    await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Carol", valid_from="2021-01-01T00:00:00Z")
    assert await held(memory, "alice", "knows") == [("Bob", "2020-01-01", "2021-01-01"), ("Carol", "2021-01-01", None)]


async def test_a_many_valued_predicate_holds_each_value_side_by_side():
    memory = await engine(many_valued=["knows"])
    await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Carol", valid_from="2021-01-01T00:00:00Z")
    assert await held(memory, "alice", "knows") == [("Bob", "2020-01-01", None), ("Carol", "2021-01-01", None)]


async def test_nothing_is_inferred_from_a_predicates_name():
    """works_at, role and email can each hold several values, yet only what
    is configured holds several: here, only knows."""
    memory = await engine(many_valued=["knows"])
    for predicate in ("works_at", "role", "email"):
        await memory.assert_fact(SPACE, "alice", predicate, "First", valid_from="2020-01-01T00:00:00Z")
        await memory.assert_fact(SPACE, "alice", predicate, "Second", valid_from="2021-01-01T00:00:00Z")
        assert await held(memory, "alice", predicate) == [("First", "2020-01-01", "2021-01-01"),
                                                          ("Second", "2021-01-01", None)], predicate


async def test_one_value_stated_again_is_a_restatement_of_it():
    memory = await engine(many_valued=["knows"])
    first = await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Carol", valid_from="2021-01-01T00:00:00Z")
    again = await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2022-01-01T00:00:00Z")
    assert again.fact_id == first.fact_id
    assert await held(memory, "alice", "knows") == [("Bob", "2020-01-01", None), ("Carol", "2021-01-01", None)]


async def test_a_value_told_late_is_bounded_only_by_the_same_value():
    """Order of arrival still never decides what held, one value at a time:
    an earlier Bob stops where the later Bob starts, and Carol, told late,
    is bounded by nothing Bob holds."""
    memory = await engine(many_valued=["knows"])
    await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2022-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Carol", valid_from="2019-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    assert await held(memory, "alice", "knows") == [
        ("Bob", "2020-01-01", "2022-01-01"), ("Bob", "2022-01-01", None), ("Carol", "2019-01-01", None)]


async def test_closing_one_value_leaves_the_others():
    memory = await engine(many_valued=["knows"])
    bob = await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Carol", valid_from="2021-01-01T00:00:00Z")
    await memory.close_fact(SPACE, bob.fact_id, "lost touch")
    assert await held(memory, "alice", "knows") == [("Bob", "2020-01-01", "2025-06-01"), ("Carol", "2021-01-01", None)]


async def test_a_predicate_is_configured_as_the_ledger_stores_it():
    memory = await engine(many_valued=["  KNOWS "])
    assert memory.many_valued == frozenset({"knows"})
    await memory.assert_fact(SPACE, "alice", "Knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "knows", "Carol", valid_from="2021-01-01T00:00:00Z")
    assert [value for value, _, until in await held(memory, "alice", "knows") if until is None] == ["Bob", "Carol"]


@pytest.mark.parametrize("many_valued, message", [
    ("knows", "a collection of predicates"),
    ([""], "predicate must not be empty"),
    (["knows", 3], "predicate"),
])
async def test_a_malformed_configuration_is_refused(many_valued, message):
    with pytest.raises(InvalidInput, match=message):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), many_valued=many_valued)


async def test_each_assertion_records_the_rule_that_placed_it():
    memory = await engine(many_valued=["knows"])
    await memory.assert_fact(SPACE, "alice", "knows", "Bob", valid_from="2020-01-01T00:00:00Z")
    await memory.assert_fact(SPACE, "alice", "lives_in", "Porto", valid_from="2020-01-01T00:00:00Z")
    asserted = [event.payload for event in await memory.events.query(SPACE, kind="fact_assert", limit=10)]
    assert sorted((payload["predicate"], payload["cardinality"]) for payload in asserted) == [
        ("knows", "many"), ("lives_in", "one")]
