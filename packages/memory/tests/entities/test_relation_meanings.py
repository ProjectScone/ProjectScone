"""What a predicate means, and what follows from it.

A graph that only holds what was said is a record of claims. Saying that
works_at is the other side of employs, that married_to reads the same
both ways, and that part_of carries through, is what makes it a graph you
can ask questions of. Meanings are configured, never guessed: nothing
here infers that two predicates are opposites because they look alike.
"""

from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import Fact
from scone_memory.entities.meanings import MAX_STEPS, RelationMeanings
from scone_memory.entities.project import Implied, project_entities

SPACE = "alpha"


def fact(fact_id: int, subject: str, predicate: str, object: str) -> Fact:
    return Fact(fact_id=fact_id, space=SPACE, subject=subject, predicate=predicate, object=object,
                valid_from="2024-01-01T00:00:00.000Z", status="active")


def projected(facts, meanings=None):
    return project_entities(SPACE, facts, revision=1, meanings=meanings)


def test_meanings_are_read_the_way_the_ledger_reads_predicates():
    """Case folded and whitespace collapsed, exactly as a stored predicate
    is, so a vocabulary written by a person meets the claims it is about."""
    meanings = RelationMeanings(inverse={"Works  At": "EMPLOYS"}, symmetric=["Married To"],
                                transitive=["Part_Of"])
    assert meanings.opposite("works at") == "employs"
    assert meanings.opposite("employs") == "works at"
    assert meanings.reads_both_ways("married to") and not meanings.reads_both_ways("works at")
    assert meanings.carries_through("part_of"), "an underscore is a letter, not a space"
    assert meanings.record()["inverse"] == {"employs": "works at", "works at": "employs"}


@pytest.mark.parametrize("bad, says", [
    ({"inverse": {"works_at": "works_at"}}, "its own opposite"),
    ({"inverse": {"works_at": "employs", "employs": "hires"}}, "one opposite"),
    ({"inverse": {"works_at": ""}}, "must not be empty"),
    ({"symmetric": "married_to"}, "collection"),
    ({"inverse": {"married_to": "spouse_of"}, "symmetric": ["married_to"]}, "reads the same both ways"),
])
def test_a_vocabulary_that_cannot_mean_what_it_says_is_refused(bad, says):
    with pytest.raises(InvalidInput, match=says):
        RelationMeanings(**bad)


def test_a_predicate_that_is_its_own_opposite_is_just_symmetric():
    meanings = RelationMeanings(symmetric=["married_to"])
    assert meanings.opposite("married_to") == "married_to"


def test_the_other_side_of_a_claim_follows_from_it():
    facts = [fact(1, "Alice Chen", "works_at", "Acme Robotics")]
    plain = projected(facts)
    assert plain.implied == ()
    graph = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    [followed] = graph.implied
    [stated] = graph.relations
    assert followed.predicate == "employs"
    assert (followed.subject_id, followed.object_id) == (stated.object_id, stated.subject_id)
    assert followed.fact_ids == stated.fact_ids, "it rests on the claim it follows from"
    assert followed.follows == "inverse" and followed.follows_from == (stated.relation_id,)


def test_a_claim_that_reads_both_ways_follows_both_ways():
    facts = [fact(1, "Alice Chen", "married_to", "Bob Stone")]
    graph = projected(facts, RelationMeanings(symmetric=["married_to"]))
    [followed] = graph.implied
    assert followed.predicate == "married_to"
    assert followed.subject_id == graph.relations[0].object_id


def test_nothing_is_implied_twice_and_nothing_already_said_is_implied():
    facts = [fact(1, "Alice Chen", "works_at", "Acme Robotics"),
             fact(2, "Acme Robotics", "employs", "Alice Chen")]
    graph = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    assert len(graph.relations) == 2
    assert graph.implied == (), "both sides were already said"


def test_what_carries_through_carries_through_the_whole_way():
    facts = [fact(1, "Shelf A", "part_of", "Aisle 3"),
             fact(2, "Aisle 3", "part_of", "Warehouse 7"),
             fact(3, "Warehouse 7", "part_of", "The Estate")]
    graph = projected(facts, RelationMeanings(transitive=["part_of"]))
    reached = {(r.subject_id, r.object_id) for r in graph.implied}
    assert len(reached) == 3, "A->W, A->E, 3->E"
    longest = max(graph.implied, key=lambda r: len(r.fact_ids))
    assert longest.fact_ids == (1, 2, 3) and longest.follows == "transitive"
    assert len(longest.follows_from) == 3


def test_what_carries_through_stops_at_a_bounded_number_of_steps():
    facts = [fact(n, f"Box {n}", "part_of", f"Box {n + 1}") for n in range(1, 30)]
    graph = projected(facts, RelationMeanings(transitive=["part_of"]))
    assert graph.implied and max(len(r.fact_ids) for r in graph.implied) == MAX_STEPS


def test_a_ring_of_claims_does_not_imply_a_thing_about_itself():
    facts = [fact(1, "A", "part_of", "B"), fact(2, "B", "part_of", "C"), fact(3, "C", "part_of", "A")]
    graph = projected(facts, RelationMeanings(transitive=["part_of"]))
    assert all(r.subject_id != r.object_id for r in graph.implied)


def test_a_different_vocabulary_is_a_different_projection():
    facts = [fact(1, "Alice Chen", "works_at", "Acme Robotics")]
    plain = projected(facts)
    graph = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    other = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    assert plain.digest != graph.digest and graph.digest == other.digest


def test_what_was_implied_is_never_mistaken_for_what_was_said():
    """A stated relation and one that follows are different types, so no
    view, count or export can read one as the other by accident."""
    facts = [fact(1, "Alice Chen", "works_at", "Acme Robotics")]
    graph = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    assert [r.predicate for r in graph.relations] == ["works_at"]
    assert not any(isinstance(r, Implied) for r in graph.relations)
    assert all(isinstance(r, Implied) for r in graph.implied)


def ended(fact_id: int, subject: str, predicate: str, object: str, since: str, until: str) -> Fact:
    return Fact(fact_id=fact_id, space=SPACE, subject=subject, predicate=predicate, object=object,
                valid_from=f"{since}T00:00:00.000Z", valid_until=f"{until}T00:00:00.000Z", status="closed")


def test_what_follows_holds_only_while_every_claim_it_rests_on_does():
    """A shelf is in the warehouse from the day the later of the two claims
    begins, and stops being in it the day either one stops."""
    facts = [fact(1, "Shelf A", "part_of", "Aisle 3"),
             ended(2, "Aisle 3", "part_of", "Warehouse 7", "2022-06-01", "2024-06-01")]
    graph = projected(facts, RelationMeanings(transitive=["part_of"]))
    [followed] = graph.implied
    assert followed.first_valid_from.startswith("2024-01-01"), "the later of the two beginnings"
    assert followed.last_valid_until is not None and followed.last_valid_until.startswith("2024-06-01")


def test_what_follows_is_no_better_supported_than_its_weakest_claim():
    facts = [Fact(fact_id=1, space=SPACE, subject="Shelf A", predicate="part_of", object="Aisle 3",
                  valid_from="2024-01-01T00:00:00.000Z", status="active",
                  source_episode_id=1, quote="Shelf A is part of Aisle 3"),
             fact(2, "Aisle 3", "part_of", "Warehouse 7")]
    graph = projected(facts, RelationMeanings(transitive=["part_of"]))
    [followed] = graph.implied
    stated = {r.relation_id: r for r in graph.relations}
    weakest = min((stated[rid] for rid in followed.follows_from), key=lambda r: r.support.quoted)
    assert followed.support == weakest.support
    assert followed.support.quoted == 0, "one leg nobody quoted makes the whole chain unquoted"
