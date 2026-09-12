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
from scone_memory.entities.read import load_projection

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


async def test_an_engine_projects_what_its_vocabulary_implies():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=RelationMeanings(inverse={"works_at": "employs"})).open()
    await memory.assert_fact(SPACE, "Alice Chen", "works_at", "Acme Robotics",
                             valid_from="2024-01-01T00:00:00Z")
    projection, _ = await load_projection(memory, SPACE)
    assert [(r.subject_id, r.predicate) for r in projection.implied] == [
        (projection.relations[0].object_id, "employs")]


async def test_an_engine_with_no_vocabulary_implies_nothing():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.assert_fact(SPACE, "Alice Chen", "works_at", "Acme Robotics",
                             valid_from="2024-01-01T00:00:00Z")
    projection, _ = await load_projection(memory, SPACE)
    assert projection.implied == () and memory.relation_meanings is None


def test_a_vocabulary_is_read_from_the_environment():
    from scone_memory.runtime.config import Settings, build_relation_meanings

    settings = Settings.from_env({"SCONE_RELATION_INVERSE": "works_at:employs, part_of:contains",
                                  "SCONE_RELATION_SYMMETRIC": "married_to",
                                  "SCONE_RELATION_TRANSITIVE": "part_of, located_in"})
    meanings = build_relation_meanings(settings)
    assert meanings is not None
    assert meanings.opposite("works_at") == "employs"
    assert meanings.reads_both_ways("married_to") and meanings.carries_through("located_in")
    assert build_relation_meanings(Settings.from_env({})) is None


def test_a_vocabulary_that_is_not_a_pair_is_refused_where_it_is_written():
    from scone_memory.runtime.config import Settings, build_relation_meanings

    with pytest.raises(InvalidInput, match="predicate:its opposite"):
        build_relation_meanings(Settings.from_env({"SCONE_RELATION_INVERSE": "works_at employs"}))


def test_a_graph_view_lists_what_follows_apart_from_what_was_said():
    """A reader sees the claim and the thing that follows from it, and can
    always tell which is which and what the second rests on."""
    from scone_memory.entities.view import knowledge_view

    facts = [fact(1, "Alice Chen", "works_at", "Acme Robotics")]
    graph = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    view = knowledge_view(graph, mode="current", as_of="2025-01-01T00:00:00.000Z", limit=10,
                          attribute_limit=10, offset=0, coverage={"reasons": []})
    [stated] = view["relations"]
    [followed] = view["implied"]
    assert stated["predicate"] == "works_at" and "follows" not in stated
    assert followed["predicate"] == "employs" and followed["follows"] == "inverse"
    assert followed["follows_from"] == [stated["id"]]
    assert followed["fact_ids"] == stated["fact_ids"]
    assert view["coverage"]["implied_total"] == 1 and view["coverage"]["implied_shown"] == 1


def test_a_graph_view_of_a_space_with_no_vocabulary_has_nothing_that_follows():
    from scone_memory.entities.view import knowledge_view

    graph = projected([fact(1, "Alice Chen", "works_at", "Acme Robotics")])
    view = knowledge_view(graph, mode="current", as_of="2025-01-01T00:00:00.000Z", limit=10,
                          attribute_limit=10, offset=0, coverage={"reasons": []})
    assert view["implied"] == [] and view["coverage"]["implied_total"] == 0


def holds(fact_id: int, subject: str, predicate: str, object: str, since: str) -> Fact:
    return Fact(fact_id=fact_id, space=SPACE, subject=subject, predicate=predicate, object=object,
                valid_from=f"{since}T00:00:00.000Z", status="active")


def test_what_follows_goes_when_a_claim_under_it_stops_holding():
    """A chain is only as present as its legs: when the aisle stops being
    part of the warehouse, the shelf stops being in the warehouse too,
    even though the shelf is still in the aisle."""
    from scone_memory.entities.view import knowledge_view

    facts = [holds(1, "Shelf A", "part_of", "Aisle 3", "2020-01-01"),
             ended(2, "Aisle 3", "part_of", "Warehouse 7", "2021-01-01", "2023-01-01"),
             # The warehouse is still spoken of, so it is still in the
             # graph: what goes is the chain through the claim that ended.
             holds(3, "Warehouse 7", "based_in", "Lisbon", "2020-01-01")]
    graph = projected(facts, RelationMeanings(transitive=["part_of"]))

    def at(moment: str):
        return knowledge_view(graph, mode="current", as_of=f"{moment}T00:00:00.000Z", limit=10,
                              attribute_limit=10, offset=0, coverage={"reasons": []})

    while_both = at("2022-01-01")
    assert [item["predicate"] for item in while_both["implied"]] == ["part_of"]
    after = at("2024-01-01")
    assert after["implied"] == [], "the leg that ended takes what followed with it"
    assert "part_of" in [r["predicate"] for r in after["relations"]], "the other leg still holds"


def test_an_entitys_page_shows_what_follows_from_the_claims_about_it():
    """Acme employs nobody in the ledger; it employs Alice because she
    works there, and the page says so in its own place."""
    from scone_memory.entities.query import neighbourhood, resolve

    facts = [fact(1, "Alice Chen", "works_at", "Acme Robotics")]
    graph = projected(facts, RelationMeanings(inverse={"works_at": "employs"}))
    [acme] = resolve(graph, "Acme Robotics").candidates
    found = neighbourhood(graph, acme.entity_id)
    assert found is not None and found.outgoing == ()
    assert [(item.predicate, item.follows) for item in found.follows] == [("employs", "inverse")]
    assert found.follows[0].object_id != acme.entity_id


def test_an_entity_page_without_a_vocabulary_has_nothing_that_follows():
    from scone_memory.entities.query import neighbourhood, resolve

    graph = projected([fact(1, "Alice Chen", "works_at", "Acme Robotics")])
    [acme] = resolve(graph, "Acme Robotics").candidates
    found = neighbourhood(graph, acme.entity_id)
    assert found is not None and found.follows == ()


async def test_the_packet_a_model_reads_says_what_follows_and_what_from():
    """An agent asking about Acme is told it employs Alice, that this
    follows rather than was said, and which claim it follows from."""
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.entities.context import graph_context

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=RelationMeanings(inverse={"works_at": "employs"})).open()
    said = await memory.remember(SPACE, "Alice Chen works at Acme Robotics.")
    await memory.assert_fact(SPACE, "Alice Chen", "works_at", "Acme Robotics",
                             valid_from="2024-01-01T00:00:00Z", source_episode_id=said.episode_id,
                             quote="Alice Chen works at Acme Robotics.")
    packet = await graph_context(memory, SPACE, names=["Acme Robotics"])
    assert "hop 1: Alice Chen works_at Acme Robotics" in packet.text, packet.text
    assert "follows: Acme Robotics employs Alice Chen (inverse)" in packet.text, packet.text


async def test_a_packet_with_no_vocabulary_says_only_what_was_claimed():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.entities.context import graph_context

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.assert_fact(SPACE, "Alice Chen", "works_at", "Acme Robotics",
                             valid_from="2024-01-01T00:00:00Z")
    packet = await graph_context(memory, SPACE, names=["Acme Robotics"])
    assert "follows:" not in packet.text


async def test_a_chain_goes_when_a_claim_under_one_leg_stops_counting():
    """What follows is only as good as every claim under it. Exclude the
    middle claim and the chain goes with it, while the claim beside it
    still shows."""
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.entities.context import graph_context

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                relation_meanings=RelationMeanings(transitive=["part_of"])).open()
    first = await memory.remember(SPACE, "Shelf A is part of Aisle 3.")
    second = await memory.remember(SPACE, "Aisle 3 is part of Warehouse 7.")
    await memory.assert_fact(SPACE, "Shelf A", "part_of", "Aisle 3", valid_from="2024-01-01T00:00:00Z",
                             source_episode_id=first.episode_id, quote="Shelf A is part of Aisle 3.")
    await memory.assert_fact(SPACE, "Aisle 3", "part_of", "Warehouse 7", valid_from="2024-01-01T00:00:00Z",
                             source_episode_id=second.episode_id, quote="Aisle 3 is part of Warehouse 7.")
    before = await graph_context(memory, SPACE, names=["Shelf A"])
    assert "follows: Shelf A part_of Warehouse 7 (transitive)" in before.text, before.text

    await memory.exclude(SPACE, 2, "the aisle was renumbered")
    after = await graph_context(memory, SPACE, names=["Shelf A"])
    assert "follows:" not in after.text, after.text
    assert "hop 1: Shelf A part_of Aisle 3" in after.text, "the claim beside it still shows"


def test_what_follows_never_borrows_the_id_of_a_claim():
    """An implied edge that happens to join the same two things by the same
    predicate must not be mistaken for the claim, anywhere it is carried."""
    from scone_memory.entities.ids import implied_id, relation_id

    facts = [fact(1, "Alice Chen", "married_to", "Bob Stone")]
    graph = projected(facts, RelationMeanings(symmetric=["married_to"]))
    [stated], [followed] = graph.relations, graph.implied
    assert followed.relation_id.startswith("imp:") and stated.relation_id.startswith("rel:")
    same = relation_id(SPACE, followed.subject_id, followed.predicate, followed.object_id)
    assert followed.relation_id != same, "the id says which kind of edge it is"
    assert followed.relation_id == implied_id(SPACE, followed.subject_id, followed.predicate, followed.object_id)
    assert {r.relation_id for r in graph.relations}.isdisjoint({i.relation_id for i in graph.implied})


def test_a_view_says_the_vocabulary_it_applied_and_where_it_stopped():
    """A bounded answer says what bounded it: which meanings were in force,
    how far a chain is followed, and how many implications it will hold."""
    from scone_memory.entities.meanings import MAX_IMPLIED
    from scone_memory.entities.view import knowledge_view

    graph = projected([fact(1, "Shelf A", "part_of", "Aisle 3")],
                      RelationMeanings(transitive=["part_of"], inverse={"works_at": "employs"}))
    view = knowledge_view(graph, mode="current", as_of="2025-01-01T00:00:00.000Z", limit=10,
                          attribute_limit=10, offset=0, coverage={"reasons": []})
    said = view["coverage"]["meanings"]
    assert said["transitive"] == ["part_of"] and said["inverse"] == {"employs": "works_at", "works_at": "employs"}
    assert said["max_steps"] == MAX_STEPS and said["max_implied"] == MAX_IMPLIED
    assert "implied_capped" not in view["coverage"], "nothing was left out"


def test_a_view_of_a_space_with_no_vocabulary_says_so():
    from scone_memory.entities.view import knowledge_view

    graph = projected([fact(1, "Shelf A", "part_of", "Aisle 3")])
    view = knowledge_view(graph, mode="current", as_of="2025-01-01T00:00:00.000Z", limit=10,
                          attribute_limit=10, offset=0, coverage={"reasons": []})
    assert view["coverage"]["meanings"] is None
