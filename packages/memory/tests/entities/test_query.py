"""Asking the graph: find an entity by name, and how two entities connect."""
from __future__ import annotations

from scone_memory.core.models import Fact
from scone_memory.entities.project import project_entities
from scone_memory.entities.query import paths_between, resolve


def fact(number: int, subject: str, predicate: str, object_: str, quote: str | None = None) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z", quote=quote, source_episode_id=1 if quote else None)


LEDGER = [
    fact(1, "alice chen", "works_at", "Acme Robotics", "Dr. Alice Chen joined Acme Robotics."),
    fact(2, "acme robotics", "based_in", "Lisbon"),
    fact(3, "bob stone", "lives_in", "Lisbon"),
    fact(4, "alice park", "knows", "Bob Stone"),
    fact(5, "carol", "joined_on", "May 2021"),
]


def ids(projection):
    return {entity.key: entity.entity_id for entity in projection.entities}


def test_a_name_resolves_by_its_best_tier_and_says_how():
    projection = project_entities("alpha", LEDGER, revision=1)
    found = resolve(projection, "ACME ROBOTICS")
    assert found.status == "resolved" and found.tier == "key"
    assert [c.entity_id for c in found.candidates] == [ids(projection)["acme robotics"]]
    assert resolve(projection, ids(projection)["lisbon"]).tier == "id"


def test_an_ambiguous_name_lists_every_candidate_instead_of_guessing():
    projection = project_entities("alpha", LEDGER, revision=1)
    found = resolve(projection, "alice")
    assert found.status == "ambiguous" and found.tier == "prefix"
    assert {c.entity_id for c in found.candidates} == {ids(projection)["alice chen"], ids(projection)["alice park"]}


def test_a_name_nobody_recorded_is_not_found():
    projection = project_entities("alpha", LEDGER, revision=1)
    assert resolve(projection, "Globex").status == "not_found"


def test_the_path_between_two_entities_keeps_direction_and_cites_facts():
    projection = project_entities("alpha", LEDGER, revision=1)
    key = ids(projection)
    result = paths_between(projection, key["alice chen"], key["bob stone"], max_hops=4, limit=3)
    assert result.status == "found"
    path = result.paths[0]
    assert [hop.predicate for hop in path.hops] == ["works_at", "based_in", "lives_in"]
    assert [hop.direction for hop in path.hops] == ["forward", "forward", "reverse"]
    assert [hop.fact_ids for hop in path.hops] == [(1,), (2,), (3,)]
    assert path.entity_ids == (key["alice chen"], key["acme robotics"], key["lisbon"], key["bob stone"])


def test_no_path_within_the_hop_limit_is_reported_as_such():
    projection = project_entities("alpha", LEDGER, revision=1)
    key = ids(projection)
    assert paths_between(projection, key["alice chen"], key["bob stone"], max_hops=2, limit=3).status == "none_within_limit"
    assert paths_between(projection, key["alice chen"], key["carol"], max_hops=6, limit=3).status == "disconnected"


def test_several_distinct_shortest_routes_are_all_returned():
    rows = [fact(1, "a", "knows", "B"), fact(2, "b", "knows", "D"), fact(3, "a", "knows", "C"), fact(4, "c", "knows", "D")]
    projection = project_entities("alpha", rows, revision=1)
    key = ids(projection)
    result = paths_between(projection, key["a"], key["d"], max_hops=3, limit=5)
    assert sorted(tuple(hop.fact_ids for hop in path.hops) for path in result.paths) == [((1,), (2,)), ((3,), (4,))]


def test_a_hub_is_not_a_shortcut():
    rows = [fact(n, "hub", "knows", f"Person {n}") for n in range(1, 8)]
    rows += [fact(20, "person 1", "reports_to", "Manager"), fact(21, "manager", "manages", "Person 2")]
    projection = project_entities("alpha", rows, revision=1)
    key = ids(projection)
    result = paths_between(projection, key["person 1"], key["person 2"], max_hops=4, limit=3, hub_degree=5)
    assert result.status == "found"
    assert all(key["hub"] not in path.entity_ids[1:-1] for path in result.paths)
    assert result.hubs_skipped == (key["hub"],)


def test_a_prefix_must_end_at_a_word_boundary():
    projection = project_entities("alpha", LEDGER, revision=1)
    assert resolve(projection, "ali").status == "not_found"
