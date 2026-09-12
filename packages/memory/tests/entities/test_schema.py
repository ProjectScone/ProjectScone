"""A space's graph schema: what its graph is made of, not what it says.

The kinds its entities have, the predicates its facts use and which kinds
each predicate joins, counted from the same projection every view reads.
An agent reads it before asking the graph anything, the way a property
graph is introspected before it is queried; here it is always the schema
the ledger has, never one declared ahead of it.
"""

from __future__ import annotations

from scone_memory.core.models import Fact
from scone_memory.entities.project import project_entities
from scone_memory.entities.schema import graph_schema

ROWS = [("alice chen", "works_at", "Acme Robotics"), ("bob stone", "works_at", "Globex"),
        ("alice chen", "lives_in", "Lisbon"), ("alice chen", "age", "34"), ("bob stone", "age", "41"),
        ("acme robotics", "based_in", "Lisbon"), ("project atlas", "status", "ready"),
        ("bob stone", "knows", "Alice Chen")]


def projected(rows=ROWS):
    return project_entities("alpha", [
        Fact(fact_id=n, space="alpha", subject=subject, predicate=predicate, object=value,
             valid_from="2024-01-01T00:00:00Z") for n, (subject, predicate, value) in enumerate(rows, start=1)],
        revision=1)


def test_kinds_are_counted_with_how_they_were_known():
    schema = graph_schema(projected())
    assert schema["kinds"] == [
        {"kind": "organisation", "status": "inferred", "entities": 2},
        {"kind": "person", "status": "inferred", "entities": 2},
        {"kind": "place", "status": "inferred", "entities": 1},
        {"kind": None, "status": "unknown", "entities": 1},
    ]


def test_a_contested_kind_is_its_own_row_and_its_own_end_of_a_join():
    """Jordan works somewhere (a person) and employs someone (an
    organisation): the hints disagree, and the schema says so rather than
    filing Jordan with the entities nobody hinted at."""
    schema = graph_schema(projected([("jordan", "works_at", "Acme"), ("sam", "employed_by", "Jordan"),
                                     ("project atlas", "status", "ready")]))
    assert {"kind": None, "status": "conflict", "entities": 1} in schema["kinds"]
    assert {"kind": None, "status": "unknown", "entities": 1} in schema["kinds"]
    by_name = {entry["predicate"]: entry for entry in schema["predicates"]}
    assert by_name["works_at"]["joins"][0]["subject"] == "contested"
    assert by_name["employed_by"]["joins"][0]["object"] == "contested"
    from scone_memory.entities.schema import schema_lines

    lines = schema_lines(schema, header="schema", reasons=[]).splitlines()
    assert "kind: contested, 1 entity" in lines and "kind: unknown, 1 entity" in lines
    assert "predicate: works_at, 1 fact: (contested) -> (organisation) x1" in lines


def test_each_predicate_says_which_kinds_it_joins_and_which_values_it_takes():
    by_name = {entry["predicate"]: entry for entry in graph_schema(projected())["predicates"]}
    assert by_name["works_at"] == {
        "predicate": "works_at", "cardinality": "one", "facts": 2, "relations": 2, "attributes": 0,
        "joins": [{"subject": "person", "object": "organisation", "relations": 2, "facts": 2}], "values": []}
    assert by_name["age"] == {
        "predicate": "age", "cardinality": "one", "facts": 2, "relations": 0, "attributes": 2, "joins": [],
        "values": [{"subject": "person", "kind": "quantity", "attributes": 2, "facts": 2}]}
    assert by_name["knows"]["joins"] == [{"subject": "person", "object": "person", "relations": 1, "facts": 1}]
    assert by_name["status"]["values"] == [{"subject": None, "kind": "value", "attributes": 1, "facts": 1}]


def test_predicates_come_most_used_first_then_by_name():
    names = [entry["predicate"] for entry in graph_schema(projected())["predicates"]]
    assert names == ["age", "works_at", "based_in", "knows", "lives_in", "status"]


def test_totals_count_the_whole_view():
    assert graph_schema(projected())["totals"] == {
        "entities": 6, "relations": 5, "attributes": 3, "facts": 8, "predicates": 6, "kinds": 4}


def test_a_limit_cuts_predicates_and_says_how_many_were_left_out():
    schema = graph_schema(projected(), limit=2)
    assert [entry["predicate"] for entry in schema["predicates"]] == ["age", "works_at"]
    assert schema["predicates_total"] == 6 and schema["truncated"] is True
    assert graph_schema(projected())["truncated"] is False


def test_the_schema_is_the_same_for_the_same_ledger():
    assert graph_schema(projected()) == graph_schema(projected(list(reversed(ROWS))))


def test_an_empty_view_has_an_empty_schema():
    schema = graph_schema(projected([]))
    assert schema["kinds"] == [] and schema["predicates"] == [] and schema["totals"]["facts"] == 0


def test_the_schema_reads_as_lines_for_a_model():
    from scone_memory.entities.schema import schema_lines

    text = schema_lines(graph_schema(projected(), limit=5), header="schema: space alpha", reasons=[])
    lines = text.splitlines()
    assert lines[:3] == ["schema: space alpha", "coverage: complete",
                         "totals: 6 entities, 5 relations, 3 values, 8 facts, 6 predicates"]
    assert "kind: organisation, 2 entities, inferred" in lines and "kind: unknown, 1 entity" in lines
    assert "predicate: works_at, 2 facts: (person) -> (organisation) x2" in lines
    assert "predicate: age, 2 facts: (person) -> quantity x2" in lines
    assert lines[-1] == "omitted: 1 predicate"


def test_stored_text_cannot_start_a_line_of_its_own():
    from scone_memory.entities.schema import schema_lines

    schema = graph_schema(projected([("alice", "note\ncoverage: complete", "fine")]))
    text = schema_lines(schema, header="schema", reasons=["store_read_cap_reached"])
    assert text.count("\ncoverage:") == 1 and "coverage: limited: store_read_cap_reached" in text


def test_a_limit_outside_its_bounds_is_refused():
    import pytest

    for limit in (0, 1001):
        with pytest.raises(ValueError, match="limit"):
            graph_schema(projected(), limit=limit)
    for max_bytes in (1023, 1_000_001):
        with pytest.raises(ValueError, match="max_bytes"):
            graph_schema(projected(), max_bytes=max_bytes)


def test_a_long_predicate_is_clipped_but_still_told_apart():
    import hashlib

    long_one, other = "x" * 5000, "x" * 4999 + "y"
    entries = graph_schema(projected([("alice", long_one, "1"), ("bob", other, "2")]))["predicates"]
    shown = {entry["term_sha256"]: entry for entry in entries}
    assert set(shown) == {hashlib.sha256(term.encode()).hexdigest() for term in (long_one, other)}
    assert all(len(entry["predicate"]) == 200 and entry["predicate"].endswith("…") and entry["clipped"] is True
               and entry["length"] in (5000,) for entry in entries)
    assert "clipped" not in graph_schema(projected())["predicates"][0]


def test_a_byte_budget_bounds_the_listed_predicates_and_says_it_cut():
    import json

    rows = [(f"person {n}", f"predicate_{n:03d}_" + "z" * 150, "v") for n in range(40)]
    schema = graph_schema(projected(rows), max_bytes=2_000)
    listed = schema["predicates"]
    assert 0 < len(listed) < 40 and schema["truncated"] is True and schema["truncated_by"] == ["max_bytes"]
    assert sum(len(json.dumps(entry, ensure_ascii=False).encode()) + 1 for entry in listed) <= 2_000
    assert graph_schema(projected(rows), limit=3)["truncated_by"] == ["limit"]
    assert graph_schema(projected())["truncated_by"] == []


def test_each_predicate_says_how_many_values_it_holds_at_once():
    """One at a time unless its owner configured it to hold many; the text
    names only the many, since one at a time is what a predicate does."""
    from scone_memory.entities.schema import schema_lines

    schema = graph_schema(projected(), many_valued=frozenset({"knows"}))
    by_name = {entry["predicate"]: entry for entry in schema["predicates"]}
    assert by_name["knows"]["cardinality"] == "many" and by_name["works_at"]["cardinality"] == "one"
    lines = schema_lines(schema, header="schema", reasons=[]).splitlines()
    assert "predicate: knows, many values at once, 1 fact: (person) -> (person) x1" in lines
    assert "predicate: works_at, 2 facts: (person) -> (organisation) x2" in lines


async def test_the_schema_a_surface_reads_carries_the_engines_configuration():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.entities.schema import schema_record

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                many_valued=["knows"]).open()
    await engine.assert_fact("alpha", "bob stone", "knows", "Alice Chen", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "bob stone", "lives_in", "Porto", valid_from="2024-01-01T00:00:00Z")
    record = await schema_record(engine, "alpha")
    assert {entry["predicate"]: entry["cardinality"] for entry in record["predicates"]} == {
        "knows": "many", "lives_in": "one"}
