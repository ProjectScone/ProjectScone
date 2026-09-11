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
        {"kind": "organisation", "entities": 2, "inferred": 2},
        {"kind": "person", "entities": 2, "inferred": 2},
        {"kind": "place", "entities": 1, "inferred": 1},
        {"kind": None, "entities": 1, "inferred": 0},
    ]


def test_each_predicate_says_which_kinds_it_joins_and_which_values_it_takes():
    by_name = {entry["predicate"]: entry for entry in graph_schema(projected())["predicates"]}
    assert by_name["works_at"] == {
        "predicate": "works_at", "facts": 2, "relations": 2, "attributes": 0,
        "joins": [{"subject": "person", "object": "organisation", "relations": 2, "facts": 2}], "values": []}
    assert by_name["age"] == {
        "predicate": "age", "facts": 2, "relations": 0, "attributes": 2, "joins": [],
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
    assert "kind: organisation, 2 entities (2 inferred)" in lines and "kind: unknown, 1 entity" in lines
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
