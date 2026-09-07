"""Choosing which memories a search may see, by what was recorded about them.

A filter narrows the candidates before anything is ranked, so a search of
one customer's notes cannot be crowded out by another's. It is a small
expression: conditions on metadata keys, combined with all and any, and
negated where that is what you mean.
"""

from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.filters import MAX_CONDITIONS, MAX_DEPTH, parse_filter

NOTE = {"category": "engineering", "priority": "5", "status": "published", "team": "eng"}


def keeps(spec, metadata=NOTE):
    return parse_filter(spec).matches(metadata)


def test_a_key_matching_exactly_is_the_common_case():
    assert keeps({"field": "category", "is": "engineering"})
    assert not keeps({"field": "category", "is": "sales"})


def test_a_key_that_was_never_recorded_matches_nothing():
    """Absent is not empty. A memory with no priority is not a memory of
    priority zero, and treating it as one quietly widens every search."""
    assert not keeps({"field": "team_size", "is": "4"})
    assert not keeps({"field": "team_size", "has": ""})
    assert not keeps({"field": "team_size", "at_least": 0})


def test_asking_only_whether_something_was_recorded():
    assert keeps({"field": "priority", "present": True})
    assert not keeps({"field": "team_size", "present": True})


def test_part_of_a_value_is_enough_when_that_is_what_was_asked():
    assert keeps({"field": "category", "has": "engine"})
    assert not keeps({"field": "category", "has": "sales"})


def test_one_of_several_replaces_a_row_of_alternatives():
    assert keeps({"field": "team", "in": ["eng", "product"]})
    assert not keeps({"field": "team", "in": ["sales", "legal"]})


@pytest.mark.parametrize("spec, kept", [
    ({"field": "priority", "at_least": 5}, True),
    ({"field": "priority", "at_least": 6}, False),
    ({"field": "priority", "above": 4}, True),
    ({"field": "priority", "above": 5}, False),
    ({"field": "priority", "at_most": 5}, True),
    ({"field": "priority", "below": 5}, False),
])
def test_numbers_recorded_as_text_still_compare_as_numbers(spec, kept):
    """Metadata values are text, so "10" sorts before "9" as a string and
    a priority filter would be nonsense. Comparison parses first."""
    assert keeps(spec) is kept


def test_a_value_that_is_not_a_number_never_satisfies_a_number_test():
    assert not keeps({"field": "status", "at_least": 1})
    assert not keeps({"field": "status", "below": 1})


def test_negating_a_condition_says_the_opposite_of_it():
    assert keeps({"field": "status", "is": "draft", "not": True})
    assert not keeps({"field": "status", "is": "published", "not": True})


def test_a_negated_condition_still_needs_the_key_to_be_there():
    """Otherwise "not draft" would match every memory that has no status
    at all, which is how a filter silently stops filtering."""
    assert not keeps({"field": "team_size", "is": "4", "not": True})


def test_conditions_can_be_required_together_or_as_alternatives():
    assert keeps({"all": [{"field": "category", "is": "engineering"},
                          {"field": "priority", "at_least": 5}]})
    assert not keeps({"all": [{"field": "category", "is": "engineering"},
                              {"field": "priority", "at_least": 9}]})
    assert keeps({"any": [{"field": "team", "is": "sales"},
                          {"field": "team", "is": "eng"}]})


def test_groups_nest_so_a_real_question_can_be_asked():
    assert keeps({"all": [
        {"field": "status", "is": "published"},
        {"any": [{"field": "team", "is": "eng"}, {"field": "team", "is": "product"}]},
    ]})


def test_an_empty_group_is_a_mistake_rather_than_everything_or_nothing():
    """All of nothing is true and any of nothing is false, and nobody
    writing a filter means either; it is a query built from a variable
    that turned out empty."""
    with pytest.raises(InvalidInput, match="at least one condition"):
        parse_filter({"all": []})
    with pytest.raises(InvalidInput, match="at least one condition"):
        parse_filter({"any": []})


@pytest.mark.parametrize("spec, complaint", [
    ({"field": "Category", "is": "x"}, "metadata key"),
    ({"field": "category"}, "exactly one test"),
    ({"field": "category", "is": "x", "has": "y"}, "exactly one test"),
    ({"is": "x"}, "field"),
    ({"field": "category", "wobble": "x"}, "exactly one test"),
    ({"all": [{"field": "category", "is": "x"}], "any": []}, "all or any"),
    ("category = x", "must be"),
    ({"field": "category", "in": []}, "at least one value"),
    ({"field": "category", "at_least": "soon"}, "number"),
], ids=["bad key", "no test", "two tests", "no field", "unknown test",
        "both groups", "not a mapping", "empty choice", "not a number"])
def test_a_filter_that_cannot_mean_anything_is_refused(spec, complaint):
    with pytest.raises(InvalidInput, match=complaint):
        parse_filter(spec)


def test_a_filter_too_large_to_be_meant_is_refused():
    """A runaway filter is a program building a query in a loop, not a
    person asking a question, and it would be the store's problem."""
    wide = {"any": [{"field": "team", "is": str(n)} for n in range(MAX_CONDITIONS + 1)]}
    with pytest.raises(InvalidInput, match="at most"):
        parse_filter(wide)


def test_a_filter_nested_deeper_than_anyone_reads_is_refused():
    deep = {"field": "team", "is": "eng"}
    for _ in range(MAX_DEPTH + 1):
        deep = {"all": [deep]}
    with pytest.raises(InvalidInput, match="nested"):
        parse_filter(deep)


def test_the_keys_a_filter_touches_are_available_without_walking_it():
    """A caller that has to authorise a filter needs to know what it
    reads without re-implementing the walk."""
    spec = {"all": [{"field": "category", "is": "engineering"},
                    {"any": [{"field": "team", "is": "eng"},
                             {"field": "priority", "at_least": 3}]}]}
    assert parse_filter(spec).fields() == {"category", "team", "priority"}


# --- turning a filter into SQL -------------------------------------------

import json
import sqlite3

CORPUS = [
    {"category": "engineering", "priority": "5", "status": "published", "team": "eng"},
    {"category": "engineering", "priority": "10", "status": "draft", "team": "product"},
    {"category": "sales", "priority": "9", "status": "published", "team": "sales"},
    {"category": "engineering", "status": "published", "team": "eng"},
    {"category": "research", "priority": "soon", "status": "review", "team": "eng"},
    {"category": "engineering", "priority": "-2.5", "status": "published", "team": "eng"},
    {},
]

FILTERS = [
    {"field": "category", "is": "engineering"},
    {"field": "category", "has": "engine"},
    {"field": "team", "in": ["eng", "product"]},
    {"field": "priority", "present": True},
    {"field": "priority", "at_least": 5},
    {"field": "priority", "above": 5},
    {"field": "priority", "below": 0},
    {"field": "priority", "at_most": 9},
    {"field": "status", "is": "draft", "not": True},
    {"field": "priority", "at_least": 5, "not": True},
    {"field": "category", "has": "sales", "not": True},
    {"all": [{"field": "category", "is": "engineering"}, {"field": "priority", "at_least": 5}]},
    {"any": [{"field": "team", "is": "sales"}, {"field": "status", "is": "review"}]},
    {"all": [{"field": "status", "is": "published"},
             {"any": [{"field": "priority", "above": 4}, {"field": "team", "is": "eng"}]}]},
]

EXACT = {"is", "has", "in", "present"}


def rows_from_sql(spec):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, metadata TEXT NOT NULL)")
    db.executemany("INSERT INTO notes (id, metadata) VALUES (?, ?)",
                   [(i, json.dumps(m)) for i, m in enumerate(CORPUS)])
    clause, params = parse_filter(spec).to_sql("notes.metadata")
    found = db.execute(f"SELECT id FROM notes WHERE {clause} ORDER BY id", params).fetchall()
    return {row[0] for row in found}


def rows_from_python(spec):
    condition = parse_filter(spec)
    return {i for i, note in enumerate(CORPUS) if condition.matches(note)}


@pytest.mark.parametrize("spec", FILTERS, ids=range(len(FILTERS)))
def test_the_sql_never_loses_a_memory_the_filter_would_keep(spec):
    """The narrowing may be generous, because SQL selects candidates and
    the filter itself decides. Generous costs a row that is then dropped.
    The other direction loses a memory that should have been found, and
    no later step can put it back."""
    assert rows_from_python(spec) <= rows_from_sql(spec)


@pytest.mark.parametrize("spec", [f for f in FILTERS if set(f) & EXACT and not f.get("not")],
                         ids=lambda f: str(sorted(f))[:40])
def test_a_test_sql_can_make_exactly_narrows_exactly(spec):
    """Where SQLite can express the test, nothing is left for the second
    pass to remove: text equality, substring, one of a list, presence."""
    assert rows_from_sql(spec) == rows_from_python(spec)


def test_a_number_stored_as_a_word_is_not_swept_in_by_a_loose_cast():
    """SQLite casts 'soon' to 0.0, so a naive at_most would return it.
    The clause has to keep it out, or a filter for low priority quietly
    returns everything that has no number in it at all."""
    assert 4 not in rows_from_sql({"field": "priority", "at_most": 3})


def test_a_negated_numeric_keeps_every_candidate_for_the_second_pass():
    """Negating a generous test would make it strict, and strict is the
    direction that loses memories. It falls back to asking only that the
    key is there."""
    assert rows_from_sql({"field": "priority", "at_least": 5, "not": True}) == \
        rows_from_sql({"field": "priority", "present": True})


def test_a_negated_exact_test_does_not_drag_in_memories_with_no_such_key():
    """An exact test has an exact opposite, so the clause can also say
    the key must be there. Without that, a question about anything not a
    draft fetches every memory that has no status at all, and the second
    pass throws them away again."""
    assert 6 not in rows_from_sql({"field": "status", "is": "draft", "not": True})
