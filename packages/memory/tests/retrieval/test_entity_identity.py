"""One rule for when two names are the same thing.

Every place that joins one claim to another has to agree on whether
"Acme Robotics" and "acme robotics" name the same entity. Four places
used to decide that four different ways. Subjects are normalised when a
claim is written and objects are kept as written, so a rule that compares
raw strings never joins across a subject at all, and the graph comes out
in disconnected pieces that should have met.
"""

from __future__ import annotations

import pytest

from scone_memory.core.models import Fact
from scone_memory.memory.engine import derivation_groups
from scone_memory.memory.identity import entity_key
from scone_memory.retrieval.adaptive_graph import exact_fact_groups
from scone_memory.retrieval.multihop import matches_subject_object


def claim(fact_id, subject, predicate, obj):
    return Fact(fact_id=fact_id, space="d", subject=subject, predicate=predicate, object=obj,
                valid_from="2024-01-01T00:00:00Z")


#: Stored the way the ledger stores them: subjects normalised, objects as written.
CHAIN = (claim(1, "alice chen", "works_at", "Acme Robotics"),
         claim(2, "acme robotics", "based_in", "Lisbon"))


@pytest.mark.parametrize("left, right", [
    ("Acme Robotics", "acme robotics"),
    ("  Acme   Robotics ", "acme robotics"),
    ("ACME ROBOTICS", "Acme Robotics"),
    ("Straße", "STRASSE"),
])
def test_names_that_differ_only_in_case_or_spacing_are_one_entity(left, right):
    assert entity_key(left) == entity_key(right)


@pytest.mark.parametrize("left, right", [
    ("Acme Robotics", "Acme Robot"),
    ("Lisbon", "Lisboa"),
    ("alice", "alice chen"),
])
def test_names_that_are_genuinely_different_stay_apart(left, right):
    """This rule is identity, not resemblance. Deciding that two different
    spellings are one entity is a resolution decision with evidence behind
    it, and it must not happen silently inside a join."""
    assert entity_key(left) != entity_key(right)


def test_a_claim_about_a_thing_joins_the_claim_that_names_it():
    """The regression: "alice works at Acme Robotics" and "acme robotics is
    based in Lisbon" are one chain. The evidence grouping returned no group
    for them at all."""
    groups = exact_fact_groups(CHAIN)
    assert groups == (("fact:1", "fact:2"),), f"expected one group joining both claims, got {groups!r}"


def test_every_join_site_agrees_on_what_connects():
    """The property that actually matters. It is not enough for one join to
    be right; they have to be right the same way, or the graph a person
    sees disagrees with the one retrieval walked."""
    by_grouping = {frozenset(g) for g in exact_fact_groups(CHAIN)}
    by_derivation = {frozenset(f"fact:{f.fact_id}" for f in g) for g in derivation_groups(list(CHAIN))}
    by_traversal = matches_subject_object(CHAIN[0].object, CHAIN[1].subject)

    assert by_grouping == {frozenset({"fact:1", "fact:2"})}
    assert by_derivation == {frozenset({"fact:1", "fact:2"})}
    assert by_traversal is True


def test_the_key_is_the_form_subjects_are_stored_in():
    """A join compares a stored subject with an object normalised at read
    time. If the two normalisations differ by so much as a collapsed space,
    the join silently misses."""
    from scone_memory.memory.engine import normalise_term

    for name in ("Acme Robotics", "  alice   chen ", "ÉCOLE Polytechnique"):
        assert entity_key(name) == normalise_term(name, "subject")
