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


#: (object, subject, how they join): the one table every join site follows.
JOINS = [
    ("Acme Robotics", "acme robotics", "normalised"),
    ("acme robotics", "acme robotics", "literal"),
    ("  NORTH   Gate  ", "north gate", "normalised"),
    ("2024-05-01", "2024-05-01", "literal"),
    ("MB", "MB", "literal"),
    ("MB", "mb", None),
    ("V2.3.1-RC", "v2.3.1-rc", None),
    ("It rained. We stayed in", "it rained. we stayed in", None),
    ('"ship it"', '"ship it"', None),
    ("it", "it", None),
    ("Lisbon", "Lisboa", None),
]


@pytest.mark.parametrize(("obj", "subject", "expected"), JOINS)
def test_one_rule_decides_whether_an_object_names_a_subject(obj, subject, expected):
    from scone_memory.memory.identity import join_match
    found = join_match(obj, subject)
    assert (found.match if found else None) == expected


@pytest.mark.parametrize(("obj", "subject", "expected"), JOINS)
async def test_every_join_site_follows_the_table(obj, subject, expected):
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor
    first, second = claim(1, "start", "relates", obj), claim(2, subject, "relates", "finish")
    by_grouping = exact_fact_groups((first, second)) == (("fact:1", "fact:2"),)
    by_derivation = len(derivation_groups([first, second])) == 1
    by_traversal = matches_subject_object(obj, subject)
    candidates = tuple(EvidenceCandidate(id=f"fact:{f.fact_id}", episode_id=f.fact_id, text="source",
                                         subject=f.subject, predicate=f.predicate, object=f.object)
                       for f in (first, second))
    requirement = EvidenceRequirement(kind="path", subject="start", predicate="relates", object="finish")
    decision = await StructuredEvidenceAssessor("question", (requirement,)).assess("question", candidates)
    by_structure = decision.status == "sufficient"
    assert (by_grouping, by_derivation, by_traversal, by_structure) == (expected is not None,) * 4


async def test_a_path_requirement_follows_names_across_their_casing():
    """The regression: the structured assessor indexed raw strings, so a
    path through 'Acme Robotics' never reached the claims about 'acme
    robotics' and alice's route to Lisbon was reported missing."""
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor
    candidates = (EvidenceCandidate(id="fact:1", episode_id=1, text="a", subject="alice chen",
                                    predicate="works_at", object="Acme Robotics"),
                  EvidenceCandidate(id="fact:2", episode_id=2, text="b", subject="acme robotics",
                                    predicate="works_at", object="Lisbon"))
    requirement = EvidenceRequirement(kind="path", subject="Alice Chen", predicate="works_at", object="lisbon")
    decision = await StructuredEvidenceAssessor("question", (requirement,)).assess("question", candidates)
    assert decision.status == "sufficient" and decision.selected_ids == ("fact:1", "fact:2")


@pytest.mark.parametrize(("premise", "derived", "restates"), [
    ("Acme", "acme", True),
    ("3 MB", "3 mb", False),
    ("3 MB", "3 MB", True),
])
def test_a_derivation_restates_a_premise_only_by_the_same_rule(premise, derived, restates):
    import json
    from scone_memory.ingestion.derive import parse_derivations
    group = {1: claim(1, "report", "size", premise), 2: claim(2, "report", "owner", "Dana")}
    text = json.dumps([{"subject": "report", "predicate": "size", "object": derived, "premises": [1, 2]}])
    accepted, rejected = parse_derivations(text, group)
    assert ([r.reason for r in rejected] == ["restates_premise"]) is restates
    assert (len(accepted) == 0) is restates


async def test_a_path_meets_a_subject_recorded_in_any_casing():
    """Candidates can come from stores that keep a subject as written."""
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor
    candidates = (EvidenceCandidate(id="fact:1", episode_id=1, text="a", subject="Alice Chen",
                                    predicate="Works_At", object="acme robotics"),
                  EvidenceCandidate(id="fact:2", episode_id=2, text="b", subject="ACME Robotics",
                                    predicate="works_at", object="Lisbon"))
    requirement = EvidenceRequirement(kind="path", subject="alice chen", predicate="works_at", object="Lisbon")
    decision = await StructuredEvidenceAssessor("question", (requirement,)).assess("question", candidates)
    assert decision.status == "sufficient" and decision.selected_ids == ("fact:1", "fact:2")


@pytest.mark.parametrize(("value", "keys"), [
    ("Acme Robotics", ("acme robotics", "Acme Robotics")),
    ("acme robotics", ("acme robotics",)),
    ("MB", ("MB",)),
    ("It rained. We stayed in", ()),
    ("it", ()),
    ("   ", ()),
])
def test_traversal_looks_up_only_the_spellings_the_rule_allows(value, keys):
    from scone_memory.memory.identity import lookup_keys
    assert lookup_keys(value) == keys
