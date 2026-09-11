"""Which fact objects name things, and which are values.

A graph whose nodes are every object string is noise: 'tired', '3 MB' and
'Monday' become things, and a unit whose case means something ('MB' against
'mb') joins what it should not. The classifier decides conservatively, and
every decision carries the rule that made it.
"""
from __future__ import annotations

import pytest

from scone_memory.entities.classify import (
    CLASSIFIER_VERSION, ClassificationContext, classify_object, is_case_safe, join_block_reason, reference_flag,
)

PLAIN = ClassificationContext()


@pytest.mark.parametrize(("text", "predicate", "expected"), [
    ("Acme Robotics", "works_at", ("entity", None, "name_shape")),
    ("Lisbon", "moved_to", ("entity", None, "name_shape")),
    ("NASA", "works_at", ("entity", None, "name_shape")),
    ("GPT-4", "uses", ("entity", None, "name_shape")),
    ("Python 3.12", "uses", ("entity", None, "name_shape")),
    ("里斯本", "lives_in", ("entity", None, "uncased_name")),
    ("lisbon", "lives_in", ("entity", None, "entity_predicate")),
    ("the Web Summit", "attended", ("entity", None, "determiner_name")),
    ("The Beatles", "likes", ("entity", None, "determiner_name")),
    ("the Acme lab", "visited", ("literal", "text", "description")),
    ("3.2 GB", "size", ("literal", "quantity", "quantity_shape")),
    ("MB", "unit", ("literal", "quantity", "quantity_shape")),
    ("mb", "unit", ("literal", "quantity", "quantity_shape")),
    ("12%", "share", ("literal", "quantity", "quantity_shape")),
    ("$5", "price", ("literal", "quantity", "quantity_shape")),
    ("42", "answer", ("literal", "quantity", "quantity_shape")),
    ("2024-05-01", "started_on", ("literal", "date", "date_shape")),
    ("2024-05-01T10:00:00Z", "started_at", ("literal", "date", "date_shape")),
    ("May 2021", "joined", ("literal", "date", "date_shape")),
    ("Monday", "meets_on", ("literal", "date", "date_shape")),
    ("1998", "born_in", ("literal", "date", "date_shape")),
    ("Q3 2024", "ships_in", ("literal", "date", "date_shape")),
    ("yesterday", "arrived", ("literal", "date", "date_shape")),
    ("v2.3.1", "runs", ("literal", "identifier", "identifier_shape")),
    ("alice@example.com", "contact", ("literal", "identifier", "identifier_shape")),
    ("https://example.com/docs", "reads", ("literal", "identifier", "identifier_shape")),
    ("/var/lib/scone", "stores_in", ("literal", "identifier", "identifier_shape")),
    ("550e8400-e29b-41d4-a716-446655440000", "id", ("literal", "identifier", "identifier_shape")),
    ("3f2a9c1d", "commit", ("literal", "identifier", "identifier_shape")),
    ("INV-2024-0042", "invoice", ("literal", "identifier", "identifier_shape")),
    ("#431", "closes", ("literal", "identifier", "identifier_shape")),
    ("true", "enabled", ("literal", "value", "value_shape")),
    ("tired", "feels", ("literal", "value", "common_value")),
    ("blue", "favourite_colour", ("literal", "value", "literal_predicate")),
    ("CTO", "role", ("literal", "value", "literal_predicate")),
    ('"move fast and fix things"', "motto", ("literal", "text", "quoted_text")),
    ("the team agreed to ship the new onboarding flow after the design review next week",
     "decided", ("literal", "text", "prose")),
    ("It rained. We stayed in", "noted", ("literal", "text", "prose")),
    ("it", "likes", ("literal", "pronoun", "pronoun")),
    ("someone", "met", ("literal", "pronoun", "pronoun")),
    ("I", "knows", ("entity", None, "name_shape")),
    ("Dr. Alice Chen", "knows", ("entity", None, "name_shape")),
    ("COVID-19", "had", ("literal", "identifier", "identifier_shape")),
])
def test_objects_are_classified_by_the_first_rule_that_fits(text: str, predicate: str,
                                                            expected: tuple[str, str | None, str]) -> None:
    result = classify_object(text, predicate, PLAIN)
    assert (result.object_class, result.literal_kind, result.basis) == expected


def test_a_value_named_as_a_subject_joins_when_its_case_cannot_matter() -> None:
    anchored = ClassificationContext(anchors=frozenset({"2024-05-01"}))
    result = classify_object("2024-05-01", "started_on", anchored)
    assert (result.object_class, result.basis) == ("entity", "subject_anchor")


def test_a_value_whose_case_can_matter_never_joins_through_folding() -> None:
    anchored = ClassificationContext(anchors=frozenset({"mb"}))
    result = classify_object("MB", "unit", anchored)
    assert (result.object_class, result.literal_kind, result.basis, result.detail) == \
        ("literal", "quantity", "quantity_shape", "case_sensitive")


def test_any_name_a_subject_carries_is_an_entity() -> None:
    anchored = ClassificationContext(anchors=frozenset({"tired"}))
    assert classify_object("tired", "feels", anchored).basis == "subject_anchor"


def test_a_decision_overrides_every_rule_and_cites_itself() -> None:
    decided = ClassificationContext(overrides={"works_at": ("literal", 812)})
    result = classify_object("Acme", "works_at", decided)
    assert (result.object_class, result.basis, result.detail) == ("literal", "decision", "812")


def test_a_key_an_identity_decision_names_is_an_entity() -> None:
    named = ClassificationContext(identity_keys=frozenset({"3 mb"}))
    assert classify_object("3 MB", "size", named).basis == "identity_decision"


def test_a_lowercase_phrase_becomes_a_concept_only_when_two_subjects_share_it() -> None:
    assert classify_object("machine learning", "interested_in", PLAIN).basis == "common_value"
    shared = ClassificationContext(shared_objects=frozenset({"machine learning"}))
    assert classify_object("machine learning", "interested_in", shared).basis == "shared_object"


@pytest.mark.parametrize(("text", "safe"), [("2024-05-01", True), ("里斯本", True), ("acme robotics", True),
                                              ("Acme", False), ("MB", False), ("3 MB", False)])
def test_case_safety(text: str, safe: bool) -> None:
    assert is_case_safe(text) is safe


@pytest.mark.parametrize(("text", "blocked"), [
    ("Acme Robotics", None), ("2024-05-01", None), ("MB", "quantity_shape"), ("v2.3.1", None),
    ("V2.3.1-RC", "identifier_shape"), ("it", "pronoun"), ('"a quote"', "quoted_text"),
    ("It rained. We stayed in", "prose"),
])
def test_join_blocks_follow_the_classifier(text: str, blocked: str | None) -> None:
    assert join_block_reason(text) == blocked


@pytest.mark.parametrize(("key", "flag"), [("i", "speaker_reference"), ("we", "speaker_reference"),
                                           ("you", "speaker_reference"), ("she", "unresolved_reference"),
                                           ("acme", None)])
def test_references_are_flagged(key: str, flag: str | None) -> None:
    assert reference_flag(key) == flag


def test_the_rules_are_versioned_and_pure() -> None:
    assert CLASSIFIER_VERSION == "objects/1"
    assert classify_object("Acme", "works_at", PLAIN) == classify_object("Acme", "works_at", PLAIN)
