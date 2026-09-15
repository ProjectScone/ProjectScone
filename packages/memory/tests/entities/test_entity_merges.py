"""Two names a person decided are one thing, honoured by the graph.

The identity rule joins names only when their keys are equal. A merge is
the recorded decision that two different keys name one entity: a ledger
claim under a reserved predicate, so it has a valid time, can be closed,
and is read by every view that reads the ledger. The projection rewrites
the alias to the entity it was merged into before anything is counted, so
relations, surface forms, kinds and every analysis run over the merged
graph -- and a closed decision leaves the two names apart again.
"""
from __future__ import annotations

from datetime import datetime, timezone

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import analyze_projection
from scone_memory.entities.ids import key_id
from scone_memory.entities.merges import SAME_ENTITY
from scone_memory.entities.project import project_entities

DAY = "2025-01-01T00:00:00Z"


def fact(number: int, subject: str, predicate: str, object_: str, **fields: object) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from=str(fields.pop("valid_from", DAY)), **fields)


def merge(number: int, alias: str, into: str, **fields: object) -> Fact:
    return fact(number, alias, SAME_ENTITY, into, **fields)


def ids(*keys: str) -> set[str]:
    return {key_id("alpha", key) for key in keys}


def chen() -> list[Fact]:
    return [fact(1, "Dr. Alice Chen", "works_at", "Acme Robotics"),
            fact(2, "Alice Chen", "leads", "Robotics Lab"),
            fact(3, "Bob Stone", "reports_to", "Dr. Alice Chen")]


def test_a_merge_redirects_every_relation_to_the_entity_it_names():
    projection = project_entities("alpha", [*chen(), merge(4, "Dr. Alice Chen", "Alice Chen")], revision=1)
    keys = {entity.key for entity in projection.entities}
    assert "dr. alice chen" not in keys and "alice chen" in keys
    chen_id = key_id("alpha", "alice chen")
    by_predicate = {relation.predicate: relation for relation in projection.relations}
    assert by_predicate["works_at"].subject_id == chen_id
    assert by_predicate["leads"].subject_id == chen_id
    assert by_predicate["reports_to"].object_id == chen_id
    [entity] = [entity for entity in projection.entities if entity.key == "alice chen"]
    assert entity.as_subject == 2 and entity.as_object == 1
    assert {form.text for form in entity.surface_forms} >= {"dr. alice chen", "Dr. Alice Chen", "alice chen"}


def test_the_decision_itself_is_never_a_relation_role_or_attribute():
    projection = project_entities("alpha", [*chen(), merge(4, "Dr. Alice Chen", "Alice Chen")], revision=1)
    assert 4 not in {role.fact_id for role in projection.roles}
    assert all(SAME_ENTITY != item.predicate for item in (*projection.relations, *projection.attributes))
    [decision] = projection.merges
    assert (decision.fact_id, decision.alias_key, decision.into_key, decision.outcome) == (
        4, "dr. alice chen", "alice chen", "applied")
    assert decision.alias_id == key_id("alpha", "dr. alice chen")


def test_a_chain_of_merges_ends_at_the_last_name():
    rows = [fact(1, "A. Chen", "knows", "Bob Stone"), fact(2, "Dr. Alice Chen", "knows", "Cara Ruiz"),
            fact(3, "Alice Chen", "knows", "Dan Wu"),
            merge(4, "A. Chen", "Dr. Alice Chen"), merge(5, "Dr. Alice Chen", "Alice Chen")]
    projection = project_entities("alpha", rows, revision=1)
    assert {relation.subject_id for relation in projection.relations} == ids("alice chen")
    assert [(item.alias_key, item.into_key) for item in projection.merges] == [
        ("a. chen", "alice chen"), ("dr. alice chen", "alice chen")]


def test_a_merge_that_would_close_a_cycle_is_refused_and_says_so():
    rows = [fact(1, "Alice Chen", "knows", "Bob Stone"), fact(2, "Dr. Alice Chen", "knows", "Cara Ruiz"),
            merge(3, "Dr. Alice Chen", "Alice Chen"), merge(4, "Alice Chen", "Dr. Alice Chen")]
    projection = project_entities("alpha", rows, revision=1)
    assert {relation.subject_id for relation in projection.relations} == ids("alice chen")
    assert [(item.fact_id, item.outcome) for item in projection.merges] == [(3, "applied"), (4, "cycle")]


def test_an_unconfirmed_or_excluded_decision_merges_nothing_and_leaves_the_digest_alone():
    plain = project_entities("alpha", chen(), revision=1)
    for decision in (merge(4, "Dr. Alice Chen", "Alice Chen", status="proposed"),
                     merge(4, "Dr. Alice Chen", "Alice Chen", excluded_reason="wrong person")):
        projection = project_entities("alpha", [*chen(), decision], revision=1)
        assert projection.merges == ()
        assert {entity.key for entity in projection.entities} == {entity.key for entity in plain.entities}
        assert projection.digest == plain.digest


def test_a_closed_decision_parts_the_names_now_and_still_joins_them_while_it_held():
    closed = merge(4, "Dr. Alice Chen", "Alice Chen", status="closed", valid_until="2025-03-01T00:00:00Z",
                   closed_reason="two people")
    now = project_entities("alpha", [*chen(), closed], revision=1)
    assert "dr. alice chen" in {entity.key for entity in now.entities} and now.merges == ()
    then = project_entities("alpha", [*chen(), closed], revision=1,
                            merges_at=datetime(2025, 2, 1, tzinfo=timezone.utc))
    assert "dr. alice chen" not in {entity.key for entity in then.entities}
    after = project_entities("alpha", [*chen(), closed], revision=1,
                             merges_at=datetime(2025, 3, 1, tzinfo=timezone.utc))
    assert "dr. alice chen" in {entity.key for entity in after.entities}


def test_a_decision_joins_a_value_the_identity_rule_would_never_join():
    rows = [fact(1, "disk", "unit", "MB"), fact(2, "megabyte", "equals", "1000000 bytes")]
    assert all(role.object_id is None for role in project_entities("alpha", rows, revision=1).roles if role.fact_id == 1)
    projection = project_entities("alpha", [*rows, merge(3, "MB", "megabyte")], revision=1)
    [role] = [role for role in projection.roles if role.fact_id == 1]
    assert role.object_id == key_id("alpha", "megabyte")
    assert role.classification.basis == "identity_decision"


def test_the_label_keeps_the_spelling_of_the_name_merged_into():
    rows = [fact(n, "Dr. Alice Chen", "knows", f"Person {n}", valid_from=DAY) for n in range(1, 4)]
    rows += [fact(9, "Bob Stone", "knows", "Alice Chen"), merge(10, "Dr. Alice Chen", "Alice Chen")]
    [entity] = [entity for entity in project_entities("alpha", rows, revision=1).entities if entity.key == "alice chen"]
    assert entity.label == "Alice Chen"


def test_rank_and_communities_are_worked_out_over_the_merged_graph():
    rows = [fact(1, "Ana", "knows", "Ben"), fact(2, "Ben", "knows", "Cho"), fact(3, "Dev", "knows", "Eli"),
            fact(4, "Eli", "knows", "Fay"), fact(5, "Cho", "knows", "C. Ho")]
    apart = analyze_projection(project_entities("alpha", rows, revision=1))
    merged_projection = project_entities("alpha", [*rows, merge(6, "Dev", "C. Ho")], revision=1)
    merged = analyze_projection(merged_projection)
    assert len(merged.importance) == len(apart.importance) - 1
    c_ho = key_id("alpha", "c. ho")
    before = next(item for item in apart.importance if item.entity_id == c_ho)
    after = next(item for item in merged.importance if item.entity_id == c_ho)
    assert after.degree == before.degree + 1 and after.pagerank > before.pagerank
    assert len(merged_projection.components()) == 1


def test_a_decision_between_two_spellings_of_one_key_is_reported_not_applied():
    rows = [fact(1, "Alice Chen", "knows", "Bob Stone"), merge(2, "Alice  CHEN", "alice chen")]
    projection = project_entities("alpha", rows, revision=1)
    assert [(item.fact_id, item.outcome) for item in projection.merges] == [(2, "same_name")]
    assert "alice chen" in {entity.key for entity in projection.entities}


def test_the_spelling_a_quote_gives_an_alias_is_kept_among_the_names():
    rows = [fact(1, "dr. alice chen", "leads", "Robotics Lab", source_episode_id=7,
                 quote="Dr. Alice CHEN leads the Robotics Lab"),
            fact(2, "alice chen", "works_at", "Acme Robotics"), merge(3, "dr. alice chen", "alice chen")]
    [entity] = [entity for entity in project_entities("alpha", rows, revision=1).entities if entity.key == "alice chen"]
    assert "Dr. Alice CHEN" in {form.text for form in entity.surface_forms}


def test_a_merged_name_or_its_old_id_resolves_to_the_entity_it_was_merged_into():
    from scone_memory.entities.query import resolve

    rows = [*chen(), fact(4, "ally chen", "knows", "Cara Ruiz"), merge(5, "Dr. Alice Chen", "Alice Chen"),
            merge(6, "Ally Chen", "Alice Chen"), merge(7, "Alice Chen", "Ally Chen")]
    projection = project_entities("alpha", rows, revision=1)
    assert [item.outcome for item in projection.merges] == ["applied", "applied", "cycle"]
    chen_id = key_id("alpha", "alice chen")
    for name, tier in (("ALLY  CHEN", "key"), (key_id("alpha", "dr. alice chen"), "id"), ("alice chen", "key")):
        found = resolve(projection, name)
        assert (found.status, found.tier, [candidate.entity_id for candidate in found.candidates]) == (
            "resolved", tier, [chen_id]), name
