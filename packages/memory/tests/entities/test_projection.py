"""The entity projection: things, how they relate, and what is known about them.

It is a pure function of fact rows. These tests pin what the audit found
missing: nodes are things rather than records, names recover their casing
from the quote, values stay values, and a chain of claims about related
things comes out connected.
"""
from __future__ import annotations

import random

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.project import project_entities

QUOTES = {
    "alice": "Dr. Alice Chen joined Acme Robotics in May 2021.",
    "acme": "Acme Robotics is based in Lisbon.",
    "lisbon": "Lisbon hosts the Web Summit every November.",
    "bob": "However, Bob feels tired on Monday. Will Alice call him?",
}


def fact(number: int, subject: str, predicate: str, object_: str, *, quote: str | None = None,
         status: str = "active", excluded: bool = False, space: str = "alpha", origin: str = "extracted",
         valid_from: str = "2025-01-01T00:00:00Z") -> Fact:
    return Fact(fact_id=number, space=space, subject=subject, predicate=predicate, object=object_,
                valid_from=valid_from, status=status, origin=origin, quote=quote,
                source_episode_id=1 if quote else None, excluded_reason="hidden" if excluded else None)


LEDGER = [
    fact(1, "alice chen", "works_at", "Acme Robotics", quote=QUOTES["alice"]),
    fact(2, "alice chen", "joined_on", "May 2021", quote=QUOTES["alice"]),
    fact(3, "acme robotics", "based_in", "Lisbon", quote=QUOTES["acme"]),
    fact(4, "lisbon", "hosts", "the Web Summit", quote=QUOTES["lisbon"]),
    fact(5, "bob", "feels", "tired", quote=QUOTES["bob"]),
    fact(6, "bob", "meets_on", "Monday", quote=QUOTES["bob"]),
    fact(7, "alice chen", "knows", "Bob", quote=QUOTES["bob"]),
    fact(8, "report", "size", "3 MB", origin="stated"),
    fact(9, "report", "size", "3 mb", origin="stated"),
]


def by_key(projection):
    return {entity.key: entity for entity in projection.entities}


def test_nodes_are_the_things_claims_are_about() -> None:
    entities = by_key(project_entities("alpha", LEDGER, revision=1))
    assert set(entities) == {"alice chen", "acme robotics", "lisbon", "the web summit", "bob", "report"}
    for noise in ("however", "monday", "will", "after", "alice", "chen", "tired", "may 2021", "3 mb"):
        assert noise not in entities


def test_labels_recover_the_casing_people_wrote() -> None:
    entities = by_key(project_entities("alpha", LEDGER, revision=1))
    assert entities["alice chen"].label == "Alice Chen"
    assert entities["acme robotics"].label == "Acme Robotics"
    assert entities["the web summit"].label == "the Web Summit"
    assert entities["bob"].label == "Bob"


def test_a_chain_of_claims_about_related_things_is_connected() -> None:
    projection = project_entities("alpha", LEDGER, revision=1)
    ids = {entity.key: entity.entity_id for entity in projection.entities}
    edges = {(relation.subject_id, relation.predicate, relation.object_id) for relation in projection.relations}
    assert (ids["alice chen"], "works_at", ids["acme robotics"]) in edges
    assert (ids["acme robotics"], "based_in", ids["lisbon"]) in edges
    assert (ids["lisbon"], "hosts", ids["the web summit"]) in edges
    assert (ids["alice chen"], "knows", ids["bob"]) in edges
    components = projection.components()
    assert any({ids["alice chen"], ids["acme robotics"], ids["lisbon"], ids["the web summit"], ids["bob"]} <= c
               for c in components)


def test_values_are_attributes_and_keep_their_exact_text() -> None:
    projection = project_entities("alpha", LEDGER, revision=1)
    ids = {entity.key: entity.entity_id for entity in projection.entities}
    attributes = {(a.entity_id, a.predicate, a.value): a for a in projection.attributes}
    assert (ids["alice chen"], "joined_on", "May 2021") in attributes
    assert attributes[(ids["bob"], "meets_on", "Monday")].literal_kind == "date"
    assert (ids["bob"], "feels", "tired") in attributes
    assert (ids["report"], "size", "3 MB") in attributes and (ids["report"], "size", "3 mb") in attributes


def test_every_relation_and_attribute_cites_the_facts_behind_it() -> None:
    projection = project_entities("alpha", LEDGER, revision=1)
    cited = sorted(fid for item in (*projection.relations, *projection.attributes) for fid in item.fact_ids)
    assert cited == [fact.fact_id for fact in LEDGER]


def test_kinds_are_inferred_from_predicates_and_name_their_evidence() -> None:
    entities = by_key(project_entities("alpha", LEDGER, revision=1))
    assert (entities["alice chen"].kind, entities["alice chen"].kind_status) == ("person", "inferred")
    assert (entities["acme robotics"].kind, entities["acme robotics"].kind_status) == ("organisation", "inferred")
    assert (entities["lisbon"].kind, entities["lisbon"].kind_status) == ("place", "inferred")
    assert 1 in entities["alice chen"].kind_basis


def test_declined_facts_never_count_and_other_statuses_are_counted_apart() -> None:
    ledger = [*LEDGER, fact(10, "alice chen", "works_at", "Globex", status="declined"),
              fact(11, "alice chen", "works_at", "Initech", status="proposed"),
              fact(12, "acme robotics", "based_in", "Lisbon", excluded=True, quote=QUOTES["acme"])]
    projection = project_entities("alpha", ledger, revision=1)
    assert "globex" not in by_key(projection)
    ids = {entity.key: entity.entity_id for entity in projection.entities}
    relation = next(r for r in projection.relations
                    if (r.subject_id, r.object_id) == (ids["acme robotics"], ids["lisbon"]))
    assert relation.fact_ids == (3, 12) and relation.support.excluded == 1 and relation.support.active == 2
    proposed = next(r for r in projection.relations if r.object_id == ids["initech"])
    assert proposed.support.proposed == 1


def test_the_same_rows_in_any_order_give_the_same_projection() -> None:
    first = project_entities("alpha", LEDGER, revision=1)
    for seed in range(20):
        shuffled = list(LEDGER)
        random.Random(seed).shuffle(shuffled)
        assert project_entities("alpha", shuffled, revision=1).digest == first.digest


def test_timestamp_spelling_and_confidence_do_not_change_the_digest() -> None:
    first = project_entities("alpha", LEDGER, revision=1)
    respelled = [f.model_copy(update={"valid_from": "2025-01-01T00:00:00+00:00", "confidence": 0.5}) for f in LEDGER]
    assert project_entities("alpha", respelled, revision=1).digest == first.digest


def test_ids_belong_to_their_space() -> None:
    alpha = by_key(project_entities("alpha", LEDGER, revision=1))
    beta = by_key(project_entities("beta", [f.model_copy(update={"space": "beta"}) for f in LEDGER], revision=1))
    assert alpha["lisbon"].entity_id != beta["lisbon"].entity_id
    assert alpha["lisbon"].entity_id == project_entities("alpha", LEDGER[::-1], revision=7).entities[
        [e.key for e in project_entities("alpha", LEDGER[::-1], revision=7).entities].index("lisbon")].entity_id


def test_a_row_from_another_space_is_refused() -> None:
    with pytest.raises(ValueError, match="space"):
        project_entities("alpha", [*LEDGER, fact(99, "x", "knows", "Y", space="beta")], revision=1)


def test_a_lowercase_object_joins_the_claims_made_about_it() -> None:
    # 'report' reads like a plain value, but another claim is about it,
    # so the two claims meet at one entity.
    projection = project_entities("alpha", [*LEDGER, fact(10, "alice chen", "reviewed", "report")], revision=1)
    ids = {entity.key: entity.entity_id for entity in projection.entities}
    edge = next(r for r in projection.relations if r.object_id == ids["report"])
    assert edge.subject_id == ids["alice chen"] and edge.fact_ids == (10,)
    role = next(role for role in projection.roles if role.fact_id == 10)
    assert role.classification.basis == "subject_anchor"


def test_a_value_meets_a_folded_subject_only_when_case_cannot_matter() -> None:
    # 'mb' as a subject was written 'MB' (megabytes); 'mb' as an object is a
    # unit that may be millibits. Folding lost the difference, so they stay apart.
    rows = [fact(1, "mb", "means", "megabytes", quote="MB means megabytes."), fact(2, "transfer", "unit", "mb")]
    projection = project_entities("alpha", rows, revision=1)
    role = next(role for role in projection.roles if role.fact_id == 2)
    assert (role.object_id, role.classification.detail) == (None, "case_sensitive")
    assert any(a.predicate == "unit" and a.value == "mb" for a in projection.attributes)


def test_code_entities_are_kinded_by_their_shape() -> None:
    from scone_memory.entities.kinds import code_kind, is_file_name

    rows = [fact(1, "pkg/a.py", "defines", "pkg/a.py:Thing"), fact(2, "pkg/a.py", "imports", "typing"),
            fact(3, "pkg/a.py", "imports", "pkg/b.py"), fact(4, "pkg/a.py:Thing", "calls", "pkg/b.py:run"),
            fact(5, "pkg/a.py", "imports", "requests"), fact(6, "pyproject.toml", "depends_on", "requests"),
            fact(7, "pkg/a.py:Thing", "cites", "ADR-12"), fact(8, "web/app.ts", "imports", "github.com/gorilla/mux"),
            fact(9, "alice chen", "works_at", "Acme")]
    entities = by_key(project_entities("alpha", rows, revision=1))
    kinds = {key: (entity.kind, entity.kind_status) for key, entity in entities.items()}
    assert kinds["pkg/a.py"] == ("file", "inferred") and kinds["pkg/b.py"] == ("file", "inferred"), "a path is a file, imported or importing"
    assert kinds["pkg/a.py:thing"] == ("declaration", "inferred") and kinds["pkg/b.py:run"] == ("declaration", "inferred")
    assert kinds["typing"] == ("module", "inferred") and kinds["github.com/gorilla/mux"] == ("module", "inferred"), "imported, no suffix: a module"
    assert kinds["requests"] == ("product", "inferred"), "a module a manifest also depends on is the package it comes from"
    assert kinds["pyproject.toml"] == ("file", "inferred")
    assert code_kind("ADR-12", "cites", "object") is None and code_kind("README.md: Getting Started", "describes", "subject") is None, \
        "a citation and a sentence about a file are neither files nor declarations"
    assert code_kind("notes.txt: don't forget", "says", "subject") is None and code_kind("pkg/a.py:Store.keep", "defines", "object") == "declaration"
    assert kinds["alice chen"] == ("person", "inferred") and kinds["acme"] == ("organisation", "inferred"), "prose is hinted as before"
    assert set(entities["pkg/a.py"].kind_basis) >= {1, 2, 3, 5}, "a shape's hint names the facts it was read from"
    assert not is_file_name("asyncio.run") and not is_file_name("1/2") and not is_file_name("12:30") and not is_file_name("https://x/y.md")
    assert not is_file_name("a b.py") and is_file_name("README.md") and is_file_name("src/x.ts") and not is_file_name("Makefile")
    assert code_kind("12:30", "starts_at", "object") is None and code_kind("asyncio.run", "calls", "object") is None
    assert code_kind("os", "imports", "object") == "module" and code_kind("os", "calls", "object") is None


def test_a_kind_conflict_shows_evidence_for_every_side() -> None:
    # Eight claims make acme a person (it 'works at' offices); one makes it an
    # organisation. The bounded evidence must still name the dissenting claim.
    rows = [fact(number, "acme", "works_at", f"Office {number}") for number in range(1, 9)]
    rows.append(fact(9, "dana", "works_at", "Acme"))
    acme = by_key(project_entities("alpha", rows, revision=1))["acme"]
    assert acme.kind_status == "conflict" and acme.kind is None
    assert 9 in acme.kind_basis and 1 in acme.kind_basis and len(acme.kind_basis) <= 8


def test_the_projection_digest_of_a_fixed_ledger_does_not_drift():
    """Ids, cursors and caches rest on the digest. A change to how it is
    computed must be a deliberate version change, never a side effect."""
    from scone_memory.core.models import Fact

    facts = [Fact(fact_id=1, space="alpha", subject="alice chen", predicate="works_at", object="Acme Robotics",
                  valid_from="2024-01-01T00:00:00Z", source_episode_id=1, quote="Alice Chen joined Acme Robotics"),
             Fact(fact_id=2, space="alpha", subject="acme robotics", predicate="based_in", object="Lisbon",
                  valid_from="2024-02-01T00:00:00Z"),
             Fact(fact_id=3, space="alpha", subject="alice chen", predicate="joined_on", object="May 2021",
                  valid_from="2024-01-01T00:00:00Z", status="closed", valid_until="2025-01-01T00:00:00Z")]
    # kinds/2 (code entities kinded by their shape) changed the digest on purpose.
    assert project_entities("alpha", facts, revision=1).digest == \
        "f5548d5261767c5db5cd2a1b493e9e1d26ce9751e2b9a7d3702a1632ea6ea9f1"
