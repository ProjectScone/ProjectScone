from __future__ import annotations

import json

import pytest

from scone_memory.retrieval.adaptive import EvidenceCandidate


def fact(number: int, subject: str | None, object_: str | None, *, text: str = "retained quote",
         episode: int = 1) -> EvidenceCandidate:
    return EvidenceCandidate(id=f"fact:{number}", episode_id=episode, text=text,
                             subject=subject, predicate="relates to", object=object_)


def test_branching_component_preserves_every_record_and_directed_join() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    candidates = (fact(1, "a", "b"), fact(2, "b", "c"), fact(3, "b", "d"))
    grouping = build_evidence_groups(candidates)
    assert len(grouping.records) == 1
    group = grouping.records[0]
    assert group["kind"] == "fact_component"
    assert group["members"] == [candidate.model_dump(mode="json") for candidate in candidates]
    assert group["joins"] == [
        {"kind": "object_subject", "match": "literal", "from_id": "fact:1", "to_id": "fact:2"},
        {"kind": "object_subject", "match": "literal", "from_id": "fact:1", "to_id": "fact:3"},
    ]
    assert grouping.members == {group["id"]: ("fact:1", "fact:2", "fact:3")}
    assert grouping.groups == grouping.members


def test_cycles_keep_all_edges_and_exclude_self_edges() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    grouping = build_evidence_groups((fact(1, "a", "b"), fact(2, "b", "a"), fact(3, "b", "b")))
    assert grouping.records[0]["joins"] == [
        {"kind": "object_subject", "match": "literal", "from_id": "fact:1", "to_id": "fact:2"},
        {"kind": "object_subject", "match": "literal", "from_id": "fact:1", "to_id": "fact:3"},
        {"kind": "object_subject", "match": "literal", "from_id": "fact:2", "to_id": "fact:1"},
        {"kind": "object_subject", "match": "literal", "from_id": "fact:3", "to_id": "fact:2"},
    ]
    isolated = build_evidence_groups((fact(4, "same", "same"),))
    assert isolated.groups == {}
    assert isolated.members == {"fact:4": ("fact:4",)}


def test_a_join_that_needed_case_or_spacing_folded_says_so() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    grouping = build_evidence_groups((fact(1, "alice chen", "Acme  Robotics"), fact(2, "acme robotics", "lisbon"),
                                      fact(3, "lisbon", "portugal")))
    assert grouping.records[0]["joins"] == [
        {"kind": "object_subject", "match": "normalised", "from_id": "fact:1", "to_id": "fact:2"},
        {"kind": "object_subject", "match": "literal", "from_id": "fact:2", "to_id": "fact:3"},
    ]


def test_components_follow_input_rank_without_losing_or_repeating_evidence() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    chunk = EvidenceCandidate(id="chunk:1", episode_id=8, text="original chunk", subject="b", object="c")
    candidates = (fact(4, "x", "y"), chunk, fact(1, "a", "b"), fact(5, "y", "z"), fact(2, "b", "c"),
                  fact(9, "isolated", "unknown"))
    grouping = build_evidence_groups(candidates)
    assert list(grouping.members.values()) == [("fact:4", "fact:5"), ("chunk:1",), ("fact:1", "fact:2"), ("fact:9",)]
    assert grouping.records[1] == {**chunk.model_dump(mode="json"), "kind": "standalone"}
    represented = [member for members in grouping.members.values() for member in members]
    assert len(represented) == len(set(represented)) == len(candidates)
    assert set(represented) == {candidate.id for candidate in candidates}
    assert len(grouping.groups) == 2


# Case and spacing are deliberately absent from this list. Subjects are
# stored normalised, so the ledger never holds a subject "Entity", only
# "entity"; an object written "Entity" naming it is the ordinary case, and
# refusing that join is what left the evidence graph in disconnected pieces.
# See test_entity_identity.py. Precomposed and decomposed accents stay apart
# because Unicode canonical equivalence is a separate decision: it would
# change how subjects are stored, which needs its own migration.
@pytest.mark.parametrize("left,right", [(None, None), ("", ""), ("  ", "  "), ("é", "e\u0301")])
def test_blank_or_nonidentical_literals_never_join(left: str | None, right: str | None) -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    grouping = build_evidence_groups((fact(1, "start", left), fact(2, right, "end")))
    assert grouping.groups == {}
    assert tuple(grouping.members) == ("fact:1", "fact:2")


def test_group_hash_is_canonical_and_changes_with_source_or_text() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    first, second = fact(1, "a", "b"), fact(2, "b", "c")
    original = build_evidence_groups((first, second))
    group_id = next(iter(original.groups))
    assert group_id.startswith("group:") and len(group_id) == 70
    assert next(iter(build_evidence_groups((second, first)).groups)) == group_id
    assert next(iter(build_evidence_groups((first, fact(2, "b", "c", episode=2))).groups)) != group_id
    assert next(iter(build_evidence_groups((first, fact(2, "b", "c", text="changed quote"))).groups)) != group_id
    assert original == build_evidence_groups((first, second))


def test_empty_input_and_hundred_isolated_records_are_complete() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    assert build_evidence_groups(()).records == ()
    candidates = tuple(fact(number, f"subject-{number}", "unmatched") for number in range(1, 101))
    result = build_evidence_groups(candidates)
    assert len(result.records) == len(result.members) == 100
    assert result.groups == {}


@pytest.mark.parametrize("bad_input", [
    [fact(1, "a", "b")],
    (fact(1, "a", "b"), fact(1, "b", "c")),
    (fact(1, "a", "b"), fact(1, "a", "b")),
    (fact(1, "a", "b").model_copy(update={"id": "secret-invalid-id"}),),
    (fact(1, "a", "b").model_copy(update={"episode_id": True}),),
    tuple(fact(number, None, None) for number in range(1, 102)),
])
def test_invalid_or_duplicate_inputs_fail_without_exposing_records(bad_input: object) -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    with pytest.raises(ValueError, match="^invalid evidence grouping input$"):
        build_evidence_groups(bad_input)  # type: ignore[arg-type]


def test_dense_component_fails_at_edge_cap_without_partial_output() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    allowed = tuple(fact(number, "shared", "shared") for number in range(1, 36))
    assert len(build_evidence_groups(allowed).records[0]["joins"]) == 1190
    with pytest.raises(ValueError, match="^evidence grouping exceeds directed edge limit$"):
        build_evidence_groups(tuple(fact(number, "shared", "shared") for number in range(1, 47)))


def test_unicode_budget_counts_utf8_bytes_and_never_clips() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    candidate = fact(1, None, None, text="é" * 60000)
    grouping = build_evidence_groups((candidate,))
    assert grouping.records[0]["text"] == candidate.text
    assert len(json.dumps(grouping.records, ensure_ascii=False, separators=(",", ":")).encode()) <= 128000
    with pytest.raises(ValueError, match="^evidence grouping exceeds serialized byte limit$"):
        build_evidence_groups((fact(1, None, None, text="é" * 64000),))


@pytest.mark.parametrize("budget", [True, 1, 128001, 2.5, "16000"])
def test_invalid_byte_budget_is_rejected(budget: object) -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    with pytest.raises(ValueError, match="^max_bytes must be an integer in 2..128000$"):
        build_evidence_groups((), max_bytes=budget)  # type: ignore[arg-type]


def test_byte_budget_includes_group_metadata_and_accepts_exact_boundary() -> None:
    from scone_memory.retrieval.evidence_groups import build_evidence_groups
    candidates = (fact(1, "a", "b"), fact(2, "b", "c"))
    records = build_evidence_groups(candidates).records
    size = len(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode())
    assert build_evidence_groups(candidates, max_bytes=size).records == records
    with pytest.raises(ValueError, match="^evidence grouping exceeds serialized byte limit$"):
        build_evidence_groups(candidates, max_bytes=size - 1)
    assert build_evidence_groups((), max_bytes=2).records == ()
