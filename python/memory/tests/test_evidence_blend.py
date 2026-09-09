"""Pure fusion preserves exact payloads, provenance, and atomic membership."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from scone_memory.retrieval.adaptive import EvidenceCandidate, evidence_payload_bytes
from scone_memory.retrieval.evidence_blend import blend_evidence


def candidate(number: int, text: str = "quote") -> EvidenceCandidate:
    return EvidenceCandidate(id=f"fact:{number}", episode_id=number, text=text)


def test_overlap_outranks_single_lane_and_tracks_delivered_provenance() -> None:
    a, b, c = candidate(1), candidate(2), candidate(3)
    result = blend_evidence((a, b), (c, b), max_candidates=2, max_bytes=128_000)
    assert result.ids == (b.id, a.id)
    assert result.original_ids == (a.id, b.id)
    assert result.selected_ids == (b.id,)
    assert result.omitted_ids == (c.id,)
    assert result.groups == ()
    assert result.reasons == ("candidate_limit",)


def test_ties_are_stable_original_then_selected_order() -> None:
    a, b, c, d = (candidate(i) for i in range(1, 5))
    for _ in range(5):
        result = blend_evidence((a, b), (c, d), max_candidates=4, max_bytes=128_000)
        assert result.ids == (a.id, c.id, b.id, d.id)
        assert result.omitted_ids == result.reasons == ()


def test_group_score_is_max_member_not_sum() -> None:
    a, b, c, d = (candidate(i) for i in range(1, 5))
    result = blend_evidence((a, b, c, d), (), original_groups=((b.id, c.id, d.id),),
        max_candidates=4, max_bytes=128_000)
    assert result.ids == (a.id, b.id, c.id, d.id)
    assert result.groups == ((b.id, c.id, d.id),)


def test_transitive_overlapping_groups_merge_across_lanes() -> None:
    a, b, c, d = (candidate(i) for i in range(1, 5))
    result = blend_evidence((a, b, c), (b, c, d),
        original_groups=((a.id, b.id), (b.id, c.id)), selected_groups=((c.id, d.id),),
        max_candidates=4, max_bytes=128_000)
    assert result.ids == (a.id, b.id, c.id, d.id)
    assert result.groups == (result.ids,)
    assert result.original_ids == (a.id, b.id, c.id)
    assert result.selected_ids == (b.id, c.id, d.id)


def test_oversized_atomic_unit_is_skipped_but_later_singleton_fits() -> None:
    a, b, c = candidate(1), candidate(2), candidate(3)
    result = blend_evidence((a, b, c), (), original_groups=((a.id, b.id),),
        max_candidates=1, max_bytes=128_000)
    assert result.ids == (c.id,)
    assert result.omitted_ids == (a.id, b.id)
    assert result.groups == ()
    assert result.reasons == ("candidate_limit", "atomic_group_omitted")


def test_unicode_json_budget_is_exact_and_group_atomic() -> None:
    a, b, c = candidate(1, '☃ "é"\\'), candidate(2, "海"), candidate(3, "x")
    payload = evidence_payload_bytes((a, b))
    exact = blend_evidence((a, b), (), original_groups=((a.id, b.id),),
        max_candidates=2, max_bytes=payload)
    assert exact.ids == (a.id, b.id)
    short = blend_evidence((a, b, c), (), original_groups=((a.id, b.id),),
        max_candidates=3, max_bytes=payload - 1)
    assert short.ids == (c.id,)
    assert short.reasons == ("max_evidence_bytes", "atomic_group_omitted")
    assert evidence_payload_bytes((c,)) <= payload - 1


def test_size_accounts_for_existing_candidates_and_json_delimiters() -> None:
    a, b = candidate(1), candidate(2)
    result = blend_evidence((a, b), (), max_candidates=2, max_bytes=evidence_payload_bytes((a, b)) - 1)
    assert result.ids == (a.id,)
    assert result.reasons == ("max_evidence_bytes",)


def test_empty_and_minimum_budget_return_immutable_tuples() -> None:
    result = blend_evidence((), (), max_candidates=1, max_bytes=2)
    assert result.ids == result.original_ids == result.selected_ids == result.omitted_ids == result.reasons == ()
    assert result.groups == ()
    with pytest.raises(FrozenInstanceError):
        setattr(result, "ids", ("fact:1",))
    tiny = blend_evidence((candidate(1),), (), max_candidates=1, max_bytes=2)
    assert tiny.ids == () and tiny.omitted_ids == ("fact:1",)


def test_equal_ids_require_equal_full_payload() -> None:
    a = candidate(1)
    for changed in (candidate(1, "changed"), a.model_copy(update={"subject": "different"})):
        with pytest.raises(ValueError):
            blend_evidence((a,), (changed,), max_candidates=2, max_bytes=128_000)
    duplicate = blend_evidence((a,), (candidate(1),), max_candidates=2, max_bytes=128_000)
    assert duplicate.ids == duplicate.original_ids == duplicate.selected_ids == (a.id,)


@pytest.mark.parametrize("bad", [True, 0, 101, 1.0, "2", None])
def test_strict_candidate_limit(bad: object) -> None:
    with pytest.raises(ValueError):
        blend_evidence((), (), max_candidates=cast(int, bad), max_bytes=128_000)


@pytest.mark.parametrize("bad", [True, 1, 128_001, 2.0, "2", None])
def test_strict_byte_limit(bad: object) -> None:
    with pytest.raises(ValueError):
        blend_evidence((), (), max_candidates=1, max_bytes=cast(int, bad))


@pytest.mark.parametrize("bad", [[], [candidate(1)], (candidate(1), candidate(1)), (object(),),
    tuple(candidate(i) for i in range(1, 102)),
    (EvidenceCandidate.model_construct(id="fact:1", episode_id="1", text="quote"),)])
def test_invalid_candidate_lanes(bad: object) -> None:
    with pytest.raises(ValueError):
        blend_evidence(cast(tuple[EvidenceCandidate, ...], bad), (), max_candidates=1, max_bytes=128_000)
    with pytest.raises(ValueError):
        blend_evidence((), cast(tuple[EvidenceCandidate, ...], bad), max_candidates=1, max_bytes=128_000)


@pytest.mark.parametrize("bad", [[], (("fact:1",),), (("fact:1", "fact:1"),),
    (("fact:1", "fact:3"),), (["fact:1", "fact:2"],), (("fact:1", 2),),
    tuple(("fact:1", "fact:2") for _ in range(101))])
def test_invalid_groups(bad: object) -> None:
    lane = (candidate(1), candidate(2))
    groups = cast(tuple[tuple[str, ...], ...], bad)
    with pytest.raises(ValueError):
        blend_evidence(lane, (), original_groups=groups, max_candidates=2, max_bytes=128_000)
    with pytest.raises(ValueError):
        blend_evidence((), lane, selected_groups=groups, max_candidates=2, max_bytes=128_000)


def test_group_members_must_belong_to_their_own_lane() -> None:
    with pytest.raises(ValueError):
        blend_evidence((candidate(1),), (candidate(2),), original_groups=(("fact:1", "fact:2"),),
            max_candidates=2, max_bytes=128_000)


def test_mutated_candidate_and_extra_fields_fail_strict_revalidation() -> None:
    for field, value in (("text", []), ("unexpected", "value"), ("id", "fact:0")):
        a = candidate(1)
        object.__setattr__(a, field, value)
        with pytest.raises(ValueError):
            blend_evidence((a,), (), max_candidates=2, max_bytes=128_000)


def test_two_full_lanes_remain_bounded_and_do_not_invent_ids() -> None:
    original = tuple(candidate(i) for i in range(1, 101))
    selected = tuple(candidate(i) for i in range(101, 201))
    result = blend_evidence(original, selected, max_candidates=100, max_bytes=128_000)
    assert len(result.ids) == len(result.omitted_ids) == 100
    assert set(result.ids).isdisjoint(result.omitted_ids)
    assert set((*result.ids, *result.omitted_ids)) == {c.id for c in (*original, *selected)}


def test_oversized_top_ranked_singleton_does_not_block_later_evidence() -> None:
    huge, small = candidate(1, "x" * 1000), candidate(2)
    result = blend_evidence((huge, small), (), max_candidates=2,
        max_bytes=evidence_payload_bytes((small,)))
    assert result.ids == (small.id,)
    assert result.omitted_ids == (huge.id,)
    assert result.reasons == ("max_evidence_bytes",)


def test_merged_group_larger_than_output_cap_is_wholly_omitted() -> None:
    original = tuple(candidate(i) for i in range(1, 101))
    selected = tuple(candidate(i) for i in range(100, 200))
    result = blend_evidence(original, selected,
        original_groups=(tuple(c.id for c in original),),
        selected_groups=(tuple(c.id for c in selected),), max_candidates=100, max_bytes=128_000)
    assert result.ids == ()
    assert result.groups == ()
    assert len(result.omitted_ids) == 199
    assert result.reasons == ("candidate_limit", "atomic_group_omitted")


def test_both_caps_are_reported_without_splitting_group() -> None:
    a, b = candidate(1), candidate(2)
    result = blend_evidence((a, b), (), original_groups=((a.id, b.id),), max_candidates=1, max_bytes=2)
    assert result.ids == ()
    assert result.reasons == ("candidate_limit", "max_evidence_bytes", "atomic_group_omitted")


def test_result_has_no_mutable_input_references() -> None:
    a, b = candidate(1), candidate(2)
    result = blend_evidence((a, b), (), original_groups=((a.id, b.id),), max_candidates=2, max_bytes=128_000)
    object.__setattr__(a, "id", "fact:99")
    assert result.ids == ("fact:1", "fact:2")
    assert result.groups == (("fact:1", "fact:2"),)


@pytest.mark.parametrize("lane_name", ["original", "selected"])
def test_input_lane_byte_cap_is_independent_of_output_budget(lane_name: str) -> None:
    overhead = evidence_payload_bytes((candidate(1, "x"),)) - 1
    exact = candidate(1, "x" * (128_000 - overhead))
    oversized = candidate(1, exact.text + "x")
    original = (exact,) if lane_name == "original" else ()
    selected = (exact,) if lane_name == "selected" else ()
    assert evidence_payload_bytes((exact,)) == 128_000
    assert blend_evidence(original, selected, max_candidates=1, max_bytes=128_000).ids == (exact.id,)
    assert blend_evidence(original, selected, max_candidates=1, max_bytes=2).ids == ()
    with pytest.raises(ValueError, match="lane.*128000"):
        blend_evidence((oversized,) if lane_name == "original" else (),
            (oversized,) if lane_name == "selected" else (), max_candidates=1, max_bytes=128_000)


@pytest.mark.parametrize("text", ["x" * 1300, "海" * 450], ids=["ascii", "unicode"])
def test_large_aggregate_lane_is_rejected_using_utf8_json_bytes(text: str) -> None:
    lane = tuple(candidate(i, text) for i in range(1, 101))
    assert all(evidence_payload_bytes((entry,)) < 128_000 for entry in lane)
    assert evidence_payload_bytes(lane) > 128_000
    with pytest.raises(ValueError, match="lane.*128000"):
        blend_evidence(lane, (), max_candidates=1, max_bytes=2)


@pytest.mark.parametrize("bound", ["count", "bytes", "fits"])
def test_joint_group_spans_lanes_and_packs_atomically(bound: str) -> None:
    a, b, c = candidate(1), candidate(2), candidate(3)
    byte_limit = evidence_payload_bytes((a, b)) - 1 if bound == "bytes" else 128_000
    count_limit = 1 if bound == "count" else 3
    result = blend_evidence((a, c), (b,), joint_groups=((a.id, b.id),),
        max_candidates=count_limit, max_bytes=byte_limit)
    if bound == "fits":
        assert result.ids == (a.id, b.id, c.id)
        assert result.groups == ((a.id, b.id),)
        assert result.original_ids == (a.id, c.id) and result.selected_ids == (b.id,)
    else:
        assert result.ids == (c.id,) and result.groups == ()
        assert result.original_ids == (c.id,) and result.selected_ids == ()
        assert result.omitted_ids == (a.id, b.id)
        assert result.reasons == (("candidate_limit" if bound == "count" else "max_evidence_bytes"),
            "atomic_group_omitted")


def test_joint_groups_preserve_reciprocal_ranks_and_lane_provenance() -> None:
    a, b, c = candidate(1), candidate(2), candidate(3)
    result = blend_evidence((a, c), (b, c), joint_groups=((a.id, b.id),),
        max_candidates=3, max_bytes=128_000)
    assert result.ids == (c.id, a.id, b.id)
    assert result.original_ids == (a.id, c.id)
    assert result.selected_ids == (b.id, c.id)
    assert result.groups == ((a.id, b.id),)


def test_joint_groups_merge_with_both_lane_contracts() -> None:
    a, b, c, d = (candidate(i) for i in range(1, 5))
    result = blend_evidence((a, b), (c, d), original_groups=((a.id, b.id),),
        selected_groups=((c.id, d.id),), joint_groups=((b.id, c.id),), max_candidates=4, max_bytes=128_000)
    assert result.ids == (a.id, b.id, c.id, d.id)
    assert result.groups == (result.ids,)


@pytest.mark.parametrize("bad", [[], (("fact:1",),), (("fact:1", "fact:1"),),
    (("fact:1", "fact:3"),), (["fact:1", "fact:2"],), (("fact:1", 2),),
    tuple(("fact:1", "fact:2") for _ in range(101)),
    (tuple(f"fact:{i}" for i in range(1, 202)),)])
def test_joint_groups_validate_shape_ids_and_caps(bad: object) -> None:
    with pytest.raises(ValueError):
        blend_evidence((candidate(1),), (candidate(2),),
            joint_groups=cast(tuple[tuple[str, ...], ...], bad), max_candidates=2, max_bytes=128_000)


def test_joint_group_accepts_all_200_union_members_before_output_cap() -> None:
    original = tuple(candidate(i) for i in range(1, 101))
    selected = tuple(candidate(i) for i in range(101, 201))
    joint = tuple(c.id for c in (*original, *selected))
    result = blend_evidence(original, selected, joint_groups=(joint,), max_candidates=100, max_bytes=128_000)
    assert result.ids == () and result.groups == ()
    assert result.omitted_ids == joint
    assert result.reasons == ("candidate_limit", "atomic_group_omitted")
