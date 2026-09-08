"""Pure, bounded fusion of original-query and selected evidence snapshots.

Atomic components compete by their strongest member's reciprocal rank score.
Packing never clips payloads or splits components; provenance follows each
input lane's order, while output IDs follow ranked units. No source freshness
or semantic accuracy is inferred here; the caller must verify retained IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adaptive import EvidenceCandidate


@dataclass(frozen=True)
class EvidenceBlend:
    ids: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    original_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    reasons: tuple[str, ...]


def _lane(candidates: tuple[EvidenceCandidate, ...]) -> tuple[EvidenceCandidate, ...]:
    from .adaptive import EvidenceCandidate, evidence_payload_bytes

    if type(candidates) is not tuple or len(candidates) > 100:
        raise ValueError("evidence lane must be a tuple of at most 100 candidates")
    validated: list[EvidenceCandidate] = []
    seen: set[str] = set()
    serialized_bytes = 2
    for candidate in candidates:
        if not isinstance(candidate, EvidenceCandidate):
            raise ValueError("evidence lane contains an invalid candidate")
        current = EvidenceCandidate.model_validate(dict(vars(candidate)), strict=True)
        if current.id in seen:
            raise ValueError("evidence lane IDs must be unique")
        serialized_bytes += evidence_payload_bytes((current,)) - 2 + bool(validated)
        if serialized_bytes > 128_000:
            raise ValueError("evidence lane must not exceed 128000 serialized bytes")
        seen.add(current.id)
        validated.append(current)
    return tuple(validated)


def _groups(groups: tuple[tuple[str, ...], ...], ids: set[str], *, max_members: int = 100) -> None:
    if type(groups) is not tuple or len(groups) > 100:
        raise ValueError("evidence groups must be a tuple of at most 100 groups")
    for group in groups:
        if type(group) is not tuple or not 2 <= len(group) <= max_members:
            raise ValueError(f"each evidence group must be a tuple of 2 to {max_members} IDs")
        if any(type(member) is not str for member in group):
            raise ValueError("evidence group IDs must be strings")
        if len(set(group)) != len(group) or not set(group).issubset(ids):
            raise ValueError("evidence group IDs must be unique members of the supplied evidence")


def blend_evidence(
    original: tuple[EvidenceCandidate, ...],
    selected: tuple[EvidenceCandidate, ...],
    *,
    original_groups: tuple[tuple[str, ...], ...] = (),
    selected_groups: tuple[tuple[str, ...], ...] = (),
    joint_groups: tuple[tuple[str, ...], ...] = (),
    max_candidates: int,
    max_bytes: int,
) -> EvidenceBlend:
    """Fuse at most 200 snapshots under exact JSON-byte and candidate caps.

    Each input lane is capped at 128000 serialized bytes independently of the
    output budget, with incremental checks before fusion.
    Shared IDs require identical payloads. Joint groups can reference either
    lane without changing lane membership or rank. Overlapping groups become
    one indivisible component. Limits and snapshots are revalidated even
    when callers bypass type annotations or frozen model construction.
    """
    from .adaptive import evidence_payload_bytes
    from .adaptive_graph import merge_groups

    if type(max_candidates) is not int or not 1 <= max_candidates <= 100:
        raise ValueError("max_candidates must be an integer from 1 to 100")
    if type(max_bytes) is not int or not 2 <= max_bytes <= 128_000:
        raise ValueError("max_bytes must be an integer from 2 to 128000")
    original = _lane(original)
    selected = _lane(selected)
    _groups(original_groups, {candidate.id for candidate in original})
    _groups(selected_groups, {candidate.id for candidate in selected})

    candidates: dict[str, EvidenceCandidate] = {}
    scores: dict[str, Fraction] = {}
    for lane in (original, selected):
        for rank, candidate in enumerate(lane, start=1):
            previous = candidates.get(candidate.id)
            if previous is not None and previous != candidate:
                raise ValueError("shared evidence IDs must have identical payloads")
            candidates[candidate.id] = candidate
            scores[candidate.id] = scores.get(candidate.id, Fraction()) + Fraction(1, 60 + rank)
    _groups(joint_groups, set(candidates), max_members=200)
    order = {identifier: index for index, identifier in enumerate(candidates)}
    merged = merge_groups((*original_groups, *selected_groups, *joint_groups))
    units = [tuple(sorted(group, key=order.__getitem__)) for group in merged]
    grouped = {identifier for group in units for identifier in group}
    units.extend((identifier,) for identifier in candidates if identifier not in grouped)
    units.sort(key=lambda unit: (-max(scores[identifier] for identifier in unit), order[unit[0]]))

    retained: list[EvidenceCandidate] = []
    retained_groups: list[tuple[str, ...]] = []
    reasons: list[str] = []
    for unit in units:
        proposed = (*retained, *(candidates[identifier] for identifier in unit))
        omitted: list[str] = []
        if len(proposed) > max_candidates:
            omitted.append("candidate_limit")
        if evidence_payload_bytes(proposed) > max_bytes:
            omitted.append("max_evidence_bytes")
        if omitted:
            if len(unit) > 1:
                omitted.append("atomic_group_omitted")
            reasons.extend(reason for reason in omitted if reason not in reasons)
            continue
        retained.extend(candidates[identifier] for identifier in unit)
        if len(unit) > 1:
            retained_groups.append(unit)

    ids = tuple(candidate.id for candidate in retained)
    delivered = set(ids)
    return EvidenceBlend(
        ids=ids,
        groups=tuple(retained_groups),
        original_ids=tuple(candidate.id for candidate in original if candidate.id in delivered),
        selected_ids=tuple(candidate.id for candidate in selected if candidate.id in delivered),
        omitted_ids=tuple(identifier for identifier in candidates if identifier not in delivered),
        reasons=tuple(reasons),
    )
