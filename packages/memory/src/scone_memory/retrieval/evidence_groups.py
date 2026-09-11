"""Pure grouping of supplied facts by directed object-to-subject joins.

A fact joins another when its object names the other's subject under
``memory.identity.join_match``: the same key (case folded, spacing collapsed,
nothing looser), never through prose, a quotation or a pronoun, and for a
value whose case carries meaning, only by its exact spelling. Each join
records whether the names were literally equal or met only after folding. Components preserve branches
and cycles; a join asserts shared naming only, not a semantic, causal, or
transitive conclusion. No retrieval or model calls occur here. Every input is
represented exactly once, or the whole call fails.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ..memory.identity import join_match
from .adaptive import EvidenceCandidate


@dataclass(frozen=True)
class EvidenceGrouping:
    """JSON-ready records and selection expansion, detached from the input.

    Record and member order follows input rank. Group IDs instead use canonical
    ID ordering, making identity independent of retrieval ranking. ``members``
    covers every selectable record; ``groups`` contains only atomic components.
    """
    records: tuple[dict[str, object], ...]
    members: dict[str, tuple[str, ...]]
    groups: dict[str, tuple[str, ...]]


def _serialize(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("invalid evidence grouping input") from None


def _validated(candidates: tuple[EvidenceCandidate, ...]) -> tuple[EvidenceCandidate, ...]:
    if not isinstance(candidates, tuple) or len(candidates) > 100:
        raise ValueError("invalid evidence grouping input")
    try:
        if any(not isinstance(candidate, EvidenceCandidate) for candidate in candidates):
            raise ValueError()
        result = tuple(EvidenceCandidate.model_validate(dict(vars(candidate)), strict=True)
                       for candidate in candidates)
        if len({candidate.id for candidate in result}) != len(result):
            raise ValueError()
        return result
    except (TypeError, ValueError):
        raise ValueError("invalid evidence grouping input") from None


def _joins(candidates: tuple[EvidenceCandidate, ...]) -> list[tuple[int, int]]:
    joins: list[tuple[int, int]] = []
    for source_index, source in enumerate(candidates):
        if not source.id.startswith("fact:") or source.object is None or not source.object.strip():
            continue
        for target_index, target in enumerate(candidates):
            if (source_index == target_index or not target.id.startswith("fact:")
                    or target.subject is None or not target.subject.strip()
                    or join_match(source.object, target.subject) is None):
                continue
            if len(joins) >= 2048:
                raise ValueError("evidence grouping exceeds directed edge limit")
            joins.append((source_index, target_index))
    return joins


def build_evidence_groups(candidates: tuple[EvidenceCandidate, ...], *, max_bytes: int = 128000) -> EvidenceGrouping:
    """Group weakly connected fact components without dropping any input.

    ``max_bytes`` bounds the full compact, non-ASCII-escaped UTF-8 JSON record
    array, including group IDs, original source fields and explicit joins.
    Inputs are strictly revalidated, including instances made by model_copy.
    """
    if type(max_bytes) is not int or not 2 <= max_bytes <= 128000:
        raise ValueError("max_bytes must be an integer in 2..128000")
    candidates = _validated(candidates)
    originals: list[dict[str, object]] = [candidate.model_dump(mode="json") for candidate in candidates]
    if len(_serialize(originals)) > max_bytes:
        raise ValueError("evidence grouping exceeds serialized byte limit")
    joins = _joins(candidates)
    neighbors: list[set[int]] = [set() for _ in candidates]
    for source, target in joins:
        neighbors[source].add(target)
        neighbors[target].add(source)
    records: list[dict[str, object]] = []
    members: dict[str, tuple[str, ...]] = {}
    groups: dict[str, tuple[str, ...]] = {}
    visited: set[int] = set()
    for anchor, candidate in enumerate(candidates):
        if anchor in visited:
            continue
        component: set[int] = set()
        pending = [anchor]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbors[current] - component)
        visited.update(component)
        ordered = sorted(component)
        ids = tuple(candidates[index].id for index in ordered)
        if len(ordered) == 1:
            records.append({**originals[anchor], "kind": "standalone"})
            members[candidate.id] = ids
            continue
        component_joins = [(source, target) for source, target in joins if source in component]
        edges: list[dict[str, object]] = [
            {"kind": "object_subject",
             "match": "literal" if candidates[source].object == candidates[target].subject else "normalised",
             "from_id": candidates[source].id, "to_id": candidates[target].id}
            for source, target in component_joins]
        canonical = {"members": [originals[index] for index in sorted(component, key=lambda index: candidates[index].id)],
                     "joins": sorted(edges, key=lambda edge: (str(edge["from_id"]), str(edge["to_id"])))}
        group_id = "group:" + hashlib.sha256(_serialize(canonical)).hexdigest()
        records.append({"id": group_id, "kind": "fact_component",
                        "members": [originals[index] for index in ordered], "joins": edges})
        members[group_id] = groups[group_id] = ids
    if len(_serialize(records)) > max_bytes:
        raise ValueError("evidence grouping exceeds serialized byte limit")
    return EvidenceGrouping(tuple(records), members, groups)
