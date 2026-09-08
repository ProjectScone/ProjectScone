"""Compact ordered paths over independently revalidated canonical evidence.

These are evidence connections, never generated conclusions. Optional ordered
quotes come verbatim from canonical claims; steps retain their claim/relation IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from pydantic import JsonValue

from .evidence_records import EvidenceRecord, EvidenceRecords
from .multihop import MultiHopEdge, MultiHopResult


@dataclass(frozen=True)
class PathEvidence:
    paths: list[EvidenceRecord]
    truncated: bool = False


def fact_ids(path: EvidenceRecord) -> list[int]:
    values = path.get("fact_ids")
    return [value for value in values if type(value) is int] if isinstance(values, list) else []


def link_ids(path: EvidenceRecord) -> list[int]:
    steps = path.get("steps")
    if not isinstance(steps, list):
        return []
    result: list[int] = []
    for step in steps:
        if isinstance(step, dict):
            value = step.get("link_id")
            if type(value) is int:
                result.append(value)
    return result


def path_records(path: EvidenceRecord, records: EvidenceRecords) -> EvidenceRecords:
    """Include every path component and competing quoted contradictions together."""
    ids, links = set(fact_ids(path)), set(link_ids(path))
    for relation in records.relations:
        if relation.get("kind") == "contradicts" and (relation.get("from_fact") in ids or relation.get("to_fact") in ids):
            for key in ("from_fact", "to_fact"):
                value = relation.get(key)
                if type(value) is int:
                    ids.add(value)
            link_id = relation.get("link_id")
            if type(link_id) is int:
                links.add(link_id)
    return EvidenceRecords([claim for claim in records.claims if claim.get("fact_id") in ids],
                           [relation for relation in records.relations if relation.get("link_id") in links])


def _verified(edge: MultiHopEdge, claims: dict[int, EvidenceRecord], relations: dict[int, EvidenceRecord]) -> bool:
    if edge.from_fact not in claims or edge.to_fact not in claims or edge.kind == "contradicts":
        return False
    if edge.kind == "subject_object":
        return (edge.link_id is None and claims[edge.from_fact].get("object") == claims[edge.to_fact].get("subject")
                and isinstance(claims[edge.from_fact].get("object"), str))
    relation = relations.get(edge.link_id) if edge.link_id is not None else None
    return relation is not None and all(relation.get(key) == value for key, value in (
        ("from_fact", edge.from_fact), ("to_fact", edge.to_fact), ("kind", edge.kind),
        ("source_episode_id", edge.source_episode_id), ("quote", edge.quote)))


def ordered_evidence_paths(expansion: MultiHopResult, records: EvidenceRecords, *,
                           max_hops: int = 3, max_paths: int = 8, max_work: int = 256,
                           include_quotes: bool = False) -> PathEvidence:
    """Prefer maximal paths, with strict caps even for dense cyclic graphs.

    Starts follow recalled seed order. Subject/object joins run forward;
    stored links preserve their original direction and record traversal direction.
    A contradiction is never traversed as a continuation. Adjacent contradiction
    evidence is supplied by ``path_records`` when a caller budgets the packet.
    ``include_quotes`` opts into duplicate verbatim claims in traversal order.
    """
    if type(include_quotes) is not bool:
        raise ValueError("include_quotes must be a boolean")
    if (any(type(value) is not int for value in (max_hops, max_paths, max_work))
            or not 1 <= max_hops <= 6 or not 1 <= max_paths <= 16 or not 1 <= max_work <= 2048):
        raise ValueError("path limits are outside supported bounds")
    claims: dict[int, EvidenceRecord] = {}
    relations: dict[int, EvidenceRecord] = {}
    for claim in records.claims:
        fact_id = claim.get("fact_id")
        quote = claim.get("quote")
        if type(fact_id) is int and type(claim.get("source_episode_id")) is int and isinstance(quote, str) and quote:
            claims[fact_id] = claim
    for relation in records.relations:
        link_id = relation.get("link_id")
        if type(link_id) is int:
            relations[link_id] = relation
    neighbors: dict[int, list[tuple[int, EvidenceRecord]]] = {}
    for edge in sorted(expansion.edges[:256], key=lambda item: item.kind != "subject_object"):
        if not _verified(edge, claims, relations):
            continue
        step: EvidenceRecord = {"from_fact": edge.from_fact, "to_fact": edge.to_fact, "kind": edge.kind, "direction": "forward"}
        if edge.link_id is not None:
            step["link_id"] = edge.link_id
        neighbors.setdefault(edge.from_fact, []).append((edge.to_fact, step))
        if edge.kind != "subject_object":
            neighbors.setdefault(edge.to_fact, []).append((edge.from_fact, {**step, "direction": "reverse"}))
    candidates: list[tuple[list[int], list[EvidenceRecord]]] = []
    work, truncated, hop_truncated = 0, False, False
    for seed in expansion.seed_fact_ids[:128]:
        if seed not in claims:
            continue
        pending: list[tuple[list[int], list[EvidenceRecord]]] = [([seed], [])]
        while pending:
            ids, steps = pending.pop()
            if steps:
                candidates.append((ids, steps))
            if len(steps) >= max_hops:
                hop_truncated = hop_truncated or any(target not in ids for target, _ in neighbors.get(ids[-1], []))
                continue
            next_paths: list[tuple[list[int], list[EvidenceRecord]]] = []
            for target, step in neighbors.get(ids[-1], []):
                if work >= max_work:
                    truncated = True
                    break
                work += 1
                if target not in ids:
                    next_paths.append(([*ids, target], [*steps, step]))
            pending.extend(reversed(next_paths))
            if truncated:
                break
        if truncated:
            break
    selected: list[tuple[list[int], list[EvidenceRecord]]] = []

    def edges(steps: list[EvidenceRecord]) -> frozenset[str]:
        return frozenset(json.dumps({key: value for key, value in step.items() if key != "direction"}, sort_keys=True)
                         for step in steps)

    for ids, steps in sorted(candidates, key=lambda item: -len(item[0])):
        # Contained prefixes/suffixes and the same path read in reverse add no
        # evidence. Preserve the first orientation anchored to ranked seeds.
        if any((set(ids) == set(kept_ids) and edges(steps) == edges(kept_steps)) or any(
                ids == kept_ids[offset:offset + len(ids)] and steps == kept_steps[offset:offset + len(steps)]
                for offset in range(len(kept_ids) - len(ids) + 1)) for kept_ids, kept_steps in selected):
            continue
        if len(selected) >= max_paths:
            truncated = True
            break
        selected.append((ids, steps))
    packets: list[EvidenceRecord] = []
    for ids, steps in selected:
        step_values: list[JsonValue] = [step for step in steps]
        packet: EvidenceRecord = {"fact_ids": [value for value in ids], "steps": step_values}
        if include_quotes:
            packet["ordered_evidence"] = [
                {"fact_id": fact_id, "source_episode_id": claims[fact_id]["source_episode_id"],
                 "quote": claims[fact_id]["quote"]} for fact_id in ids
            ]
        packets.append(packet)
    return PathEvidence(packets, truncated or hop_truncated)
