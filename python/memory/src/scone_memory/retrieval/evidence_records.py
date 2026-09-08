"""Canonical sourced ledger data for inference, separate from graph layout.

The graph reader already validated source retention, scope and exact quotes.
Only complete quote-backed records cross this boundary. Fingerprints retain
no evidence text and allow cached receipts to revalidate the same records.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import JsonValue

from .evidence_graph import QueryEvidenceGraph

EvidenceRecord = dict[str, JsonValue]


@dataclass(frozen=True)
class EvidenceRecords:
    claims: list[EvidenceRecord]
    relations: list[EvidenceRecord]


def fingerprint(record: EvidenceRecord) -> str:
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_evidence(graph: QueryEvidenceGraph) -> EvidenceRecords:
    claims: list[EvidenceRecord] = []
    for node in graph.nodes:
        if (node.kind != "claim" or node.data.get("provenance_status") != "retained"
                or node.data.get("record_complete") is not True or not node.data.get("quote")):
            continue
        claims.append({key: node.data.get(key) for key in (
            "fact_id", "subject", "predicate", "object", "origin", "status", "valid_from", "valid_until",
            "confidence", "source_episode_id", "quote")})
    ids = {record["fact_id"] for record in claims if isinstance(record["fact_id"], int)}
    relations: list[EvidenceRecord] = []
    for edge in graph.edges:
        if (edge.kind not in {"extends", "derived_from", "contradicts", "supports"}
                or edge.data.get("provenance_status") != "retained" or not edge.data.get("quote")
                or edge.data.get("record_complete") is not True):
            continue
        left = edge.source.removeprefix("claim:")
        right = edge.target.removeprefix("claim:")
        if not left.isdecimal() or not right.isdecimal() or int(left) not in ids or int(right) not in ids:
            continue
        relations.append({"link_id": edge.data.get("link_id"), "from_fact": int(left), "to_fact": int(right),
                          "kind": edge.kind, "source_episode_id": edge.data.get("source_episode_id"), "quote": edge.data.get("quote")})
    return EvidenceRecords(claims, relations)


def restrict_graph(graph: QueryEvidenceGraph, fact_ids: set[int], link_ids: set[int], chunk_ids: set[int] | None = None) -> QueryEvidenceGraph:
    """Keep only ledger records actually supplied, removing newly added links."""
    graph = graph.model_copy(deep=True)
    claim_ids = {f"claim:{value}" for value in fact_ids}
    graph.nodes = [node for node in graph.nodes if (node.kind != "claim" or node.id in claim_ids)
                   and (chunk_ids is None or node.kind != "chunk" or node.data.get("chunk_id") in chunk_ids)]
    ids = {node.id for node in graph.nodes}
    # Supersession is inspectable in a query graph, but canonical model
    # records do not supply that field or fingerprint it for an earlier reply.
    graph.edges = [edge for edge in graph.edges if edge.source in ids and edge.target in ids
                   and edge.kind != "superseded_by"
                   and (edge.kind not in {"extends", "derived_from", "contradicts", "supports"} or edge.data.get("link_id") in link_ids)
                   and (edge.kind != "relation" or edge.data.get("fact_id") in fact_ids)]
    connected = {value for edge in graph.edges for value in (edge.source, edge.target)}
    link_sources = {f"episode:{edge.data.get('source_episode_id')}" for edge in graph.edges
                    if edge.kind in {"extends", "derived_from", "contradicts", "supports"}}
    graph.nodes = [node for node in graph.nodes if node.kind not in {"concept", "episode"}
                   or node.id in connected or node.id in link_sources]
    graph.counts = {kind: sum(node.kind == kind for node in graph.nodes) for kind in sorted({node.kind for node in graph.nodes})}
    return graph
