"""Pure, bounded topology summaries of an already scoped query evidence graph.

Communities use original Scone deterministic label propagation, with connected
components for trees and pairs. Hubs, density (cohesion), and bridges describe
an undirected projection, not relevance, confidence, causality, or proof. Edge
provenance is reported from the supplied graph, never newly verified here.
No labels, quotes, prompts, store reads, external code, or model calls are used.
"""
from __future__ import annotations

from collections import deque
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .evidence_graph import EvidenceEdge, EvidenceNode, QueryEvidenceGraph

_ID = re.compile(r'[A-Za-z0-9:_-]{1,128}\Z')


class GraphAnalysisLimits(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    max_nodes: int = Field(default=256, ge=1, le=512)
    max_edges: int = Field(default=1024, ge=0, le=2048)
    max_iterations: int = Field(default=30, ge=1, le=60)
    max_work: int = Field(default=100000, ge=1, le=500000)
    max_hubs: int = Field(default=10, ge=0, le=32)
    max_bridges: int = Field(default=64, ge=0, le=256)


class GraphCommunity(BaseModel):
    id: str
    node_ids: list[str]
    isolated: bool
    internal_edges: int
    boundary_edges: int
    cohesion: float | None


class GraphHub(BaseModel):
    node_id: str
    community_id: str
    degree: int
    in_degree: int
    out_degree: int


class GraphBridge(BaseModel):
    source: str
    target: str
    relation_count: int
    cut_edge: bool
    cross_community: bool


class GraphAnalysisCoverage(BaseModel):
    input_truncated: bool = False
    provenance_missing: int = 0
    provenance_omitted: int = 0
    recorded_relations: int = 0
    inferred_relations: int = 0
    unverified_relations: int = 0
    literal_mentions: int = 0
    retrieval_links: int = 0
    ignored_edges: int = 0
    hubs_omitted: int = 0
    bridges_omitted: int = 0
    reasons: list[str] = Field(default_factory=list)


class GraphAnalysisCounts(BaseModel):
    nodes: int = 0
    concepts: int = 0
    edges: int = 0
    topology_edges: int = 0
    iterations: int = 0
    work: int = 0


class GraphAnalysisResult(BaseModel):
    """Versioned analysis; ``method`` describes the resulting partition.

    ``components`` can also mean that label propagation ran on a cyclic
    component without splitting it. ``counts.iterations`` records that work.
    """

    schema_version: Literal[1] = 1
    algorithm: Literal['scone_label_propagation_v1'] = 'scone_label_propagation_v1'
    status: Literal['complete', 'partial', 'unavailable'] = 'complete'
    basis: Literal['recorded_concept_relation_topology'] = 'recorded_concept_relation_topology'
    projection: Literal['undirected_unique_pairs'] = 'undirected_unique_pairs'
    method: Literal['label_propagation', 'components', 'unlinked'] = 'unlinked'
    communities: list[GraphCommunity] = Field(default_factory=list)
    hubs: list[GraphHub] = Field(default_factory=list)
    bridges: list[GraphBridge] = Field(default_factory=list)
    coverage: GraphAnalysisCoverage = Field(default_factory=GraphAnalysisCoverage)
    counts: GraphAnalysisCounts = Field(default_factory=GraphAnalysisCounts)

    @classmethod
    def unavailable(cls, reason: str) -> GraphAnalysisResult:
        return cls(status='unavailable', coverage=GraphAnalysisCoverage(reasons=[reason]))


class _Limit(Exception):
    pass


class _Work:
    def __init__(self, maximum: int, counts: GraphAnalysisCounts) -> None:
        self.maximum, self.counts = maximum, counts

    def use(self, amount: int = 1) -> None:
        if self.counts.work + amount > self.maximum:
            raise _Limit('max_work')
        self.counts.work += amount


def _identifier(value: str) -> bool:
    return isinstance(value, str) and len(value) <= 128 and _ID.fullmatch(value) is not None


def _record_id(value: object) -> bool:
    return type(value) is int and 0 < value <= 2**63 - 1


def _recorded(edge: EvidenceEdge) -> bool:
    data = edge.data
    return (data.get('category') == 'fact_relation' and data.get('provenance_status') == 'retained'
            and _record_id(data.get('fact_id')) and _record_id(data.get('source_episode_id'))
            and isinstance(data.get('quote'), str) and bool(data['quote'])
            and data.get('origin') in ('stated', 'extracted', 'inferred'))


def _components(ids: list[str], neighbors: dict[str, tuple[str, ...]], work: _Work) -> list[list[str]]:
    remaining = set(ids)
    groups: list[list[str]] = []
    for seed in sorted(ids):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        queue = deque([seed])
        group = []
        while queue:
            work.use()
            current = queue.popleft()
            group.append(current)
            for target in neighbors[current]:
                work.use()
                if target in remaining:
                    remaining.remove(target)
                    queue.append(target)
        groups.append(sorted(group))
    return groups


def _partition(neighbors: dict[str, tuple[str, ...]], limits: GraphAnalysisLimits,
               result: GraphAnalysisResult, work: _Work) -> list[list[str]]:
    groups = _components(list(neighbors), neighbors, work)
    fixed, cyclic = [], []
    isolated = []
    linked_components = 0
    for group in groups:
        work.use(len(group))
        edge_count = sum(len(neighbors[node]) for node in group) // 2
        if edge_count == 0:
            isolated.extend(group)
            continue
        linked_components += 1
        if edge_count < len(group):
            fixed.append(group)
        else:
            cyclic.extend(group)
    if cyclic:
        labels = {node: node for node in cyclic}
        sequence = sorted(cyclic, key=lambda node: (len(neighbors[node]), node))
        for _ in range(limits.max_iterations):
            changed = False
            result.counts.iterations += 1
            for node in sequence:
                votes: dict[str, int] = {}
                for target in neighbors[node]:
                    work.use()
                    label = labels[target]
                    votes[label] = votes.get(label, 0) + 1
                best = max(votes.values())
                current = labels[node]
                winner = current if votes.get(current) == best else min(label for label, count in votes.items() if count == best)
                if winner != current:
                    labels[node] = winner
                    changed = True
            if not changed:
                break
        else:
            result.coverage.reasons.append('max_iterations')
        by_label: dict[str, list[str]] = {}
        for node in sorted(cyclic):
            by_label.setdefault(labels[node], []).append(node)
        for members in by_label.values():
            # A shared propagated label alone must not join disconnected sets.
            fixed.extend(_components(members, neighbors, work))
    fixed.sort(key=lambda group: group[0])
    result.method = 'label_propagation' if len(fixed) > linked_components else 'components' if linked_components else 'unlinked'
    if isolated:
        fixed.append(sorted(isolated))
    return fixed


def _cut_edges(neighbors: dict[str, tuple[str, ...]], work: _Work) -> set[tuple[str, str]]:
    """Linear DFS low-link calculation; recursion is bounded by max_nodes <=512."""
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    cuts: set[tuple[str, str]] = set()

    def visit(node: str, parent: str | None) -> None:
        work.use()
        discovery[node] = low[node] = len(discovery)
        for target in neighbors[node]:
            work.use()
            if target == parent:
                continue
            if target not in discovery:
                visit(target, node)
                low[node] = min(low[node], low[target])
                if low[target] > discovery[node]:
                    cuts.add((node, target) if node < target else (target, node))
            else:
                low[node] = min(low[node], discovery[target])

    for node in sorted(neighbors):
        if node not in discovery:
            visit(node, None)
    return cuts


def _summarize(neighbors: dict[str, tuple[str, ...]], incoming: dict[str, set[str]],
               outgoing: dict[str, set[str]], pairs: dict[tuple[str, str], int],
               limits: GraphAnalysisLimits, result: GraphAnalysisResult, work: _Work) -> None:
    membership: dict[str, str] = {}
    for members in _partition(neighbors, limits, result, work):
        inside = set(members)
        internal, boundary = 0, 0
        for node in members:
            for target in neighbors[node]:
                work.use()
                if target in inside:
                    internal += 1
                else:
                    boundary += 1
        internal //= 2
        isolated = internal == 0 and boundary == 0
        identity = 'unlinked' if isolated else 'group:' + members[0]
        size = len(members)
        result.communities.append(GraphCommunity(id=identity, node_ids=members, isolated=isolated,
            internal_edges=internal, boundary_edges=boundary,
            cohesion=round(2 * internal / (size * (size - 1)), 6) if size > 1 and not isolated else None))
        membership.update((node, identity) for node in members)
    ranked = sorted((node for node in neighbors if neighbors[node]), key=lambda node: (-len(neighbors[node]), node))
    result.coverage.hubs_omitted = max(0, len(ranked) - limits.max_hubs)
    result.hubs = [GraphHub(node_id=node, community_id=membership[node], degree=len(neighbors[node]),
                           in_degree=len(incoming[node]), out_degree=len(outgoing[node])) for node in ranked[:limits.max_hubs]]
    cuts = _cut_edges(neighbors, work)
    for (left, right), count in sorted(pairs.items()):
        work.use()
        cross = membership[left] != membership[right]
        cut = (left, right) in cuts
        if not cross and not cut:
            continue
        if len(result.bridges) >= limits.max_bridges:
            result.coverage.bridges_omitted += 1
            continue
        result.bridges.append(GraphBridge(source=left, target=right, relation_count=count,
                                         cut_edge=cut, cross_community=cross))
    if result.coverage.bridges_omitted:
        result.coverage.reasons.append('max_bridges')


def analyze_evidence_graph(graph: QueryEvidenceGraph, *, limits: GraphAnalysisLimits | None = None) -> GraphAnalysisResult:
    """Analyze supplied topology only, with stable IDs and no caller mutation.

    Input row caps reject whole oversized graphs, so input permutations cannot
    choose a different prefix. Only bounded ASCII IDs and constant metadata
    fields are inspected; arbitrary label/text payloads are never traversed,
    encoded or copied. Work counts row and adjacency visits (sorting is bounded
    separately by row caps). Exhausted work returns no partial topology claims.
    Iteration exhaustion returns a deterministic partition marked partial.
    """
    selected = GraphAnalysisLimits.model_validate(limits.model_dump()) if limits is not None else GraphAnalysisLimits()
    result = GraphAnalysisResult(counts=GraphAnalysisCounts(nodes=len(graph.nodes), edges=len(graph.edges)))
    work = _Work(selected.max_work, result.counts)
    try:
        if len(graph.nodes) > selected.max_nodes:
            raise _Limit('max_nodes')
        if len(graph.edges) > selected.max_edges:
            raise _Limit('max_edges')
        work.use(len(graph.nodes) + len(graph.edges))
        ids: dict[str, str] = {}
        for node in graph.nodes:
            if not isinstance(node, EvidenceNode) or not _identifier(node.id) or node.id in ids:
                raise _Limit('invalid_node_identity')
            ids[node.id] = node.kind
        for count in (graph.provenance_missing, graph.provenance_omitted):
            if type(count) is not int or not 0 <= count <= 1000000000:
                raise _Limit('invalid_provenance_counts')
        coverage = result.coverage
        coverage.input_truncated = graph.truncated
        coverage.provenance_missing = graph.provenance_missing
        coverage.provenance_omitted = graph.provenance_omitted
        if graph.truncated:
            coverage.reasons.append('input_truncated')
        if graph.provenance_missing or graph.provenance_omitted:
            coverage.reasons.append('incomplete_provenance')
        if graph.notices:
            coverage.reasons.append('input_notices')
        adjacent: dict[str, set[str]] = {node: set() for node in sorted(ids) if ids[node] == 'concept'}
        incoming: dict[str, set[str]] = {node: set() for node in adjacent}
        outgoing: dict[str, set[str]] = {node: set() for node in adjacent}
        pairs: dict[tuple[str, str], int] = {}
        for edge in graph.edges:
            if not isinstance(edge, EvidenceEdge) or not _identifier(edge.source) or not _identifier(edge.target):
                raise _Limit('invalid_edge_identity')
            if edge.source not in ids or edge.target not in ids:
                coverage.ignored_edges += 1
                if 'dangling_edges' not in coverage.reasons:
                    coverage.reasons.append('dangling_edges')
                continue
            if edge.kind == 'mentions':
                coverage.literal_mentions += 1
            if edge.kind == 'returned':
                coverage.retrieval_links += 1
            if edge.kind != 'relation' or edge.source not in adjacent or edge.target not in adjacent:
                coverage.ignored_edges += 1
                continue
            if not _recorded(edge):
                coverage.unverified_relations += 1
                coverage.ignored_edges += 1
                continue
            coverage.recorded_relations += 1
            coverage.inferred_relations += int(edge.data.get('origin') == 'inferred')
            if edge.source == edge.target:
                coverage.ignored_edges += 1
                continue
            pair = (edge.source, edge.target) if edge.source < edge.target else (edge.target, edge.source)
            pairs[pair] = pairs.get(pair, 0) + 1
            adjacent[edge.source].add(edge.target)
            adjacent[edge.target].add(edge.source)
            outgoing[edge.source].add(edge.target)
            incoming[edge.target].add(edge.source)
        if coverage.unverified_relations:
            coverage.reasons.append('unverified_relations')
        result.counts.concepts = len(adjacent)
        result.counts.topology_edges = len(pairs)
        neighbors = {node: tuple(sorted(targets)) for node, targets in adjacent.items()}
        _summarize(neighbors, incoming, outgoing, pairs, selected, result, work)
        coverage.reasons = sorted(set(coverage.reasons))
        result.status = 'partial' if coverage.reasons else 'complete'
        return result
    except _Limit as error:
        result.status = 'unavailable'
        result.coverage.reasons = sorted(set([*result.coverage.reasons, str(error)]))
        result.communities, result.hubs, result.bridges = [], [], []
        return result
