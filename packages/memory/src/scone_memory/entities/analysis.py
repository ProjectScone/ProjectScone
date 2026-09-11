"""Whole-space graph analysis over an entity projection.

Communities, central and bridging entities, surprising connections and
questions worth asking, all computed from recorded relations and nothing
else. Every figure is deterministic (sorted iteration, stable tie-breaks,
rounded floats), every surprise and suggestion lists the relations and facts
behind it, and anything left out for size is counted.

Communities come from a modularity optimisation written for this package:
nodes move one at a time, in id order, to the neighbouring community that
raises modularity most; communities then collapse into single nodes and the
moves repeat on the smaller graph until nothing improves. Each community is
finally split into its connected parts, so no community is two groups that
merely share a label. Weights are the number of facts behind each pair.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
from typing import Literal

from .project import Entity, EntityProjection

ANALYSIS_VERSION = "scone.analysis/1"
_EXACT_BETWEENNESS = 500
_SAMPLED_SOURCES = 64
_MAX_LEVELS = 10
_MAX_PASSES = 20
_PAGERANK_ITERATIONS = 100
_DAMPING = 0.85


@dataclass(frozen=True)
class Community:
    community_id: str
    label: str
    members: tuple[str, ...]
    top_entities: tuple[str, ...]
    internal_links: int
    boundary_links: int
    #: Share of member pairs that are directly linked; None for one member.
    cohesion: float | None
    kinds: tuple[tuple[str, int], ...]
    predicates: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Importance:
    entity_id: str
    community_id: str
    degree: int
    weight: int
    pagerank: float
    betweenness: float
    #: How evenly an entity's links spread across communities: 0 inside one.
    participation: float


@dataclass(frozen=True)
class Surprise:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    fact_ids: tuple[int, ...]
    communities: tuple[str, str]
    links_between_communities: int
    reason: str


@dataclass(frozen=True)
class Suggestion:
    kind: Literal["connection", "bridge", "kind_conflict", "isolated"]
    text: str
    entity_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    fact_ids: tuple[int, ...]


@dataclass(frozen=True)
class AnalysisCoverage:
    entities_total: int
    entities_analysed: int
    isolated_entities: int
    truncated: bool
    reasons: tuple[str, ...]
    betweenness: str
    levels: int


@dataclass(frozen=True)
class GraphAnalysis:
    projection_digest: str
    modularity: float
    communities: tuple[Community, ...]
    importance: tuple[Importance, ...]
    surprising_connections: tuple[Surprise, ...]
    suggestions: tuple[Suggestion, ...]
    coverage: AnalysisCoverage
    version: str = ANALYSIS_VERSION


Adjacency = dict[str, dict[str, int]]


def _round(value: float) -> float:
    return round(value, 9)


def _local_moves(graph: Adjacency) -> tuple[dict[str, str], bool]:
    total = sum(sum(neighbours.values()) for neighbours in graph.values())
    if total == 0:
        return {node: node for node in graph}, False
    degree = {node: sum(neighbours.values()) for node, neighbours in graph.items()}
    community = {node: node for node in graph}
    totals: dict[str, int] = dict(degree)
    moved_any = False
    for _ in range(_MAX_PASSES):
        moved = False
        for node in sorted(graph):
            current = community[node]
            links: dict[str, int] = defaultdict(int)
            for neighbour, weight in graph[node].items():
                if neighbour != node:
                    links[community[neighbour]] += weight
            totals[current] -= degree[node]
            best, best_gain = current, links.get(current, 0) - totals[current] * degree[node] / total
            for candidate in sorted(links):
                gain = links[candidate] - totals[candidate] * degree[node] / total
                if gain > best_gain + 1e-12:
                    best, best_gain = candidate, gain
            totals[best] += degree[node]
            if best != current:
                community[node] = best
                moved = moved_any = True
        if not moved:
            break
    return community, moved_any


def _aggregate(graph: Adjacency, community: dict[str, str]) -> Adjacency:
    # Every community is a node of the next level, including one with no
    # links (a kept entity whose neighbours all fell outside the budget).
    merged: Adjacency = defaultdict(lambda: defaultdict(int))
    for node in graph:
        merged[community[node]]
    for node, neighbours in graph.items():
        for neighbour, weight in neighbours.items():
            merged[community[node]][community[neighbour]] += weight
    return {node: dict(neighbours) for node, neighbours in merged.items()}


def _components(members: list[str], graph: Adjacency) -> list[list[str]]:
    inside, seen, parts = set(members), set(), []
    for start in sorted(members):
        if start in seen:
            continue
        part, queue = [], deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            part.append(node)
            for neighbour in sorted(graph[node]):
                if neighbour in inside and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        parts.append(sorted(part))
    return parts


def _partition(graph: Adjacency) -> tuple[list[list[str]], int]:
    assignment = {node: node for node in graph}
    level_graph, levels = graph, 0
    for _ in range(_MAX_LEVELS):
        moves, moved = _local_moves(level_graph)
        if not moved:
            break
        levels += 1
        assignment = {node: moves[label] for node, label in assignment.items()}
        level_graph = _aggregate(level_graph, moves)
    groups: dict[str, list[str]] = defaultdict(list)
    for node, label in assignment.items():
        groups[label].append(node)
    parts = [part for group in groups.values() for part in _components(group, graph)]
    return sorted(parts, key=lambda part: (-len(part), part[0])), levels


def _modularity(graph: Adjacency, membership: dict[str, str]) -> float:
    total = sum(sum(neighbours.values()) for neighbours in graph.values())
    if total == 0:
        return 0.0
    inside: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for node, neighbours in graph.items():
        for neighbour, weight in neighbours.items():
            degree[membership[node]] += weight
            if membership[node] == membership[neighbour]:
                inside[membership[node]] += weight
    return _round(sum(inside[c] / total - (degree[c] / total) ** 2 for c in degree))


def _pagerank(graph: Adjacency) -> dict[str, float]:
    nodes = sorted(graph)
    if not nodes:
        return {}
    count = len(nodes)
    rank = {node: 1.0 / count for node in nodes}
    strength = {node: sum(graph[node].values()) for node in nodes}
    for _ in range(_PAGERANK_ITERATIONS):
        spill = sum(rank[node] for node in nodes if strength[node] == 0)
        following = {node: (1 - _DAMPING) / count + _DAMPING * spill / count for node in nodes}
        for node in nodes:
            if strength[node]:
                share = _DAMPING * rank[node] / strength[node]
                for neighbour in sorted(graph[node]):
                    following[neighbour] += share * graph[node][neighbour]
        change = sum(abs(following[node] - rank[node]) for node in nodes)
        rank = following
        if change < 1e-12:
            break
    return {node: _round(value) for node, value in rank.items()}


def _betweenness(graph: Adjacency) -> tuple[dict[str, float], str]:
    nodes = sorted(graph)
    ordered = {node: tuple(neighbour for neighbour in sorted(graph[node]) if neighbour != node) for node in nodes}
    score = {node: 0.0 for node in nodes}
    if len(nodes) <= _EXACT_BETWEENNESS:
        sources, method = nodes, "exact"
    else:
        step = len(nodes) / _SAMPLED_SOURCES
        sources, method = [nodes[int(i * step)] for i in range(_SAMPLED_SOURCES)], f"sampled:{_SAMPLED_SOURCES}"
    for source in sources:
        order: list[str] = []
        parents: dict[str, list[str]] = {node: [] for node in nodes}
        paths = dict.fromkeys(nodes, 0)
        depth = dict.fromkeys(nodes, -1)
        paths[source], depth[source] = 1, 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in ordered[node]:
                if depth[neighbour] < 0:
                    depth[neighbour] = depth[node] + 1
                    queue.append(neighbour)
                if depth[neighbour] == depth[node] + 1:
                    paths[neighbour] += paths[node]
                    parents[neighbour].append(node)
        credit = dict.fromkeys(nodes, 0.0)
        for node in reversed(order):
            for parent in parents[node]:
                credit[parent] += paths[parent] / paths[node] * (1 + credit[node])
            if node != source:
                score[node] += credit[node]
    count = len(nodes)
    scale = (count / len(sources) if sources else 0) / ((count - 1) * (count - 2) or 1)
    return {node: _round(value * scale) for node, value in score.items()}, method


def _community_id(members: list[str]) -> str:
    return "com:" + hashlib.sha256("\x1f".join(members).encode("utf-8")).hexdigest()[:16]


def analyze_projection(projection: EntityProjection, *, max_entities: int = 20_000,
                       max_surprises: int = 20, max_suggestions: int = 10) -> GraphAnalysis:
    entities = {entity.entity_id: entity for entity in projection.entities}
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for relation in projection.relations:
        if relation.subject_id != relation.object_id:
            pair = tuple(sorted((relation.subject_id, relation.object_id)))
            pairs[(pair[0], pair[1])] += len(relation.fact_ids)
    strength: Counter[str] = Counter()
    for (left, right), weight in pairs.items():
        strength[left] += weight
        strength[right] += weight
    linked = sorted(strength, key=lambda node: (-strength[node], node))
    reasons: list[str] = []
    if len(linked) > max_entities:
        linked = linked[:max_entities]
        reasons.append("entity_limit")
    kept = set(linked)
    graph: Adjacency = {node: {} for node in sorted(kept)}
    for (left, right), weight in pairs.items():
        if left in kept and right in kept:
            graph[left][right] = weight
            graph[right][left] = weight

    parts, levels = _partition(graph)
    membership = {node: _community_id(part) for part in parts for node in part}
    rank = _pagerank(graph)
    between, method = _betweenness(graph)
    predicates_by_node: dict[str, Counter[str]] = defaultdict(Counter)
    for relation in projection.relations:
        if relation.subject_id in kept and relation.object_id in kept:
            predicates_by_node[relation.subject_id][relation.predicate] += len(relation.fact_ids)

    communities = []
    for part in parts:
        inside = set(part)
        internal = sum(1 for node in part for neighbour in graph[node] if neighbour in inside) // 2
        boundary = sum(1 for node in part for neighbour in graph[node] if neighbour not in inside)
        local = {node: sum(weight for neighbour, weight in graph[node].items() if neighbour in inside) for node in part}
        top = sorted(part, key=lambda node: (-local[node], -rank.get(node, 0.0), node))[:3]
        kinds: Counter[str] = Counter(str(entities[node].kind) for node in part if entities[node].kind)
        predicates = sum((predicates_by_node[node] for node in part), Counter())
        communities.append(Community(
            membership[part[0]], " · ".join(entities[node].label for node in top), tuple(part), tuple(top),
            internal, boundary,
            _round(2 * internal / (len(part) * (len(part) - 1))) if len(part) > 1 else None,
            tuple(sorted(kinds.items(), key=lambda item: (-item[1], item[0]))),
            tuple(sorted(predicates.items(), key=lambda item: (-item[1], item[0]))[:5])))

    importance = []
    for node in graph:
        spread: Counter[str] = Counter()
        for neighbour, weight in graph[node].items():
            spread[membership[neighbour]] += weight
        total = sum(spread.values())
        participation = 1 - sum((value / total) ** 2 for value in spread.values()) if total else 0.0
        importance.append(Importance(node, membership[node], len(graph[node]), strength[node], rank[node],
                                     between[node], _round(participation)))
    importance.sort(key=lambda item: (-item.pagerank, item.entity_id))

    between_communities: Counter[tuple[str, str]] = Counter()
    for (left, right) in pairs:
        if left in kept and right in kept and membership[left] != membership[right]:
            ends = sorted((membership[left], membership[right]))
            between_communities[(ends[0], ends[1])] += 1
    labels = {community.community_id: community.label for community in communities}
    surprises = []
    for relation in projection.relations:
        subject, obj = relation.subject_id, relation.object_id
        if subject not in kept or obj not in kept or membership[subject] == membership[obj]:
            continue
        ends = sorted((membership[subject], membership[obj]))
        links = between_communities[(ends[0], ends[1])]
        surprises.append((links, min(len(graph[subject]), len(graph[obj])), relation.relation_id, Surprise(
            relation.relation_id, subject, relation.predicate, obj, relation.fact_ids, (ends[0], ends[1]), links,
            f"{'the only link' if links == 1 else f'one of {links} links'} between "
            f"'{labels[membership[subject]]}' and '{labels[membership[obj]]}'")))
    surprises.sort(key=lambda item: item[:3])
    surprising = tuple(item[3] for item in surprises[:max_surprises])

    suggestions: list[Suggestion] = []
    for surprise in surprising[:3]:
        suggestions.append(Suggestion(
            "connection", f"How is {entities[surprise.subject_id].label} connected to {entities[surprise.object_id].label}?",
            (surprise.subject_id, surprise.object_id), (surprise.relation_id,), surprise.fact_ids))
    for item in sorted(importance, key=lambda item: (-item.participation, -item.degree, item.entity_id))[:3]:
        if item.participation < 0.3 or item.degree < 2:
            continue
        touched = sorted({membership[neighbour] for neighbour in graph[item.entity_id]} - {item.community_id})
        crossing = [relation for relation in projection.relations
                    if item.entity_id in (relation.subject_id, relation.object_id)
                    and relation.subject_id in kept and relation.object_id in kept
                    and membership[relation.subject_id] != membership[relation.object_id]]
        suggestions.append(Suggestion(
            "bridge", f"What role does {entities[item.entity_id].label} play between "
                      f"{' and '.join(repr(labels[community]) for community in [item.community_id, *touched][:3])}?",
            (item.entity_id,), tuple(relation.relation_id for relation in crossing[:8]),
            tuple(sorted({fact_id for relation in crossing for fact_id in relation.fact_ids}))[:8]))
    for entity in sorted(projection.entities, key=lambda entity: entity.entity_id):
        if entity.kind_status == "conflict":
            suggestions.append(Suggestion("kind_conflict", f"What kind of thing is {entity.label}?",
                                          (entity.entity_id,), (), entity.kind_basis))
    attributes_by_entity: dict[str, list[int]] = defaultdict(list)
    for attribute in projection.attributes:
        attributes_by_entity[attribute.entity_id].extend(attribute.fact_ids)
    isolated = [entity for entity in projection.entities if entity.entity_id not in strength]
    for entity in sorted(isolated, key=lambda entity: (-len(attributes_by_entity[entity.entity_id]), entity.entity_id)):
        if len(attributes_by_entity[entity.entity_id]) >= 2:
            suggestions.append(Suggestion("isolated", f"What is {entity.label} connected to?", (entity.entity_id,),
                                          (), tuple(sorted(attributes_by_entity[entity.entity_id]))[:8]))
    return GraphAnalysis(
        projection.digest, _modularity(graph, membership), tuple(communities), tuple(importance), surprising,
        tuple(suggestions[:max_suggestions]),
        AnalysisCoverage(len(strength), len(kept), len(isolated), bool(reasons), tuple(reasons), method, levels))
