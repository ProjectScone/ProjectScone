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

from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
import os
from typing import Literal, Sequence

from .project import Entity, EntityProjection

ANALYSIS_VERSION = "scone.analysis/2"
#: A community over this share of the analysed graph is re-partitioned on its own links...
_MAX_SHARE = 0.25
#: ...once it has at least this many members, so a small graph is never split for being small.
_MIN_SPLIT = 10
#: A community of at least _NESTED_MIN members is re-partitioned on its own links when that
#: partition reaches this modularity: it holds communities of its own.
_NESTED_MODULARITY = 0.3
_NESTED_MIN = 50
_EXACT_BETWEENNESS = 500
_SAMPLED_SOURCES = 64
_MAX_LEVELS = 10
_MAX_PASSES = 20
_PAGERANK_ITERATIONS = 100
_DAMPING = 0.85


@dataclass(frozen=True)
class Community:
    """``members`` holds the graph's own members and, after them, the
    externals attached for reading; the link counts and cohesion are of
    the own members alone, since an attached external is not a tie the
    partition made."""

    community_id: str
    label: str
    members: tuple[str, ...]
    top_entities: tuple[str, ...]
    internal_links: int
    boundary_links: int
    #: Share of own-member pairs that are directly linked; None for one member.
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
    #: Named by the graph and never read by it: the object of imports,
    #: dependencies or citations that is the subject of nothing (`typing`,
    #: `pydantic.BaseModel`, `ADR-12`). Ranked apart from the graph's own.
    external: bool = False


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
    #: The modularity resolution the communities were found at.
    resolution: float = 1.0
    #: Entities named but never read (see `Importance.external`): kept out
    #: of the partition and the central ranking, attached to the community
    #: of what names them most, and listed apart.
    external_entities: int = 0
    #: The degree percentile above which the graph's own entities were held
    #: out of the partition as hubs, and how many were.
    exclude_hubs: float | None = None
    hubs_held_apart: int = 0
    #: Communities over a quarter of the graph split on their own links.
    split_oversized: int = 0
    #: Large communities holding communities of their own, split on their own links.
    split_nested: int = 0
    #: Communities a guard looked at whose own partition was one piece, or (for a
    #: large one) too weak to be structure, kept whole.
    unsplittable: int = 0
    #: With ``detach_hubs``: the degree percentile above which entities were
    #: left out while communities were found, and how many were.
    detach_hubs: float | None = None
    hubs_detached: int = 0
    #: When a guard split a community: the modularity of the partition before
    #: it did, beside the final one. Finer communities often score lower on
    #: modularity over the whole graph, and a reader weighing them should see
    #: both. None when no guard split anything.
    modularity_before_guards: float | None = None

    def record(self) -> dict[str, object]:
        """What the analysis covered, for a caller to show beside its scores:
        apart from any paging of the view that shows them."""
        return {"entities_total": self.entities_total, "entities_analysed": self.entities_analysed,
                "isolated_entities": self.isolated_entities, "truncated": self.truncated,
                "reasons": list(self.reasons), "betweenness": self.betweenness,
                "betweenness_estimated": self.betweenness != "exact", "levels": self.levels,
                "resolution": self.resolution, "external_entities": self.external_entities,
                "exclude_hubs": self.exclude_hubs, "hubs_held_apart": self.hubs_held_apart,
                "split_oversized": self.split_oversized,
                "split_nested": self.split_nested, "unsplittable": self.unsplittable,
                "detach_hubs": self.detach_hubs, "hubs_detached": self.hubs_detached,
                "modularity_before_guards": self.modularity_before_guards}


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
    #: The entities named but never read, by id.
    external: frozenset[str] = frozenset()
    #: The graph's own entities held out of the partition as hubs, by id.
    hubs: frozenset[str] = frozenset()


#: A relation whose object can be something the graph only names: a module
#: imported, a package depended on, a document cited, a type used.
NAMED_ONLY = frozenset(("imports", "imports_when_called", "imports_for_types", "depends_on", "develops_with", "cites",
                        "uses_type", "references", "runs_with", "requires_env", "connects_to"))


def external_entities(projection: EntityProjection) -> frozenset[str]:
    """The entities a graph names and never reads: objects of imports,
    dependencies, citations or type uses that are the subject of no
    relation at all. `typing` is imported by every file and defines
    nothing here; a module of the codebase is imported too, but it also
    defines its own things, so it is the graph's own. Deterministic, from
    the relations alone, and the reason a report's central entities are
    the codebase's and not the standard library's."""
    subjects = {relation.subject_id for relation in projection.relations}
    return frozenset(relation.object_id for relation in projection.relations
                     if relation.predicate in NAMED_ONLY and relation.object_id not in subjects)


Adjacency = dict[str, dict[str, int]]


def _round(value: float) -> float:
    return round(value, 9)


def _local_moves(graph: Adjacency, resolution: float) -> tuple[dict[str, str], bool]:
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
            best, best_gain = current, links.get(current, 0) - resolution * totals[current] * degree[node] / total
            for candidate in sorted(links):
                gain = links[candidate] - resolution * totals[candidate] * degree[node] / total
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


def _partition(graph: Adjacency, resolution: float) -> tuple[list[list[str]], int]:
    assignment = {node: node for node in graph}
    level_graph, levels = graph, 0
    for _ in range(_MAX_LEVELS):
        moves, moved = _local_moves(level_graph, resolution)
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


def _induced(graph: Adjacency, members: list[str]) -> Adjacency:
    inside = set(members)
    return {node: {other: weight for other, weight in graph[node].items() if other in inside} for node in members}


def _guarded(graph: Adjacency, parts: list[list[str]], resolution: float) -> tuple[list[list[str]], dict[str, int]]:
    """``parts`` with two kinds of community re-partitioned on their own links, at the same resolution.

    - One over a quarter of ``graph`` (and at least ``_MIN_SPLIT`` members) is
      split into whatever pieces its own links give, since a map by community
      draws it as one blob.
    - One of at least ``_NESTED_MIN`` members is split when its own partition
      reaches ``_NESTED_MODULARITY``: modularity over a large graph merges
      small modules into one community, and this one holds several.

    The share of member pairs linked is not the test for the second: in a
    sparse graph it falls with size, about 2/n, so it fires on nearly every
    large community whatever its structure. The resolution is never raised to
    force a split, because high enough it cuts a tight clique. A community a
    guard looked at and kept whole is counted, so a guard never reads as
    having found structure it did not. The pieces of a split are looked at
    in turn: a partition of one community can still leave a piece holding
    two."""
    fired = {"split_oversized": 0, "split_nested": 0, "unsplittable": 0}
    largest = max(_MIN_SPLIT, len(graph) * _MAX_SHARE)
    guarded: list[list[str]] = []
    waiting = list(parts)
    while waiting:
        part = waiting.pop()
        oversized = len(part) > largest
        if not oversized and len(part) < _NESTED_MIN:
            guarded.append(part)
            continue
        inside = _induced(graph, part)
        pieces, _ = _partition(inside, resolution)
        strong = len(pieces) > 1 and _modularity(
            inside, {node: str(index) for index, piece in enumerate(pieces) for node in piece}, resolution
        ) >= _NESTED_MODULARITY
        if len(pieces) < 2 or not (oversized or strong):
            fired["unsplittable"] += 1
            guarded.append(part)
            continue
        fired["split_oversized" if oversized else "split_nested"] += 1
        waiting.extend(pieces)  # each piece is smaller than its part, so this ends
    return sorted(guarded, key=lambda part: (-len(part), part[0])), fired


def _rejoined(graph: Adjacency, parts: list[list[str]], hubs: frozenset[str]) -> list[list[str]]:
    """``parts`` with each hub, in name order, added to the part most of its link weight goes to."""
    joined = [list(part) for part in parts]
    for hub in sorted(hubs):
        index_of = {node: index for index, part in enumerate(joined) for node in part}
        pull: Counter[int] = Counter()
        for other, weight in graph[hub].items():
            if other in index_of:
                pull[index_of[other]] += weight
        if pull:
            joined[min(pull, key=lambda index: (-pull[index], index))].append(hub)
        else:
            joined.append([hub])
    return joined


def _modularity(graph: Adjacency, membership: dict[str, str], resolution: float = 1.0) -> float:
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
    return _round(sum(inside[c] / total - resolution * (degree[c] / total) ** 2 for c in degree))


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
    # Scaling a sample up to the whole graph can overshoot; no node lies on
    # more than every shortest path, so an estimate stops at 1.
    return {node: _round(min(1.0, value * scale)) for node, value in score.items()}, method


def _community_id(members: list[str]) -> str:
    return "com:" + hashlib.sha256("\x1f".join(members).encode("utf-8")).hexdigest()[:16]


#: Analyses kept by projection digest and resolution: one is the same for an
#: unchanged graph, and the work grows with the graph.
_KEPT = 8
_ANALYSES: OrderedDict[tuple[str, float, float | None], GraphAnalysis] = OrderedDict()


def cached_analysis(projection: EntityProjection, resolution: float = 1.0,
                    exclude_hubs: float | None = None, detach_hubs: float | None = None) -> GraphAnalysis:
    """The projection's analysis, computed once per digest, resolution and
    hub percentiles."""
    key = (projection.digest, float(resolution), None if exclude_hubs is None else float(exclude_hubs),
           None if detach_hubs is None else float(detach_hubs))
    if key in _ANALYSES:
        _ANALYSES.move_to_end(key)
        return _ANALYSES[key]
    found = _ANALYSES[key] = analyze_projection(projection, resolution=resolution, exclude_hubs=exclude_hubs,
                                                detach_hubs=detach_hubs)
    while len(_ANALYSES) > _KEPT:
        _ANALYSES.popitem(last=False)
    return found


def _hubs_above(graph: Adjacency, external: frozenset[str], percentile: float | None) -> frozenset[str]:
    """The graph's own entities with more neighbours than the given
    percentile of them (nearest rank), or none when no percentile is set."""
    if percentile is None:
        return frozenset()
    own = [node for node in graph if node not in external]
    if not own:
        return frozenset()
    degrees = sorted(len(graph[node]) for node in own)
    cut = degrees[max(0, math.ceil(len(degrees) * percentile / 100) - 1)]
    return frozenset(node for node in own if len(graph[node]) > cut)


def _place(labels: Sequence[str]) -> str | None:
    """The directory a community of files is named by: the deepest one
    that holds at least half of its paths, when at least two members are
    paths and most members are. A community of files is what its
    directory is called, not the three file names that happen to lead it;
    and a community that spans two packages is named by the one most of
    it sits in, since a shared root would name nothing."""
    paths = [label.split(":", 1)[0] for label in labels if "/" in label.split(":", 1)[0]]
    if len(paths) < 2 or len(paths) * 2 < len(labels):
        return None
    holding: Counter[str] = Counter()
    for path in paths:
        parts = path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            holding["/".join(parts[:depth])] += 1
    enough = max(2, -(-len(paths) // 2))
    deep = [place for place, count in holding.items() if count >= enough]
    return max(deep, key=lambda place: (place.count("/"), -holding[place], place)) if deep else None


def _label(names: Sequence[str], members: Sequence[str]) -> str:
    """A community's name: the directory its files share, then its most
    central members with that directory left off; or the members alone."""
    place = _place(members)
    if place is None:
        return " · ".join(names)
    return " · ".join([place + "/", *(name[len(place) + 1:] if name.startswith(place + "/") else name for name in names)])


def analyze_projection(projection: EntityProjection, *, max_entities: int = 20_000,
                       max_surprises: int = 20, max_suggestions: int = 10, resolution: float = 1.0,
                       exclude_hubs: float | None = None, detach_hubs: float | None = None) -> GraphAnalysis:
    """``resolution`` sets how fine the communities are: above 1 favours
    smaller ones, below 1 larger ones, as in modularity with a resolution.
    ``exclude_hubs`` (a degree percentile, 50 to 100) holds the graph's own
    entities with more neighbours than that out of the partition, as
    externals are held: a base class every file inherits, or a person every
    note mentions, would otherwise glue every community into one.

    ``detach_hubs``, a degree percentile from 50 to 100, leaves the entities
    linked to more neighbours than that out only while communities are
    found, so an entity everything touches does not pull unrelated groups
    into one; each then joins the community most of its link weight goes
    to, as a full member. Detached hubs are picked among what is left once
    externals and held-apart hubs are out."""
    if not resolution > 0:
        raise ValueError("resolution must be positive")
    if exclude_hubs is not None and not 50 <= exclude_hubs <= 100:
        raise ValueError("exclude_hubs must be a percentile from 50 to 100")
    if detach_hubs is not None and (not isinstance(detach_hubs, (int, float)) or not 50 <= detach_hubs <= 100):
        raise ValueError("detach_hubs is a degree percentile from 50 to 100")
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

    # What the graph names and never reads is kept out of the partition:
    # `typing`, imported by every file, would otherwise be the strongest
    # tie between any two communities and the most central thing in each.
    external = external_entities(projection) & kept
    # A hub is held apart the same way when asked: out of the partition,
    # the naming, the surprises and the counts between communities, and
    # attached afterwards to the community it links most.
    hubs = _hubs_above(graph, external, exclude_hubs)
    apart = external | hubs
    own: Adjacency = {node: {other: weight for other, weight in graph[node].items() if other not in apart}
                      for node in graph if node not in apart}
    # Detached hubs are the graph's own entities with the most links to each
    # other, among what is left once externals and held-apart hubs are out:
    # they leave only while communities are found, and rejoin as members.
    detached = _hubs_above(own, frozenset(), detach_hubs)
    without = {node: {other: weight for other, weight in neighbours.items() if other not in detached}
               for node, neighbours in own.items() if node not in detached}
    parts, levels = _partition(without, resolution)
    found = parts
    parts, fired = _guarded(without, parts, resolution)
    parts = _rejoined(own, parts, detached)
    # Measured like the final modularity, over the graph's own entities with the detached hubs rejoined,
    # so the two compare.
    before = (_modularity(own, {node: str(index) for index, part in enumerate(_rejoined(own, found, detached))
                                for node in part}, resolution)
              if fired["split_oversized"] or fired["split_nested"] else None)
    parts = sorted((sorted(part) for part in parts), key=lambda part: (-len(part), part[0]))
    membership = {node: _community_id(part) for part in parts for node in part}
    # An external belongs, for reading, with the community that names it
    # most (by the weight of what names it; a tie to the smaller id). One
    # whose namers were all cut by the entity budget stands alone, as any
    # linkless node did before.
    attached: dict[str, list[str]] = defaultdict(list)
    alone: list[list[str]] = []
    for node in sorted(apart):
        naming: Counter[str] = Counter()
        for other, weight in graph[node].items():
            if other in membership and other not in apart:
                naming[membership[other]] += weight
        if naming:
            membership[node] = min(naming, key=lambda community: (-naming[community], community))
            attached[membership[node]].append(node)
        else:
            membership[node] = _community_id([node])
            alone.append([node])
    parts = [[*part, *attached[membership[part[0]]]] for part in parts] + alone
    parts.sort(key=lambda part: (-len(part), part[0]))
    rank = _pagerank(graph)
    between, method = _betweenness(graph)
    predicates_by_node: dict[str, Counter[str]] = defaultdict(Counter)
    for relation in projection.relations:
        if relation.subject_id in kept and relation.object_id in kept:
            predicates_by_node[relation.subject_id][relation.predicate] += len(relation.fact_ids)

    communities = []
    for part in parts:
        # Links are counted among the graph's own members: an attached
        # external is not a tie the partition made, and `typing` imported
        # from two communities is not a link between them.
        members = [node for node in part if node not in apart] or list(part)
        inside = set(members)
        internal = sum(1 for node in members for neighbour in graph[node] if neighbour in inside) // 2
        boundary = sum(1 for node in members for neighbour in graph[node]
                       if neighbour not in inside and neighbour not in apart)
        local = {node: sum(weight for neighbour, weight in graph[node].items() if neighbour in inside) for node in part}
        # Named by the graph's own members: a community is not called `typing`.
        top = sorted(members, key=lambda node: (-local[node], -rank.get(node, 0.0), node))[:3]
        kinds: Counter[str] = Counter(str(entities[node].kind) for node in part if entities[node].kind)
        predicates = sum((predicates_by_node[node] for node in part), Counter())
        communities.append(Community(
            membership[part[0]], _label([entities[node].label for node in top], [entities[node].label for node in members]),
            tuple(part), tuple(top),
            internal, boundary,
            _round(2 * internal / (len(members) * (len(members) - 1))) if len(members) > 1 else None,
            tuple(sorted(kinds.items(), key=lambda item: (-item[1], item[0]))),
            tuple(sorted(predicates.items(), key=lambda item: (-item[1], item[0]))[:5])))

    importance = []
    for node in graph:
        spread: Counter[str] = Counter()
        for neighbour, weight in graph[node].items():
            # What an entity imports from outside the graph spreads it
            # across nothing: only ties to the graph's own count.
            if neighbour not in apart:
                spread[membership[neighbour]] += weight
        total = sum(spread.values())
        participation = (1 - sum((value / total) ** 2 for value in spread.values())
                         if total and node not in apart else 0.0)
        importance.append(Importance(node, membership[node], len(graph[node]), strength[node], rank[node],
                                     between[node], _round(participation), node in external))
    importance.sort(key=lambda item: (-item.pagerank, item.entity_id))

    between_communities: Counter[tuple[str, str]] = Counter()
    for (left, right) in pairs:
        if left in apart or right in apart:
            continue
        if left in kept and right in kept and membership[left] != membership[right]:
            ends = sorted((membership[left], membership[right]))
            between_communities[(ends[0], ends[1])] += 1
    labels = {community.community_id: community.label for community in communities}
    surprises = []
    for relation in projection.relations:
        subject, obj = relation.subject_id, relation.object_id
        if subject not in kept or obj not in kept or membership[subject] == membership[obj]:
            continue
        if obj in apart or subject in apart:
            # Everything imports `typing`; that a second community does too
            # is no surprise, and neither is a link to a hub held apart.
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
    for item in sorted((item for item in importance if item.entity_id not in apart),
                       key=lambda item: (-item.participation, -item.degree, item.entity_id))[:3]:
        if item.participation < 0.3 or item.degree < 2:
            continue
        touched = sorted({membership[neighbour] for neighbour in graph[item.entity_id]} - {item.community_id})
        crossing = [relation for relation in projection.relations
                    if item.entity_id in (relation.subject_id, relation.object_id)
                    and relation.subject_id in kept and relation.object_id in kept
                    and relation.object_id not in apart
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
        projection.digest, _modularity(own, {node: membership[node] for node in own}, resolution),
        tuple(communities), tuple(importance), surprising,
        tuple(suggestions[:max_suggestions]),
        AnalysisCoverage(len(strength), len(kept), len(isolated), bool(reasons), tuple(reasons), method, levels,
                         resolution, len(external), exclude_hubs=exclude_hubs, hubs_held_apart=len(hubs),
                         split_oversized=fired["split_oversized"], split_nested=fired["split_nested"],
                         unsplittable=fired["unsplittable"], detach_hubs=detach_hubs, hubs_detached=len(detached),
                         modularity_before_guards=before), external=frozenset(external), hubs=hubs)
