"""Bounded, payload-free trajectory reports for standalone clients."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ._wire import boolean, integer, invalid, items, record


@dataclass(frozen=True)
class TrapNode:
    node_id: int
    first_round: int
    visits: int
    comparable: bool

    @classmethod
    def from_json(cls, value: object) -> TrapNode:
        row = record(value)
        if set(row) != {'node_id', 'first_round', 'visits', 'comparable'}:
            raise invalid('trap node fields')
        return cls(integer(row['node_id'], 1, 16), integer(row['first_round'], 1, 16),
                   integer(row['visits'], 1, 16), boolean(row['comparable']))


@dataclass(frozen=True)
class TrapEdge:
    source: int
    target: int
    count: int

    @classmethod
    def from_json(cls, value: object) -> TrapEdge:
        row = record(value)
        if set(row) != {'source', 'target', 'count'}:
            raise invalid('trap edge fields')
        return cls(integer(row['source'], 1, 16), integer(row['target'], 1, 16),
                   integer(row['count'], 1, 15))


@dataclass(frozen=True)
class TrapGraph:
    nodes: tuple[TrapNode, ...]
    edges: tuple[TrapEdge, ...]
    path: tuple[int, ...]
    pattern: tuple[int, ...]
    repetitions: int

    @classmethod
    def from_json(cls, value: object) -> TrapGraph:
        row = record(value)
        if set(row) != {'nodes', 'edges', 'path', 'pattern', 'repetitions'}:
            raise invalid('trap graph fields')
        result = cls(
            tuple(TrapNode.from_json(node) for node in items(row['nodes'], 16)),
            tuple(TrapEdge.from_json(edge) for edge in items(row['edges'], 15)),
            tuple(integer(node, 1, 16) for node in items(row['path'], 16)),
            tuple(integer(node, 1, 16) for node in items(row['pattern'], 8)),
            integer(row['repetitions'], 2, 16),
        )
        if len(result.path) < 2 or not result.pattern:
            raise invalid('trap graph size')
        order = tuple(dict.fromkeys(result.path))
        if order != tuple(range(1, len(order) + 1)) or len(result.nodes) != len(order):
            raise invalid('trap node identities')
        visits = Counter(result.path)
        for expected, node in zip(order, result.nodes):
            if (node.node_id, node.first_round, node.visits) != (
                expected, result.path.index(expected) + 1, visits[expected]
            ) or (not node.comparable and node.visits != 1):
                raise invalid('trap node counts')
        transitions = Counter(zip(result.path, result.path[1:]))
        edges = {(edge.source, edge.target): edge.count for edge in result.edges}
        if len(edges) != len(result.edges) or edges != dict(transitions):
            raise invalid('trap transitions')
        repeated = result.pattern * result.repetitions
        if len(repeated) > len(result.path) or result.path[-len(repeated):] != repeated:
            raise invalid('trap repetition')
        if any(not result.nodes[node - 1].comparable for node in result.pattern):
            raise invalid('trap comparability')
        return result
