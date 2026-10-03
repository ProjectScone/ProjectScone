"""Bounded graphs of exact read-only action/observation repetition.

A repeated state is an intervention signal, not a factuality judgment. Reports
contain numeric graph identities only; queries and tool outputs stay private.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class TrapNode:
    node_id: int
    first_round: int
    visits: int
    comparable: bool


@dataclass(frozen=True)
class TrapEdge:
    source: int
    target: int
    count: int


@dataclass(frozen=True)
class TrapGraph:
    nodes: tuple[TrapNode, ...]
    edges: tuple[TrapEdge, ...]
    path: tuple[int, ...]
    pattern: tuple[int, ...]
    repetitions: int


class AgentTrapDetected(RuntimeError):
    def __init__(self, graph: TrapGraph) -> None:
        super().__init__('agent_no_progress')
        self.graph = graph


class ObservationGraph:
    """One invocation, at most sixteen tool rounds, no execution authority."""

    def __init__(self, repetitions: int) -> None:
        if type(repetitions) is not int or not 2 <= repetitions <= 16:
            raise ValueError('invalid repetition threshold')
        self._repetitions = repetitions
        self._states: dict[bytes, int] = {}
        self._path: list[int] = []
        self._comparable: dict[int, bool] = {}
        self._segment_start = 0

    def observe(self, observation: bytes | None) -> TrapGraph | None:
        if len(self._path) >= 16:
            raise ValueError('observation graph round limit')
        if observation is not None and not isinstance(observation, bytes):
            raise ValueError('invalid observation')
        if observation is not None:
            key = hashlib.sha256(observation).digest()
            node = self._states.get(key)
            if node is None:
                node = len(self._comparable) + 1
                self._states[key] = node
        else:
            node = len(self._comparable) + 1
            self._segment_start = len(self._path) + 1
        self._comparable[node] = observation is not None
        self._path.append(node)
        segment = self._path[self._segment_start:]
        for size in range(1, len(segment) // self._repetitions + 1):
            pattern = segment[-size:]
            if segment[-size * self._repetitions:] == pattern * self._repetitions:
                return self.snapshot(tuple(pattern))
        return None

    def snapshot(self, pattern: tuple[int, ...] = ()) -> TrapGraph:
        nodes = tuple(TrapNode(node, self._path.index(node) + 1, self._path.count(node), comparable)
                      for node, comparable in self._comparable.items())
        pairs: dict[tuple[int, int], int] = {}
        for pair in zip(self._path, self._path[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1
        return TrapGraph(nodes, tuple(TrapEdge(*pair, count) for pair, count in pairs.items()),
                         tuple(self._path), pattern, self._repetitions if pattern else 0)


def validate_trap_graph(graph: TrapGraph) -> None:
    """Verify a bounded report against its path before publishing or replaying it."""
    if not isinstance(graph, TrapGraph) or any(
        type(value) is not tuple for value in (graph.nodes, graph.edges, graph.path, graph.pattern)
    ):
        raise ValueError('invalid trap graph')
    if not 2 <= len(graph.path) <= 16 or not 1 <= len(graph.pattern) <= 8 or len(graph.edges) > 15:
        raise ValueError('invalid trap graph size')
    if type(graph.repetitions) is not int or not 2 <= graph.repetitions <= 16:
        raise ValueError('invalid trap repetitions')
    if any(type(node) is not int or not 1 <= node <= 16 for node in (*graph.path, *graph.pattern)):
        raise ValueError('invalid trap path')
    order = tuple(dict.fromkeys(graph.path))
    if order != tuple(range(1, len(order) + 1)) or len(graph.nodes) != len(order):
        raise ValueError('invalid trap node identities')
    visits = Counter(graph.path)
    for expected, node in zip(order, graph.nodes):
        if not isinstance(node, TrapNode) or any(
            type(value) is not int for value in (node.node_id, node.first_round, node.visits)
        ) or type(node.comparable) is not bool:
            raise ValueError('invalid trap node')
        if (node.node_id, node.first_round, node.visits) != (
            expected, graph.path.index(expected) + 1, visits[expected]
        ) or (not node.comparable and node.visits != 1):
            raise ValueError('inconsistent trap node')
    expected_edges = Counter(zip(graph.path, graph.path[1:]))
    edges: dict[tuple[int, int], int] = {}
    for edge in graph.edges:
        if not isinstance(edge, TrapEdge) or any(
            type(value) is not int for value in (edge.source, edge.target, edge.count)
        ) or (edge.source, edge.target) in edges:
            raise ValueError('invalid trap edge')
        edges[edge.source, edge.target] = edge.count
    if edges != dict(expected_edges):
        raise ValueError('inconsistent trap edges')
    repeated = graph.pattern * graph.repetitions
    if len(repeated) > len(graph.path) or graph.path[-len(repeated):] != repeated:
        raise ValueError('inconsistent trap repetition')
    if any(not graph.nodes[node - 1].comparable for node in graph.pattern):
        raise ValueError('trap crosses incomparable observation')
