"""Bounded graphs of exact read-only action/observation repetition.

A repeated state is an intervention signal, not a factuality judgment. Reports
contain numeric graph identities only; queries and tool outputs stay private.
"""

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
