"""What runs when this runs, and who reaches it.

``affected`` answers "what breaks if I change this": a set, walked
backwards over every kind of dependency this graph records. A call flow
answers the other question a code graph is asked, and ours could not:
**follow the calls, from here**. Downstream is what a declaration
reaches when it runs; upstream is who reaches it. The reference draws
this as a call-flow diagram; here it is a read with a drawing beside it,
so a caller can have the chain as data or as a picture.

Four rules, each with a test:

- **One predicate.** Only ``calls``. An import is not a call: a file
  that imports another may never run a line of it, and a flow that mixed
  the two would draw control that never happens.
- **Direction is kept, both ways.** ``A calls B`` puts B downstream of A
  and A upstream of B, and neither side may leak into the other. Walking
  undirected would answer both questions wrongly at once.
- **A cycle is drawn once and ends.** Recursion is ordinary code, not an
  error and not an infinite walk.
- **Every bound says it bit.** Hops each way and a node count stop the
  walk, and each says so, because "the flow stops here" and "we stopped
  looking here" are different answers and only one of them is about the
  code.

Nothing here calls a model, and an empty flow means nothing *in this
graph* calls or is called -- never that nothing does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable, Optional

from ..core.errors import InvalidInput
from ..memory.engine import check_space
from .affected import _read_graph, labels_of
from .view import StatusMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: The one predicate a flow follows.
CALLS = "calls"
#: Hops a flow may take in either direction.
MAX_HOPS = 8
#: Declarations a flow may name, over both directions.
MAX_REACHED = 500
#: Bytes of a name; echoed in the answer, so bounded like affected's.
MAX_NAME = 200


@dataclass(frozen=True)
class Step:
    """One declaration the flow reached, and the call that reached it."""

    label: str
    hop: int
    predicate: str
    fact_id: Optional[int] = None
    quote: str = ""
    #: The declaration this one was reached from.
    through: str = ""

    def record(self) -> dict[str, object]:
        return {"label": self.label, "hop": self.hop, "predicate": self.predicate,
                "through": self.through, **({"fact_id": self.fact_id} if self.fact_id is not None else {}),
                **({"quote": self.quote} if self.quote else {})}


@dataclass(frozen=True)
class Flow:
    """A call flow: what the root reaches, and what reaches it."""

    root: str
    status: str
    downstream: list[Step] = field(default_factory=list)
    upstream: list[Step] = field(default_factory=list)
    #: Which bounds stopped the walk, by name; a bound that did not bite is absent.
    bounds: dict[str, bool] = field(default_factory=dict)
    why: str = ""
    mode: str = "current"
    as_of: str = ""
    settled: bool = True

    def record(self) -> dict[str, object]:
        return {"root": self.root, "status": self.status, "why": self.why,
                "downstream": [step.record() for step in self.downstream],
                "upstream": [step.record() for step in self.upstream],
                "bounds": dict(self.bounds), "mode": self.mode, "as_of": self.as_of,
                "settled": self.settled}


def _whole(value: object, name: str, top: int) -> int:
    if type(value) is not int or not 1 <= value <= top:
        raise InvalidInput(f"{name} is a whole number in 1..={top}, got {value!r}")
    return value


def module_form(label: str) -> str:
    """The dotted name a declaration is called by from another file.

    A declaration is labelled by its file (``pkg/api.py:put``) and a call
    to it from elsewhere is recorded under the name the caller imported
    (``pkg.api.put``), because that is what the caller's own source says.
    They are two labels for one thing, and nothing joined them: a chain
    reached the edge of its file and stopped. This is the join, and it is
    a pure function of the label so it can be checked on its own.
    """
    path, _, rest = label.partition(":")
    if not rest or "/" not in path and not path.endswith(".py"):
        return ""
    stem = path[:-len(".py")] if path.endswith(".py") else path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return f"{stem.replace('/', '.')}.{rest}"


def _joined(labels: dict[str, str]) -> dict[str, str]:
    """Entity id -> the declaration's id it names, where exactly one
    declaration answers to that dotted name.

    The single-definition rule this graph holds everywhere else: two
    declarations that would answer to one imported name join to neither,
    because an edge nobody can check is worse than no edge.
    """
    by_form: dict[str, list[str]] = {}
    for entity_id, label in labels.items():
        form = module_form(label)
        if form:
            by_form.setdefault(form, []).append(entity_id)
    declared = {form: ids[0] for form, ids in by_form.items() if len(ids) == 1}
    return {entity_id: declared[label] for entity_id, label in labels.items()
            if label in declared and declared[label] != entity_id}


def _edges(projection: object, labels: dict[str, str]) -> tuple[
        dict[str, list[tuple[str, object]]], dict[str, list[tuple[str, object]]], int]:
    """The call graph, forwards and backwards, by entity id, with an
    imported name standing for the declaration it names. Returns the
    number of edges that join crossed."""
    joins = _joined(labels)
    forwards: dict[str, list[tuple[str, object]]] = {}
    backwards: dict[str, list[tuple[str, object]]] = {}
    crossed = 0
    for relation in projection.relations:  # type: ignore[attr-defined]
        if relation.predicate != CALLS:
            continue
        subject, obj = relation.subject_id, relation.object_id
        joined_to = joins.get(obj)
        if joined_to is not None:
            obj, crossed = joined_to, crossed + 1
        forwards.setdefault(subject, []).append((obj, relation))
        backwards.setdefault(obj, []).append((subject, relation))
    return forwards, backwards, crossed


def _walk(start: str, edges: dict[str, list[tuple[str, object]]], labels: dict[str, str],
          hops: int, room: int) -> tuple[list[Step], bool, bool]:
    """Breadth-first from ``start``, nearest first. Returns the steps and
    whether the hop bound and the node bound each bit."""
    seen = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    steps: list[Step] = []
    hop_bound = node_bound = False
    while queue:
        entity_id, depth = queue.popleft()
        if depth >= hops:
            hop_bound = hop_bound or bool(edges.get(entity_id))
            continue
        for other, relation in edges.get(entity_id, []):
            if other in seen:
                continue  # a cycle is drawn once
            if len(steps) >= room:
                node_bound = True
                return steps, hop_bound, node_bound
            seen.add(other)
            fact = next(iter(getattr(relation, "fact_ids", ()) or ()), None)
            steps.append(Step(labels.get(other, other), depth + 1, CALLS, fact,
                              getattr(relation, "quote", "") or "", labels.get(entity_id, entity_id)))
            queue.append((other, depth + 1))
    return steps, hop_bound, node_bound


async def call_flow(engine: "MemoryEngine", space: str, name: str, *, downstream_hops: int = 3,
                    upstream_hops: int = 3, max_reached: int = MAX_REACHED,
                    when: Optional[datetime] = None, mode: StatusMode = "current") -> Flow:
    """The calls out of ``name`` and the calls into it, nearest first.

    Read at ``mode`` and ``when``, both on the answer, for the reason
    ``affected`` gives: a flow is a question about the calls that hold
    now, and reading every edge ever recorded would draw control that was
    removed years ago.
    """
    check_space(space)
    _whole(downstream_hops, "downstream_hops", MAX_HOPS)
    _whole(upstream_hops, "upstream_hops", MAX_HOPS)
    _whole(max_reached, "max_reached", MAX_REACHED)
    if not isinstance(name, str) or not name.strip():
        raise InvalidInput("name the declaration to follow the calls from")
    cost = len(name.encode())
    if cost > MAX_NAME:
        raise InvalidInput(f"a name is at most {MAX_NAME} bytes, not {cost}")

    moment = when or datetime.now(timezone.utc)
    projection, _covered, moved = await _read_graph(engine, space, mode, moment)
    stamp = moment.isoformat(timespec="seconds").replace("+00:00", "Z")
    labels = {entity.entity_id: entity.label for entity in projection.entities}
    start = next((entity_id for entity_id, label in labels.items() if label == name), "")
    if not start:
        return Flow(name, "unknown", why=f"no declaration called {name!r} is in this graph; it may exist "
                                         f"in code this memory never read", mode=str(mode), as_of=stamp,
                    settled=not moved)
    forwards, backwards, crossed = _edges(projection, labels)
    down, down_hop, down_nodes = _walk(start, forwards, labels, downstream_hops, max_reached)
    up, up_hop, up_nodes = _walk(start, backwards, labels, upstream_hops, max(1, max_reached - len(down)))
    bounds = {name: True for name, bit in (("downstream_hops", down_hop), ("upstream_hops", up_hop),
                                           ("max_reached", down_nodes or up_nodes)) if bit}
    if not down and not up:
        why = (f"nothing in this graph calls {name!r} and it calls nothing here; a call this memory "
               f"never read is a call it cannot draw")
        return Flow(name, "empty", bounds=bounds, why=why, mode=str(mode), as_of=stamp, settled=not moved)
    reached = f"{len(down)} called, {len(up)} calling"
    if crossed:
        reached += f"; {crossed} call edge(s) joined an imported name to the declaration it names"
    why = f"{reached}. " + ("A bound stopped the walk, so there is further to go: "
                            + ", ".join(sorted(bounds)) if bounds else
                            "No bound bit: this is every call edge this graph holds from here.")
    return Flow(name, "found", downstream=down, upstream=up, bounds=bounds, why=why,
                mode=str(mode), as_of=stamp, settled=not moved)


def _node(label: str) -> str:
    """A Mermaid node id and its text. A name may hold quotes, brackets
    and arrows; the id is made safe and the text is escaped, because a
    drawing that silently drops a name is worse than an ugly one."""
    label = label or "(unnamed)"
    safe = "".join(character if character.isalnum() else "_" for character in label) or "node"
    text = (label.replace("&", "#amp;").replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")
            .replace("[", "#91;").replace("]", "#93;").replace("(", "#40;").replace(")", "#41;"))
    return f'n_{safe}["{text}"]'


def mermaid(flow: Flow) -> str:
    """The flow as a Mermaid flowchart: callers above, the root marked,
    what it calls below. Drawn from the same steps the record carries, so
    the picture and the data cannot disagree."""
    lines = ["flowchart TD"]
    drawn: set[str] = set()

    def edge(caller: str, called: str) -> None:
        line = f"    {_node(caller)} --> {_node(called)}"
        if line not in drawn:
            drawn.add(line)
            lines.append(line)

    for step in flow.upstream:
        edge(step.label, step.through)
    for step in flow.downstream:
        edge(step.through, step.label)
    root = "".join(character if character.isalnum() else "_" for character in flow.root) or "node"
    lines.append(f"    n_{root}:::root")
    lines.append("    classDef root stroke-width:3px")
    return "\n".join(lines)
