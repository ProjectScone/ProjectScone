"""What breaks if I change this?

A code graph exists to answer that, and ours could not. ``graph changes``
answers a different question -- what changed between two moments -- and
nothing walked the dependency edges **backwards**, from a symbol to
everything resting on it. The reference calls it blast radius, and it is
the question people bring to a code graph before touching anything.

Three rules, each with a test:

- **Direction is the whole point.** ``A calls B`` means changing B
  affects A, not the reverse. Walking undirected would report everything
  B calls as affected by changing B, which is exactly backwards and would
  look plausible.
- **A bound that bites is reported.** Depth, count and bytes each stop
  the walk; each says so; and none of them may read as "nothing further
  depends on it".
- **The answer is what this graph holds.** A memory holds the code
  someone gave it. An empty blast radius means nothing *here* depends on
  the symbol, never that nothing does, and every report says so.

No model is called and nothing is re-embedded.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional

from .view import StatusMode

from ..core.errors import InvalidInput
from ..memory.engine import check_space
from .query import resolve

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Relations where the subject rests on the object, so a change to the
#: object reaches the subject. ``defines`` is deliberately absent: a file
#: containing a symbol is not a file depending on it, and following
#: containment would make every symbol's blast radius its own file.
DEPENDS_ON = ("calls", "imports", "inherits", "mixes_in")
#: Hops the walk may take. Past this it is the whole component, not a
#: blast radius.
MAX_HOPS = 8
#: Entities one answer may name.
MAX_REACHED = 1_000
#: Bytes one answer may reach.
MAX_BYTES = 64_000
#: Characters a name may be. A 20,000-character name echoed into `target`
#: and `why` produced a 40,000-byte answer under a 512-byte budget: the
#: same metadata-outside-the-budget fault the context receipts had.
MAX_NAME = 200


@dataclass(frozen=True)
class Reached:
    """One thing that rests on the target, and how the walk got there."""

    entity_id: str
    label: str
    #: Hops from the target. 1 is a direct dependant.
    depth: int
    #: The relation that brought it in, and what it depends on directly.
    through: str
    depends_on: str


@dataclass(frozen=True)
class Blast:
    """What rests on a symbol, and everything the answer does not cover."""

    target: str = ""
    status: Literal["found", "nothing", "unknown", "ambiguous"] = "nothing"
    reached: tuple[Reached, ...] = ()
    #: How many were reached at each depth, including any the bounds then
    #: excluded -- so a caller can see the shape, not just the sample.
    by_depth: dict[int, int] = field(default_factory=dict)
    #: The deepest hop actually walked.
    deepest: int = 0
    #: Whether the walk stopped because it reached ``max_hops`` with more
    #: to follow, rather than because it ran out of dependants.
    stopped_at_depth: bool = False
    #: Entities reached but not listed, because the count or the byte
    #: budget was spent.
    not_listed: int = 0
    #: The status mode and moment the graph was read at, because an
    #: answer about dependencies depends entirely on which facts counted.
    mode: str = "current"
    at: str = ""
    #: Whether the read behind this answer was itself truncated. An
    #: answer of "nothing" from an incomplete read is not an answer.
    partial_read: bool = False
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"target": self.target, "status": self.status, "reached": len(self.reached),
                "mode": self.mode, "at": self.at, "partial_read": self.partial_read,
                "by_depth": {str(k): v for k, v in sorted(self.by_depth.items())},
                "deepest": self.deepest, "stopped_at_depth": self.stopped_at_depth,
                "not_listed": self.not_listed, "why": self.why,
                "entities": [[r.label, r.depth, r.through, r.depends_on] for r in self.reached]}


def _whole(value: object, name: str, top: int) -> int:
    if type(value) is not int or not 1 <= value <= top:
        raise InvalidInput(f"{name} is a whole number from 1 to {top}, not {value!r}")
    return value


async def affected(engine: "MemoryEngine", space: str, name: str, *, max_hops: int = 4,
                   limit: int = MAX_REACHED, max_bytes: int = MAX_BYTES,
                   when: Optional[datetime] = None,
                   mode: StatusMode = "current") -> Blast:
    """Everything in this graph that rests on ``name``, nearest first.

    Read at ``mode`` and ``when``, both reported. The default is
    **current**: "what breaks if I change this" is a question about the
    dependencies that hold now, and reading `all` would answer it with
    edges closed years ago and edges that do not start until 2030. An
    ``as_of`` that implies a filter it does not apply is worse than no
    filter at all, so the mode and the moment are on every answer.
    """
    check_space(space)
    _whole(max_hops, "max_hops", MAX_HOPS)
    _whole(limit, "limit", MAX_REACHED)
    _whole(max_bytes, "max_bytes", MAX_BYTES)
    if not isinstance(name, str) or not name.strip():
        raise InvalidInput("name the symbol, file or module to start from")
    if len(name) > MAX_NAME:
        raise InvalidInput(f"a name is at most {MAX_NAME} characters, not {len(name)} -- a name "
                           f"is echoed in the answer, so an unbounded one is an unbounded answer")

    moment = when or datetime.now(timezone.utc)
    projection, covered = await engine.entities.projection(space, mode=mode, when=moment)
    read = str(covered.get("facts_read", ""))
    seen, allowed = covered.get("facts_read"), covered.get("facts_limit")
    limit_hit = bool(covered.get("reasons")) or (
        isinstance(seen, int) and isinstance(allowed, int) and seen >= allowed)
    said = f"read as {mode} at {moment.isoformat()}"
    if limit_hit:
        said += (f"; the read behind this answer was itself truncated at {read} fact(s), so what "
                 f"is missing here may be missing from the graph it walked rather than from the "
                 f"codebase")
    # The rule merging and windowing follow and this did not: a source
    # confirmed gone is dropped, never served. A projection is a snapshot
    # taken before this line, so a space deleted while it was being read
    # would otherwise be answered from rows that no longer exist.
    await engine._living(space)

    found = resolve(projection, name.strip())
    if found.status == "not_found" or not found.candidates:
        return Blast(target=name, status="unknown",
                     mode=mode, at=moment.isoformat(), partial_read=limit_hit,
                     why=f"{name!r} is not in this graph ({said}) -- which is not a finding that "
                         f"nothing depends on it, only that this memory has not been given it")
    if found.status == "ambiguous":
        # The resolver's own verdict, not a score comparison of my own
        # invention: it knows what "ambiguous" means here and I do not.
        return Blast(target=name, status="ambiguous",
                     mode=mode, at=moment.isoformat(), partial_read=limit_hit,
                     why=f"{name!r} names more than one thing in this graph "
                         f"({', '.join(one.label for one in found.candidates[:4])}); "
                         f"ask for one of them by its own name ({said})")
    target = found.candidates[0].entity_id
    label = {entity.entity_id: entity.label for entity in projection.entities}

    # Reverse adjacency over dependency relations only: object -> the
    # subjects that rest on it.
    rests_on: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relation in projection.relations:
        if relation.predicate in DEPENDS_ON and relation.subject_id != relation.object_id:
            rests_on[relation.object_id].append((relation.subject_id, relation.predicate))

    seen = {target}
    counted: dict[int, int] = {}
    listed: list[Reached] = []
    spent = 0
    not_listed = 0
    queue: deque[tuple[str, int]] = deque([(target, 0)])
    deepest = 0
    more_beyond = False
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            # Something still to follow at the bound: that is the whole
            # reason `stopped_at_depth` exists.
            more_beyond = more_beyond or bool(
                [one for one, _ in rests_on.get(node, ()) if one not in seen])
            continue
        for dependant, predicate in sorted(rests_on.get(node, ())):
            if dependant in seen:
                continue
            seen.add(dependant)
            step = depth + 1
            deepest = max(deepest, step)
            counted[step] = counted.get(step, 0) + 1
            one = Reached(entity_id=dependant, label=label.get(dependant, dependant),
                          depth=step, through=predicate, depends_on=label.get(node, node))
            size = len(one.label.encode()) + len(one.depends_on.encode()) + len(predicate)
            if len(listed) >= limit or spent + size > max_bytes:
                not_listed += 1
            else:
                listed.append(one)
                spent += size
            queue.append((dependant, step))
    if not listed and not not_listed:
        return Blast(target=label.get(target, name), status="nothing", by_depth=counted,
                     mode=mode, at=moment.isoformat(), partial_read=limit_hit,
                     why=f"nothing in this graph rests on {label.get(target, name)!r} through "
                         f"{', '.join(DEPENDS_ON)} ({said}) -- a memory holds the code it was "
                         f"given, so this is not a finding that nothing depends on it")

    why = (f"{len(listed)} thing(s) rest on {label.get(target, name)!r}, nearest first, through "
           f"{', '.join(DEPENDS_ON)}")
    if more_beyond:
        why += (f"; the walk stopped at {max_hops} hop(s) with more to follow, so what lies "
                f"beyond is unexplored rather than absent")
    if not_listed:
        why += (f"; {not_listed} more were reached and not listed, because the limit of {limit} "
                f"or the budget of {max_bytes} byte(s) was spent -- they are counted in by_depth")
    why += (f"; {said}; this is what this graph holds, which is the code it was given and not "
            f"the whole of any codebase")
    return Blast(target=label.get(target, name), status="found", reached=tuple(listed),
                 by_depth=counted, deepest=deepest, stopped_at_depth=more_beyond,
                 not_listed=not_listed, mode=mode, at=moment.isoformat(),
                 partial_read=limit_hit, why=why)
