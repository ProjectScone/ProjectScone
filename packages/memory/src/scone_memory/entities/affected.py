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

import json

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional

from .view import StatusMode

from ..core.errors import InvalidInput
from ..memory.engine import check_space
from .query import resolve

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine
    from .project import EntityProjection

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
#: **Bytes** a name may cost, not characters. A 20,000-character name
#: echoed into `target` and `why` produced a 40,000-byte answer under a
#: 512-byte budget: the same metadata-outside-the-budget fault the
#: context receipts had. Counting characters did not close it -- 200
#: emoji are 200 characters, 800 UTF-8 bytes and 2,400 once JSON escapes
#: them, so the bound has to be in the unit the answer is paid in.
MAX_NAME = 200
#: Reads allowed while the ledger moves under the walk. The same fence,
#: and the same count, that `memory/catalog.py` uses for a profile.
ATTEMPTS = 2


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
    #: Whether the ledger moved under every read, so the graph walked
    #: here was never the graph at any single moment.
    graph_moved: bool = False
    #: Whether the byte budget was spent on the answer's own framing
    #: before a single dependant could be listed.
    framing_spent_budget: bool = False
    #: What this answer serializes to, which is what ``max_bytes``
    #: bounds and what the caller receives. Reported rather than left to
    #: be inferred: a budget enforced invisibly is one nobody can check.
    #: It sits inside the record it measures, so it is settled by
    #: measuring until it stops moving.
    bytes_spent: int = 0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"target": self.target, "status": self.status, "reached": len(self.reached),
                "mode": self.mode, "at": self.at, "partial_read": self.partial_read,
                "by_depth": {str(k): v for k, v in sorted(self.by_depth.items())},
                "deepest": self.deepest, "stopped_at_depth": self.stopped_at_depth,
                "not_listed": self.not_listed, "graph_moved": self.graph_moved,
                "framing_spent_budget": self.framing_spent_budget,
                "bytes_spent": self.bytes_spent, "why": self.why,
                "entities": [[r.label, r.depth, r.through, r.depends_on] for r in self.reached]}


def _why(label: str, said: str, listed: int, not_listed: int, more_beyond: bool,
         max_hops: int, limit: int) -> str:
    """The answer's own prose, built in one place because the answer is
    measured before it is returned and an estimate would drift from the
    sentence it estimates.

    ``max_bytes`` is deliberately not named here. It would put the budget
    inside the thing the budget measures, so the answer's size would move
    with the number it was given -- and then the refusal below could not
    promise that the size it names is a budget that works. ``bytes_spent``
    and ``not_listed`` carry the same information without that knot."""
    why = (f"{listed} thing(s) rest on {label!r}, nearest first, through "
           f"{', '.join(DEPENDS_ON)}")
    if more_beyond:
        why += (f"; the walk stopped at {max_hops} hop(s) with more to follow, so what lies "
                f"beyond is unexplored rather than absent")
    if not_listed:
        why += (f"; {not_listed} more were reached and not listed, because the limit of {limit} "
                f"or the byte budget was spent -- bytes_spent says what this answer cost, and "
                f"all of them are counted in by_depth")
    return why + (f"; {said}; this is what this graph holds, which is the code it was given "
                  f"and not the whole of any codebase")


async def _read_graph(engine: "MemoryEngine", space: str, mode: StatusMode,
                      moment: datetime) -> tuple["EntityProjection", dict[str, object], bool]:
    """The projection, what its read covered, and whether the ledger held
    still while it was taken.

    ``_living`` fences the space and nothing fenced the facts, so a
    dependency excluded the instant after the snapshot was still reported
    as current -- an answer stamped with a moment it was no longer true
    at. The projection carries the revision it was built from, so the
    fence is exact rather than a window around the call.
    """
    for _ in range(ATTEMPTS - 1):
        projection, covered = await engine.entities.projection(space, mode=mode, when=moment)
        if await engine.documents.revision(space) == projection.revision:
            return projection, covered, False
    # The last read is taken whatever happens, so there is always an
    # answer; only whether it settled is still in question.
    projection, covered = await engine.entities.projection(space, mode=mode, when=moment)
    return projection, covered, (await engine.documents.revision(space)) != projection.revision


def _serialized(blast: "Blast") -> int:
    """The bytes a caller actually receives. ``runtime/cli.py`` prints
    exactly ``json.dumps(record())``, and that is the thing ``max_bytes``
    has to bound -- not the raw strings inside it. JSON writes one emoji
    as twelve bytes and puts every key around them, so a budget charged
    against UTF-8 was short by a factor of three on a name like that.
    Measured, because an estimate of a serializer is a guess about a
    serializer."""
    return len(json.dumps(blast.record()).encode())


def _settle(blast: "Blast") -> "Blast":
    """``blast`` with ``bytes_spent`` telling the truth about itself.

    The number sits inside the record it measures, so writing it changes
    it. Measure again until it stops moving; only a digit's worth of
    steps is ever needed, and the cap keeps a pathological case finite
    rather than trusting that claim."""
    spent = 0
    for _ in range(4):
        one = replace(blast, bytes_spent=spent)
        measured = _serialized(one)
        if measured == spent:
            return one
        spent = measured
    return replace(blast, bytes_spent=spent)


def _within(blast: "Blast", max_bytes: int) -> "Blast":
    """``blast`` if it fits, otherwise a refusal naming what it would cost.

    An answer with nothing left to drop cannot be made smaller without
    dropping what it says about itself, and an answer that has quietly
    stopped saying what it covers is the failure this whole module is
    written against. So the bound is kept and the caller is told the
    number that would hold it."""
    one = _settle(blast)
    if one.bytes_spent > max_bytes:
        raise InvalidInput(
            f"this answer serializes to {one.bytes_spent} bytes and max_bytes is {max_bytes}, with nothing "
            f"left to drop -- what remains is the target, the status and what the read covered, "
            f"and an answer that stops saying those is worse than no answer; ask again with "
            f"max_bytes at least {one.bytes_spent}, which this answer does not vary with")
    return one


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
    cost = len(name.encode())
    if cost > MAX_NAME:
        raise InvalidInput(f"a name is at most {MAX_NAME} bytes, not {cost} -- a name is echoed "
                           f"in the answer, so an unbounded one is an unbounded answer, and "
                           f"characters are not the unit an answer is paid in")

    moment = when or datetime.now(timezone.utc)
    projection, covered, moved = await _read_graph(engine, space, mode, moment)
    read = str(covered.get("facts_read", ""))
    seen, allowed = covered.get("facts_read"), covered.get("facts_limit")
    limit_hit = bool(covered.get("reasons")) or (
        isinstance(seen, int) and isinstance(allowed, int) and seen >= allowed)
    said = f"read as {mode} at {moment.isoformat()}"
    if limit_hit:
        said += (f"; the read behind this answer was itself truncated at {read} fact(s), so what "
                 f"is missing here may be missing from the graph it walked rather than from the "
                 f"codebase")
    if moved:
        said += (f"; the ledger moved under all {ATTEMPTS} reads, so this walk crossed facts from "
                 f"either side of a write and was never the graph at any one moment")
    # The rule merging and windowing follow and this did not: a source
    # confirmed gone is dropped, never served. A projection is a snapshot
    # taken before this line, so a space deleted while it was being read
    # would otherwise be answered from rows that no longer exist.
    await engine._living(space)

    found = resolve(projection, name.strip())
    if found.status == "not_found" or not found.candidates:
        return _within(Blast(target=name, status="unknown", graph_moved=moved,
                     mode=mode, at=moment.isoformat(), partial_read=limit_hit,
                     why=f"{name!r} is not in this graph ({said}) -- which is not a finding that "
                         f"nothing depends on it, only that this memory has not been given it"),
                       max_bytes)
    if found.status == "ambiguous":
        # The resolver's own verdict, not a score comparison of my own
        # invention: it knows what "ambiguous" means here and I do not.
        return _within(Blast(target=name, status="ambiguous", graph_moved=moved,
                     mode=mode, at=moment.isoformat(), partial_read=limit_hit,
                     why=f"{name!r} names more than one thing in this graph "
                         f"({', '.join(one.label for one in found.candidates[:4])}); "
                         f"ask for one of them by its own name ({said})"), max_bytes)
    target = found.candidates[0].entity_id
    label = {entity.entity_id: entity.label for entity in projection.entities}

    # Reverse adjacency over dependency relations only: object -> the
    # subjects that rest on it.
    rests_on: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relation in projection.relations:
        if relation.predicate in DEPENDS_ON and relation.subject_id != relation.object_id:
            rests_on[relation.object_id].append((relation.subject_id, relation.predicate))

    # `target` and `why` are in the answer too, and `max_bytes` bounded
    # neither -- so a 512-byte budget returned five kilobytes and the
    # bound read as a fact. Charge the framing first, at its widest, so
    # what the walk may list is what the budget has left rather than the
    # whole of it. Widest, because the prose grows with the clauses the
    # walk turns out to need and the charge is made before the walk.
    shown = label.get(target, name)
    framing = len(shown.encode()) + len(
        _why(shown, said, limit, limit, True, max_hops, limit).encode())
    room = max_bytes - framing
    framing_spent = room <= 0

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
            if len(listed) >= limit or spent + size > room:
                not_listed += 1
            else:
                listed.append(one)
                spent += size
            queue.append((dependant, step))
    if not listed and not not_listed:
        return _within(Blast(target=shown, status="nothing", by_depth=counted, graph_moved=moved,
                     mode=mode, at=moment.isoformat(), partial_read=limit_hit,
                     why=f"nothing in this graph rests on {shown!r} through "
                         f"{', '.join(DEPENDS_ON)} ({said}) -- a memory holds the code it was "
                         f"given, so this is not a finding that nothing depends on it"), max_bytes)

    reached = len(listed)

    def answer(kept: int) -> Blast:
        """The answer carrying the first ``kept`` dependants, claiming to
        cost ``spent`` bytes. Nothing here is said in a sentence of its
        own when ``kept`` is zero: the clause naming what was dropped and
        why already covers it, and an extra sentence at exactly one size
        would break the monotonicity the fit below relies on."""
        dropped = not_listed + reached - kept
        return Blast(target=shown, status="found", reached=tuple(listed[:kept]),
                     by_depth=counted, deepest=deepest, stopped_at_depth=more_beyond,
                     not_listed=dropped, mode=mode, at=moment.isoformat(),
                     partial_read=limit_hit, graph_moved=moved,
                     framing_spent_budget=not kept and bool(reached or not_listed),
                     why=_why(shown, said, kept, dropped, more_beyond, max_hops, limit))

    whole = _settle(answer(reached))
    if whole.bytes_spent <= max_bytes:
        return whole
    # Every answer below the whole one carries the same clauses, so its
    # serialized length rises with the count and the largest that fits
    # can be bisected rather than walked down one dependant at a time.
    low, high = 0, reached - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _settle(answer(middle)).bytes_spent <= max_bytes:
            low = middle
        else:
            high = middle - 1
    return _within(answer(low), max_bytes)
