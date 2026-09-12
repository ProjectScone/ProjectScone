"""Search each part of a multi-part question, so no part loses the blend.

One query over a question with two halves returns one blend of passages,
and the half whose words are commoner in the corpus tends to win all the
room. Searching the parts separately and giving each part its turn fixes
that, and costs nothing but a second search: no model is called here
either.

Two things this reports that a single search cannot, and that a caller
needs:

- **A part nothing answered is named.** "What did I decide about billing,
  and where did I moor the sailboat?" can come back full of billing and
  silent about the sailboat. Said plainly, that is a caller's cue to stop
  rather than to answer half a question confidently.
- **The order is the merge, not a ranking.** Items arrive one per part in
  turn, and each item's ``score`` was computed against *its own part's*
  query. Scores from different queries are not comparable, so this does
  not sort by them and neither should a reader; ``placed_by`` says which
  part put each item where it is.

For the same reason there is no single confidence number for a multi-part
answer. The floor was measured per query, the parts can differ, and one
number over the lot would be a number a reader could mistake for a
judgment about the whole question. ``weak`` names the parts whose own
evidence fell below the floor instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.models import RecallItem
from ..memory.engine import check_space
from .decompose import MAX_PARTS, Decomposition, Part, decompose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Items one part may return before the merge. Kept equal to the caller's
#: limit so a part always has a full slate to offer.
MAX_LIMIT = 100


@dataclass(frozen=True)
class PartResult:
    """What one part of the question found, and what of it survived."""

    part: Part
    #: Items this part's own search returned.
    found: int
    #: Items of the merged answer this part put there. Less than ``found``
    #: when another part had already taken them, or when the limit ran
    #: out — reporting one number for both would hide that.
    contributed: int
    #: True when this part's own evidence fell below the engine's floor;
    #: None when no floor is configured.
    weak: Optional[bool] = None
    degraded: tuple[str, ...] = ()


@dataclass(frozen=True)
class PartedRecall:
    """A merged answer to a question that asked more than one thing."""

    decomposition: Decomposition
    items: tuple[RecallItem, ...]
    per_part: tuple[PartResult, ...]
    #: For each item, the index of the part that placed it. The order is
    #: the merge, not a ranking.
    placed_by: tuple[int, ...]
    #: Chunk id to every part whose search returned it, so a passage that
    #: answers two parts is returned once and credited twice.
    by_chunk: dict[int, tuple[int, ...]]
    #: Parts whose search returned nothing at all.
    unanswered: tuple[str, ...]
    #: Parts whose evidence fell below the engine's floor.
    weak: tuple[str, ...]
    #: True when a floor actually judged the parts' evidence. When False,
    #: an empty ``weak`` says nothing was measured, not that every part
    #: was answered well.
    judged: bool
    why: str

    def record(self) -> dict[str, object]:
        return {"why": self.why, "judged": self.judged,
                "decomposition": self.decomposition.record(),
                "items": [item.model_dump() for item in self.items],
                "placed_by": list(self.placed_by),
                "by_chunk": {str(k): list(v) for k, v in self.by_chunk.items()},
                "unanswered": list(self.unanswered), "weak": list(self.weak),
                "per_part": [{"part": p.part.text, "found": p.found,
                              "contributed": p.contributed, "weak": p.weak,
                              "degraded": list(p.degraded)} for p in self.per_part]}

    def text(self) -> str:
        lines = [f"{self.why}"]
        for n, item in zip(self.placed_by, self.items):
            lines.append(f"[{self.decomposition.parts[n].text}] {item.text.strip()[:200]}")
        for part in self.unanswered:
            lines.append(f"[{part}] nothing found")
        return "\n".join(lines)


async def recall_parts(
    engine: "MemoryEngine",
    space: str,
    question: str,
    *,
    limit: int = 5,
    as_of: Optional[str] = None,
    tags: Sequence[str] = (),
    where: Mapping[str, str] | None = None,
    kind: Optional[str] = None,
    source_prefix: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    conditions: Mapping[str, object] | None = None,
    rerank: bool = True,
    graph_boost: bool = False,
    parts_limit: int = MAX_PARTS,
) -> PartedRecall:
    """Read the question as its parts, search each, and merge them so every
    part that found something is represented."""
    check_space(space)
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("a question must have something in it")
    if not 1 <= limit <= MAX_LIMIT:
        raise InvalidInput(f"limit must be from 1 to {MAX_LIMIT}")

    read = decompose(question, limit=parts_limit)
    narrowing = {"as_of": as_of, "tags": tags, "where": where, "kind": kind,
                 "source_prefix": source_prefix, "since": since, "until": until,
                 "conditions": conditions, "rerank": rerank, "graph_boost": graph_boost}
    results = [await engine.recall(space, part.text, limit=limit, **narrowing)  # type: ignore[arg-type]
               for part in read.parts]

    by_chunk: dict[int, tuple[int, ...]] = {}
    for index, result in enumerate(results):
        for item in result.items:
            by_chunk[item.chunk_id] = by_chunk.get(item.chunk_id, ()) + (index,)

    taken: dict[int, RecallItem] = {}
    placed_by: list[int] = []
    deepest = max((len(result.items) for result in results), default=0)
    for depth in range(deepest):
        if len(taken) >= limit:
            break
        for index, result in enumerate(results):
            if len(taken) >= limit:
                break
            if depth < len(result.items) and result.items[depth].chunk_id not in taken:
                taken[result.items[depth].chunk_id] = result.items[depth]
                placed_by.append(index)

    per_part = tuple(
        PartResult(part=part, found=len(result.items),
                   contributed=sum(1 for index in placed_by if index == n),
                   weak=result.low_confidence, degraded=tuple(result.degraded))
        for n, (part, result) in enumerate(zip(read.parts, results)))
    unanswered = tuple(part.part.text for part in per_part if not part.found)
    weak = tuple(part.part.text for part in per_part if part.weak)

    why = read.why if not read.split else (
        f"it asks {len(read.parts)} parts, searched on their own and merged so each has its turn")
    if unanswered:
        why += f"; {len(unanswered)} part(s) found nothing"
    judged = any(part.weak is not None for part in per_part)
    if not judged:
        why += ("; no similarity floor is configured, so a part returning passages is "
                "not evidence that part was answered")
    return PartedRecall(decomposition=read, items=tuple(taken.values()), per_part=per_part,
                        placed_by=tuple(placed_by), by_chunk=by_chunk,
                        unanswered=unanswered, weak=weak, judged=judged, why=why)
