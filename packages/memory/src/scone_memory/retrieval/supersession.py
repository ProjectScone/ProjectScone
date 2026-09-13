"""Evidence the ledger retired should not outrank what replaced it.

``fusion.demote_restated`` already reorders chunks that **lexically**
restate one another. It needs a shared prefix of four words covering 60%
of the shorter text, so it sees a statement whose replacement differs at
the **end** -- and for that shape it is perfect. Measured over 40
subjects: MRR 1.000.

The two ordinary shapes it cannot see scored 0.500, the retired passage
first every time:

    person works at Northwind as a staff engineer.       (retired)
    person works at Brightlake as a principal engineer.  (current)

The changed value sits mid-sentence, so no prefix is shared; reword the
replacement and there is nothing lexical left at all. Meanwhile the
ledger knows exactly: one fact carries ``superseded_by`` pointing at the
other. Nothing in the recall path had ever read it -- the framework's two
halves, a ledger and a retriever, each holding a piece of the answer.

The question is about the **passages the reader received**, not about the
facts the query matched. So this reads the claims each returned episode
stated -- one indexed read per episode, proportional to the result and
never to the ledger -- and asks of each claim: had it ended by the
boundary asked about, and is what replaced it among the passages too?

Three rules, all borrowed from the mechanism beside it:

- **Order only.** Nothing is dropped, because a caller may be asking
  about the past.
- **Only against evidence that is present.** A retired passage is moved
  below its own successor, never below something the reader did not get
  and never below a coexisting value that did not replace it --
  otherwise the one result they did receive is pushed down for nothing.
- **The reader's boundary decides what is current.** A claim retired
  after ``when`` still held then: ask about March and March's statement
  leads, unmarked.

Two things the ledger says are honoured as it says them. A claim a
person **excluded** is suppressed from recall, so it neither marks nor
moves anything, and an excluded successor cannot pull its passage ahead;
the intervals it left behind stand, so the claim it replaced is still
retired. A claim **closed** without a replacement is retired too -- its
passage says so -- and there is no successor to order it under.
"""

from __future__ import annotations

from heapq import heapify, heappop, heappush
from typing import Optional

from ..core import graph_read
from ..core.models import Fact, RecallItem
from ..core.ports import DocumentStore
from ..core.timeutil import parse_rfc3339


def retired_at(fact: Fact, when: str) -> bool:
    """The claim held once and had ended by ``when``, replaced or closed.
    One that had not begun by then is not retired, it is not yet; a
    proposal or a declined claim never held at all."""
    if not fact.in_ledger or fact.valid_until is None:
        return False
    boundary = parse_rfc3339(when)
    return parse_rfc3339(fact.valid_from) <= boundary and parse_rfc3339(fact.valid_until) <= boundary


def _reader(documents: DocumentStore) -> Optional[graph_read.GraphFactReader]:
    """A store that can read the facts one episode stated, or None. A
    matching name alone is not enough; custom stores must supply a
    callable."""
    if isinstance(documents, graph_read.GraphFactReader) and callable(getattr(documents, "facts_for_graph", None)):
        return documents
    return None


async def demote_superseded(documents: DocumentStore, space: str, items: list[RecallItem],
                            when: str, *, degraded: list[str]) -> list[RecallItem]:
    """``items`` marked and reordered: a passage whose claim the ledger
    retired by ``when`` says so, and follows its replacement when the
    reader received one.

    **Reordering runs after the result was cut to ``limit``**, as the
    lexical rule beside it does, so at a small limit the replacement may
    already have been discarded and there is nothing to move below. The
    mark still lands, which is why it is computed for every retired
    passage rather than only for the ones that can be reordered.

    What could not be read is written to ``degraded`` rather than passed
    over: a store that cannot read facts by episode, and an episode whose
    claims exceed the read cap, of which only the first ``cap`` are seen.
    """
    if not items:
        return items
    reader = _reader(documents)
    if reader is None:
        degraded.append("supersession: store cannot read facts by episode; retired passages are not marked")
        return items
    present: list[int] = []
    for item in items:
        if item.episode_id not in present:
            present.append(item.episode_id)
    # Read at call time, so a test that lowers the cap lowers it for the
    # store's read and this check alike.
    cap = graph_read.MAX_GRAPH_FACTS
    stated: dict[int, Fact] = {}
    for episode_id in present:
        rows = await reader.facts_for_graph(space, episode_id, cap + 1)
        if len(rows) > cap:
            degraded.append(f"supersession: episode {episode_id} stated more than {cap} facts; "
                            f"only the first {cap} were read")
            rows = rows[:cap]
        for fact in rows:
            if fact.space == space and fact.source_episode_id == episode_id:
                stated[fact.fact_id] = fact
    retired: set[int] = set()
    follows: dict[int, set[int]] = {}
    for fact in stated.values():
        if fact.excluded or fact.source_episode_id is None or not retired_at(fact, when):
            continue
        retired.add(fact.source_episode_id)
        # The successor is known by id. Its episode is present exactly
        # when the successor is among what the returned episodes stated.
        successor = stated.get(fact.superseded_by) if fact.superseded_by is not None else None
        if (successor is None or successor.excluded or successor.source_episode_id is None
                or successor.source_episode_id == fact.source_episode_id):
            continue
        follows.setdefault(fact.source_episode_id, set()).add(successor.source_episode_id)
    if not retired:
        return items
    # Marked whether or not the replacement is present. That is the case
    # where the mark matters most: there is nothing to reorder against,
    # so without it the reader is handed a retired claim with nothing to
    # distinguish it.
    items = [item.model_copy(update={"superseded": True}) if item.episode_id in retired else item
             for item in items]
    return _replacements_first(items, follows) if follows else items


def _replacements_first(items: list[RecallItem], follows: dict[int, set[int]]) -> list[RecallItem]:
    """The passages involved fill their own positions, a replacement
    before what it replaced and otherwise in the reader's order; every
    other passage keeps its place.

    A passage may replace one claim and be replaced under another, so
    this is an ordering of a graph, not a swap within a pair. Where two
    passages each replace a claim the other made, no order puts every
    replacement first, and the reader's order stands for that pair.
    """
    involved = set(follows).union(*follows.values())
    positions = [index for index, item in enumerate(items) if item.episode_id in involved]
    at: dict[int, list[int]] = {}
    for index in positions:
        at.setdefault(items[index].episode_id, []).append(index)
    waits = {index: 0 for index in positions}
    releases: dict[int, list[int]] = {index: [] for index in positions}
    for retired, replacements in follows.items():
        for replacement in replacements:
            for lead in at.get(replacement, ()):
                for late in at.get(retired, ()):
                    releases[lead].append(late)
                    waits[late] += 1
    ready = [index for index in positions if waits[index] == 0]
    heapify(ready)
    pending = set(positions)
    placed: list[int] = []
    while pending:
        if not ready:
            # Everything left waits on something left: a cycle. Release
            # the earliest, which keeps the reader's order for it.
            ready = [min(pending)]
        index = heappop(ready)
        if index not in pending:
            continue
        pending.discard(index)
        placed.append(index)
        for late in releases[index]:
            waits[late] -= 1
            if waits[late] == 0:
                heappush(ready, late)
    ordered = list(items)
    for position, index in zip(positions, placed):
        ordered[position] = items[index]
    return ordered
