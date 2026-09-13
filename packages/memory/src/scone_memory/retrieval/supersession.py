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

Three rules, all borrowed from the mechanism beside it:

- **Order only.** Nothing is dropped, because a caller may be asking
  about the past.
- **Only against evidence that is present.** A retired passage is moved
  below its replacement, never below something the reader did not get --
  otherwise the one result they did receive is pushed down for nothing.
- **The reader's boundary decides what is current.** The facts paired
  against are already the ones holding at ``when``, so ``as_of`` needs no
  special case: ask about March and March's statement leads.

The cost is proportionate: nothing at all when the query matched no
fact, and one chain read per fact that it did.
"""

from __future__ import annotations

from typing import Sequence

from ..core.models import Fact, RecallItem
from ..core.ports import DocumentStore


async def demote_superseded(documents: DocumentStore, space: str, items: list[RecallItem],
                            facts: Sequence[Fact], when: str) -> list[RecallItem]:
    """``items`` marked and reordered: a retired passage says so, and
    follows its replacement when the reader received one.

    **Reordering runs after the result was cut to ``limit``**, as the
    lexical rule beside it does, so at a small limit the replacement may
    already have been discarded and there is nothing to move below. The
    mark still lands, which is why it is computed for every retired
    passage rather than only for the ones that can be reordered.
    """
    # Not `len(items) < 2`: a lone retired passage cannot be reordered and
    # is precisely the one that must say so. Reordering checks its own
    # group size further down.
    if not items or not facts:
        return items
    present = {item.episode_id for item in items}
    replaced: dict[int, int] = {}
    retired: set[int] = set()
    for fact in facts:
        # Deliberately not skipped when the replacement's own episode is
        # absent: whether `other` was retired does not depend on that, and
        # the reader who did not receive the replacement is exactly the one
        # who needs telling. Presence gates the **reordering** below, not
        # the mark.
        current = fact.source_episode_id
        for other in await documents.facts_for(space, fact.subject, fact.predicate):
            # Believed once and not now: superseded at or before the
            # reader's boundary, rather than at some point after it.
            #
            # **Unproven, and deliberately kept.** For this clause to
            # matter, two facts must hold at the boundary with one of them
            # already carrying `superseded_by` -- and no fixture here
            # reaches that. A single-valued predicate has one fact holding
            # at a time, and it is the same one this pairs against; a
            # many-valued predicate supersedes only on a repeated object,
            # which deduplicates instead. "I could not construct the
            # state" is weaker than "the state cannot exist", and
            # demoting a claim that still holds at the date asked about
            # would be wrong, so the clause stays.
            if (other.fact_id != fact.fact_id and other.superseded_by is not None
                    and not other.holds_at(when)
                    and other.source_episode_id is not None
                    and other.source_episode_id != current):
                retired.add(other.source_episode_id)
                if (current is not None and current in present
                        and other.source_episode_id in present):
                    replaced[other.source_episode_id] = current
    if retired:
        # Marked whether or not the replacement is present. That is the
        # case where the mark matters most: there is nothing to reorder
        # against, so without it the reader is handed a retired claim
        # with nothing to distinguish it.
        items = [item.model_copy(update={"superseded": True})
                 if item.episode_id in retired else item for item in items]
    if not replaced:
        return items
    leaders = set(replaced.values())
    groups: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        head = replaced.get(item.episode_id)
        if head is not None:
            groups.setdefault(head, []).append(index)
        elif item.episode_id in leaders:
            groups.setdefault(item.episode_id, []).append(index)
    ordered = list(items)
    for head, positions in groups.items():
        if len(positions) < 2:
            continue
        members = [items[position] for position in positions]
        leads = [member for member in members if member.episode_id == head]
        rest = sorted((member for member in members if member.episode_id != head),
                      key=lambda item: (item.created_at, item.chunk_id), reverse=True)
        for position, member in zip(positions, leads + rest):
            ordered[position] = member
    return ordered
