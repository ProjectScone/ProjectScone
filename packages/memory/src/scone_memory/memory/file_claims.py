"""What a file's claims do when the file changes.

A claim read out of a file -- what it defines, imports, calls, depends
on -- holds while the file says it. ``replace`` and ``sync`` store a
changed file as an update: the old episode is forgotten, the new one
stored, and the new one's claims read. Forget's contract leaves claims
standing, rightly, since a person's memory of a fact survives deleting
its source. It was wrong for what a reader extracted: the ledger held
what the file used to say beside what it says now, so a module that
dropped an import still imported it and a function that was removed was
still defined.

So when a file is replaced, the extracted claims its old episode
grounded that the new content did not restate are closed, with a reason
naming the file; when a file is removed by a sync asked to remove, all of
them are. Restated means the same subject, predicate and object came out
of the new content -- compared as the ledger stored them, not by
affirmation bookkeeping, which records nothing for a restatement at the
same instant. What a person stated about the episode is left alone, and
a plain ``forget`` still touches no claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Optional

from ..core import graph_read
from . import fact_review


@dataclass(frozen=True)
class Retired:
    #: Claims closed. None when the store cannot read claims by episode,
    #: in which case nothing was closed, rather than closed on a guess.
    closed: Optional[int]
    #: The episode grounded more claims than one read returns, so some
    #: were not examined and may still stand.
    unread: bool = False


def _triple(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
    return subject, predicate, obj


async def close_unstated(runtime: fact_review.FactReviewRuntime, space: str, episode_id: int, *,
                         kept: Collection[tuple[str, str, str]], reason: str, kind: str) -> Retired:
    """Close the extracted claims ``episode_id`` grounded whose subject,
    predicate and object are not in ``kept``. ``kind`` names why on the
    event: ``source_changed`` or ``source_removed``, never ``manual``."""
    documents = runtime.documents
    if not (isinstance(documents, graph_read.GraphFactReader)
            and callable(getattr(documents, "facts_for_graph", None))):
        return Retired(closed=None)
    cap = graph_read.MAX_GRAPH_FACTS
    rows = await documents.facts_for_graph(space, episode_id, cap + 1)
    unread = len(rows) > cap
    still = set(_triple(*item) for item in kept)
    closed = 0
    for fact in rows[:cap]:
        if (fact.space != space or fact.source_episode_id != episode_id or fact.origin != "extracted"
                or fact.status != "active" or _triple(fact.subject, fact.predicate, fact.object) in still):
            continue
        await fact_review.close_fact(runtime, space, fact.fact_id, reason, kind=kind)
        closed += 1
    return Retired(closed=closed, unread=unread)
