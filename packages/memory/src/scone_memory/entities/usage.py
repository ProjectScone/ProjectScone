"""How much of the graph is used: the facts recent recalls returned.

Every recall records, in the event log, the ids of the facts it returned.
Read back over the most recent ``MAX_RECALLS`` recalls (since a moment, if
given), those ids say which facts answer questions and which sit unread,
and through their roles, which entities and relations. Only the counts
leave the log: no query text is read or shown. A recall that returned two
facts of one entity counts once for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: Recall events read, newest first; more than this is reported as cut.
MAX_RECALLS = 1_000


@dataclass(frozen=True)
class Usage:
    """The fact ids each recall read returned, one set per recall."""

    returned: tuple[frozenset[int], ...] = ()
    available: bool = True
    truncated: bool = False
    since: str | None = None
    recalls_read: int = field(default=0)

    def record(self) -> dict[str, object]:
        return {"available": self.available, "recalls_read": self.recalls_read, "truncated": self.truncated,
                "since": self.since}


async def recall_usage(engine: "MemoryEngine", space: str, *, since: str | None = None) -> Usage:
    """The fact ids the space's recent recalls returned, per recall."""
    if engine.events is None:
        return Usage(available=False, since=since)
    events = await engine.events.query(space, kind="recall", since=since, limit=MAX_RECALLS + 1)
    returned = []
    for event in events[:MAX_RECALLS]:
        ids = event.payload.get("fact_ids")
        returned.append(frozenset(int(fact_id) for fact_id in ids) if isinstance(ids, list) else frozenset())
    return Usage(returned=tuple(returned), truncated=len(events) > MAX_RECALLS, since=since,
                 recalls_read=len(returned))
