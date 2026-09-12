"""How much of the graph is used: the facts recent recalls returned.

Every recall records, in the event log, the ids of the facts it returned,
current and, when asked for, those that held before. Read back over the
most recent ``MAX_RECALLS`` recalls (since a moment, if given), those ids
say which facts answer questions and which sit unread, and through their
roles, which entities and relations. Only the counts leave the log: no
query text is read or shown. A recall that returned two facts of one
entity counts once for it.

The log keeps what its retention keeps, so this is a window, never all
time: the oldest recall read and the log's retention are said with the
counts. An event this reader cannot count, of another payload version or
whose ids are not whole numbers, is counted as such and read no further.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

from ..core.ports import EVENT_SCHEMA_VERSION

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine
    from .project import FactRole

#: Recall events read, newest first; more than this is reported as cut.
MAX_RECALLS = 1_000
#: What an event log may say it keeps.
_RETENTION = ("max_events", "max_age_days")


@dataclass(frozen=True)
class Usage:
    """The fact ids each recall read returned, one set per recall."""

    returned: tuple[frozenset[int], ...] = ()
    available: bool = True
    truncated: bool = False
    since: str | None = None
    recalls_read: int = 0
    #: When the oldest recall event read was made; older ones are not read.
    oldest: str | None = None
    #: Recall events read but not counted: of a payload version this reader
    #: does not know, or whose fact ids are not whole numbers.
    unsupported: int = 0
    malformed: int = 0
    #: Recalls counted that were recorded before recalls kept the history
    #: they returned: facts they returned only as history are not counted.
    history_unrecorded: int = 0
    #: What the event log says it keeps, when it says.
    retention: Optional[dict[str, object]] = None

    def record(self) -> dict[str, object]:
        return {"available": self.available, "recalls_read": self.recalls_read, "truncated": self.truncated,
                "since": self.since, "oldest": self.oldest, "unsupported": self.unsupported,
                "malformed": self.malformed, "history_unrecorded": self.history_unrecorded,
                "retention": self.retention}


def recalled_by_entity(usage: Usage, roles: Mapping[int, "FactRole"]) -> Counter[str]:
    """Per entity, the recalls that returned a fact it takes part in, each
    recall counted once however many of its facts it returned."""
    counted: Counter[str] = Counter()
    for returned in usage.returned:
        counted.update({end for fact_id in returned if (role := roles.get(fact_id)) is not None
                        for end in (role.subject_id, role.object_id) if end is not None})
    return counted


def _ids(value: object) -> Optional[frozenset[int]]:
    """Fact ids as the recall recorded them, or None when they are not a
    list of whole numbers (True is not fact 1, nor is 1.9)."""
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        return None
    return frozenset(value)


async def recall_usage(engine: "MemoryEngine", space: str, *, since: str | None = None) -> Usage:
    """The fact ids the space's recent recalls returned, per recall."""
    log = engine.events
    if log is None:
        return Usage(available=False, since=since)
    events = await log.query(space, kind="recall", since=since, limit=MAX_RECALLS + 1)
    read = events[:MAX_RECALLS]
    returned: list[frozenset[int]] = []
    unsupported = malformed = unrecorded = 0
    for event in read:
        if event.schema_version != EVENT_SCHEMA_VERSION:
            unsupported += 1
            continue
        current = _ids(event.payload.get("fact_ids", []))
        history = _ids(event.payload.get("history_fact_ids", []))
        if current is None or history is None:
            malformed += 1
            continue
        if "history_fact_ids" not in event.payload:
            unrecorded += 1
        returned.append(current | history)
    kept = {name: getattr(log, name) for name in _RETENTION if hasattr(log, name)}
    return Usage(returned=tuple(returned), truncated=len(events) > MAX_RECALLS, since=since,
                 recalls_read=len(returned), oldest=read[-1].ts if read else None, unsupported=unsupported,
                 malformed=malformed, history_unrecorded=unrecorded, retention=kept or None)
