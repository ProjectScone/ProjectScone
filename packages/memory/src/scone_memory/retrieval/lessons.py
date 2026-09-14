"""What people said about a passage, folded into a lesson.

A person marks a returned passage useful or not, and the event is kept.
Here those events are read back: per passage, the latest judgement of each
recall counts, each judgement weighs 1 and halves every ``half_life_days``
(positive when useful, negative when not), and the passage gets a state:

- ``preferred``: at least ``min_corroboration`` useful judgements and none against;
- ``dead_end``: judged, and never useful;
- ``contested``: judged both ways;
- ``tentative``: useful, but not yet corroborated.

A lesson is shown beside a passage, never used to move it: nothing here
has been measured to rank better, so the score is information, not order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Iterable, Literal, Optional

from ..core.errors import InvalidInput
from ..core.ports import Event
from ..core.timeutil import parse_rfc3339

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

LESSONS_VERSION = "lessons-v1"
#: Feedback events one read folds; past it the newest are kept and the read says it was cut.
MAX_FEEDBACK_EVENTS = 5_000
State = Literal["preferred", "tentative", "contested", "dead_end"]
Evidence = Literal["present", "gone", "unverified"]


@dataclass(frozen=True)
class Lesson:
    chunk_id: int
    state: State
    #: Judgements weighed by age: +1 useful, -1 not, halving every half-life.
    score: float
    useful: int
    not_useful: int
    last_at: str
    #: Whether the passage judged can still be read.
    evidence: Evidence = "unverified"

    def record(self) -> dict[str, object]:
        return {"state": self.state, "score": round(self.score, 6), "useful": self.useful,
                "not_useful": self.not_useful, "last_at": self.last_at, "evidence": self.evidence}


@dataclass(frozen=True)
class Lessons:
    lessons: dict[int, Lesson]
    as_of: str
    window_days: int
    half_life_days: float
    min_corroboration: int
    events_read: int
    events_cut: bool

    def record(self) -> dict[str, object]:
        return {"version": LESSONS_VERSION, "as_of": self.as_of, "window_days": self.window_days,
                "half_life_days": self.half_life_days, "min_corroboration": self.min_corroboration,
                "events_read": self.events_read, "events_cut": self.events_cut,
                "lessons": {str(chunk): lesson.record() for chunk, lesson in sorted(self.lessons.items())}}


def _check(half_life_days: float, min_corroboration: int) -> None:
    if isinstance(half_life_days, bool) or not isinstance(half_life_days, (int, float)) \
            or not math.isfinite(half_life_days) or half_life_days <= 0:
        raise InvalidInput("half_life_days must be a positive number of days")
    if isinstance(min_corroboration, bool) or not isinstance(min_corroboration, int) or min_corroboration < 1:
        raise InvalidInput("min_corroboration must be a whole number from 1")


def fold_lessons(events: Iterable[Event], *, now: str, half_life_days: float = 30,
                 min_corroboration: int = 2) -> dict[int, Lesson]:
    """One lesson per judged passage, from feedback events up to ``now``."""
    _check(half_life_days, min_corroboration)
    moment = parse_rfc3339(now)
    latest: dict[tuple[int, int], Event] = {}
    for event in sorted(events, key=lambda event: (parse_rfc3339(event.ts), event.event_id)):
        if event.kind != "feedback" or parse_rfc3339(event.ts) > moment:
            continue
        latest[(int(event.payload["recall_event_id"]), int(event.payload["chunk_id"]))] = event  # type: ignore[call-overload]
    by_chunk: dict[int, list[Event]] = {}
    for (_, chunk), event in latest.items():
        by_chunk.setdefault(chunk, []).append(event)
    folded: dict[int, Lesson] = {}
    for chunk, judged in by_chunk.items():
        useful = sum(bool(event.payload["useful"]) for event in judged)
        against = len(judged) - useful
        score = sum((1.0 if event.payload["useful"] else -1.0)
                    * 0.5 ** ((moment - parse_rfc3339(event.ts)) / timedelta(days=1) / half_life_days)
                    for event in judged)
        state: State = ("contested" if useful and against else "dead_end" if against
                        else "preferred" if useful >= min_corroboration else "tentative")
        latest_judged = max(judged, key=lambda event: parse_rfc3339(event.ts))  # instants, not strings
        folded[chunk] = Lesson(chunk, state, score, useful, against, latest_judged.ts)
    return folded


async def read_lessons(engine: "MemoryEngine", space: str, *, window_days: int = 90, half_life_days: float = 30,
                       min_corroboration: int = 2, max_events: int = MAX_FEEDBACK_EVENTS,
                       chunk_ids: Optional[Iterable[int]] = None) -> Lessons:
    """Lessons from the feedback of the last ``window_days``, each checked for whether its
    passage can still be read. With ``chunk_ids``, only those passages are checked and returned."""
    _check(half_life_days, min_corroboration)
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1:
        raise InvalidInput("window_days must be a whole number of days from 1")
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
        raise InvalidInput("max_events must be a whole number from 1")
    if engine.events is None:
        raise InvalidInput("no event log is attached, so there is no feedback to read")
    now = engine.clock()
    since = (parse_rfc3339(now) - timedelta(days=window_days)).isoformat().replace("+00:00", "Z")
    read = await engine.events.query(space, kind="feedback", since=since, limit=max_events + 1)
    cut = len(read) > max_events
    kept = read[:max_events]  # newest first, so a cut drops the oldest
    folded = fold_lessons(kept, now=now, half_life_days=half_life_days, min_corroboration=min_corroboration)
    if chunk_ids is not None:
        wanted = set(chunk_ids)
        folded = {chunk: lesson for chunk, lesson in folded.items() if chunk in wanted}
    present = {chunk.chunk_id for chunk in await engine.documents.get_chunks(space, list(folded))} if folded else set()
    checked = {chunk: Lesson(lesson.chunk_id, lesson.state, lesson.score, lesson.useful, lesson.not_useful,
                             lesson.last_at, "present" if chunk in present else "gone")
               for chunk, lesson in folded.items()}
    return Lessons(checked, now, window_days, float(half_life_days), min_corroboration, len(kept), cut)


def read_summary(found: Lessons) -> dict[str, object]:
    """What a set of lessons was read from, for an answer that shows them one passage at a time."""
    return {"window_days": found.window_days, "half_life_days": found.half_life_days,
            "min_corroboration": found.min_corroboration, "events_read": found.events_read,
            "events_cut": found.events_cut}
