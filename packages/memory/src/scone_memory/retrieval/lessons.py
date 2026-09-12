"""What a search taught us, and whether it still applies.

A framework that never learns from being wrong repeats itself. Recording
how a recall turned out costs a person one word, and the accumulated
judgements are worth reading back — the leading code-graph tool keeps
exactly this kind of work memory and aggregates it into lessons.

What makes it safe rather than dangerous is the part that is easy to
leave out: **a lesson about evidence that has since changed must say so
rather than be quietly applied.** A confident verdict carried onto text
nobody judged is worse than no verdict. So each judgement records the
content hash of the episode it was about, and a lesson reports one of
four states:

- ``fresh`` — the evidence is the text that was judged.
- ``changed`` — it was replaced afterwards, so the lesson wants
  re-verifying rather than applying.
- ``gone`` — the episode was forgotten; the lesson refers to text this
  space no longer holds.
- ``unknown`` — the judgement carries no hash, so freshness **cannot be
  determined**. Reported as its own state, because "I did not check" is
  not "I checked and it is fine".

Three outcomes rather than a boolean, because "this was a dead end" and
"this was wrong, and here is the right answer" carry different
information, and one count for both loses the half that says what to do.

No model is called. The judgements are events, read back and grouped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence, cast

from ..core.errors import InvalidInput, NotFound
from ..memory.engine import check_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.ports import Event
    from ..memory.engine import MemoryEngine

#: What a person can say about a returned passage. ``corrected`` carries a
#: note saying what the answer should have been.
OUTCOMES = ("useful", "dead_end", "corrected")
#: The event a judgement is written as.
JUDGED = "recall.outcome"
#: Judgements read when building lessons. More than this and the report
#: says how many it read, never implying it read them all.
MAX_JUDGEMENTS = 2_000
#: Lessons returned. The count of what was found is reported beside it.
MAX_LESSONS = 100
#: Corrections kept per lesson. The rest are counted, not dropped in
#: silence.
MAX_CORRECTIONS = 5


@dataclass(frozen=True)
class Lesson:
    """What was learned about one episode, and whether it still holds."""

    episode_id: int
    source: Optional[str]
    useful: int = 0
    dead_end: int = 0
    corrected: int = 0
    #: What people said the answer should have been, newest first.
    corrections: tuple[str, ...] = ()
    corrections_found: int = 0
    last_judged: str = ""
    #: fresh, changed, gone or unknown. See the module docstring.
    evidence: str = "unknown"
    why: str = ""

    @property
    def judgements(self) -> int:
        return self.useful + self.dead_end + self.corrected

    def record(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "source": self.source, "useful": self.useful,
                "dead_end": self.dead_end, "corrected": self.corrected,
                "corrections": list(self.corrections),
                "corrections_found": self.corrections_found,
                "last_judged": self.last_judged, "evidence": self.evidence, "why": self.why}


@dataclass(frozen=True)
class Lessons:
    """What the judgements add up to, and what was not read."""

    lessons: tuple[Lesson, ...] = ()
    #: Judgements read, and how many the log holds for this space when
    #: that is more. Never one number for both.
    judgements_read: int = 0
    #: Episodes with at least one judgement. ``lessons`` may be fewer.
    found: int = 0
    changed: int = 0
    gone: int = 0
    unknown: int = 0
    why: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def capped(self) -> bool:
        return len(self.lessons) < self.found

    def record(self) -> dict[str, object]:
        return {"judgements_read": self.judgements_read, "found": self.found,
                "capped": self.capped, "changed": self.changed, "gone": self.gone,
                "unknown": self.unknown, "why": self.why, "notes": list(self.notes),
                "lessons": [one.record() for one in self.lessons]}

    def text(self) -> str:
        head = (f"lessons: {len(self.lessons)} of {self.found} episode(s) judged, from "
                f"{self.judgements_read} judgement(s)")
        said = [head]
        if self.changed or self.gone or self.unknown:
            said.append(f"{self.changed} about evidence that changed, {self.gone} about evidence "
                        f"that is gone, {self.unknown} whose freshness cannot be told")
        return "; ".join(said + list(self.notes))


async def record_outcome(engine: "MemoryEngine", space: str, recall_event_id: int, chunk_id: int,
                         outcome: str, note: Optional[str] = None) -> "Event":
    """Record how one returned passage turned out.

    The episode's content hash is recorded with the judgement, so a later
    reader can tell whether the lesson is about the text that is there
    now.
    """
    check_space(space)
    if outcome not in OUTCOMES:
        raise InvalidInput(f"an outcome is one of {', '.join(OUTCOMES)}, not {outcome!r}")
    if note is not None and len(note) > 500:
        raise InvalidInput("a note must be at most 500 characters")
    if engine.events is None:
        raise InvalidInput("no event log is attached, so an outcome cannot be kept")
    recall = await engine.events.get(space, recall_event_id)
    if recall is None or recall.kind != "recall":
        raise NotFound(f"recall event {recall_event_id} not found in {space!r}")
    returned = {int(item["chunk_id"]): int(item.get("episode_id") or 0)
                for item in cast(Sequence[dict], recall.payload.get("items", []))}
    if chunk_id not in returned:
        raise InvalidInput(f"chunk {chunk_id} was not returned by recall {recall_event_id}, so "
                           f"there is nothing to judge")
    episode_id = returned[chunk_id]
    digest = None
    try:
        digest = (await engine.episode(space, episode_id)).content_hash
    except Exception:
        # The evidence is already unreadable. The judgement still stands
        # as a judgement; its freshness will read as unknown, which is
        # the truth rather than a guess in either direction.
        digest = None
    return cast("Event", await engine._emit(space, JUDGED, {
        "recall_event_id": recall_event_id, "chunk_id": chunk_id, "episode_id": episode_id,
        "outcome": outcome, "note": note, "content_hash": digest}))


async def lessons(engine: "MemoryEngine", space: str, *, limit: int = MAX_LESSONS,
                  judgements: int = MAX_JUDGEMENTS) -> Lessons:
    """What the judgements in this space add up to, per episode."""
    check_space(space)
    if not 1 <= limit <= MAX_LESSONS:
        raise InvalidInput(f"limit must be from 1 to {MAX_LESSONS}, not {limit}")
    if engine.events is None:
        return Lessons(why="no event log is attached, so nothing has been judged here",
                       notes=("this is not a finding that no lesson exists",))

    read = await engine.events.query(space, kind=JUDGED, limit=judgements)
    counted: dict[int, dict] = {}
    for event in read:
        episode_id = event.payload.get("episode_id")
        outcome = event.payload.get("outcome")
        if not isinstance(episode_id, int) or outcome not in OUTCOMES:
            continue
        held = counted.setdefault(episode_id, {"useful": 0, "dead_end": 0, "corrected": 0,
                                               "corrections": [], "last": event.ts,
                                               "hash": event.payload.get("content_hash")})
        held[outcome] = int(held[outcome]) + 1
        note = event.payload.get("note")
        if outcome == "corrected" and isinstance(note, str) and note.strip():
            cast(list, held["corrections"]).append(note)
        if event.ts > str(held["last"]):
            held["last"] = event.ts
            held["hash"] = event.payload.get("content_hash")

    built: list[Lesson] = []
    changed = missing = unknown = 0
    for episode_id, held in sorted(counted.items()):
        state, why, source = await _evidence(engine, space, episode_id, held["hash"])
        changed += state == "changed"
        missing += state == "gone"
        unknown += state == "unknown"
        said = cast(list, held["corrections"])
        built.append(Lesson(
            episode_id=episode_id, source=source, useful=int(held["useful"]),
            dead_end=int(held["dead_end"]), corrected=int(held["corrected"]),
            corrections=tuple(said[:MAX_CORRECTIONS]), corrections_found=len(said),
            last_judged=str(held["last"]), evidence=state, why=why))

    built.sort(key=lambda one: (-one.judgements, one.episode_id))
    notes: list[str] = []
    if len(read) >= judgements:
        notes.append(f"only the newest {judgements} judgement(s) were read; there may be more")
    return Lessons(lessons=tuple(built[:limit]), judgements_read=len(read), found=len(built),
                   changed=changed, gone=missing, unknown=unknown,
                   why=f"{len(built)} episode(s) have been judged in this space",
                   notes=tuple(notes))


async def _evidence(engine: "MemoryEngine", space: str, episode_id: int,
                    digest: object) -> tuple[str, str, Optional[str]]:
    """Whether the lesson is about the text that is there now."""
    try:
        episode = await engine.episode(space, episode_id)
    except Exception:
        return ("gone", "the episode this was judged on is no longer there, so the lesson is "
                        "about text this space does not hold", None)
    if not isinstance(digest, str) or not digest:
        return ("unknown", "this judgement recorded no content hash, so whether the evidence "
                           "still matches cannot tell -- which is not the same as it being "
                           "unchanged", episode.source)
    if digest != episode.content_hash:
        return ("changed", "the source changed after this was judged, so re-verify before "
                           "applying the lesson", episode.source)
    return ("fresh", "the evidence is the text that was judged", episode.source)
