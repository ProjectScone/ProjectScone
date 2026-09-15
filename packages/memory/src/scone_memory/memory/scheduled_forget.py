"""Forget what a memory's own schedule says is due.

A memory written with ``forget_after`` carries the instant it is to be forgotten
on its metadata (``core.forget_after``). From that instant recall leaves it out
whether or not anything has swept it; this is the sweep that then forgets it.

Each forget is the ordinary one -- its retirement record, retry, tombstone,
event and receipt, and the claims policy of ``forget`` -- so a scheduled forget
takes exactly what a manual forget of the same episode takes, and nothing
here decides what that is.

A pass is bounded twice and says when either bound bit. The walk reads at most
``MAX_SCANNED`` episodes, newest id first; ``scan_complete`` false says it
stopped before the oldest, and ``resume_before`` is where to walk on. The pass
forgets at most ``limit``, most overdue first; ``limited`` says more were due.
A ``now`` later than the engine's clock is refused: a sweep forgets what is
due, not what will be.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from ..core import forget_after
from ..core.errors import Gone, InvalidInput, NotFound
from ..core.forget_after import MAX_DURATION_DAYS as MAX_DURATION_DAYS
from ..core.models import ForgetDueReport, ScheduledForget
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import check_space, normalise_time

if TYPE_CHECKING:
    from .engine import MemoryEngine

#: Episodes one pass reads before it stops and says so.
MAX_SCANNED = 10_000
#: Episodes read per page of the walk.
PAGE = 200
#: Episodes one pass may forget, at most.
MAX_PASS = 1_000


async def forget_due(engine: "MemoryEngine", space: str, *, now: Optional[str] = None, limit: int = 100,
                     dry_run: bool = False, with_claims: Literal["keep", "exclude"] = "keep",
                     before: Optional[int] = None) -> ForgetDueReport:
    check_space(space)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PASS:
        raise InvalidInput(f"limit must be an integer in 1..{MAX_PASS}")
    if with_claims not in ("keep", "exclude"):
        raise InvalidInput(f"with_claims must be keep or exclude, not {with_claims!r}")
    if before is not None and (isinstance(before, bool) or not isinstance(before, int) or not 1 <= before <= 2**63 - 1):
        raise InvalidInput("before must be a positive signed 64-bit episode ID")
    clock = parse_rfc3339(engine.clock())
    if now is not None and not isinstance(now, str):
        raise InvalidInput(f"now must be an RFC 3339 time, not {now!r}")
    moment = parse_rfc3339(normalise_time(now)) if now is not None else clock
    if moment > clock:
        raise InvalidInput(f"now ({format_rfc3339(moment)}) is later than the engine clock ({format_rfc3339(clock)}); "
                           f"a sweep forgets what is due, not what will be")
    page = getattr(engine.documents, "page_episodes", None)
    if not callable(page):
        raise InvalidInput("this document store does not implement source inventory, which the sweep walks")

    due: list[tuple[datetime, int, str]] = []
    unreadable: list[int] = []
    scanned, cursor, complete = 0, before, False
    while scanned < MAX_SCANNED:
        asked = min(PAGE, MAX_SCANNED - scanned)
        batch = await page(space, cursor, asked, None)
        for episode in batch:
            scanned += 1
            cursor = episode.episode_id
            raw = episode.metadata.get(forget_after.KEY)
            if raw is None:
                continue
            when = forget_after.read(raw)
            if when is None:
                unreadable.append(episode.episode_id)
            elif when <= moment:
                due.append((when, episode.episode_id, format_rfc3339(when)))
        if len(batch) < asked:
            complete = True
            break
    if not complete:
        # The bound was reached on a full page: the end of the space and the
        # end of what was read are told apart by looking for one more.
        complete = not await page(space, cursor, 1, None)

    due.sort()
    at = format_rfc3339(moment)
    report = ForgetDueReport(space=space, now=at, limit=limit, dry_run=dry_run, with_claims=with_claims,
                             scanned=scanned, scan_complete=complete, resume_before=None if complete else cursor,
                             due=len(due), limited=len(due) > limit, unreadable=sorted(unreadable))
    skipped = 0
    for _, episode_id, stamp in due[:limit]:
        reason = f"forget_after {stamp} had passed at {at}"
        if dry_run:
            report.items.append(ScheduledForget(episode_id=episode_id, forget_after=stamp, reason=reason,
                                                outcome="would_forget"))
            continue
        try:
            receipt = await engine.forget(space, episode_id, with_claims=with_claims)
        except NotFound as error:
            # Forgotten by someone else between the walk and this forget.
            gone = "already forgotten" if isinstance(error, Gone) else "no longer found"
            report.items.append(ScheduledForget(episode_id=episode_id, forget_after=stamp, outcome="skipped",
                                                reason=f"{reason}; {gone} when its forget ran"))
            skipped += 1
            continue
        report.items.append(ScheduledForget(episode_id=episode_id, forget_after=stamp, reason=reason,
                                            outcome="forgotten", receipt=receipt))
        report.forgotten.append(episode_id)
    report.remaining = len(due) - len(report.forgotten) - skipped
    if not dry_run:
        await engine._emit(space, "forget_due", {
            "now": at, "due": report.due, "forgotten": len(report.forgotten), "skipped": skipped,
            "remaining": report.remaining, "limit": limit, "limited": report.limited, "scanned": scanned,
            "scan_complete": complete, "unreadable": len(unreadable), "with_claims": with_claims,
        })
    return report
