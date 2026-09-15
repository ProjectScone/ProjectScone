"""When a memory is to be forgotten: the time a write asks for, and whether it has come.

A write names the time as an RFC 3339 instant, a bare date (midnight UTC), or a
duration counted from the engine's clock: whole numbers of weeks, days, hours,
minutes or seconds written together, ``30d``, ``1d12h``, ``90m``. Months and
years are not units, because their length depends on where they start and a
schedule should mean one instant. Whatever was asked, the episode keeps the
resolved instant, in the engine's one timestamp form, under the ``forget_after``
metadata key -- so every store carries it, an export carries it, and the time
it was resolved against does not have to be remembered to read it back.

Comparisons are on parsed instants, never on the stored text.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .errors import InvalidInput
from .timeutil import format_rfc3339, parse_rfc3339

#: The metadata key the resolved time is kept under.
KEY = "forget_after"
#: The longest duration a write may ask for. Longer is a typo more often than
#: a plan, and a date says a far-off time more plainly.
MAX_DURATION_DAYS = 36_525
#: Longest text read as a schedule.
MAX_TEXT = 64

_SECONDS = {"w": 7 * 86_400, "d": 86_400, "h": 3_600, "m": 60, "s": 1}
_DURATION = re.compile(r"(?:[0-9]+[wdhms])+")
_PART = re.compile(r"([0-9]+)([wdhms])")

_FORMS = ("forget_after must be an RFC 3339 time, a date, or a duration of whole weeks, days, hours, "
          "minutes or seconds such as 30d or 1d12h")


@dataclass(frozen=True)
class Resolved:
    """An instant a write resolved from what its caller asked, and checked
    against the engine clock, before its work began. An ingestion path passes
    it down instead of the text, so a parse or a walk that outlasts a short
    schedule stores the instant it was asked for rather than being refused
    part way, for a time that was ahead when the caller asked: the memory is
    then due at once, and the sweep takes it with what it attached."""

    instant: str


def instant(value: Optional[Resolved]) -> Optional[str]:
    """The stored form of a resolved schedule, None for none."""
    return None if value is None else value.instant


def text(value: object) -> str:
    """``value`` when it is schedule text, refused as ``resolve`` refuses it
    otherwise; for a caller that keeps what was asked before resolving it."""
    if not isinstance(value, str) or len(value) > MAX_TEXT:
        raise InvalidInput(f"{_FORMS}, got {value!r}")
    return value


def resolve(value: object, now: str, *, allow_past: bool = False) -> str:
    """The instant ``value`` names, counted from ``now`` when it is a duration,
    in the engine's timestamp form. Refused when it cannot be read, and --
    unless ``allow_past`` or ``value`` is ``Resolved`` -- when it is not after
    ``now``."""
    if isinstance(value, Resolved):
        return resolve(value.instant, now, allow_past=True)
    written = text(value).strip()
    base = parse_rfc3339(now)
    if _DURATION.fullmatch(written):
        seconds = sum(int(count) * _SECONDS[unit] for count, unit in _PART.findall(written))
        if seconds > MAX_DURATION_DAYS * 86_400:
            raise InvalidInput(f"forget_after as a duration is at most {MAX_DURATION_DAYS} days, got {written!r}; "
                               f"name a date for a time further off")
        when = base + timedelta(seconds=seconds)
    else:
        try:
            when = parse_rfc3339(written)
        except (ValueError, OverflowError):
            raise InvalidInput(f"{_FORMS}, got {written!r}") from None
    stamp = format_rfc3339(when)
    if not allow_past and when <= base:
        raise InvalidInput(f"forget_after {stamp} is not after the engine clock ({format_rfc3339(base)}): "
                           f"a memory cannot be scheduled to be forgotten at a time already past")
    return stamp


def asked(value: object, now: str) -> Optional[Resolved]:
    """``resolve`` for a write that may name no schedule: None when it names
    none. An ingestion path calls it before it stores anything, so a refused
    schedule leaves no attachment behind, and passes the result down; a
    ``Resolved`` value is kept as it is."""
    return None if value is None else Resolved(resolve(value, now))


def read(value: str) -> Optional[datetime]:
    """A stored schedule as an instant, or None when it cannot be read."""
    try:
        return parse_rfc3339(value)
    except (OverflowError, ValueError):
        return None


def reason(stamp: str, at: str) -> str:
    """Why a memory was taken: the one wording the sweep and a write use."""
    return f"forget_after {stamp} had passed at {at}"


def is_due(metadata: Mapping[str, str], now: datetime) -> bool:
    """Whether the memory's time has come. No schedule, or one that cannot be
    read, is never due: an unreadable value is reported by the sweep, not
    guessed at."""
    raw = metadata.get(KEY)
    if raw is None:
        return False
    when = read(raw)
    return when is not None and when <= now
