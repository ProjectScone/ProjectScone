"""Dates a question refers to, resolved against the moment it is asked.

Memory questions are anchored in time: "in May 2023", "on the 3rd of
June", "three weeks ago", "last month", "the past two months". Each
reference becomes a window of whole days, the first day in and the
first day out, carrying the words that made it, so whoever uses it can
say what was understood.

The readings are calendar ones, the way people count: "two weeks ago"
is the Monday-to-Sunday week two before this one, "last month" the
whole of the month before this one, "the past two months" the days from
this day two months back up to today. A month named without a year is
the latest one begun; "the past month" runs up to today, where "last
month" is the calendar one. Nothing ahead of the moment asked is read
("tomorrow"), since memory holds what has happened.

Words that are dates only in context are read only in it: "May" after a
word that sets a date ("in May", "since May"), so "I may go" is left
alone, and a year after one too ("in 2021"), so "room 2021" is a room.

The same readings resolve what a remembered message meant by its own
relative dates. ``relative_dates`` reads "yesterday" or "last Friday" in a
message against the day it was said, and ``date_note`` puts the days
beside the words for a model, which otherwise has to do the calendar
itself and often lands a few days out. A message is not a question: it
can plan ahead, and a week back is two different weeks to different
people, so only readings that name past days without doubt are used
there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re

from ..core.timeutil import parse_rfc3339

_MONTH_NAMES = (("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",),
                ("june", "jun"), ("july", "jul"), ("august", "aug"), ("september", "sept", "sep"),
                ("october", "oct"), ("november", "nov"), ("december", "dec"))
_MONTHS = {name: number for number, names in enumerate(_MONTH_NAMES, start=1) for name in names}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
#: Words after which a month or a year alone is a date.
_SETTERS = frozenset({"in", "during", "since", "until", "till", "from", "by", "of", "throughout", "before",
                      "after", "early", "late", "mid", "around"})

_MONTH = "(?P<{}>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_COUNT = r"(?P<count>\d{1,3}|" + "|".join(_COUNTS) + ")"
_UNIT = r"(?P<unit>day|week|month|year)s?"
_ORDINAL = r"(?:st|nd|rd|th)?"
_READINGS = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in (
    ("iso", r"\b(?P<iy>\d{4})-(?P<im>\d{2})-(?P<id>\d{2})\b"),
    ("slashed", r"\b(?P<sy>\d{4})/(?P<sm>\d{1,2})/(?P<sd>\d{1,2})\b"),
    ("month_day_year", r"\b" + _MONTH.format("mdy_m") + r"\s+(?P<mdy_d>\d{1,2})" + _ORDINAL
     + r",?\s+(?P<mdy_y>\d{4})\b"),
    ("day_month_year", r"\b(?P<dmy_d>\d{1,2})" + _ORDINAL + r"\s+(?:of\s+)?" + _MONTH.format("dmy_m")
     + r",?\s+(?P<dmy_y>\d{4})\b"),
    ("month_year", r"\b" + _MONTH.format("my_m") + r",?\s+(?P<my_y>\d{4})\b"),
    ("next_to", r"\b(?P<nx>yesterday|today)\b"),
    ("ago", r"\b" + _COUNT + r"\s+" + _UNIT + r"\s+ago\b"),
    ("past", r"\b(?:the\s+)?(?:past|last)\s+" + _COUNT.replace("count", "pcount") + r"\s+"
     + _UNIT.replace("unit", "punit") + r"\b"),
    ("past_one", r"\bthe\s+(?:past|last)\s+(?P<ounit>day|week|month|year)\b"),
    ("calendar", r"\b(?P<which>last|this)\s+(?P<cunit>week|month|year)\b"),
    ("weekday", r"\b(?P<wlast>last\s+)?(?P<wday>" + "|".join(_WEEKDAYS) + r")\b"),
    ("year", r"\b(?P<yr>(?:19|20)\d{2})\b"),
    ("month", r"\b" + _MONTH.format("m") + r"(?!\s*\d)"),
)))


@dataclass(frozen=True)
class DateWindow:
    """Whole days a question refers to: ``start`` is the first day in and
    ``end`` the first day out, and ``words`` are the words that said so."""

    start: date
    end: date
    words: str

    def holds(self, moment: str) -> bool:
        """Whether an RFC 3339 moment falls on one of the window's days."""
        return self.start <= parse_rfc3339(moment).date() < self.end

    def record(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "words": self.words}


def _month(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    return first, date(year + month // 12, month % 12 + 1, 1)


def _months_back(day: date, months: int) -> date:
    """The same day ``months`` calendar months back, or the month's last."""
    index = day.year * 12 + day.month - 1 - months
    year, month = divmod(index, 12)
    last = (_month(year, month + 1)[1] - timedelta(days=1)).day
    return date(year, month + 1, min(day.day, last))


def _unit_back(today: date, unit: str, count: int) -> tuple[date, date]:
    """The whole calendar unit ``count`` back from the one ``today`` is in."""
    if unit == "day":
        day = today - timedelta(days=count)
        return day, day + timedelta(days=1)
    if unit == "week":
        monday = today - timedelta(days=today.weekday() + 7 * count)
        return monday, monday + timedelta(days=7)
    if unit == "month":
        year, month = divmod(today.year * 12 + today.month - 1 - count, 12)
        return _month(year, month + 1)
    return date(today.year - count, 1, 1), date(today.year - count + 1, 1, 1)


def _count(word: str) -> int:
    return int(word) if word.isdigit() else _COUNTS[word]


def _read(match: re.Match[str], text: str, today: date) -> tuple[date, date] | None:
    """The days one reading refers to, or None when it is not a date here."""
    found = match.groupdict()
    kind = match.lastgroup
    try:
        if kind in ("iso", "slashed", "month_day_year", "day_month_year"):
            prefix = {"iso": "i", "slashed": "s", "month_day_year": "mdy_", "day_month_year": "dmy_"}[kind]
            month = found[prefix + "m"]
            day = date(int(found[prefix + "y"]), int(month) if month.isdigit() else _MONTHS[month],
                       int(found[prefix + "d"]))
            return day, day + timedelta(days=1)
    except ValueError:
        return None
    if kind == "month_year":
        return _month(int(found["my_y"]), _MONTHS[found["my_m"]])
    if kind == "next_to":
        day = today - timedelta(days=1 if found["nx"] == "yesterday" else 0)
        return day, day + timedelta(days=1)
    if kind == "ago":
        count = _count(found["count"])
        return _unit_back(today, found["unit"], count) if count else None
    if kind in ("past", "past_one"):
        count, unit = (_count(found["pcount"]), found["punit"]) if kind == "past" else (1, found["ounit"])
        if not count:
            return None
        start = (today - timedelta(days=count * (7 if unit == "week" else 1)) if unit in ("day", "week")
                 else _months_back(today, count * (12 if unit == "year" else 1)))
        return start, today + timedelta(days=1)
    if kind == "calendar":
        unit = found["cunit"]
        return _unit_back(today, unit, 1 if found["which"] == "last" else 0)
    before = text[:match.start()].split()
    setter = bool(before) and before[-1] in _SETTERS
    if kind == "weekday":
        # "on Tuesday" and "on the Tuesday" both name a day; a weekday
        # alone is as often a name ("Friday Night Lights") as a date.
        said = before[-2:] if before[-1:] == ["the"] else before[-1:]
        if not found["wlast"] and said[:1] != ["on"]:
            return None
        back = (today.weekday() - _WEEKDAYS.index(found["wday"]) - 1) % 7 + 1
        day = today - timedelta(days=back)
        return day, day + timedelta(days=1)
    if kind == "year":
        return (date(int(found["yr"]), 1, 1), date(int(found["yr"]) + 1, 1, 1)) if setter else None
    if kind == "month" and setter:
        month = _MONTHS[found["m"]]
        return _month(today.year if month <= today.month else today.year - 1, month)
    return None


def date_windows(question: str, *, now: str | datetime) -> list[DateWindow]:
    """Each date the question refers to, in the order it says them, the
    longest reading of any words first."""
    today = (parse_rfc3339(now) if isinstance(now, str) else now).date()
    text = question.casefold()
    windows = []
    for match in _READINGS.finditer(text):
        days = _read(match, text, today)
        if days is None:
            continue
        words = match.group(0)
        if match.lastgroup == "weekday" and not match.group("wlast"):
            words = match.group("wday")
        if match.lastgroup == "month":
            words = match.group("m")
        windows.append(DateWindow(days[0], days[1], words))
    return windows


@dataclass(frozen=True)
class RelativeDate:
    """A past-relative expression in a text and the whole days it means:
    ``words`` as written, ``at`` their offset in the text, ``start`` the
    first day in and ``end`` the first day out."""

    words: str
    at: int
    start: date
    end: date

    def days(self) -> str:
        """The days in words: one day with its weekday, a whole calendar
        month or year by name, any other span by its first and last day."""
        return describe_days(self.start, self.end)


def describe_days(start: date, end: date) -> str:
    """Whole days ``start`` (in) to ``end`` (out) in words."""
    if end - start == timedelta(days=1):
        return f"{start:%A} {start.day} {start:%B %Y}"
    if (start.month, start.day) == (1, 1) and end == date(start.year + 1, 1, 1):
        return str(start.year)
    if start.day == 1 and end == _month(start.year, start.month)[1]:
        return f"{start:%B %Y}"
    last = end - timedelta(days=1)
    return f"{start:%A} {start.day} {start:%B %Y} to {last:%A} {last.day} {last:%B %Y}"


def _certain_past(match: re.Match[str]) -> bool:
    """Whether a reading in a message names past days without doubt.

    A bare weekday is not: "on Friday" in a question is the Friday gone,
    but a message can plan ahead ("I'm going on Friday"). "Last week" and
    "two weeks ago" are not either: people mean the calendar week before
    or the seven days before, and either range would claim more than the
    words do. "Today" and "this month" say nothing the message's own date
    does not.
    """
    kind = match.lastgroup
    if kind == "ago":
        return match.group("unit") != "week"
    if kind in ("past", "past_one"):
        return True
    if kind == "next_to":
        return match.group("nx") == "yesterday"
    if kind == "calendar":
        return match.group("which") == "last" and match.group("cunit") != "week"
    if kind == "weekday":
        return bool(match.group("wlast"))
    return False


def _day_of(said: str | datetime | date) -> date:
    if isinstance(said, str):
        return parse_rfc3339(said).date()
    if isinstance(said, datetime):
        return said.date()
    return said


def relative_dates(text: str, *, said: str | datetime | date) -> tuple[RelativeDate, ...]:
    """Each past-relative expression in a message, in order, with the days
    it means when said on ``said`` (an RFC 3339 moment, a datetime or a
    date): "yesterday", "N days/months/years ago", "last <weekday>", "last
    month/year", "the past N ..."."""
    day = _day_of(said)
    # Fold one character at a time and keep any that fold to more than one
    # ("ß" to "ss"), so offsets in the folded text are offsets in the text.
    folded = "".join(f if len(f := c.casefold()) == 1 else c for c in text)
    found = []
    for match in _READINGS.finditer(folded):
        if not _certain_past(match):
            continue
        days = _read(match, folded, day)
        if days is None:
            continue
        found.append(RelativeDate(text[match.start():match.end()], match.start(), days[0], days[1]))
    return tuple(found)


def date_note(text: str, *, said: str | datetime | date) -> str | None:
    """The days a message's relative dates mean, beside the words, for a
    model reading it: '"last Friday" = Friday 14 July 2023; "yesterday" =
    ...'. None when it has none. The message itself is left as it is, so
    quotes taken from it still match it."""
    found = relative_dates(text, said=said)
    if not found:
        return None
    return "; ".join(f'"{item.words}" = {item.days()}' for item in found)
