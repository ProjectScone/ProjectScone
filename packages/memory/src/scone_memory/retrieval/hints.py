"""What a question says about where to look, read as the scope it asks for.

"What did we decide about the launch in last week's notes?" carries two
things: a question (what was decided about the launch) and a scope (the
notes, last week). The reference framework has a model infer metadata
filters from a question; here the words that name a scope are read by
rule, each recorded with the words that said it: a date the question
says (the readings ``dates.py`` already makes for temporal questions), a
kind of memory it names ("in my notes", "from the files"), a tag it
names ("tagged urgent", "#urgent"), a place it names ("under docs/",
"in README.md"). The scope is applied only when the caller asks for it,
never over a filter the caller set by hand, and the answer says what was
read and from which words, so a wrong reading is visible rather than a
silent narrowing. The words stay in the question the lanes search: a
word that names a scope can still name what the passage says.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Optional

from ..core.errors import InvalidInput
from .dates import date_windows

MAX_QUESTION_CHARS = 4_000
_KINDS = {"note": "note", "notes": "note", "file": "file", "files": "file", "document": "file", "documents": "file",
          "pdf": "file", "pdfs": "file", "conversation": "conversation", "conversations": "conversation",
          "chat": "conversation", "chats": "conversation", "observation": "observation", "observations": "observation"}
_KIND = re.compile(r"\b(?:in|from|among|within|across)\s+(?:my|our|your|the|these|those|his|her|their)?\s*"
                   r"(?P<kind>" + "|".join(sorted(_KINDS, key=len, reverse=True)) + r")\b", re.I)
_TAGGED = re.compile(r"\btagged\s+#?(?P<tag>[A-Za-z][\w-]*)", re.I)
_HASH = re.compile(r"(?<![\w&])#(?P<tag>[A-Za-z][\w-]*)")
_UNDER = re.compile(r"\b(?:under|in|from|inside)\s+(?P<path>(?:[\w.-]+/)+)")
_NAMED = re.compile(r"\b(?:in|from)\s+(?P<file>[\w.-]+\.(?:md|txt|rst|py|pdf|csv|json|html|docx|xlsx))\b", re.I)


@dataclass(frozen=True)
class Reading:
    """One thing the question said about where to look, and the words that said it."""

    filter: str
    value: str
    words: str

    def record(self) -> dict[str, str]:
        return {"filter": self.filter, "value": self.value, "words": self.words}


@dataclass(frozen=True)
class InferredScope:
    since: Optional[str] = None
    until: Optional[str] = None
    kind: Optional[str] = None
    tags: tuple[str, ...] = ()
    source_prefix: Optional[str] = None
    readings: tuple[Reading, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.readings

    def record(self) -> dict[str, object]:
        return {"since": self.since, "until": self.until, "kind": self.kind, "tags": list(self.tags),
                "source_prefix": self.source_prefix, "readings": [r.record() for r in self.readings],
                "notice": "Read from the question's own words by rule; a filter the caller set is never replaced."}


def infer_scope(question: str, *, now: str | datetime) -> InferredScope:
    """The scope a question names: dates, a kind, tags and a place."""
    if not isinstance(question, str):
        raise InvalidInput("a question is text")
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidInput(f"a question is read for its scope up to {MAX_QUESTION_CHARS} characters")
    readings: list[Reading] = []
    since = until = kind = prefix = None
    windows = date_windows(question, now=now)
    if windows:
        first = windows[0]
        since = datetime.combine(first.start, datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        last = datetime.combine(first.end, datetime.min.time(), tzinfo=timezone.utc) - timedelta(seconds=1)
        until = last.isoformat().replace("+00:00", "Z")
        readings.append(Reading("since", since, first.words))
        readings.append(Reading("until", until, first.words))
    named = _KIND.search(question)
    if named:
        kind = _KINDS[named.group("kind").lower()]
        readings.append(Reading("kind", kind, named.group(0)))
    tags: list[str] = []
    for found in (*_TAGGED.finditer(question), *_HASH.finditer(question)):
        tag = found.group("tag").lower()
        if tag not in tags:
            tags.append(tag)
            readings.append(Reading("tag", tag, found.group(0)))
    place = _UNDER.search(question) or _NAMED.search(question)
    if place:
        prefix = place.group("path") if "path" in place.groupdict() and place.group("path") else place.group("file")
        readings.append(Reading("source_prefix", prefix, place.group(0)))
    return InferredScope(since, until, kind, tuple(tags), prefix, tuple(readings))


@dataclass(frozen=True)
class AppliedScope:
    """The filters to search with: the caller's values, and the question's where the caller set none."""

    since: Optional[str]
    until: Optional[str]
    kind: Optional[str]
    tags: tuple[str, ...]
    source_prefix: Optional[str]
    applied: tuple[Reading, ...]

    def record(self, scope: InferredScope) -> dict[str, object]:
        return {**scope.record(), "applied": [r.record() for r in self.applied]}


def apply_scope(scope: InferredScope, *, since: Optional[str], until: Optional[str], kind: Optional[str],
                tags: list[str], source_prefix: Optional[str]) -> AppliedScope:
    """The filters to search with, the caller's values kept over the
    question's, and which readings were applied."""
    applied: list[Reading] = []
    kept_tags = list(tags)
    for reading in scope.readings:
        if reading.filter == "tag":
            if reading.value not in kept_tags:
                kept_tags.append(reading.value)
                applied.append(reading)
        elif reading.filter in ("since", "until"):
            if since is None and until is None:  # a caller's window, whichever end they set, is theirs
                applied.append(reading)
        elif reading.filter == "kind" and kind is None:
            applied.append(reading)
        elif reading.filter == "source_prefix" and source_prefix is None:
            applied.append(reading)
    chosen = {reading.filter: reading.value for reading in applied}
    return AppliedScope(since if since is not None else chosen.get("since"), until if until is not None else chosen.get("until"),
                        kind if kind is not None else chosen.get("kind"), tuple(kept_tags),
                        source_prefix if source_prefix is not None else chosen.get("source_prefix"), tuple(applied))
