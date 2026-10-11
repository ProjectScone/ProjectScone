"""RFC 3339 timestamps, the one wire format for every date in the engine.

Storage keeps timestamps as strings so a MongoDB document, a Qdrant
payload, and an in-process dict all carry the same bytes. Comparisons
happen on parsed UTC datetimes, never on the strings, because
``2024-01-05T00:00:00+02:00`` and ``2024-01-04T22:00:00Z`` are the same
instant and sort differently as text.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re

RFC3339 = "%Y-%m-%dT%H:%M:%S.%fZ"


def now() -> datetime:
    return datetime.now(timezone.utc)


def now_rfc3339() -> str:
    return format_rfc3339(now())


def format_rfc3339(value: datetime) -> str:
    """Millisecond UTC, ``Z`` suffix, like the Rust core emits."""
    utc = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def parse_rfc3339(text: str) -> datetime:
    """Accept ``Z`` or an offset, with or without fractional seconds.

    A bare date (``2024-02-01``) means midnight UTC on that day, which
    is how a note "from Feb 1" without a clock time should date.
    """
    raw = text.strip()
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        raw += "T00:00:00Z"
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except OverflowError:
        # 0001-01-01T00:00+01:00 is a valid text whose UTC instant is not.
        raise ValueError(f"{text!r} is outside the instants UTC can represent") from None


def epoch_seconds(text: str) -> float:
    """For stores that range-filter on numbers, not strings (Qdrant)."""
    return parse_rfc3339(text).timestamp()


#: The one shape ``format_rfc3339`` writes: millisecond UTC with a ``Z``. Two timestamps of this shape sort as
#: text exactly as they sort as instants, so a store may index them as text. Any other shape may not be
#: compared as text.
_CANONICAL = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def is_canonical(text: str) -> bool:
    """Whether ``text`` has the shape ``format_rfc3339`` writes. The shape alone: it is not checked as a date."""
    return _CANONICAL.fullmatch(text) is not None


def canonical_floor(text: str) -> str:
    """``text``'s instant in the canonical shape, cut down to the millisecond.

    For any canonical timestamp ``c``: ``c`` is after ``text`` exactly when ``c > canonical_floor(text)`` as text.
    That is what lets a store answer "which canonical rows start after this instant" by comparing text."""
    return format_rfc3339(parse_rfc3339(text))


def epoch_micros(text: str) -> int:
    """Whole microseconds since the Unix epoch, exactly: the resolution timestamps are compared at. Two
    timestamps order as their ``epoch_micros`` do, so a store may index instants as these integers."""
    delta = parse_rfc3339(text) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def is_before_or_at(candidate: str, boundary: str) -> bool:
    return parse_rfc3339(candidate) <= parse_rfc3339(boundary)
