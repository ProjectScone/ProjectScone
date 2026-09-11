"""RFC 3339 timestamps, the one wire format for every date in the engine.

Storage keeps timestamps as strings so a MongoDB document, a Qdrant
payload, and an in-process dict all carry the same bytes. Comparisons
happen on parsed UTC datetimes, never on the strings, because
``2024-01-05T00:00:00+02:00`` and ``2024-01-04T22:00:00Z`` are the same
instant and sort differently as text.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


def is_before_or_at(candidate: str, boundary: str) -> bool:
    return parse_rfc3339(candidate) <= parse_rfc3339(boundary)
