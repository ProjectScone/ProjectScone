"""Shared memory input rules, independent of storage and orchestration."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from .errors import InvalidInput
from .timeutil import format_rfc3339, parse_rfc3339

SPACE_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")

METADATA_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

MAX_METADATA_KEYS = 16

MAX_METADATA_VALUE = 256

ORIGINS = ("stated", "extracted", "inferred")

STATUSES = ("active", "closed", "proposed", "declined")

KINDS = ("note", "file", "conversation", "observation", "connector")

MAX_QUERY = 1_000

MAX_LIMIT = 50

MAX_SOURCE = 1_000

def retention_policy(policy: Mapping[str, float]) -> dict[str, float]:
    """A retention policy: episode kinds to the days they are kept. Only
    episodes have retention; facts, links and tombstones are not kinds."""
    clean: dict[str, float] = {}
    for kind, days in dict(policy or {}).items():
        if kind not in KINDS:
            raise InvalidInput(f"retention kind must be one of {KINDS}, got {kind!r}")
        if isinstance(days, bool) or not isinstance(days, (int, float)) or not math.isfinite(days) or days <= 0:
            raise InvalidInput(f"retention days for {kind} must be a positive number, got {days!r}")
        clean[kind] = float(days)
    return clean

def check_space(space: str) -> None:
    if not SPACE_NAME.match(space or ""):
        raise InvalidInput(f"space name must be 1..=64 chars of [a-z0-9-_], got {space!r}")

def normalise_tags(tags: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for tag in tags:
        clean = tag.strip().casefold()
        if not clean:
            continue
        if len(clean) > 64:
            raise InvalidInput(f"tag too long: {tag!r}")
        if clean not in seen:
            seen.append(clean)
    return tuple(seen)

def normalise_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    if len(metadata) > MAX_METADATA_KEYS:
        raise InvalidInput(f"at most {MAX_METADATA_KEYS} metadata keys")
    clean: dict[str, str] = {}
    for key, value in metadata.items():
        if not METADATA_KEY.match(key or ""):
            raise InvalidInput(f"metadata key must match [a-z][a-z0-9_]{{0,31}}, got {key!r}")
        if not isinstance(value, str) or not value or len(value) > MAX_METADATA_VALUE:
            raise InvalidInput(f"metadata value for {key!r} must be 1..={MAX_METADATA_VALUE} chars")
        clean[key] = value
    return clean

def entity_key(name: str) -> str:
    """The one rule for when two names denote the same thing.

    Case is folded, including the cases lower() misses ("Straße" and
    "STRASSE" are one name), and runs of whitespace become one space.
    Nothing else: this is identity, not resemblance. Deciding that two
    different spellings are one entity is a resolution decision with
    evidence behind it, and must never happen silently inside a join.

    Every join, grouping and traversal compares names through this, and so
    does storing a subject. Four join sites once each had their own rule;
    one compared raw strings, so a claim never met the claim naming its
    object, and the graph came out in pieces that should have connected.
    """
    return " ".join(name.casefold().split())

def normalise_term(value: str, what: str) -> str:
    clean = entity_key(value)
    if not clean:
        raise InvalidInput(f"{what} must not be empty")
    return clean

def normalise_time(value: str) -> str:
    try:
        return format_rfc3339(parse_rfc3339(value))
    except ValueError as e:
        raise InvalidInput(f"not an RFC 3339 timestamp: {value!r}") from e
