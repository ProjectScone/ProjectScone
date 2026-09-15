"""A profile's claims in two buckets: static and dynamic.

Some of what a profile holds is settled -- a name, a role, a diet -- and
some is what its subject is in the middle of: this quarter's project, this
week's city. A reader uses the two differently, so a profile can be read in
two buckets. Which bucket a claim goes in is a rule the ledger can check,
never a guess at what its words mean. The first rule that applies decides:

1. ``override``: the predicate is named in ``static_predicates`` or
   ``dynamic_predicates``.
2. ``changes``: the claim's slot (its subject and predicate; with the
   object too for a many-valued predicate, as the ledger keys it) changed
   value at least ``dynamic_changes`` times in the ``change_window_days``
   ending now. The slot's first value is not a change, nor is the same
   value stated again after a gap. A value closed at the instant it began
   (superseded at the same instant) never held, so it is no change and no
   value. Dynamic, however long this value held.
3. ``tenure``: the claim's value has held unbroken for at least
   ``static_after_days``: static; otherwise dynamic. Unbroken means across
   the slot's earlier rows of the same value that reach its start, as when
   a value is backfilled to an earlier day or stated again from the
   instant it was closed; a different value, or a gap, breaks it.

The static bucket keeps the profile's own order: most restated, then
newest. The dynamic bucket is ordered by weight, ``stated * 0.5 ** (days
since last stated / half_life_days)``, where ``stated`` counts the claim
and its restatements up to now: each statement adds as much weight as the
first, and each half-life since the last one halves it. So a claim stated
``n`` times falls behind one stated once only after ``log2(n)`` half-lives
more; with the default 30 days, six statements two months ago still lead
one yesterday. Static claims do not decay.

Each bucket has its own count and byte bound, and says which cut it. The
bytes are those of the bucket's claim records as compact JSON, the same
records a conversation is shown. Every shown claim names the rule that
placed it and the numbers the rule read.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, cast

from ..core.errors import InvalidInput
from ..core.models import Fact
from ..core.timeutil import parse_rfc3339

INCLUDES = ("static", "dynamic", "both")
BUCKETS = ("static", "dynamic")
#: The most claims one bucket may show.
MAX_BUCKET_LIMIT = 50
#: Bytes one bucket's claims may take, at least and at most.
MIN_BUCKET_BYTES, MAX_BUCKET_BYTES = 100, 16000
_DAY_SECONDS = 86400.0
#: When a row that is still open ends: after every moment it is compared with.
_OPEN = datetime.max.replace(tzinfo=timezone.utc)
#: One row of a slot's history: when it began, its id, its value, when it ended.
_Row = tuple[datetime, int, str, datetime]


def _days(value: object, what: str, *, above_zero: bool) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (above_zero and value == 0)):
        floor = "above 0" if above_zero else "of 0 or more"
        raise InvalidInput(f"{what} must be a finite number of days {floor}")
    return float(value)


@dataclass(frozen=True)
class BucketRules:
    """The thresholds and overrides that place a claim in a bucket."""

    static_predicates: frozenset[str] = frozenset()
    dynamic_predicates: frozenset[str] = frozenset()
    static_after_days: float = 90.0
    dynamic_changes: int = 2
    change_window_days: float = 365.0
    half_life_days: float = 30.0

    def __post_init__(self) -> None:
        both = self.static_predicates & self.dynamic_predicates
        if both:
            raise InvalidInput("a predicate cannot be in both static_predicates and dynamic_predicates: "
                               + ", ".join(sorted(both)))
        _days(self.static_after_days, "static_after_days", above_zero=False)
        if type(self.dynamic_changes) is not int or self.dynamic_changes < 1:
            raise InvalidInput("dynamic_changes must be an integer of 1 or more")
        _days(self.change_window_days, "change_window_days", above_zero=True)
        _days(self.half_life_days, "half_life_days", above_zero=True)

    @classmethod
    def of(cls, static_predicates: object = (), dynamic_predicates: object = (), static_after_days: float = 90.0,
           dynamic_changes: int = 2, change_window_days: float = 365.0,
           half_life_days: float = 30.0) -> "BucketRules":
        """Rules with predicates named as the ledger names them."""
        from .catalog import _predicates

        return cls(_predicates(static_predicates, "static_predicates"),
                   _predicates(dynamic_predicates, "dynamic_predicates"),
                   static_after_days, dynamic_changes, change_window_days, half_life_days)

    def record(self) -> dict[str, object]:
        return {"static_predicates": sorted(self.static_predicates),
                "dynamic_predicates": sorted(self.dynamic_predicates),
                "static_after_days": self.static_after_days, "dynamic_changes": self.dynamic_changes,
                "change_window_days": self.change_window_days, "half_life_days": self.half_life_days}


@dataclass(frozen=True)
class BucketBounds:
    """Which buckets a reader asked for, and how many claims and bytes each may show."""

    include: str = "both"
    static_limit: int = 10
    dynamic_limit: int = 10
    static_max_bytes: int = 2000
    dynamic_max_bytes: int = 2000

    def __post_init__(self) -> None:
        if self.include not in INCLUDES:
            raise InvalidInput(f"include must be one of {', '.join(INCLUDES)}")
        for bucket in BUCKETS:
            limit, most = getattr(self, f"{bucket}_limit"), getattr(self, f"{bucket}_max_bytes")
            if type(limit) is not int or not 1 <= limit <= MAX_BUCKET_LIMIT:
                raise InvalidInput(f"{bucket}_limit must be an integer from 1 to {MAX_BUCKET_LIMIT}")
            if type(most) is not int or not MIN_BUCKET_BYTES <= most <= MAX_BUCKET_BYTES:
                raise InvalidInput(f"{bucket}_max_bytes must be an integer from {MIN_BUCKET_BYTES} "
                                   f"to {MAX_BUCKET_BYTES}")

    @property
    def buckets(self) -> tuple[str, ...]:
        return BUCKETS if self.include == "both" else (self.include,)


def requested(include: Optional[str], **bounds: Optional[int]) -> Optional[BucketBounds]:
    """The bounds a route or command asked for, or None when it asked for no
    buckets. A bound given without buckets would bound nothing, so it is refused."""
    given = {name: value for name, value in bounds.items() if value is not None}
    if include is None:
        if given:
            raise InvalidInput(f"{', '.join(sorted(given))} bound a bucketed profile: ask for buckets too")
        return None
    return BucketBounds(include, **given)


@dataclass(frozen=True)
class Placement:
    """Which bucket a claim went in, by which rule, and what the rule read."""

    bucket: str
    rule: str
    #: Days the claim's value has held unbroken, up to now.
    held_days: float
    changes: int
    #: Dynamic claims only: when the claim was last stated, and its weight.
    last_stated: Optional[str] = None
    weight: Optional[float] = None

    def record(self) -> dict[str, object]:
        out: dict[str, object] = {"bucket": self.bucket, "rule": self.rule,
                                  "held_days": round(self.held_days, 2), "changes": self.changes}
        if self.weight is not None:
            out.update(last_stated=self.last_stated, weight=round(self.weight, 4))
        return out


@dataclass
class ProfileBuckets:
    static: list[Fact] = field(default_factory=list)
    dynamic: list[Fact] = field(default_factory=list)
    #: What was asked for, the rules in force, each bucket's bounds and
    #: whether they cut, and the placement of every shown claim.
    coverage: dict = field(default_factory=dict)
    placements: dict[int, Placement] = field(default_factory=dict)

    def record(self, fact: Fact) -> dict[str, object]:
        """The claim as a bucket's bytes count it and a conversation is shown it."""
        return claim_record(fact, self.placements[fact.fact_id])


def claim_record(fact: Fact, placement: Placement) -> dict[str, object]:
    return {"fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate,
            "object": fact.object, "valid_from": fact.valid_from, "rule": placement.rule}


def _slot(fact: Fact, many_valued: frozenset[str]) -> tuple[str, ...]:
    if fact.predicate in many_valued:
        return (fact.subject, fact.predicate, fact.object)
    return (fact.subject, fact.predicate)


def _histories(ledger: Sequence[Fact], wanted: set[tuple[str, ...]],
               many_valued: frozenset[str]) -> dict[tuple[str, ...], list[_Row]]:
    """The rows each wanted slot held, by when each began, then by id.

    Only the slots of the claims being placed are read: a ledger read holds
    up to 20,000 facts, and a profile places at most 200. A row closed at or
    before the instant it began never held at any moment -- the ledger
    closes a value that way when another supersedes it at the same instant
    -- so, like a proposed or declined claim, it is not history."""
    slots: dict[tuple[str, ...], list[_Row]] = {}
    for fact in ledger:
        slot = _slot(fact, many_valued)
        if not fact.in_ledger or slot not in wanted:
            continue
        began = parse_rfc3339(fact.valid_from)
        ended = _OPEN if fact.valid_until is None else parse_rfc3339(fact.valid_until)
        if ended <= began:
            continue
        slots.setdefault(slot, []).append((began, fact.fact_id, fact.object, ended))
    for held in slots.values():
        held.sort()  # by when each began, then by id: ids are unique, so no further field is compared
    return slots


def _changes(held: Sequence[_Row], since: datetime) -> int:
    """How many times a slot took a different value after ``since``.

    No change after now is looked for: a later value in a slot ends the one
    before it, and the profile shows only active claims, so a claim with a
    later change in its slot is never placed."""
    return sum(1 for before, after in zip(held, held[1:]) if after[2] != before[2] and since < after[0])


def _held_since(fact_id: int, held: Sequence[_Row]) -> datetime:
    """When a claim's value began to hold unbroken: its own start, or the
    start of an earlier row of the same value that reaches it. The rows are
    in order of start, so walking back stops at the first different value."""
    index = next(at for at, row in enumerate(held) if row[1] == fact_id)
    began, _, value, _ = held[index]
    for earlier, _, other, ended in reversed(held[:index]):
        if other != value:
            break
        if ended >= began:
            began = earlier
    return began


def _place(fact: Fact, rules: BucketRules, now: datetime, changes: int, held_since: datetime,
           restated: Sequence[str]) -> Placement:
    held = (now - held_since).total_seconds() / _DAY_SECONDS
    if fact.predicate in rules.static_predicates:
        bucket, rule = "static", "override"
    elif fact.predicate in rules.dynamic_predicates:
        bucket, rule = "dynamic", "override"
    elif changes >= rules.dynamic_changes:
        bucket, rule = "dynamic", "changes"
    elif held >= rules.static_after_days:
        bucket, rule = "static", "tenure"
    else:
        bucket, rule = "dynamic", "tenure"
    if bucket == "static":
        return Placement(bucket, rule, held, changes)
    # A restatement dated after now has not been made yet: it is neither when
    # the claim was last stated nor weight.
    stated = [fact.valid_from, *(when for when in restated if parse_rfc3339(when) <= now)]
    last = max(stated, key=parse_rfc3339)
    since = (now - parse_rfc3339(last)).total_seconds() / _DAY_SECONDS
    weight = len(stated) * 0.5 ** (since / rules.half_life_days)
    return Placement(bucket, rule, held, changes, last, weight)


def _bounded(facts: list[Fact], placements: Mapping[int, Placement], limit: int,
             max_bytes: int) -> tuple[list[Fact], dict[str, object]]:
    kept: list[Fact] = []
    size, cut = 2, None  # the brackets of an empty JSON array
    for fact in facts:
        if len(kept) == limit:
            cut = "count"
            break
        more = len(json.dumps(claim_record(fact, placements[fact.fact_id]), ensure_ascii=False,
                              separators=(",", ":")).encode()) + (1 if kept else 0)
        if size + more > max_bytes:
            cut = "bytes"
            break
        kept.append(fact)
        size += more
    return kept, {"candidates": len(facts), "shown": len(kept), "omitted": len(facts) - len(kept),
                  "bytes": size, "limit": limit, "max_bytes": max_bytes, "cut": cut}


def bucketed(ranked: Sequence[Fact], ledger: Sequence[Fact], *, restated: Mapping[int, Sequence[str]],
             rules: BucketRules, bounds: BucketBounds, now: str, many_valued: frozenset[str],
             candidates_truncated: bool) -> ProfileBuckets:
    """Place the profile's candidates, already in profile order, and bound each bucket.

    ``ledger`` is every fact the profile read, in every status: the slot
    history the ``changes`` and ``tenure`` rules read. ``restated`` holds the days each
    candidate was stated again."""
    moment = parse_rfc3339(now)
    histories = _histories(ledger, {_slot(fact, many_valued) for fact in ranked}, many_valued)
    since = moment - timedelta(days=rules.change_window_days)
    placements: dict[int, Placement] = {}
    for fact in ranked:
        held = histories[_slot(fact, many_valued)]
        placements[fact.fact_id] = _place(fact, rules, moment, _changes(held, since),
                                          _held_since(fact.fact_id, held), restated[fact.fact_id])
    static = [fact for fact in ranked if placements[fact.fact_id].bucket == "static"]
    dynamic = sorted((fact for fact in ranked if placements[fact.fact_id].bucket == "dynamic"),
                     key=lambda fact: -cast(float, placements[fact.fact_id].weight))
    answer = ProfileBuckets(coverage={"include": bounds.include, "rules": rules.record(),
                                      "candidates": len(ranked), "candidates_truncated": candidates_truncated})
    for bucket, members in (("static", static), ("dynamic", dynamic)):
        if bucket not in bounds.buckets:
            continue
        kept, report = _bounded(members, placements, getattr(bounds, f"{bucket}_limit"),
                                getattr(bounds, f"{bucket}_max_bytes"))
        setattr(answer, bucket, kept)
        answer.coverage[bucket] = report
    shown = [*answer.static, *answer.dynamic]
    answer.placements = {fact.fact_id: placements[fact.fact_id] for fact in shown}
    answer.coverage["placed"] = {str(fact.fact_id): placements[fact.fact_id].record() for fact in shown}
    return answer
