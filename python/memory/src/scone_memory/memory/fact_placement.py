"""Temporal fact placement through storage, clock and event ports.

The public engine owns space lifecycle and relationship validation. Placement
preserves historical intervals, restatement identity and proposal isolation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ..core.errors import InvalidInput, NotFound
from ..core.models import Fact
from ..core.ports import DocumentStore, NewFact
from ..core.timeutil import parse_rfc3339
from ..core.validation import ORIGINS, check_space, normalise_term, normalise_time


@dataclass(frozen=True)
class Placement:
    covering: list[Fact]
    restates: Optional[Fact]
    bound: Optional[str]
    bound_reason: Optional[str]
    bound_by: Optional[int]


@dataclass(frozen=True)
class FactPlacementRuntime:
    """Storage snapshot plus the host's clock and live placement/event hooks."""

    documents: DocumentStore
    clock: Callable[[], str]
    emit: Callable[[str, str, dict[str, object]], Awaitable[object]]
    place: Callable[[str, str, str, str, str], Awaitable[Placement]]
    truncate: Callable[[list[Fact], str, int], Awaitable[None]]


async def assert_placed(
    runtime: FactPlacementRuntime,
    space: str,
    subject: str,
    predicate: str,
    object: str,
    valid_from: Optional[str] = None,
    confidence: float = 1.0,
    source_episode_id: Optional[int] = None,
    origin: str = "stated",
    proposed: bool = False,
    quote: Optional[str] = None,
) -> Fact:
    """Record that ``subject predicate object`` holds from ``valid_from``.

    ``quote`` is the exact substring of the source episode the claim
    rests on. When both a source and a quote are given, the quote must
    be non-empty and must occur in that episode's content, else the
    assertion is refused as ungrounded rather than stored. A source with
    no quote is stored and reads as ungrounded (``Fact.grounded`` is
    False); a quote with no source is refused, since there is nothing
    to check it against.

    ``origin`` says who is speaking: a person or trusted program
    (stated), a model reading an episode (extracted), or a derivation
    (inferred). ``proposed`` parks the fact for a person to approve;
    until then it is outside the ledger and answers nothing.

    Every ledger fact with the same subject and predicate takes part,
    whatever its status, so the ledger stays a partition of time:

    - a fact with the same object whose interval covers the start is a
      restatement and is returned unchanged;
    - any fact whose interval covers the start (open or closed) ends at
      the start, with the reason naming what superseded it. A reason a
      person wrote is kept; only the bound moves;
    - the first fact that starts after the new one bounds it: the new
      fact is stored closed at that point. A stale record arriving late
      never overwrites a fresher one.
    """
    check_space(space)
    subject = normalise_term(subject, "subject")
    predicate = normalise_term(predicate, "predicate")
    object = object.strip()
    if not object:
        raise InvalidInput("object must not be empty")
    if not 0.0 <= confidence <= 1.0:
        raise InvalidInput("confidence must be within 0..=1")
    if origin not in ORIGINS:
        raise InvalidInput(f"origin must be one of {ORIGINS}, got {origin!r}")
    if quote is not None:
        if not quote.strip():
            raise InvalidInput("quote must not be empty when given")
        if len(quote) > 2000:
            raise InvalidInput("quote must be at most 2000 characters")
        if source_episode_id is None:
            raise InvalidInput("a quote needs a source_episode_id to be checked against")
        episode = await runtime.documents.get_episode(space, source_episode_id)
        if episode is None:
            raise NotFound(f"source episode {source_episode_id} not found in {space!r}")
        if quote not in episode.content:
            raise InvalidInput(f"quote is not a substring of episode {source_episode_id}; the claim is ungrounded and was not stored")
    start = normalise_time(valid_from) if valid_from else runtime.clock()
    started = time.perf_counter()

    if proposed:
        fact = await runtime.documents.insert_fact(
            NewFact(
                space=space, subject=subject, predicate=predicate, object=object, valid_from=start,
                confidence=confidence, status="proposed", source_episode_id=source_episode_id, origin=origin, quote=quote,
            )
        )
        await runtime.documents.bump_revision(space)
        await runtime.emit(space, "fact_assert", {
            "fact_id": fact.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
            "outcome": "proposed", "superseded": [], "source_episode_id": source_episode_id,
            "grounded": fact.grounded, "latency_ms": _ms(started),
        })
        return fact

    placement = await runtime.place(space, subject, predicate, object, start)
    if placement.restates is not None:
        await runtime.emit(space, "fact_assert", {
            "fact_id": placement.restates.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
            "outcome": "restated", "superseded": [], "latency_ms": _ms(started),
        })
        return placement.restates
    fact = await runtime.documents.insert_fact(
        NewFact(
            space=space, subject=subject, predicate=predicate, object=object, valid_from=start,
            valid_until=placement.bound, confidence=confidence,
            status="closed" if placement.bound else "active",
            closed_reason=placement.bound_reason, superseded_by=placement.bound_by,
            source_episode_id=source_episode_id, origin=origin, quote=quote,
        )
    )
    await runtime.truncate(placement.covering, start, fact.fact_id)
    await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_assert", {
        "fact_id": fact.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
        "outcome": "new_closed" if placement.bound else "new_active",
        "superseded": [r.fact_id for r in placement.covering],
        "source_episode_id": source_episode_id,
        "latency_ms": _ms(started),
    })
    return fact


async def place(documents: DocumentStore, space: str, subject: str, predicate: str, object: str, start: str, exclude_id: Optional[int] = None) -> "Placement":
    """Where a fact starting at ``start`` sits among the ledger facts
    (active or closed) with the same subject and predicate."""
    start_dt = parse_rfc3339(start)
    rivals = [
        r for r in await documents.facts_for(space, subject, predicate)
        if r.in_ledger and r.fact_id != exclude_id
    ]
    covering = [r for r in rivals if _covers(r, start_dt)]
    restates = next((r for r in covering if r.object == object), None)
    later = [r for r in rivals if parse_rfc3339(r.valid_from) > start_dt]
    successor = min(later, key=lambda f: (parse_rfc3339(f.valid_from), f.fact_id)) if later else None
    return Placement(
        covering=[] if restates else covering,
        restates=restates,
        bound=successor.valid_from if successor else None,
        bound_reason=f"superseded by fact {successor.fact_id}" if successor else None,
        bound_by=successor.fact_id if successor else None,
    )


async def truncate(documents: DocumentStore, covering: list[Fact], start: str, by_fact_id: int) -> None:
    for rival in covering:
        reason = rival.closed_reason
        if reason is None or reason.startswith("superseded by fact "):
            reason = f"superseded by fact {by_fact_id}"
        await documents.update_fact(
            rival.model_copy(update={"status": "closed", "valid_until": start, "closed_reason": reason, "superseded_by": by_fact_id})
        )


def _covers(fact: Fact, instant) -> bool:
    """True when ``instant`` lies in ``[valid_from, valid_until)``."""
    if parse_rfc3339(fact.valid_from) > instant:
        return False
    return fact.valid_until is None or parse_rfc3339(fact.valid_until) > instant


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)
