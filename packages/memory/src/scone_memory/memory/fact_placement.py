"""Temporal fact placement through storage, clock and event ports.

The public engine owns space lifecycle and relationship validation. Placement
preserves historical intervals, restatement identity and proposal isolation.
"""
from __future__ import annotations

import time
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence, cast

from ..core.affirmations import Affirmation, NewAffirmation, affirmation_store
from ..core.errors import InvalidInput, NotFound
from ..core.models import Fact
from ..core.ports import DocumentStore, NewFact, NewFactLink
from ..core.timeutil import parse_rfc3339
from ..core.validation import ORIGINS, check_space, normalise_term, normalise_time


@dataclass(frozen=True)
class Resumption:
    """A covering fact's claim, stated again after the new fact's start: once
    the new fact cuts the covering fact short, the claim resumes from
    ``first``, and the ``later`` affirmations go with it."""

    covering: Fact
    first: Affirmation
    later: tuple[Affirmation, ...]


@dataclass(frozen=True)
class Placement:
    covering: list[Fact]
    restates: Optional[Fact]
    bound: Optional[str]
    bound_reason: Optional[str]
    bound_by: Optional[int]
    resumes: Optional[Resumption] = None


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
    links: Sequence[tuple[str, int]] = (),
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
      never overwrites a fresher one;
    - a restatement from a later day than the fact it restates is kept
      as an affirmation of that fact, with ``links``, the (kind, fact id)
      dependency links it was stated with. Should a fact with another
      object later cut the restated fact short before that day, the claim
      resumes from it as a new fact resting on those links, and bounds
      the one that cut it, so the order facts arrive in never decides
      what held.
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
        async with atomic(runtime.documents):
            affirmed = await affirm(runtime.documents, space, placement.restates, start, runtime.clock(),
                                    confidence=confidence, source_episode_id=source_episode_id, origin=origin,
                                    quote=quote, links=links)
            if affirmed:
                await runtime.documents.bump_revision(space)
        await runtime.emit(space, "fact_assert", {
            "fact_id": placement.restates.fact_id, "subject": subject, "predicate": predicate, "origin": origin,
            "outcome": "restated", "affirmed": affirmed, "superseded": [], "latency_ms": _ms(started),
        })
        return placement.restates
    # The continuation, the new fact, the cut and the revision commit
    # together where the store can hold a transaction.
    async with atomic(runtime.documents):
        placement = await resume(runtime.documents, space, placement)
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
    resumes = None
    if restates is None:
        # A covering fact's claim stated again after the new start resumes
        # there; it lies inside the covering fact, so before any successor.
        for rival in covering:
            if (resumes := await resumption(documents, space, rival, start_dt)) is not None:
                break
    if resumes is not None:
        return Placement(covering=covering, restates=None, bound=resumes.first.valid_from, bound_reason=None,
                         bound_by=None, resumes=resumes)
    return Placement(
        covering=[] if restates else covering,
        restates=restates,
        bound=successor.valid_from if successor else None,
        bound_reason=f"superseded by fact {successor.fact_id}" if successor else None,
        bound_by=successor.fact_id if successor else None,
    )


def atomic(documents: DocumentStore) -> AbstractAsyncContextManager[None]:
    """The store's transaction, when it can hold one across several writes,
    else nothing: each write then stands alone, as it always has."""
    hold = getattr(documents, "atomic", None)
    return cast(AbstractAsyncContextManager[None], hold()) if callable(hold) else _alone()


@asynccontextmanager
async def _alone() -> AsyncIterator[None]:
    yield


async def affirm(documents: DocumentStore, space: str, held: Fact, start: str, recorded_at: str, *,
                 confidence: float, source_episode_id: Optional[int], origin: str, quote: Optional[str],
                 links: Sequence[tuple[str, int]] = ()) -> bool:
    """Keep a restatement's later start beside the fact it restates, with
    its evidence. True only when that day was not kept already; one at the
    fact's own start adds nothing."""
    store = affirmation_store(documents)
    if store is None or parse_rfc3339(start) <= parse_rfc3339(held.valid_from):
        return False
    if any(kept.valid_from == start for kept in await store.affirmations(space, held.fact_id)):
        return False
    await store.add_affirmation(NewAffirmation(
        space=space, fact_id=held.fact_id, valid_from=start, recorded_at=recorded_at, confidence=confidence,
        source_episode_id=source_episode_id, origin=origin, quote=quote, links=tuple(links)))
    return True


async def resumption(documents: DocumentStore, space: str, held: Fact, after: datetime) -> Optional[Resumption]:
    """How ``held``'s claim resumes once it is cut at ``after``: from its
    first affirmation after that moment and before the fact's own end. One
    at or past the end resumes nothing, since the claim would end before
    it began."""
    store = affirmation_store(documents)
    if store is None:
        return None
    end = parse_rfc3339(held.valid_until) if held.valid_until is not None else None
    inside = [a for a in await store.affirmations(space, held.fact_id)
              if parse_rfc3339(a.valid_from) > after and (end is None or parse_rfc3339(a.valid_from) < end)]
    return Resumption(held, inside[0], tuple(inside[1:])) if inside else None


async def resume(documents: DocumentStore, space: str, placement: Placement) -> Placement:
    """Write the fact a resumption calls for, before the fact that cuts the
    covering one is written: the covering fact's claim from its first later
    affirmation to where the covering fact ended, ending the same way, with
    the affirmation's evidence and the covering fact's exclusion. The later
    affirmations move to it, and the placement is bounded by it."""
    if placement.resumes is None:
        return placement
    resumed = await write_resumption(documents, space, placement.resumes)
    return replace(placement, bound_reason=f"superseded by fact {resumed.fact_id}", bound_by=resumed.fact_id)


async def write_resumption(documents: DocumentStore, space: str, resumes: Resumption) -> Fact:
    """The fact a resumption resumes as, written with the links its first
    affirmation was stated with; the later affirmations move to it."""
    held, first, later = resumes.covering, resumes.first, resumes.later
    store = affirmation_store(documents)
    assert store is not None, "a resumption is only found on a store that keeps affirmations"
    resumed = await documents.insert_fact(NewFact(
        space=space, subject=held.subject, predicate=held.predicate, object=held.object, valid_from=first.valid_from,
        valid_until=held.valid_until, confidence=first.confidence,
        status="closed" if held.valid_until is not None else "active",
        closed_reason=held.closed_reason if held.valid_until is not None else None,
        superseded_by=held.superseded_by if held.valid_until is not None else None,
        source_episode_id=first.source_episode_id, origin=first.origin, quote=first.quote,
        excluded_reason=held.excluded_reason))
    # A new fact has nothing depending on it, so no link from it closes a cycle.
    for kind, to_fact in first.links:
        await documents.insert_fact_link(NewFactLink(space=space, from_fact=resumed.fact_id, to_fact=to_fact,
                                                     kind=kind, created_at=first.recorded_at))
    for affirmation in later:
        await store.add_affirmation(NewAffirmation(
            space=space, fact_id=resumed.fact_id, valid_from=affirmation.valid_from,
            recorded_at=affirmation.recorded_at, confidence=affirmation.confidence,
            source_episode_id=affirmation.source_episode_id, origin=affirmation.origin, quote=affirmation.quote,
            links=tuple(affirmation.links)))
    await store.drop_affirmations(space, [first.affirmation_id, *(a.affirmation_id for a in later)])
    return resumed


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
