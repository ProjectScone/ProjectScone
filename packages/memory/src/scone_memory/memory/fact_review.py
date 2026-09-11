"""Human review and visibility changes for the fact ledger.

Batch decisions check the starting revision, then apply in historical order.
Storage mutations and audit events retain their existing order; a batch is not
an atomic transaction and cancellation leaves completed decisions in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol, Sequence

from ..core.errors import Conflict, InvalidInput, NotFound
from ..core.models import DEPENDENCY_KINDS, BatchDecision, DecisionOutcome, Fact
from ..core.ports import DocumentStore
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space
from . import fact_placement
from .fact_placement import Placement


DECISIONS = ("approve", "decline", "exclude")
MAX_DECISIONS = 500


class PlaceFact(Protocol):
    def __call__(self, space: str, subject: str, predicate: str, object: str,
                 start: str, exclude_id: Optional[int] = None) -> Awaitable[Placement]: ...


class ApproveFact(Protocol):
    def __call__(self, space: str, fact_id: int,
                 actor: Optional[str] = None) -> Awaitable[Fact]: ...


class ReasonedDecision(Protocol):
    def __call__(self, space: str, fact_id: int, reason: str,
                 actor: Optional[str] = None) -> Awaitable[Fact]: ...


@dataclass(frozen=True)
class FactReviewRuntime:
    """Ledger ports and bound callbacks, preserving host overrides."""

    documents: DocumentStore
    clock: Callable[[], str]
    emit: Callable[[str, str, dict[str, object]], Awaitable[object]]
    place: PlaceFact
    truncate: Callable[[list[Fact], str, int], Awaitable[None]]
    proposed: Callable[[str, int], Awaitable[Fact]]
    approve: ApproveFact
    decline: ReasonedDecision
    exclude: ReasonedDecision
    decide_one: Callable[[str, str, int, Optional[str], Optional[str]], Awaitable[DecisionOutcome]]


async def approve(runtime: FactReviewRuntime, space: str, fact_id: int, actor: Optional[str] = None) -> Fact:
    """A person accepts a proposed fact: it enters the ledger exactly as
    an assertion at its own valid_from would, truncating what it
    covers and bounded by what starts later. A proposal that restates
    a fact already held is marked declined as a duplicate and the held
    fact is returned."""
    check_space(space)
    fact = await runtime.proposed(space, fact_id)
    placement = await runtime.place(space, fact.subject, fact.predicate, fact.object, fact.valid_from, exclude_id=fact.fact_id)
    if placement.restates is not None:
        # Approved, the proposal restates what holds; its later start is
        # kept as an affirmation, as an assertion's would be, with the
        # dependency links it was proposed with.
        links = [(link.kind, link.to_fact) for link in await runtime.documents.fact_links(space, fact.fact_id)
                 if link.from_fact == fact.fact_id and link.kind in DEPENDENCY_KINDS]
        async with fact_placement.atomic(runtime.documents):
            await fact_placement.affirm(runtime.documents, space, placement.restates, fact.valid_from,
                                        runtime.clock(), confidence=fact.confidence,
                                        source_episode_id=fact.source_episode_id, origin=fact.origin,
                                        quote=fact.quote, links=links)
            await runtime.documents.update_fact(
                fact.model_copy(update={"status": "declined",
                                        "closed_reason": f"duplicate of fact {placement.restates.fact_id}"}))
            await runtime.documents.bump_revision(space)
        await runtime.emit(space, "fact_review", {"fact_id": fact_id, "decision": "duplicate", "of": placement.restates.fact_id, "actor": actor})
        return placement.restates
    async with fact_placement.atomic(runtime.documents):
        placement = await fact_placement.resume(runtime.documents, space, placement)
        accepted = fact.model_copy(update={
            "status": "closed" if placement.bound else "active",
            "valid_until": placement.bound,
            "closed_reason": placement.bound_reason,
            "superseded_by": placement.bound_by,
        })
        await runtime.documents.update_fact(accepted)
        await runtime.truncate(placement.covering, fact.valid_from, fact.fact_id)
        await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_review", {
        "fact_id": fact_id, "decision": "approved", "superseded": [r.fact_id for r in placement.covering], "actor": actor,
    })
    return accepted


async def decide(
    runtime: FactReviewRuntime,
    space: str,
    decision: str,
    fact_ids: Sequence[int],
    reason: Optional[str] = None,
    actor: Optional[str] = None,
    expect_revision: Optional[int] = None,
) -> BatchDecision:
    """One reviewed batch, settled together.

    Three things a loop of single decisions cannot do. The batch is
    applied by (valid_from, fact_id), not in the order the caller
    listed, because inside one subject and predicate each approval
    truncates what it covers and is bounded by what starts later, so
    the caller's draw order would otherwise decide the ledger. With
    ``expect_revision`` the space is checked once, before anything is
    applied, so a batch built from a stale reading is refused whole
    rather than half applied. And every id comes back with its own
    outcome, keyed by the id that was sent, so one refusal does not
    hide what the rest did.
    """
    check_space(space)
    if decision not in DECISIONS:
        raise InvalidInput(f"decision must be one of {DECISIONS}, got {decision!r}")
    if not fact_ids:
        raise InvalidInput("decide needs at least one fact id")
    if len(fact_ids) > MAX_DECISIONS:
        raise InvalidInput(f"decide takes at most {MAX_DECISIONS} ids, got {len(fact_ids)}")
    if decision in ("decline", "exclude"):
        reason = _reason(reason or "")
    current = await runtime.documents.revision(space)
    if expect_revision is not None and expect_revision != current:
        raise Conflict(
            f"space {space!r} is at revision {current}, the batch was built from {expect_revision}",
            revision=current,
        )
    known = {}
    outcomes: dict[int, DecisionOutcome] = {}
    for fact_id in fact_ids:
        if fact_id in outcomes or fact_id in known:
            continue
        found = await runtime.documents.get_fact(space, fact_id)
        if found is None:
            outcomes[fact_id] = DecisionOutcome(fact_id=fact_id, outcome="not_found")
        else:
            known[fact_id] = found
    for fact in sorted(known.values(), key=lambda f: (parse_rfc3339(f.valid_from), f.fact_id)):
        outcomes[fact.fact_id] = await runtime.decide_one(space, decision, fact.fact_id, reason, actor)
    applied = sum(1 for o in outcomes.values() if o.outcome not in ("not_found", "refused"))
    return BatchDecision(
        results=[outcomes[fact_id] for fact_id in dict.fromkeys(fact_ids)],
        applied=applied,
        revision=await runtime.documents.revision(space),
    )


async def decide_one(
    runtime: FactReviewRuntime, space: str, decision: str, fact_id: int, reason: Optional[str], actor: Optional[str]
) -> DecisionOutcome:
    try:
        if decision == "approve":
            landed = await runtime.approve(space, fact_id, actor=actor)
            if landed.fact_id != fact_id:
                return DecisionOutcome(fact_id=fact_id, outcome="duplicate_of", held_fact_id=landed.fact_id)
            return DecisionOutcome(fact_id=fact_id, outcome="approved")
        if decision == "decline":
            await runtime.decline(space, fact_id, reason or "", actor=actor)
            return DecisionOutcome(fact_id=fact_id, outcome="declined")
        await runtime.exclude(space, fact_id, reason or "", actor=actor)
        return DecisionOutcome(fact_id=fact_id, outcome="excluded")
    except NotFound as e:
        return DecisionOutcome(fact_id=fact_id, outcome="not_found", error=str(e))
    except InvalidInput as e:
        return DecisionOutcome(fact_id=fact_id, outcome="refused", error=str(e))


async def decline(runtime: FactReviewRuntime, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
    """A person rejects a proposed fact. It never held; the reason is kept."""
    check_space(space)
    reason = _reason(reason)
    fact = await runtime.proposed(space, fact_id)
    declined = fact.model_copy(update={"status": "declined", "closed_reason": reason})
    await runtime.documents.update_fact(declined)
    await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_review", {"fact_id": fact_id, "decision": "declined", "actor": actor})
    return declined


async def proposed(runtime: FactReviewRuntime, space: str, fact_id: int) -> Fact:
    fact = await runtime.documents.get_fact(space, fact_id)
    if fact is None:
        raise NotFound(f"fact {fact_id} not found in {space!r}")
    if fact.status != "proposed":
        raise InvalidInput(f"fact {fact_id} is {fact.status}, not proposed")
    return fact


async def exclude(runtime: FactReviewRuntime, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
    """Suppress a ledger fact from recall without touching its interval
    or its history. The third operation beside close (it stopped
    holding) and forget (the data is gone)."""
    check_space(space)
    reason = _reason(reason)
    fact = await runtime.documents.get_fact(space, fact_id)
    if fact is None:
        raise NotFound(f"fact {fact_id} not found in {space!r}")
    if not fact.in_ledger:
        raise InvalidInput(f"fact {fact_id} is {fact.status}; only ledger facts can be excluded")
    excluded = fact.model_copy(update={"excluded_reason": reason})
    await runtime.documents.update_fact(excluded)
    await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_exclude", {"fact_id": fact_id, "action": "exclude", "actor": actor})
    return excluded


async def include(runtime: FactReviewRuntime, space: str, fact_id: int, actor: Optional[str] = None) -> Fact:
    """Undo exclude."""
    check_space(space)
    fact = await runtime.documents.get_fact(space, fact_id)
    if fact is None:
        raise NotFound(f"fact {fact_id} not found in {space!r}")
    if not fact.excluded:
        return fact
    included = fact.model_copy(update={"excluded_reason": None})
    await runtime.documents.update_fact(included)
    await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_exclude", {"fact_id": fact_id, "action": "include", "actor": actor})
    return included


async def close_fact(runtime: FactReviewRuntime, space: str, fact_id: int, reason: str, actor: Optional[str] = None) -> Fact:
    check_space(space)
    reason = _reason(reason)
    fact = await runtime.documents.get_fact(space, fact_id)
    if fact is None:
        raise NotFound(f"fact {fact_id} not found in {space!r}")
    if fact.status == "closed":
        return fact
    if not fact.in_ledger:
        raise InvalidInput(f"fact {fact_id} is {fact.status}; decline a proposal instead of closing it")
    end = runtime.clock()
    if parse_rfc3339(end) < parse_rfc3339(fact.valid_from):
        raise InvalidInput(f"fact {fact_id} begins at {fact.valid_from}, after now; "
                           "closing it now would end it before it begins")
    closed = fact.model_copy(update={"status": "closed", "valid_until": end, "closed_reason": reason})
    # The claim stated again from a day after the close resumes there, as
    # it would had it been stated after the close.
    resumes = await fact_placement.resumption(runtime.documents, space, fact, parse_rfc3339(end))
    async with fact_placement.atomic(runtime.documents):
        resumed = await fact_placement.write_resumption(runtime.documents, space, resumes) if resumes else None
        await runtime.documents.update_fact(closed)
        await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_close", {"fact_id": fact_id, "reason_kind": "manual", "actor": actor,
                                             "resumed": resumed.fact_id if resumed else None})
    return closed


def _reason(reason: str) -> str:
    reason = reason.strip()
    if not reason or len(reason) > 500:
        raise InvalidInput("reason must be 1..=500 chars")
    return reason
