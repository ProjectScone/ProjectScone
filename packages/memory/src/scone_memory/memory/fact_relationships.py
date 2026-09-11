"""Fact relationships: validated assertions, source quotes and dependency cycles.

Storage and host callbacks are explicit ports. The engine retains its public
API and lifecycle ownership; this module never imports or owns an engine.
Writes preserve the existing sequence and are not a cross-operation transaction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol, Sequence

from ..core.errors import InvalidInput, NotFound
from ..core.models import DEPENDENCY_KINDS, LINK_KINDS, Fact, FactLink
from ..core.ports import DocumentStore, NewFactLink
from ..core.validation import check_space


class PlaceFact(Protocol):
    async def __call__(
        self, space: str, subject: str, predicate: str, object: str,
        valid_from: Optional[str] = None, confidence: float = 1.0,
        source_episode_id: Optional[int] = None, origin: str = "stated",
        proposed: bool = False, quote: Optional[str] = None, links: Sequence[tuple[str, int]] = (),
    ) -> Fact: ...


class LinkFacts(Protocol):
    async def __call__(
        self, space: str, from_fact: int, to_fact: int, kind: str, *,
        source_episode_id: Optional[int] = None, quote: Optional[str] = None,
    ) -> FactLink: ...


@dataclass(frozen=True)
class FactRelationshipsRuntime:
    documents: DocumentStore
    clock: Callable[[], str]
    emit: Callable[[str, str, dict[str, object]], Awaitable[object]]
    living: Callable[[str], Awaitable[None]]
    assert_placed: PlaceFact
    link_facts: LinkFacts
    depends_on: Callable[[str, int, int], Awaitable[bool]]


async def assert_fact(
    runtime: FactRelationshipsRuntime,
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
    extends: Optional[int] = None,
    derived_from: Sequence[int] = (),
) -> Fact:
    """Record that ``subject predicate object`` holds from ``valid_from``;
    the host placement callback maintains its temporal ledger interval.

    ``extends`` names a fact this one adds detail to: both stay as they
    are and a link records the relation, so an extension may not share
    the extended fact's subject and predicate (that would supersede it;
    assert an update instead). ``derived_from`` names the ledger facts
    this one was inferred from; the claim is stored as ``inferred`` and
    a link records each premise. Both are checked before anything is
    written: an unknown target, one in another space, or one that is
    only proposed or declined refuses the whole assertion."""
    await runtime.living(space)
    check_space(space)
    premises = [int(f) for f in derived_from]
    if premises:
        if origin == "stated":
            origin = "inferred"
        elif origin != "inferred":
            raise InvalidInput("a derived claim is inferred; it cannot claim another origin")
    targets = {}
    for target_id in ([extends] if extends is not None else []) + premises:
        target = await runtime.documents.get_fact(space, target_id)
        if target is None:
            raise NotFound(f"fact {target_id} not found in {space!r}")
        if not target.in_ledger:
            raise InvalidInput(f"fact {target_id} is {target.status}; only a held or closed fact can be built on")
        targets[target_id] = target
    if extends is not None and (targets[extends].subject, targets[extends].predicate) == (subject, predicate):
        raise InvalidInput("an extension cannot supersede what it extends; assert an update instead")
    # Placement keeps them too, beside a restatement it keeps as an
    # affirmation, so a claim that resumes from it rests on its own.
    links = [("extends", extends)] if extends is not None else []
    links += [("derived_from", premise) for premise in premises]
    fact = await runtime.assert_placed(space, subject, predicate, object, valid_from=valid_from, confidence=confidence,
                                     source_episode_id=source_episode_id, origin=origin, proposed=proposed, quote=quote,
                                     links=links)
    if extends is not None:
        await runtime.link_facts(space, fact.fact_id, extends, "extends")
    for premise in premises:
        await runtime.link_facts(space, fact.fact_id, premise, "derived_from")
    return fact


async def link_facts(
    runtime: FactRelationshipsRuntime, space: str, from_fact: int, to_fact: int, kind: str, *,
    source_episode_id: Optional[int] = None, quote: Optional[str] = None,
) -> FactLink:
    """Relate two facts of one space: ``from_fact`` extends / is derived
    from / contradicts / supports ``to_fact``. The same link twice is one
    link. A quote must sit in the source episode it names. A dependency
    (extends, derived_from) that would close a cycle is refused."""
    await runtime.living(space)
    check_space(space)
    if kind not in LINK_KINDS:
        raise InvalidInput(f"link kind must be one of {LINK_KINDS}, got {kind!r}")
    if from_fact == to_fact:
        raise InvalidInput("a fact cannot be linked to itself")
    for fact_id in (from_fact, to_fact):
        if await runtime.documents.get_fact(space, fact_id) is None:
            raise NotFound(f"fact {fact_id} not found in {space!r}")
    if quote is not None:
        if not quote.strip() or len(quote) > 2000:
            raise InvalidInput("quote must be 1..2000 characters when given")
        if source_episode_id is None:
            raise InvalidInput("a quote needs a source_episode_id to be checked against")
        episode = await runtime.documents.get_episode(space, source_episode_id)
        if episode is None:
            raise NotFound(f"source episode {source_episode_id} not found in {space!r}")
        if quote not in episode.content:
            raise InvalidInput(f"quote is not a substring of episode {source_episode_id}; the link was not stored")
    for existing in await runtime.documents.fact_links(space, from_fact):
        if (existing.from_fact, existing.to_fact, existing.kind) == (from_fact, to_fact, kind):
            return existing
    if kind in DEPENDENCY_KINDS and await runtime.depends_on(space, to_fact, from_fact):
        raise InvalidInput(f"fact {from_fact} cannot {kind.replace('_', ' ')} fact {to_fact}: that would close a dependency cycle")
    link = await runtime.documents.insert_fact_link(NewFactLink(
        space=space, from_fact=from_fact, to_fact=to_fact, kind=kind, created_at=runtime.clock(),
        source_episode_id=source_episode_id, quote=quote,
    ))
    await runtime.documents.bump_revision(space)
    await runtime.emit(space, "fact_link", {"link_id": link.link_id, "from_fact": from_fact, "to_fact": to_fact,
                                          "kind": kind, "source_episode_id": source_episode_id})
    return link


async def depends_on(documents: DocumentStore, space: str, start: int, target: int) -> bool:
    """Whether ``start`` reaches ``target`` along dependency links."""
    seen, frontier = {start}, [start]
    while frontier:
        fact_id = frontier.pop()
        for link in await documents.fact_links(space, fact_id):
            if link.from_fact != fact_id or link.kind not in DEPENDENCY_KINDS or link.to_fact in seen:
                continue
            if link.to_fact == target:
                return True
            seen.add(link.to_fact)
            frontier.append(link.to_fact)
        if len(seen) > 10_000:
            raise InvalidInput("dependency chain too long to check")
    return False
