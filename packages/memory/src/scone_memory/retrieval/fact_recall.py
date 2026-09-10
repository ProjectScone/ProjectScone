"""Scoped fact retrieval and history over the document-store port.

Indexed results are validated against retained records before delivery. Stores
without an index use the same bounded per-query source-eligibility cache.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy

from ..core.models import Fact
from ..core.ports import DocumentStore, TextFilter
from .episode_scope import episode_fits
from .fact_search import IndexedFactSearch
from .lexical import tokenize

FACT_SCOPE_CACHE_LIMIT = 256


async def fact_fits_scope(documents: DocumentStore, space: str, fact: Fact, scope: TextFilter | None) -> bool:
    if scope is None:
        return True
    if fact.source_episode_id is None:
        return False
    episode = await documents.get_episode(space, fact.source_episode_id)
    if episode is None or episode.space != space:
        return False
    return (set(scope.tags).issubset(episode.tags)
            and all(episode.metadata.get(key) == value for key, value in scope.where.items())
            and (scope.as_of is None or episode.created_at <= scope.as_of)
            and episode_fits(episode, scope.kind, scope.source_prefix, scope.since, scope.until, scope.conditions))


async def facts_for_query(documents: DocumentStore, space: str, query: str, when: str, limit: int = 10,
                           scope: TextFilter | None = None, degraded: list[str] | None = None) -> list[Fact]:
    terms = set(tokenize(query))
    if not terms:
        return []
    if isinstance(documents, IndexedFactSearch):
        try:
            raw: object = await documents.search_facts(space, query, when, limit, scope=deepcopy(scope))
            if not isinstance(raw, list) or len(raw) > limit:
                raise ValueError("invalid indexed facts")
            # Detach snapshots before any awaited point reads; mutable store
            # objects must not change the evidence being checked in place.
            facts: list[Fact] = []
            for candidate in raw:
                if not isinstance(candidate, Fact):
                    raise ValueError("invalid indexed fact")
                facts.append(Fact.model_validate(candidate.model_dump(), strict=True))
            if len({fact.fact_id for fact in facts}) != len(facts):
                raise ValueError("duplicate indexed facts")
            for fact in facts:
                if (fact.space != space or fact.excluded or not fact.holds_at(when)
                        or not terms.intersection(tokenize(f"{fact.subject} {fact.predicate} {fact.object}"))
                        or not await fact_fits_scope(documents, space, fact, scope)):
                    raise ValueError("ineligible indexed fact")
                current = await documents.get_fact(space, fact.fact_id)
                if current is None or current != fact:
                    raise ValueError("stale indexed fact")
            facts.sort(key=lambda fact: (-len(terms.intersection(tokenize(
                f"{fact.subject} {fact.predicate} {fact.object}"))), -fact.confidence, fact.fact_id))
            return facts
        except Exception:
            if degraded is not None:
                degraded.append("fact_index: unavailable")
    return await scan_facts_for_query(documents, space, query, when, limit, scope)


async def scan_facts_for_query(documents: DocumentStore, space: str, query: str, when: str, limit: int = 10,
                           scope: TextFilter | None = None) -> list[Fact]:
    terms = set(tokenize(query))
    if not terms:
        return []
    scored = []
    # Keep only eligibility booleans, never mutable source records. A source
    # decision lasts for this scan (until eviction), not for later queries.
    source_eligibility: OrderedDict[int, bool] = OrderedDict()
    # Closed facts are included on purpose: asked about 2023, the
    # fact that held in 2023 is the answer even if it closed since.
    for fact in await documents.list_facts(space, include_closed=True):
        if fact.excluded or not fact.holds_at(when):
            continue
        overlap = len(terms & set(tokenize(f"{fact.subject} {fact.predicate} {fact.object}")))
        if not overlap:
            continue
        source_id = fact.source_episode_id
        if scope is not None and source_id is not None:
            if source_id in source_eligibility:
                eligible = source_eligibility[source_id]
                source_eligibility.move_to_end(source_id)
            else:
                eligible = await fact_fits_scope(documents, space, fact, scope)
                source_eligibility[source_id] = eligible
                if len(source_eligibility) > FACT_SCOPE_CACHE_LIMIT:
                    source_eligibility.popitem(last=False)
        else:
            eligible = await fact_fits_scope(documents, space, fact, scope)
        if eligible:
            scored.append((-overlap, -fact.confidence, fact.fact_id, fact))
    scored.sort(key=lambda t: t[:3])
    return [t[3] for t in scored[:limit]]


async def history_for(documents: DocumentStore, space: str, facts: Sequence[Fact], when: str, limit: int = 20,
                       scope: TextFilter | None = None) -> list[Fact]:
    """The closed ledger facts sharing a subject and predicate with any of
    ``facts`` and begun by ``when``, oldest first: the chain of what was
    believed before. Nothing that started after the reader's boundary
    leaks through; excluded facts stay out, as everywhere; proposals and
    declined candidates never held, so they are not history."""
    if not facts:
        return []
    keys = {(f.subject, f.predicate) for f in facts}
    shown = {f.fact_id for f in facts}
    chain = [
        f for f in await documents.list_facts(space, include_closed=True)
        if (f.subject, f.predicate) in keys and f.fact_id not in shown
        and f.status == "closed" and not f.excluded and f.valid_from <= when
        and await fact_fits_scope(documents, space, f, scope)
    ]
    chain.sort(key=lambda f: (f.valid_from, f.fact_id))
    return chain[:limit]
