"""Offline, bounded traversal of retained ledger evidence.

Seeds come from recall or caller-authorized IDs. Stored links preserve their
kind and direction. Object-to-subject joins use the ledger's subject normalization
and retain exact-spelling compatibility for directly inserted records. They are
labeled as joins, never
as inferred semantic relationships. No model, embedding, browser or write is
involved. Capability-free stores return verified seeds with explicit coverage.
"""
from __future__ import annotations

from collections import deque
from itertools import chain, islice
from typing import Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..core.models import Episode, Fact, FactLink, RecallResult
from ..core.ports import TextFilter
from ..core.timeutil import now_rfc3339, parse_rfc3339
from ..memory.engine import check_space, normalise_term


class MultiHopDocuments(Protocol):
    async def get_fact(self, space: str, fact_id: int) -> Fact | None: ...
    async def get_episode(self, space: str, episode_id: int) -> Episode | None: ...


@runtime_checkable
class BoundedIncidentLinks(Protocol):
    async def fact_links_from(self, space: str, fact_id: int, limit: int) -> list[FactLink]: ...


@runtime_checkable
class BoundedSubjectFacts(Protocol):
    async def facts_by_subject(self, space: str, subject: str, limit: int) -> list[Fact]: ...


@runtime_checkable
class PointFactLinks(Protocol):
    async def get_fact_link(self, space: str, link_id: int) -> FactLink | None: ...


@runtime_checkable
class RevisionedDocuments(Protocol):
    async def revision(self, space: str) -> int: ...


class MultiHopLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_hops: int = Field(default=3, ge=0, le=6)
    max_nodes: int = Field(default=32, ge=1, le=128)
    max_edges: int = Field(default=64, ge=0, le=256)
    max_bytes: int = Field(default=65_536, ge=512, le=262_144)
    max_store_calls: int = Field(default=256, ge=1, le=1024)
    max_candidates: int = Field(default=256, ge=1, le=2048)
    per_node_limit: int = Field(default=32, ge=1, le=128)


class MultiHopEdge(BaseModel):
    id: str
    from_fact: int
    to_fact: int
    kind: Literal["extends", "derived_from", "contradicts", "supports", "subject_object"]
    link_id: int | None = None
    source_episode_id: int | None = None
    quote: str | None = None
    # Joins are backed by the two quoted facts, not an invented link quote.
    source_fact_ids: list[int] = Field(default_factory=list)


class MultiHopPath(BaseModel):
    fact_ids: list[int]
    edge_ids: list[str] = Field(default_factory=list)
    directions: list[Literal["forward", "reverse"]] = Field(default_factory=list)


class MultiHopCoverage(BaseModel):
    complete: bool = True
    truncated: bool = False
    reasons: list[str] = Field(default_factory=list)


class MultiHopCounts(BaseModel):
    store_calls: int = 0
    candidates: int = 0
    facts: int = 0
    edges: int = 0
    sources: int = 0
    rejected: int = 0
    output_bytes: int = 0


class MultiHopResult(BaseModel):
    facts: list[Fact] = Field(default_factory=list)
    edges: list[MultiHopEdge] = Field(default_factory=list)
    paths: list[MultiHopPath] = Field(default_factory=list)
    seed_fact_ids: list[int] = Field(default_factory=list)
    coverage: MultiHopCoverage = Field(default_factory=MultiHopCoverage)
    counts: MultiHopCounts = Field(default_factory=MultiHopCounts)


def _source_matches(episode: Episode, scope: TextFilter, excluded_session: str | None) -> bool:
    if (not all(tag in episode.tags for tag in scope.tags)
            or not all(episode.metadata.get(key) == value for key, value in scope.where.items())
            or (scope.conditions is not None and not scope.conditions.matches(episode.metadata))
            or (scope.kind is not None and episode.kind != scope.kind)
            or (scope.source_prefix is not None and (episode.source is None or not episode.source.startswith(scope.source_prefix)))
            or (excluded_session is not None and (episode.metadata.get("session_id") == excluded_session
                                                  or episode.source == excluded_session))):
        return False
    created = parse_rfc3339(episode.created_at)
    return ((scope.since is None or created >= parse_rfc3339(scope.since))
            and (scope.until is None or created <= parse_rfc3339(scope.until))
            and (scope.as_of is None or created <= parse_rfc3339(scope.as_of)))


class _Walker:
    def __init__(self, documents: MultiHopDocuments, space: str, scope: TextFilter,
                 limits: MultiHopLimits, include_history: bool, exclude_session_id: str | None) -> None:
        self.documents, self.space, self.scope, self.limits = documents, space, scope, limits
        self.include_history, self.exclude_session_id = include_history, exclude_session_id
        self.boundary = scope.as_of or now_rfc3339()
        # Validate caller timestamps before any store access, including an empty seed set.
        for value in (self.boundary, scope.since, scope.until):
            if value is not None:
                parse_rfc3339(value)
        self.result = MultiHopResult()
        self.facts: dict[int, Fact | None] = {}
        self.sources: dict[int, Episode | None] = {}
        self.accepted: dict[int, MultiHopPath] = {}
        self.edge_ids: set[str] = set()
        self.seen_links: set[int] = set()
        self.retained_links: dict[int, FactLink] = {}
        self.queue: deque[tuple[int, int]] = deque()

    def incomplete(self, reason: str, *, truncated: bool = True) -> None:
        coverage = self.result.coverage
        coverage.complete = False
        coverage.truncated = coverage.truncated or truncated
        if reason not in coverage.reasons:
            coverage.reasons.append(reason)

    def call(self) -> bool:
        if self.result.counts.store_calls >= self.limits.max_store_calls:
            self.incomplete("max_store_calls")
            return False
        self.result.counts.store_calls += 1
        return True

    def candidate(self) -> bool:
        if self.result.counts.candidates >= self.limits.max_candidates:
            self.incomplete("max_candidates")
            return False
        self.result.counts.candidates += 1
        return True

    async def source(self, episode_id: int | None, quote: str | None) -> bool:
        if episode_id is None or not quote:
            return False
        if episode_id not in self.sources:
            if not self.call():
                return False
            source = await self.documents.get_episode(self.space, episode_id)
            if source is not None:
                source = source.model_copy(deep=True)
            try:
                valid = (source is not None and source.space == self.space and source.episode_id == episode_id
                         and _source_matches(source, self.scope, self.exclude_session_id)
                         and parse_rfc3339(source.created_at) <= parse_rfc3339(self.boundary))
            except ValueError:
                valid = False
            self.sources[episode_id] = source if valid else None
        retained = self.sources[episode_id]
        return retained is not None and quote in retained.content

    async def fact(self, fact_id: int, expected: Fact | None = None) -> Fact | None:
        if fact_id not in self.facts:
            if not self.call():
                return None
            found = await self.documents.get_fact(self.space, fact_id)
            if found is not None:
                found = found.model_copy(deep=True)
            valid = found is not None and found.fact_id == fact_id and found.space == self.space
            if found is not None and valid:
                try:
                    valid = (not found.excluded and found.in_ledger
                             and parse_rfc3339(found.valid_from) <= parse_rfc3339(self.boundary)
                             and (self.include_history or found.holds_at(self.boundary))
                             and (self.scope.as_of is not None or self.include_history or found.status == "active"))
                except ValueError:
                    valid = False
                if valid:
                    valid = await self.source(found.source_episode_id, found.quote)
            self.facts[fact_id] = found if valid else None
        retained = self.facts[fact_id]
        if retained is None or (expected is not None and retained != expected):
            self.result.counts.rejected += 1
            return None
        return retained

    def add(self, fact: Fact, *, parent: int | None = None, edge: MultiHopEdge | None = None,
            direction: Literal["forward", "reverse"] = "forward", depth: int = 0) -> bool:
        existing = fact.fact_id in self.accepted
        if not existing and len(self.result.facts) >= self.limits.max_nodes:
            self.incomplete("max_nodes")
            return False
        if edge is not None and edge.id in self.edge_ids:
            return True
        if edge is not None and len(self.result.edges) >= self.limits.max_edges:
            self.incomplete("max_edges")
            return False
        path = MultiHopPath(fact_ids=[fact.fact_id])
        if parent is not None and edge is not None:
            previous = self.accepted[parent]
            path = MultiHopPath(fact_ids=[*previous.fact_ids, fact.fact_id],
                                edge_ids=[*previous.edge_ids, edge.id], directions=[*previous.directions, direction])
        if not existing:
            self.result.facts.append(fact)
            self.result.paths.append(path)
            if parent is None:
                self.result.seed_fact_ids.append(fact.fact_id)
        if edge is not None:
            self.result.edges.append(edge)
        # Reserve space for final counters and coverage notices. Full records are
        # either retained or omitted; quotes are never silently shortened.
        if len(self.result.model_dump_json().encode()) + 384 > self.limits.max_bytes:
            if edge is not None:
                self.result.edges.pop()
            if not existing:
                self.result.facts.pop()
                self.result.paths.pop()
                if parent is None:
                    self.result.seed_fact_ids.pop()
            self.incomplete("max_bytes")
            return False
        if not existing:
            self.accepted[fact.fact_id] = path
            self.queue.append((fact.fact_id, depth))
        if edge is not None:
            self.edge_ids.add(edge.id)
        return True

    def window(self) -> int:
        remaining = self.limits.max_candidates - self.result.counts.candidates
        if remaining <= 0:
            self.incomplete("max_candidates")
            return 0
        # A one-record sentinel detects an incomplete bounded candidate window.
        return min(self.limits.per_node_limit + 1, remaining)

    async def links(self, current: Fact, depth: int) -> None:
        if not isinstance(self.documents, BoundedIncidentLinks):
            self.incomplete("unsupported_bounded_links", truncated=False)
            return
        limit = self.window()
        if not limit or not self.call():
            return
        links = await self.documents.fact_links_from(self.space, current.fact_id, limit)
        if len(links) >= limit:
            self.incomplete("candidate_window")
        for link in links[:limit]:
            link = link.model_copy(deep=True)
            if not self.candidate():
                break
            if link.link_id in self.seen_links:
                continue
            self.seen_links.add(link.link_id)
            if link.space != self.space or current.fact_id not in (link.from_fact, link.to_fact):
                self.result.counts.rejected += 1
                continue
            try:
                valid_time = parse_rfc3339(link.created_at) <= parse_rfc3339(self.boundary)
            except ValueError:
                valid_time = False
            if not valid_time or not await self.source(link.source_episode_id, link.quote):
                self.result.counts.rejected += 1
                continue
            target = link.to_fact if link.from_fact == current.fact_id else link.from_fact
            fact = await self.fact(target)
            if fact is None:
                continue
            self.retained_links[link.link_id] = link
            self.add(fact, parent=current.fact_id, depth=depth + 1,
                     edge=MultiHopEdge(id=f"link:{link.link_id}", from_fact=link.from_fact, to_fact=link.to_fact,
                         kind=link.kind, link_id=link.link_id, source_episode_id=link.source_episode_id,
                         quote=link.quote, source_fact_ids=[link.from_fact, link.to_fact]),
                     direction="forward" if link.from_fact == current.fact_id else "reverse")

    async def subjects(self, current: Fact, depth: int) -> None:
        if not isinstance(self.documents, BoundedSubjectFacts):
            self.incomplete("unsupported_bounded_subjects", truncated=False)
            return
        if not current.object.strip():
            return
        subjects = dict.fromkeys((normalise_term(current.object, "subject"), current.object))
        remaining = self.limits.per_node_limit + 1
        for subject in subjects:
            limit = min(self.window(), remaining)
            if not limit or not self.call():
                return
            candidates = await self.documents.facts_by_subject(self.space, subject, limit)
            if len(candidates) >= limit:
                self.incomplete("candidate_window")
            remaining -= min(len(candidates), limit)
            for candidate in candidates[:limit]:
                if not self.candidate():
                    return
                if candidate.space != self.space or candidate.subject != subject:
                    self.result.counts.rejected += 1
                    continue
                fact = await self.fact(candidate.fact_id, expected=candidate)
                if fact is None:
                    continue
                self.add(fact, parent=current.fact_id, depth=depth + 1,
                         edge=MultiHopEdge(id=f"chain:{current.fact_id}:{fact.fact_id}", from_fact=current.fact_id,
                             to_fact=fact.fact_id, kind="subject_object", source_fact_ids=[current.fact_id, fact.fact_id]))

    async def revalidate(self) -> None:
        """Fresh bounded ledger/source checks, then prune dependent paths.

        A native revision guard also detects engine writes during these reads.
        Direct adapter writes must provide their own transaction discipline.
        Exhausted work budgets never turn unchecked cached quotes into output.
        """
        result = self.result
        if not result.facts:
            return
        revision: int | None = None
        if isinstance(self.documents, RevisionedDocuments):
            if self.call():
                revision = await self.documents.revision(self.space)
        checked_facts: list[Fact] = []
        for fact in result.facts:
            if self.call() and await self.documents.get_fact(self.space, fact.fact_id) == fact:
                checked_facts.append(fact)
        checked_edges: list[MultiHopEdge] = []
        for edge in result.edges:
            if edge.link_id is None:
                checked_edges.append(edge)
            elif not isinstance(self.documents, PointFactLinks):
                self.incomplete("unsupported_link_revalidation", truncated=False)
            elif self.call() and await self.documents.get_fact_link(self.space, edge.link_id) == self.retained_links[edge.link_id]:
                checked_edges.append(edge)
        self.sources.clear()
        valid_facts: set[int] = set()
        for fact in checked_facts:
            if await self.source(fact.source_episode_id, fact.quote):
                valid_facts.add(fact.fact_id)
        valid_edges: set[str] = set()
        for edge in checked_edges:
            if (edge.from_fact in valid_facts and edge.to_fact in valid_facts
                    and (edge.link_id is None or await self.source(edge.source_episode_id, edge.quote))):
                valid_edges.add(edge.id)
        if isinstance(self.documents, RevisionedDocuments):
            if not self.call() or revision is None or await self.documents.revision(self.space) != revision:
                valid_facts.clear()
                valid_edges.clear()
        paths = [path for path in result.paths if set(path.fact_ids) <= valid_facts and set(path.edge_ids) <= valid_edges]
        reachable = {path.fact_ids[-1] for path in paths}
        if len(reachable) != len(result.facts) or len(valid_edges) != len(result.edges):
            self.incomplete("stale_evidence", truncated=False)
        result.facts = [fact for fact in result.facts if fact.fact_id in reachable]
        result.paths = paths
        result.edges = [edge for edge in result.edges if edge.id in valid_edges
                        and edge.from_fact in reachable and edge.to_fact in reachable]
        result.seed_fact_ids = [fact_id for fact_id in result.seed_fact_ids if fact_id in reachable]

    def finish(self) -> MultiHopResult:
        result = self.result
        result.counts.facts, result.counts.edges = len(result.facts), len(result.edges)
        result.counts.sources = len({f.source_episode_id for f in result.facts}
            | {edge.source_episode_id for edge in result.edges if edge.source_episode_id is not None})
        # Decimal digit growth converges immediately; include the counter itself.
        for _ in range(4):
            result.counts.output_bytes = len(result.model_dump_json().encode())
        return result


async def expand_multihop(documents: MultiHopDocuments, space: str, *, seeds: RecallResult | None = None,
                          seed_fact_ids: Sequence[int] = (), scope: TextFilter | None = None,
                          limits: MultiHopLimits | None = None, include_history: bool = False,
                          exclude_session_id: str | None = None) -> MultiHopResult:
    """Return verified facts, retained relations and one discovery path per fact.

    ``seed_fact_ids`` must already be authorized by the caller. A recall seed
    must still equal its retained record. Scope and source quotes are checked
    for every seed, expanded fact and stored link. History admits closed facts
    that began by the boundary, while ordinary reads require interval validity.
    Contradictions remain typed edges, with both eligible claims retained.

    Coverage means exhaustion of these seeds' reachable, eligible ledger under
    the two supported traversal rules. It never claims query/answer completeness.
    Candidate windows are postfiltered and may miss eligible records; this is
    reported. Store work is bounded by calls and rows, output by serialized UTF-8
    bytes. Individual source reads use the existing document API and may load a
    large retained episode; this API does not promise a bound on source bytes.
    """
    check_space(space)
    walker = _Walker(documents, space, scope or TextFilter(), limits or MultiHopLimits(),
                     include_history, exclude_session_id)
    recalled = chain(seeds.facts, seeds.history if include_history else ()) if seeds is not None else iter(())
    candidates = chain(((fact.fact_id, fact) for fact in recalled), ((fact_id, None) for fact_id in seed_fact_ids))
    cap = walker.limits.max_candidates
    selected = list(islice(candidates, cap + 1))
    if len(selected) > cap:
        walker.incomplete("max_candidates")
    if seeds is not None and seeds.degraded:
        walker.incomplete("degraded_recall", truncated=False)
    for fact_id, expected in selected[:cap]:
        if not walker.candidate():
            break
        fact = await walker.fact(fact_id, expected=expected)
        if fact is not None:
            walker.add(fact)
    if not selected:
        walker.incomplete("no_seeds", truncated=False)
    while walker.queue:
        fact_id, depth = walker.queue.popleft()
        if depth >= walker.limits.max_hops:
            walker.incomplete("max_hops")
            continue
        fact = walker.facts[fact_id]
        if fact is not None:
            await walker.links(fact, depth)
            await walker.subjects(fact, depth)
    await walker.revalidate()
    return walker.finish()
