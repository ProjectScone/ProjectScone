"""Read-only source inventory, ledger selection, profiles and status.

Hosts provide a document store and the clock or live identity callback needed
by a query. No engine instance, model call, or storage mutation is required.
Source-page scans are bounded; aggregate counts retain their existing scans.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, TypedDict

from ..core.affirmations import affirmation_store
from ..core.errors import InvalidInput
from ..core.models import Episode, Fact, Status
from ..core.ports import DocumentStore, SourcePage
from ..core.validation import (KINDS, STATUSES, check_space, normalise_metadata, normalise_term, normalise_time)

if TYPE_CHECKING:
    from .engine import MemoryEngine

SOURCE_WALK_PAGE = 200
SOURCE_WALK_READS = 25


class CatalogIdentity(TypedDict):
    embedder: str
    document_store: str
    vector_index: str
    abstention: Optional[dict]


@dataclass(frozen=True)
class RecentActivity:
    """One ``dynamic`` excerpt with the episode it was cut from."""

    episode_id: int
    excerpt: str
    created_at: str


@dataclass
class Profile:
    static_facts: list[Fact] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)
    #: ``dynamic`` with its evidence: same order, same excerpts, newest first.
    recent: list[RecentActivity] = field(default_factory=list)
    #: What the profile was made from: how many facts were read, what the
    #: read left out, the policy it kept to, and how often each shown
    #: claim was stated again.
    coverage: dict = field(default_factory=dict)


#: Facts read for one profile; past this the oldest are not read.
MAX_PROFILE_FACTS = 20_000
#: Claims whose restatements are counted before the profile is cut.
MAX_PROFILE_CANDIDATES = 200


def _predicates(names: object, what: str) -> frozenset[str]:
    if isinstance(names, (str, bytes)):
        raise InvalidInput(f"{what} takes a collection of predicates, not one string")
    if not isinstance(names, Iterable):
        raise InvalidInput(f"{what} takes a collection of predicates")
    chosen = set()
    for name in names:
        if not isinstance(name, str):
            raise InvalidInput(f"a profile predicate must be text, not {type(name).__name__}")
        chosen.add(normalise_term(name, "predicate"))
    return frozenset(chosen)


@dataclass(frozen=True)
class ProfilePolicy:
    """Which claims a profile is made of: only these predicates when any
    are named, and never those left out. Named as the ledger names
    predicates; nothing is inferred from a predicate's meaning."""

    predicates: frozenset[str] = frozenset()
    without: frozenset[str] = frozenset()

    @classmethod
    def of(cls, predicates: object = (), without: object = ()) -> "ProfilePolicy":
        return cls(_predicates(predicates, "predicates"), _predicates(without, "without"))

    def keeps(self, fact: Fact) -> bool:
        if self.predicates and fact.predicate not in self.predicates:
            return False
        return fact.predicate not in self.without

    def record(self) -> dict[str, list[str]]:
        return {"predicates": sorted(self.predicates), "without": sorted(self.without)}


async def facts(
    documents: DocumentStore,
    space: str,
    include_closed: bool = False,
    as_of: Optional[str] = None,
    status: Optional[str] = None,
    include_excluded: bool = False,
) -> list[Fact]:
    """Ledger facts: active by default, closed too with include_closed,
    those holding at ``as_of`` when given. ``status`` selects one
    status instead (``proposed`` lists what awaits review). Excluded
    facts are left out unless asked for."""
    check_space(space)
    if status is not None and status not in STATUSES:
        raise InvalidInput(f"status must be one of {STATUSES}, got {status!r}")
    found = await documents.list_facts(space, include_closed=True)
    if status is not None:
        found = [f for f in found if f.status == status]
    elif as_of is not None:
        boundary = normalise_time(as_of)
        found = [f for f in found if f.holds_at(boundary)]
    else:
        allowed = ("active", "closed") if include_closed else ("active",)
        found = [f for f in found if f.status in allowed]
    if not include_excluded:
        found = [f for f in found if not f.excluded]
    return sorted(found, key=lambda f: f.fact_id)


async def profile(engine: "MemoryEngine", space: str, limit: int = 10, *,
                  policy: Optional[ProfilePolicy] = None) -> Profile:
    """Who the space is about, without being asked a question: the claims
    that hold, what the space says most often and most recently first,
    each with the evidence it rests on; then its recent activity.

    Which predicates count is the policy's to say. The read behind it is
    bounded and disclosed: a profile of what was read is not a profile of
    everything, and it does not pretend to be."""
    from ..entities.read import read_ledger

    check_space(space)
    limit = max(1, min(limit, 50))
    documents = engine.documents
    now = engine.clock()
    kept = policy or ProfilePolicy()
    read = await read_ledger(engine, space, max_facts=MAX_PROFILE_FACTS)
    active = [fact for fact in read.facts
              if fact.status == "active" and not fact.excluded and fact.holds_at(now) and kept.keeps(fact)]
    # Newest first, and only as many as restatements will be counted for:
    # each count is a read of its own.
    active.sort(key=lambda fact: (fact.valid_from, fact.fact_id), reverse=True)
    said: dict[int, int] = {}
    affirmations = affirmation_store(documents)
    for fact in active[:MAX_PROFILE_CANDIDATES]:
        said[fact.fact_id] = len(await affirmations.affirmations(space, fact.fact_id)) if affirmations else 0
    # Newest first, then stably by how often each was stated again, so the
    # newer of two claims said as often still leads.
    ranked = sorted(active[:MAX_PROFILE_CANDIDATES],
                    key=lambda fact: (fact.valid_from, fact.fact_id), reverse=True)
    ranked.sort(key=lambda fact: -said[fact.fact_id])
    shown = ranked[:limit]
    recent = [RecentActivity(e.episode_id, e.content[:200], e.created_at)
              for e in await documents.recent_episodes(space, limit)]
    return Profile(
        static_facts=shown,
        dynamic=[r.excerpt for r in recent],
        recent=recent,
        coverage={"facts_read": len(read.facts), "reasons": list(read.reasons), "policy": kept.record(),
                  "restatements": {str(fact.fact_id): said[fact.fact_id] for fact in shown},
                  "candidates": min(len(active), MAX_PROFILE_CANDIDATES)},
    )



async def tags(documents: DocumentStore, space: str) -> dict[str, int]:
    check_space(space)
    return dict(sorted((await documents.counts(space)).tags.items()))


async def pending_distillation(documents: DocumentStore, space: str) -> int:
    """Episodes no claim cites yet. The same definition the distiller
    and memory_pending use, so the three surfaces agree."""
    check_space(space)
    counts = await documents.counts(space)
    if counts.episodes == 0:
        return 0
    referenced = {f.source_episode_id for f in await documents.list_facts(space, include_closed=True)}
    return sum(1 for e in await documents.recent_episodes(space, counts.episodes) if e.episode_id not in referenced)


async def cited_episode_ids(documents: DocumentStore, space: str) -> set[int]:
    """The episodes some claim cites, whatever the claim's status."""
    check_space(space)
    return {f.source_episode_id for f in await documents.list_facts(space, include_closed=True)
            if f.source_episode_id is not None}


async def source_page(documents: DocumentStore, space: str, *, before: Optional[int] = None,
                      limit: int = 25, kind: Optional[str] = None,
                      conditions: Mapping[str, object] | None = None,
                      walk_page: int = SOURCE_WALK_PAGE, walk_reads: int = SOURCE_WALK_READS) -> SourcePage:
    """Browse retained sources by descending ID, not relevance or source date.

    Newer inserts are found by restarting the walk. Deleting the boundary
    record does not invalidate the next page. This is not a frozen snapshot.

    With ``conditions``, the walk keeps reading until the page is full,
    rather than filtering one page and handing back what survives. The
    latter turns a page of twenty-five into a page of one and makes the
    page size mean nothing. The walk is bounded, because a filter that
    matches nothing would otherwise read a whole space to prove it, and
    reaching that bound is reported as more to come rather than as the
    end.
    """
    check_space(space)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise InvalidInput("limit must be an integer from 1 through 100")
    if before is not None and (isinstance(before, bool) or not isinstance(before, int) or not 1 <= before <= 2**63-1):
        raise InvalidInput("before must be a positive signed 64-bit episode ID")
    if kind is not None and kind not in KINDS:
        raise InvalidInput(f"kind must be one of {KINDS}")
    page = getattr(documents, "page_episodes", None)
    if not callable(page):
        raise InvalidInput("this document store does not implement source inventory")
    if conditions is None:
        rows = await page(space, before, limit + 1, kind)
        more = len(rows) > limit
        return SourcePage(rows[:limit], more, rows[limit-1].episode_id if more else None)

    from ..retrieval.filters import parse_filter

    narrow = parse_filter(conditions)
    kept: list[Episode] = []
    cursor, reads, ended = before, 0, False
    while len(kept) < limit and reads < walk_reads:
        batch = await page(space, cursor, walk_page, kind)
        reads += 1
        filled = False
        for episode in batch:
            cursor = episode.episode_id
            if narrow.matches(episode.metadata):
                kept.append(episode)
                if len(kept) == limit:
                    filled = True
                    break
        if filled:
            # Stopped on a full page, not on the end of the store: the
            # rest of this batch has not been looked at yet.
            break
        if len(batch) < walk_page:
            # The store had nothing more to give, so this is the end of
            # the space rather than the end of what was read.
            ended = True
            break
    return SourcePage(kept, not ended, None if ended else cursor)


async def episodes(documents: DocumentStore, space: str, where: Mapping[str, str], limit: Optional[int] = None) -> list[Episode]:
    """The episodes whose metadata matches every ``where`` pair, oldest
    first by (created_at, episode_id); with ``limit``, the newest N of
    them in that same order. This is a walk over the space's episodes,
    fine for a session's turns, not a query language."""
    check_space(space)
    clean = normalise_metadata(where)
    if not clean:
        raise InvalidInput("episodes() needs at least one where pair")
    counts = await documents.counts(space)
    found = [
        e for e in await documents.recent_episodes(space, max(counts.episodes, 1))
        if all(e.metadata.get(k) == v for k, v in clean.items())
    ]
    found.sort(key=lambda e: (e.created_at, e.episode_id))
    return found[-limit:] if limit else found


async def scopes(documents: DocumentStore, space: str) -> dict[str, dict[str, int]]:
    """Episode counts per metadata key and value: which users, agents
    and sessions have memory here. Walks the episodes, which is fine
    for an overview and avoids asking every store for a new query."""
    check_space(space)
    counts = await documents.counts(space)
    out: dict[str, dict[str, int]] = {}
    for episode in await documents.recent_episodes(space, max(counts.episodes, 1)):
        for key, value in episode.metadata.items():
            out.setdefault(key, {})
            out[key][value] = out[key].get(value, 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


async def status(documents: DocumentStore, space: str, *, identity: Callable[[], CatalogIdentity]) -> Status:
    check_space(space)
    counts = await documents.counts(space)
    pending = sum(1 for f in await documents.list_facts(space, include_closed=True) if f.status == "proposed")
    return Status(
        space=space,
        episodes=counts.episodes,
        chunks=counts.chunks,
        bytes=counts.bytes,
        pending_review=pending,
        revision=await documents.revision(space),
        **identity(),
    )
