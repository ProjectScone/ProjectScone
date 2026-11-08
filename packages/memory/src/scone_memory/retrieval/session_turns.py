"""The turns around a recalled turn, read from its own session.

A conversation is remembered one turn per record, and a search finds the
turn whose words match the question. The turns either side of it often
hold what that turn leaves out: the question it answered, the reason given
after it, the date said the line before. So each turn a search found brings
up to ``radius`` turns either side of it from the same session, placed
right after it in the order they were said. A turn the search ranked below
its neighbour moves up to follow it; one it ranked above stays.

Which turns are neighbours is read from the store, not ranked: episodes of
the turn's kind whose ids lie within ``reach`` of its id, since episode ids
are the order the store received its records, and whose session field
(``session_id`` unless named) holds the same value. A conversation written
in order, as one batch or turn by turn, keeps its turns in that order;
other sessions' turns written between them are passed over, as long as the
neighbour lies within ``reach`` ids. A side that found fewer turns than
``radius`` is counted in ``short``: that is the session's first or last
turn, or a neighbour more than ``reach`` ids away; the count does not say
which.

A turn without a session field brings nothing and is counted. A store that
cannot page its episodes brings nothing, and says so; one that fails leaves
the result whole, and says so. A neighbour outside the caller's scope, in
an excluded session, or dated after ``now`` is not brought, and keeps its
place in the window: the turn beyond it is not brought in its stead, so no
turn shown is presented as following on from one it did not follow. Brought
passages score 0.0, since they were read rather than ranked, and are marked
superseded from the ledger as recall marks its own. A long neighbour brings
its first ``MAX_TURN_CHUNKS`` chunks only, counted in ``shortened``.

Nothing here calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.errors import InvalidInput
from ..core.models import Episode, RecallItem, RecallResult
from ..core.ports import DocumentStore, EpisodeInventory, TextFilter
from ..core.timeutil import now_rfc3339, parse_rfc3339
from . import supersession
from .multihop import _source_matches

#: Turns either side of a recalled turn one call may bring at most.
MAX_RADIUS = 4
#: Recalled turns one result may look around at most; each costs a few store reads.
MAX_SEEDS = 20
#: Episode ids looked at either side of a recalled turn: the default, and the most.
DEFAULT_REACH = 32
MAX_REACH = 100
#: A long neighbour's first chunks, in order, at most.
MAX_TURN_CHUNKS = 2
SESSION_FIELD = "session_id"


@dataclass(frozen=True)
class SessionTurns:
    """How many recalled turns were looked around, what that brought, and
    what it could not, or why nothing was looked at."""

    #: Recalled turns with a session that were looked around.
    seeds: int = 0
    #: Neighbouring turns found for them, each counted once.
    found: int = 0
    #: Turns the search had not returned at all.
    added: int = 0
    #: Turns the search ranked below their recalled turn, moved up to follow it.
    moved: int = 0
    #: Recalled turns past ``seeds`` that were not looked around.
    not_looked: int = 0
    #: Recalled turns among those looked at that named no session.
    no_session: int = 0
    #: Sides that found fewer turns than ``radius`` within ``reach``.
    short: int = 0
    #: Neighbours longer than ``MAX_TURN_CHUNKS`` chunks, brought in part.
    shortened: int = 0
    skipped: Optional[str] = None
    #: The passages placed after a recalled turn, added or moved, so a
    #: caller that packs the result can say how many of them it showed.
    brought_chunk_ids: tuple[int, ...] = ()

    def record(self) -> dict[str, object]:
        return {"seeds": self.seeds, "found": self.found, "added": self.added, "moved": self.moved,
                "not_looked": self.not_looked, "no_session": self.no_session, "short": self.short,
                "shortened": self.shortened, "skipped": self.skipped}


def _checked(radius: int, seeds: int, reach: int, session_field: str) -> None:
    for name, value, top, bottom in (("radius", radius, MAX_RADIUS, 0), ("seeds", seeds, MAX_SEEDS, 0),
                                     ("reach", reach, MAX_REACH, 1)):
        if type(value) is not int or not bottom <= value <= top:
            raise InvalidInput(f"{name} must be a whole number from {bottom} to {top}, not {value!r}")
    if not isinstance(session_field, str) or not session_field.strip():
        raise InvalidInput(f"session_field must name a metadata field, not {session_field!r}")


def _visible(episode: Episode, *, session: str, session_field: str, scope: TextFilter,
             excluded_session: Optional[str], now: str) -> bool:
    """Whether ``episode`` is a turn of ``session`` the caller may see."""
    return (episode.metadata.get(session_field) == session
            and parse_rfc3339(episode.created_at) <= parse_rfc3339(now)
            and _source_matches(episode, scope, excluded_session))


async def _around(documents: DocumentStore, space: str, seed: Episode, session: str, *, radius: int, reach: int,
                  session_field: str, scope: TextFilter, excluded_session: Optional[str],
                  now: str) -> tuple[list[Episode], int]:
    """Up to ``radius`` turns of ``session`` either side of ``seed`` within
    ``reach`` ids, in the order said, and how many sides came up short."""
    assert isinstance(documents, EpisodeInventory)
    seed_id = seed.episode_id
    # Newest first: the ids below the seed, nearest first.
    below = await documents.page_episodes(space, seed_id, reach, seed.kind)
    # The ids up to seed + reach (the page stops there); those above the seed
    # are the turns after it.
    above = await documents.page_episodes(space, seed_id + reach + 1, reach, seed.kind)

    def ours(episode: Episode) -> bool:
        return episode.metadata.get(session_field) == session

    # The window is the session's own turns, whoever may see them: a turn
    # the caller may not see keeps its place and is left out, so the next
    # one shown is never presented as following on directly.
    before = [e for e in below if seed_id - reach <= e.episode_id < seed_id and ours(e)][:radius]
    after = sorted((e for e in above if e.episode_id > seed_id and ours(e)),
                   key=lambda e: e.episode_id)[:radius]
    short = (len(before) < radius) + (len(after) < radius)
    window = sorted(before, key=lambda e: e.episode_id) + after
    return [e for e in window if _visible(e, session=session, session_field=session_field, scope=scope,
                                         excluded_session=excluded_session, now=now)], short


async def add_session_turns(result: RecallResult, documents: DocumentStore, space: str, *, radius: int,
                            scope: Optional[TextFilter] = None, excluded_session: Optional[str] = None,
                            seeds: int = MAX_SEEDS, reach: int = DEFAULT_REACH,
                            session_field: str = SESSION_FIELD,
                            now: Optional[str] = None) -> tuple[RecallResult, SessionTurns]:
    """``result`` with up to ``radius`` turns either side of each of its
    first ``seeds`` recalled turns, from the same session, placed right
    after that turn in the order said, and what that did.

    ``scope`` is the filter the search ran under, and ``excluded_session``
    a session the caller must not see; a turn outside either is not
    brought. ``now`` bounds what may be read (the wall clock by default) and
    is the ledger boundary for marking superseded."""
    _checked(radius, seeds, reach, session_field)
    if radius == 0:
        return result, SessionTurns(skipped="no turns either side were asked for")
    if not isinstance(documents, EpisodeInventory) or not callable(getattr(documents, "page_episodes", None)):
        return result, SessionTurns(skipped="the store cannot page its episodes")
    scope = scope if scope is not None else TextFilter()
    now = now if now is not None else now_rfc3339()
    order: list[int] = []
    session_of: dict[int, Optional[str]] = {}
    for item in result.items:
        if item.episode_id not in session_of:
            order.append(item.episode_id)
            session_of[item.episode_id] = item.metadata.get(session_field)
    looked, not_looked = order[:seeds], len(order) - min(len(order), seeds)
    no_session = sum(1 for episode_id in looked if not session_of[episode_id])
    present = {item.chunk_id for item in result.items}
    brought: dict[int, list[RecallItem]] = {}
    short = 0
    found: set[int] = set()
    long_turns: set[int] = set()
    try:
        for episode_id in looked:
            session = session_of[episode_id]
            if not session:
                continue
            seed = await documents.get_episode(space, episode_id)
            if seed is None:
                continue
            turns, seed_short = await _around(documents, space, seed, session, radius=radius, reach=reach,
                                              session_field=session_field, scope=scope,
                                              excluded_session=excluded_session, now=now)
            short += seed_short
            items: list[RecallItem] = []
            for neighbour in turns:
                found.add(neighbour.episode_id)
                chunks = sorted(await documents.chunks_of(space, neighbour.episode_id), key=lambda c: c.ordinal)
                if len(chunks) > MAX_TURN_CHUNKS:
                    long_turns.add(neighbour.episode_id)
                items.extend(
                    RecallItem(chunk_id=c.chunk_id, episode_id=neighbour.episode_id, text=c.text, score=0.0,
                               created_at=c.created_at, source=neighbour.source, tags=neighbour.tags,
                               metadata=dict(neighbour.metadata), start=c.start, end=c.end)
                    for c in chunks[:MAX_TURN_CHUNKS])
            brought[episode_id] = items
        degraded = list(result.degraded)
        new = list({item.chunk_id: item for items in brought.values() for item in items
                    if item.chunk_id not in present}.values())
        marked = {item.chunk_id: item for item in
                  await supersession.demote_superseded(documents, space, new, now, degraded=degraded)}
    except Exception as error:  # noqa: BLE001 - the result stands, and the failure is said
        failed = sorted({*result.degraded, f"session_turns: {type(error).__name__}: {error}"})
        return (result.model_copy(update={"degraded": failed}),
                SessionTurns(seeds=len(looked) - no_session, not_looked=not_looked, no_session=no_session,
                             skipped=f"the store failed: {type(error).__name__}"))
    # Each passage once, at the earliest of its places: a neighbour the
    # search ranked above its turn stays there; one it ranked below moves up
    # to follow the turn, with its own score, so a caller that cuts the list
    # keeps them together.
    ranked = {item.chunk_id: item for item in result.items}
    merged: list[RecallItem] = []
    added: list[int] = []
    moved: list[int] = []
    seen: set[int] = set()
    for item in result.items:
        if item.chunk_id not in seen:
            merged.append(item)
            seen.add(item.chunk_id)
        for extra in brought.get(item.episode_id, []):
            if extra.chunk_id in seen:
                continue
            seen.add(extra.chunk_id)
            if extra.chunk_id in present:
                merged.append(ranked[extra.chunk_id])
                moved.append(extra.chunk_id)
            else:
                merged.append(marked[extra.chunk_id])
                added.append(extra.chunk_id)
    return (result.model_copy(update={"items": merged, "degraded": sorted(set(degraded))}),
            SessionTurns(seeds=len(looked) - no_session, found=len(found), added=len(added), moved=len(moved),
                         not_looked=not_looked, no_session=no_session, short=short, shortened=len(long_turns),
                         brought_chunk_ids=tuple(added + moved)))
