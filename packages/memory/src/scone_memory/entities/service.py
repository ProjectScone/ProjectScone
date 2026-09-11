"""Hold each space's projection between graph requests.

Reading a whole ledger and projecting it is the cost of every graph view,
so the engine keeps, per space, the ledger read and the views built from
it (``engine.entities``):

- **Same revision:** a view comes back without touching the store beyond
  one revision check.
- **New revision:** the ledger is read again. If every fact is as it was
  (an episode was stored, say), the views are kept and restamped with the
  new revision; otherwise they are dropped and rebuilt on demand.
- **Moments:** a ``current`` or ``history`` view counts facts by their
  valid_from and valid_until, so one built for a moment is exact until the
  next such boundary. Views are keyed by that window, and time passing
  costs a rebuild but no read.
- **Bounds:** an inconsistent read is never kept, a deleted space is
  forgotten, and the facts held across spaces stay under
  ``MAX_HELD_FACTS``, dropping the least recently used space first.

Requests that arrive together for a space share one read. Nothing on a
recall path builds a projection; it is built only for a graph request.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from ..core.models import Fact
from ..core.timeutil import parse_rfc3339
from .project import EntityProjection, project_entities
from . import read as reading
from .read import LedgerRead, read_ledger

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine
    from .view import StatusMode

MAX_HELD_SPACES = 8
#: Facts held across every cached space; past it, the least recent go.
MAX_HELD_FACTS = 200_000
MAX_VIEWS_PER_SPACE = 16

_Key = tuple[str, datetime | None, datetime | None]


def _ledger_digest(facts: tuple[Fact, ...]) -> str:
    """Changes when any field of any fact does. repr is exact for every
    string the ledger holds, lone surrogates included."""
    rows = hashlib.sha256()
    for fact in facts:
        rows.update(repr(sorted(vars(fact).items())).encode("utf-8", "surrogatepass"))
    return rows.hexdigest()


def _boundaries(facts: tuple[Fact, ...]) -> list[datetime]:
    moments = {parse_rfc3339(fact.valid_from) for fact in facts}
    moments |= {parse_rfc3339(fact.valid_until) for fact in facts if fact.valid_until is not None}
    return sorted(moments)


@dataclass
class _Held:
    ledger: LedgerRead
    digest: str
    boundaries: list[datetime]
    views: OrderedDict[_Key, tuple[EntityProjection, int]]

    def key(self, mode: "StatusMode", when: datetime) -> _Key:
        if mode not in ("current", "history"):
            return (mode, None, None)
        index = bisect.bisect_right(self.boundaries, when)
        return (mode, self.boundaries[index - 1] if index else None,
                self.boundaries[index] if index < len(self.boundaries) else None)


class EntityService:
    def __init__(self, engine: "MemoryEngine") -> None:
        self._engine = engine
        self._held: OrderedDict[str, _Held] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def cached(self, space: str) -> EntityProjection | None:
        """The most recently used view held for the space, of any revision,
        with no I/O; None when nothing is held."""
        held = self._held.get(space)
        if held is None or not held.views:
            return None
        return next(reversed(held.views.values()))[0]

    def forget(self, space: str) -> None:
        self._held.pop(space, None)
        self._locks.pop(space, None)

    def clear(self) -> None:
        self._held.clear()
        self._locks.clear()

    async def projection(self, space: str, *, mode: "StatusMode",
                         when: datetime) -> tuple[EntityProjection, dict[str, object]]:
        """The projection of the facts that count in ``mode`` at ``when``,
        with what the read behind it covered."""
        from .view import counts

        held = await self._ledger(space)
        key = held.key(mode, when)
        found = held.views.get(key)
        if found is None:
            facts = [fact for fact in held.ledger.facts
                     if counts(fact.status, fact.excluded, fact.valid_from, fact.valid_until, mode, when)]
            found = (project_entities(space, facts, revision=held.ledger.revision), len(facts))
            if self._held.get(space) is held:
                held.views[key] = found
                while len(held.views) > MAX_VIEWS_PER_SPACE:
                    held.views.popitem(last=False)
        else:
            held.views.move_to_end(key)
        projection, counted = found
        return projection, {"facts_read": len(held.ledger.facts), "facts_counted": counted,
                            "facts_limit": held.ledger.limit,
                            "reasons": list(held.ledger.reasons), "read_mode": held.ledger.read_mode}

    async def _ledger(self, space: str) -> _Held:
        held = self._current(space, await self._engine.revision(space))
        if held is not None:
            return held
        async with self._locks.setdefault(space, asyncio.Lock()):
            previous = self._held.get(space)
            held = self._current(space, await self._engine.revision(space))
            if held is not None:
                return held
            ledger = await read_ledger(self._engine, space)
            digest = _ledger_digest(ledger.facts)
            if previous is not None and previous.digest == digest:
                views = OrderedDict((key, (replace(projection, revision=ledger.revision), counted))
                                    for key, (projection, counted) in previous.views.items())
                fresh = _Held(ledger, digest, previous.boundaries, views)
            else:
                fresh = _Held(ledger, digest, _boundaries(ledger.facts), OrderedDict())
            if ledger.consistent:
                self._keep(space, fresh)
            return fresh

    def _current(self, space: str, revision: int) -> _Held | None:
        held = self._held.get(space)
        if held is None or held.ledger.revision != revision or held.ledger.limit != reading.MAX_FACTS:
            return None
        self._held.move_to_end(space)
        return held

    def _keep(self, space: str, held: _Held) -> None:
        self._held[space] = held
        self._held.move_to_end(space)
        while self._held and (len(self._held) > MAX_HELD_SPACES
                              or sum(len(item.ledger.facts) for item in self._held.values()) > MAX_HELD_FACTS):
            self._held.popitem(last=False)
