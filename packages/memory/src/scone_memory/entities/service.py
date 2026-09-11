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

Requests that arrive together for a space share one read, and one view
asked for at once is built once. A view over more than ``INLINE_FACTS``
facts is projected in a worker thread, so the event loop keeps serving
other requests. A caller may give a budget: past it the build carries on
in the background and the caller gets ``ProjectionBuilding`` (HTTP 503 with
Retry-After), and the next request finds the view ready. Closing the
engine cancels what is still building. Nothing on a recall path builds a
projection; it is built only for a graph request.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for_futures
from functools import partial
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
#: Seconds a graph request waits for a projection before it is told to
#: come back; the build continues meanwhile.
BUILD_TIMEOUT = 5.0
#: Views over more facts than this are projected off the event loop.
INLINE_FACTS = 2_000
#: Builds in flight at once, waiting or running. A request past it is told
#: to come back and starts nothing.
MAX_BUILDS = 8
#: Worker threads projecting at once, per engine.
WORKERS = 2


class ProjectionBuilding(Exception):
    """The projection is still being built; the request should come back."""

    def __init__(self, space: str) -> None:
        super().__init__(f"the entity projection for {space!r} is still being built; try again shortly")

_Key = tuple[str, datetime | None, datetime | None]
#: One build: the view's key within one ledger read (revision, cap and rows).
_Slot = tuple[str, int, int, str, _Key]


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
        self._projecting: dict[_Slot, asyncio.Future[tuple[EntityProjection, int]]] = {}
        self._background: set[asyncio.Task[tuple[EntityProjection, dict[str, object]]]] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._workers: set[Future[EntityProjection]] = set()

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
        """Forget every space and cancel whatever is still building."""
        for task in list(self._background):
            task.cancel()
        for pending in list(self._projecting.values()):
            pending.cancel()
        self._held.clear()
        self._locks.clear()
        self._projecting.clear()

    async def aclose(self) -> None:
        """Cancel every build and return only once no worker thread is still
        projecting: a running thread cannot be cancelled, so it is waited for."""
        self.clear()
        executor, self._executor = self._executor, None
        if executor is None:
            return
        executor.shutdown(wait=False, cancel_futures=True)
        running = [worker for worker in self._workers if not worker.done()]
        if running:
            await asyncio.to_thread(wait_for_futures, running)
        self._workers.clear()

    def building(self) -> set[asyncio.Task[tuple[EntityProjection, dict[str, object]]]]:
        """Builds still running, including those whose caller stopped waiting."""
        return {task for task in self._background if not task.done()}

    async def projection(self, space: str, *, mode: "StatusMode", when: datetime,
                         timeout: float | None = None) -> tuple[EntityProjection, dict[str, object]]:
        """The projection of the facts that count in ``mode`` at ``when``,
        with what the read behind it covered. Past ``timeout`` seconds the
        build continues in the background and ``ProjectionBuilding`` is raised;
        with ``MAX_BUILDS`` already in flight, it is raised at once and nothing
        starts."""
        if timeout is not None and len(self.building()) >= MAX_BUILDS:
            raise ProjectionBuilding(space)
        task = asyncio.ensure_future(self._view(space, mode, when))
        self._background.add(task)
        task.add_done_callback(self._settled)
        if timeout is None:
            return await task
        # asyncio.wait never cancels the build and never mistakes a failure
        # inside it (a store's own TimeoutError, say) for the budget running out.
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            raise ProjectionBuilding(space)
        return task.result()

    def _settled(self, task: asyncio.Task[tuple[EntityProjection, dict[str, object]]]) -> None:
        self._background.discard(task)
        if not task.cancelled():
            task.exception()  # a failure nobody waited for is still retrieved, not logged as lost

    def _projected(self, slot: _Slot, _done: object) -> None:
        self._projecting.pop(slot, None)

    async def _project(self, space: str, facts: list[Fact], revision: int) -> tuple[EntityProjection, int]:
        if len(facts) <= INLINE_FACTS:
            return project_entities(space, facts, revision=revision), len(facts)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="scone-projection")
        worker = self._executor.submit(partial(project_entities, space, facts, revision=revision))
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)
        return await asyncio.wrap_future(worker), len(facts)

    async def _view(self, space: str, mode: "StatusMode",
                    when: datetime) -> tuple[EntityProjection, dict[str, object]]:
        from .view import counts

        held = await self._ledger(space)
        key = held.key(mode, when)
        found = held.views.get(key)
        if found is None:
            slot = (space, held.ledger.revision, held.ledger.limit, held.digest, key)
            pending = self._projecting.get(slot)
            if pending is None:
                facts = [fact for fact in held.ledger.facts
                         if counts(fact.status, fact.excluded, fact.valid_from, fact.valid_until, mode, when)]
                pending = asyncio.ensure_future(self._project(space, facts, held.ledger.revision))
                self._projecting[slot] = pending
                pending.add_done_callback(partial(self._projected, slot))
            found = await asyncio.shield(pending)
            if self._held.get(space) is held and key not in held.views:
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
