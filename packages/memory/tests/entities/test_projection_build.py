"""A slow projection never holds a graph request past its budget.

A build that runs past the budget keeps going in the background, and the
request is told to come back; the next request finds the view ready. The
same view asked for at once is built once. A large build runs off the
event loop, so other requests are served meanwhile, and closing the
engine stops what is still building.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import service as service_module
from scone_memory.entities.service import ProjectionBuilding
from scone_memory.core.timeutil import parse_rfc3339

WHEN = parse_rfc3339("2025-01-01T00:00:00Z")


class Slow(InMemoryDocumentStore):
    delay = 0.0

    async def page_facts(self, space, before_id, limit):
        await asyncio.sleep(self.delay)
        return await super().page_facts(space, before_id, limit)


async def engine_with(store) -> MemoryEngine:
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    return engine


async def test_a_build_past_its_budget_finishes_in_the_background():
    store = Slow()
    engine = await engine_with(store)
    store.delay = 0.2
    with pytest.raises(ProjectionBuilding):
        await engine.entities.projection("alpha", mode="current", when=WHEN, timeout=0.01)
    await asyncio.sleep(0.4)
    projection, _ = await engine.entities.projection("alpha", mode="current", when=WHEN, timeout=0.01)
    assert {entity.key for entity in projection.entities} == {"alice", "acme"}


async def test_one_view_asked_for_at_once_is_built_once(monkeypatch):
    engine = await engine_with(Slow())
    built = []
    real = service_module.project_entities
    monkeypatch.setattr(service_module, "project_entities", lambda *a, **k: built.append(1) or real(*a, **k))
    await asyncio.gather(*(engine.entities.projection("alpha", mode="current", when=WHEN) for _ in range(4)))
    assert built == [1]


async def test_a_large_build_runs_off_the_event_loop(monkeypatch):
    engine = await engine_with(Slow())
    monkeypatch.setattr(service_module, "INLINE_FACTS", 0)
    threads = []
    real = service_module.project_entities
    monkeypatch.setattr(service_module, "project_entities",
                        lambda *a, **k: threads.append(threading.get_ident()) or real(*a, **k))
    await engine.entities.projection("alpha", mode="current", when=WHEN)
    assert threads and threads[0] != threading.get_ident()


async def test_closing_the_engine_stops_what_is_still_building():
    store = Slow()
    engine = await engine_with(store)
    store.delay = 5.0
    with pytest.raises(ProjectionBuilding):
        await engine.entities.projection("alpha", mode="current", when=WHEN, timeout=0.01)
    running = list(engine.entities.building())
    assert running
    await engine.close()
    await asyncio.sleep(0)
    assert all(task.cancelled() or task.done() for task in running) and not engine.entities.building()


class Gated(service_module.EntityService):
    """Holds each projection build until released."""

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.inputs: list[int] = []

    async def _project(self, space, facts, revision, *args, **kwargs):
        # `*args` on purpose: this override exists to hold a build open,
        # not to restate the signature it is standing in for. Spelling
        # out three positional parameters is how it drifted -- the real
        # `_project` gained `meanings`, `source` and `why` with the
        # vocabulary work, the override kept taking three, and every
        # call raised `TypeError` *before* `started.set()` ran. The test
        # then waited on an event nothing would ever set, which is how
        # one changed signature became a suite that hung at 41% with no
        # failing test name.
        self.inputs.append(len(facts))
        self.started.set()
        await self.release.wait()
        return await super()._project(space, facts, revision, *args, **kwargs)


async def begun(gated, build, *, seconds=5.0):
    """Wait for a held build to begin, or raise what stopped it.

    `await event.wait()` on its own waits forever when the code that
    should set the event raised first, and reports nothing about why.
    Racing the event against the build's own task means a failure
    surfaces as that failure, with its traceback, and a genuine hang
    surfaces as a bounded assertion instead of a dead run.
    """
    waiting = asyncio.ensure_future(gated.started.wait())
    done, pending = await asyncio.wait({waiting, build}, timeout=seconds,
                                       return_when=asyncio.FIRST_COMPLETED)
    if waiting in done:
        return
    for task in pending:
        task.cancel()
    if build in done:
        build.result()  # re-raises whatever actually happened
    raise AssertionError(f"the build did not begin within {seconds}s")


async def test_a_build_is_shared_only_by_requests_reading_the_same_ledger(monkeypatch):
    """A build begun under one read is not handed to a request whose read
    differs (here, a lower cap): its projection would not match its coverage."""
    from scone_memory.entities import read

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(3):
        await engine.assert_fact("alpha", f"person {number}", "knows", "Bob", valid_from="2024-01-01T00:00:00Z")
    engine.entities = gated = Gated(engine)
    monkeypatch.setattr(read, "MAX_FACTS", 3)
    first = asyncio.ensure_future(gated.projection("alpha", mode="all", when=WHEN))
    await begun(gated, first)
    monkeypatch.setattr(read, "MAX_FACTS", 1)
    second = asyncio.ensure_future(gated.projection("alpha", mode="all", when=WHEN))
    for _ in range(10):
        await asyncio.sleep(0)
    gated.release.set()
    (wide, wide_read), (narrow, narrow_read) = await asyncio.gather(first, second)
    assert len(wide.roles) == wide_read["facts_counted"] == 3
    assert len(narrow.roles) == narrow_read["facts_counted"] == narrow_read["facts_read"] == 1


class Failing(InMemoryDocumentStore):
    async def page_facts(self, space, before_id, limit):
        raise TimeoutError("storage read timed out")


async def test_a_store_that_times_out_is_a_failure_not_a_build_in_progress():
    engine = await MemoryEngine(Failing(), InMemoryVectorIndex(), HashEmbedder()).open()
    with pytest.raises(TimeoutError, match="storage read timed out"):
        await engine.entities.projection("alpha", mode="current", when=WHEN, timeout=1.0)
    assert not engine.entities.building()


async def test_admission_counts_build_jobs_not_the_requests_waiting_on_them(monkeypatch):
    """Past MAX_BUILDS jobs, a new view waits (library call) or is turned away
    at once (request with a budget); a view already built is still served,
    and requests joining a build in flight take no slot of their own."""
    from datetime import timedelta

    monkeypatch.setattr(service_module, "MAX_BUILDS", 2)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for day in range(1, 7):
        await engine.assert_fact("alpha", f"person {day}", "knows", "Bob", valid_from=f"2025-01-0{day}T00:00:00Z")
    engine.entities = gated = Gated(engine)
    gated.release.set()
    warm, _ = await gated.projection("alpha", mode="all", when=WHEN)
    gated.release.clear()
    windows = [parse_rfc3339(f"2025-01-0{day}T12:00:00Z") for day in range(1, 6)]
    untimed = [asyncio.ensure_future(gated.projection("alpha", mode="current", when=when)) for when in windows]
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(gated._projecting) == 2
    joiners = await asyncio.gather(*(gated.projection("alpha", mode="current", when=windows[0], timeout=0.01)
                                     for _ in range(8)), return_exceptions=True)
    assert all(isinstance(outcome, ProjectionBuilding) for outcome in joiners) and len(gated._projecting) == 2
    waiting = len(gated.building())
    with pytest.raises(ProjectionBuilding):
        await gated.projection("alpha", mode="current", when=windows[0] + timedelta(days=5), timeout=0.01)
    await asyncio.sleep(0)
    assert len(gated.building()) == waiting  # turned away, not left queued to start later
    served, _ = await gated.projection("alpha", mode="all", when=WHEN, timeout=0.01)
    assert served is warm
    gated.release.set()
    done = await asyncio.gather(*untimed)
    assert [len(projection.roles) for projection, _ in done] == [1, 2, 3, 4, 5]
    await engine.close()


async def test_closing_waits_for_a_worker_thread_still_projecting(monkeypatch):
    """A thread cannot be cancelled; close returns only once it has finished,
    so nothing of the engine is still running afterwards."""
    import time

    monkeypatch.setattr(service_module, "INLINE_FACTS", 0)
    finished = []
    real = service_module.project_entities

    def slow_projection(*args, **kwargs):
        time.sleep(0.3)
        result = real(*args, **kwargs)
        finished.append(True)
        return result

    monkeypatch.setattr(service_module, "project_entities", slow_projection)
    engine = await engine_with(Slow())
    with pytest.raises(ProjectionBuilding):
        await engine.entities.projection("alpha", mode="current", when=WHEN, timeout=0.01)
    await engine.close()
    assert finished == [True]


async def test_nothing_is_admitted_or_kept_once_closing_begins(monkeypatch):
    """While close waits for a worker, a new request must not start another
    worker (or a new pool), and nothing built may land in the cache after."""
    from scone_memory.entities.service import EntityServiceClosed

    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    real = service_module.project_entities

    def gated(*args, **kwargs):
        entered[1 if entered[0].is_set() else 0].set()
        release.wait(3)
        return real(*args, **kwargs)

    monkeypatch.setattr(service_module, "INLINE_FACTS", 0)
    monkeypatch.setattr(service_module, "project_entities", gated)
    engine = await engine_with(Slow())
    try:
        with pytest.raises(ProjectionBuilding):
            await engine.entities.projection("alpha", mode="all", when=WHEN, timeout=0.01)
        await asyncio.to_thread(entered[0].wait, 1)
        closing = asyncio.ensure_future(engine.close())
        await asyncio.sleep(0.02)
        assert not closing.done()
        with pytest.raises(EntityServiceClosed):
            await engine.entities.projection("alpha", mode="current", when=WHEN, timeout=0.01)
        release.set()
        await asyncio.wait_for(closing, 2)
        await asyncio.sleep(0.02)
        assert not entered[1].is_set() and engine.entities.cached("alpha") is None and not engine.entities.building()
    finally:
        release.set()
