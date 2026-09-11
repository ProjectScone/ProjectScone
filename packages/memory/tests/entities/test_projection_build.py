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

    async def _project(self, space, facts, revision):
        self.inputs.append(len(facts))
        self.started.set()
        await self.release.wait()
        return await super()._project(space, facts, revision)


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
    await gated.started.wait()
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
