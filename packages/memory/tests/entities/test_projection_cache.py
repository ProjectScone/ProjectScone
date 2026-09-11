"""Graph views reuse the projection until the ledger, or the moment, moves.

A projection is held per space with the ledger read it came from. Asked
again at the same revision, the space is not read at all. A write makes
the next view read again; a write that left every fact as it was (an
episode, say) re-reads but keeps its views, restamped to the new
revision. A view for a moment stays exact until the next valid_from or
valid_until passes. An inconsistent read is never kept, a deleted space
is forgotten, and the held facts across spaces stay under a budget.
"""
from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import service as service_module
from scone_memory.entities.read import load_projection


class Counting(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self.pages = 0
        self.moving = 0

    async def page_facts(self, space, before_id, limit):
        self.pages += 1
        await asyncio.sleep(0)
        rows = await super().page_facts(space, before_id, limit)
        if self.moving:
            self.moving -= 1
            await self.bump_revision(space)
        return rows


class Clock:
    def __init__(self, now: str) -> None:
        self.now = now

    def __call__(self) -> str:
        return self.now


async def engine_with(store: Counting, clock: Clock | None = None) -> MemoryEngine:
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                **({"clock": clock} if clock else {})).open()
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "acme", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    return engine


async def test_a_second_view_at_the_same_revision_reads_nothing():
    store = Counting()
    engine = await engine_with(store)
    first, _ = await load_projection(engine, "alpha", mode="current")
    read = store.pages
    second, coverage = await load_projection(engine, "alpha", mode="current")
    assert store.pages == read and second is first and coverage["facts_counted"] == 2


async def test_a_new_fact_makes_the_next_view_read_again():
    store = Counting()
    engine = await engine_with(store)
    first, _ = await load_projection(engine, "alpha", mode="current")
    await engine.assert_fact("alpha", "bob", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    second, _ = await load_projection(engine, "alpha", mode="current")
    assert second.digest != first.digest and "bob" in {entity.key for entity in second.entities}
    assert second.revision == await engine.revision("alpha")


async def test_a_write_that_changed_no_fact_keeps_the_view_restamped(monkeypatch):
    store = Counting()
    engine = await engine_with(store)
    first, _ = await load_projection(engine, "alpha", mode="current")
    built = []
    real = service_module.project_entities
    monkeypatch.setattr(service_module, "project_entities", lambda *a, **k: built.append(1) or real(*a, **k))
    await engine.remember("alpha", "A note about the weather.")
    assert await engine.revision("alpha") > first.revision
    second, _ = await load_projection(engine, "alpha", mode="current")
    assert built == [] and second.digest == first.digest and second.revision == await engine.revision("alpha")


async def test_a_view_follows_the_clock_past_a_fact_that_begins_later():
    clock = Clock("2025-01-01T00:00:00Z")
    store = Counting()
    engine = await engine_with(store, clock)
    await engine.assert_fact("alpha", "carol", "works_at", "Acme", valid_from="2026-01-01T00:00:00Z")
    before, _ = await load_projection(engine, "alpha", mode="current")
    read = store.pages
    clock.now = "2025-06-01T00:00:00Z"
    still, _ = await load_projection(engine, "alpha", mode="current")
    clock.now = "2026-06-01T00:00:00Z"
    after, _ = await load_projection(engine, "alpha", mode="current")
    assert still is before and "carol" not in {entity.key for entity in before.entities}
    assert "carol" in {entity.key for entity in after.entities} and store.pages == read


async def test_an_inconsistent_read_is_never_kept():
    store = Counting()
    engine = await engine_with(store)
    store.moving = 100
    _, coverage = await load_projection(engine, "alpha", mode="current")
    assert "ledger_changed_during_read" in coverage["reasons"]
    # Not even the accessor that asks for no I/O may hand out a torn read.
    assert engine.entities.cached("alpha") is None
    store.moving = 0
    read = store.pages
    _, again = await load_projection(engine, "alpha", mode="current")
    assert store.pages > read and again["reasons"] == []


async def test_views_asked_at_once_share_one_read():
    store = Counting()
    engine = await engine_with(store)
    await asyncio.gather(*(load_projection(engine, "alpha", mode=mode) for mode in ("current", "history", "all")))
    assert store.pages == 1


async def test_a_deleted_space_is_forgotten():
    store = Counting()
    engine = await engine_with(store)
    await load_projection(engine, "alpha", mode="current")
    assert engine.entities.cached("alpha") is not None
    await engine.delete_space("alpha")
    assert engine.entities.cached("alpha") is None


async def test_held_facts_stay_under_the_budget(monkeypatch):
    monkeypatch.setattr(service_module, "MAX_HELD_FACTS", 3)
    store = Counting()
    engine = await engine_with(store)
    await engine.assert_fact("beta", "zed", "works_at", "Globex", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("beta", "yan", "works_at", "Globex", valid_from="2024-01-01T00:00:00Z")
    await load_projection(engine, "alpha", mode="current")
    await load_projection(engine, "beta", mode="current")
    assert engine.entities.cached("alpha") is None and engine.entities.cached("beta") is not None
