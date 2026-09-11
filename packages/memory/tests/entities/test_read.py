"""The projection reader reports every cap that bites."""
from __future__ import annotations

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import read


async def engine_with(count: int) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(count):
        await engine.assert_fact("alpha", f"team {number}", "based_in", "Lisbon")
    return engine


async def test_a_store_that_may_have_cut_the_read_short_is_named(monkeypatch) -> None:
    engine = await engine_with(3)
    monkeypatch.setattr(read, "_SILENT_CAPS", {"memory": 3})
    _, coverage = await read.load_projection(engine, "alpha")
    assert "store_read_cap_reached" in coverage["reasons"]


async def test_past_the_fact_limit_only_the_newest_facts_are_projected(monkeypatch) -> None:
    engine = await engine_with(5)
    monkeypatch.setattr(read, "MAX_FACTS", 2)
    projection, coverage = await read.load_projection(engine, "alpha")
    assert coverage["reasons"] == ["fact_limit"] and coverage["facts_read"] == 2
    assert sorted(role.fact_id for role in projection.roles) == [4, 5]


async def test_an_ordinary_read_reports_nothing_left_out() -> None:
    engine = await engine_with(3)
    projection, coverage = await read.load_projection(engine, "alpha")
    assert coverage["reasons"] == [] and coverage["facts_read"] == 3 and projection.revision >= 1
