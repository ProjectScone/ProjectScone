"""The projection reader reports every cap that bites."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import read
from scone_memory.entities.mentions import MENTIONS, mention_predicate


class Unpaged(InMemoryDocumentStore):
    """A store without a ledger pager, read by one whole-ledger list."""
    page_facts = None


async def engine_with(count: int, store: InMemoryDocumentStore | None = None) -> MemoryEngine:
    engine = await MemoryEngine(store or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(count):
        await engine.assert_fact("alpha", f"team {number}", "based_in", "Lisbon")
    return engine


async def test_a_store_that_may_have_cut_the_read_short_is_named(monkeypatch) -> None:
    engine = await engine_with(3, Unpaged())
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


# -- mentions read under a budget of their own ------------------------------------

WHEN = "2024-03-01T00:00:00Z"


async def mention(engine: MemoryEngine, name: str) -> None:
    """A recognizer's mention, written the way the recorder writes one."""
    added = await engine.remember("alpha", f"{name} was named.", source="notes.md", created_at=WHEN)
    await engine._assert_placed("alpha", "notes.md", mention_predicate("organisation"), name, valid_from=WHEN,
                                source_episode_id=added.episode_id, origin="extracted", quote=f"{name} was named.")


@pytest.mark.parametrize("store", [InMemoryDocumentStore, Unpaged], ids=["paged", "unpaged"])
async def test_mentions_never_displace_claims_from_the_graph_read(monkeypatch, store) -> None:
    monkeypatch.setattr(read, "MAX_FACTS", 2)
    monkeypatch.setattr(read, "MAX_MENTION_FACTS", 1)
    monkeypatch.setattr(read, "MAX_LEDGER_PAGE", 1)  # so the walk decides where to stop row by row
    engine = await MemoryEngine(store(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", "Alice Chen", "works_at", "Acme", valid_from=WHEN)
    await engine.assert_fact("alpha", "Alice Chen", "lives_in", "Lisbon", valid_from=WHEN)
    for name in ("Globex", "Initech", "Umbrella", "Hooli", "Wayne"):
        await mention(engine, name)
    projection, coverage = await read.load_projection(engine, "alpha")
    stated = sorted(role.predicate for role in projection.roles if not role.predicate.startswith(MENTIONS))
    assert stated == ["lives_in", "works_at"], "both claims, though five newer mentions exist"
    assert sum(role.predicate.startswith(MENTIONS) for role in projection.roles) == 1
    assert coverage["reasons"] == ["mention_limit"] and coverage["mentions_limit"] == 1


async def test_the_walk_stops_at_the_claims_cap_and_reads_no_older_mention(monkeypatch) -> None:
    """A read cut at the claims' cap covers the time back to its oldest
    claim; a mention older than that is outside it like an older claim,
    and the walk reads no further than a ledger without mentions would."""
    monkeypatch.setattr(read, "MAX_FACTS", 2)
    monkeypatch.setattr(read, "MAX_MENTION_FACTS", 5)
    monkeypatch.setattr(read, "MAX_LEDGER_PAGE", 1)
    engine = await engine_with(0)
    await mention(engine, "Acme")
    for number in range(4):
        await engine.assert_fact("alpha", f"team {number}", "based_in", "Lisbon", valid_from=WHEN)
    projection, coverage = await read.load_projection(engine, "alpha")
    assert coverage["reasons"] == ["fact_limit"]
    assert [role.predicate for role in projection.roles] == ["based_in", "based_in"]


async def test_a_reader_with_no_use_for_mentions_keeps_none() -> None:
    engine = await engine_with(1)
    await mention(engine, "Acme")
    ledger = await read.read_ledger(engine, "alpha", max_mentions=0)
    assert [fact.predicate for fact in ledger.facts] == ["based_in"] and "mention_limit" in ledger.reasons
