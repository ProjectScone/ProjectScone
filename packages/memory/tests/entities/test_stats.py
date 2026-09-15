"""The graph counted, and its hubs ranked, over recorded data alone."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import stats as stats_module
from scone_memory.entities.stats import MAX_HUBS, StatsError, graph_hubs, graph_stats
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def engine_with(*triples, **options) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    for subject, predicate, obj in triples:
        await engine.assert_fact("alpha", subject, predicate, obj, valid_from=DAY, **options)
    return engine


PEOPLE = (("alice chen", "works_at", "Acme Robotics"), ("bob stone", "works_at", "Acme Robotics"),
          ("carol diaz", "works_at", "Acme Robotics"), ("alice chen", "lives_in", "Lisbon"),
          ("bob stone", "lives_in", "Lisbon"), ("Acme Robotics", "based_in", "Lisbon"),
          ("alice chen", "born_on", "1990-04-02"))


async def test_the_stats_count_what_the_projection_and_the_analysis_hold():
    engine = await engine_with(*PEOPLE, origin="stated")
    await engine.assert_fact("alpha", "dan poe", "works_at", "Globex", valid_from=DAY, origin="extracted")
    hidden = await engine.assert_fact("alpha", "alice chen", "met", "Mallory", valid_from=DAY, origin="stated")
    await engine.exclude("alpha", hidden.fact_id, "not for recall")
    try:
        found = await graph_stats(engine, "alpha")
        everything = await graph_stats(engine, "alpha", status="all")
    finally:
        await engine.close()
    assert everything.facts["standing"] == {"active": 8, "excluded": 1} and everything.totals["facts"] == 9, \
        "under status=all an excluded fact is counted as excluded, not as active"
    totals = found.totals
    # Seven entities: a date is a value, not an entity.
    assert totals["entities"] == 7 and totals["relations"] == 7 and totals["attributes"] == 1
    assert totals["facts"] == 8 and totals["communities"] >= 1 and totals["isolated_entities"] == 0
    assert found.facts["origin"] == {"stated": 7, "extracted": 1}
    assert found.facts["grounding"] == {"unsourced": 8} and found.facts["standing"] == {"active": 8}
    assert dict(found.predicates) == {"works_at": 4, "lives_in": 2, "based_in": 1}
    assert found.predicates[0] == ("works_at", 4), "most common first"
    assert sum(count for _, count in found.kinds) == 7 and all(kind for kind, _ in found.kinds)
    assert found.text.startswith("stats: space alpha, current facts as of 2025-06-01") and "coverage: complete" in found.text
    assert "totals: 7 entities, 7 relations (0 implied), 1 attributes, 8 facts" in found.text
    assert "facts by origin: stated 7, extracted 1" in found.text and "facts by grounding: unsourced 8" in found.text
    record = found.record("alpha", status="current", as_of="2025-06-01T00:00:00.000Z")
    assert record["schema_version"] == 1 and record["totals"] == totals and record["coverage"]["reasons"] == []
    assert record["predicates"][0] == ["works_at", 4] and "analysis" in record["coverage"]


async def test_more_kinds_or_predicates_than_are_listed_are_counted_and_said(monkeypatch):
    monkeypatch.setattr(stats_module, "MAX_LISTED", 2)
    engine = await engine_with(*PEOPLE)
    try:
        found = await graph_stats(engine, "alpha")
    finally:
        await engine.close()
    assert len(found.predicates) == 2 and found.totals["predicates"] == 3
    assert "predicates_listed 2 of 3" in found.coverage["reasons"] and "coverage: limited: " in found.text
    assert "predicates (3): works_at 3, lives_in 2" in found.text


async def test_hubs_are_the_most_linked_first_and_the_percentile_holds_the_rest_out():
    engine = await engine_with(*PEOPLE)
    try:
        found = await graph_hubs(engine, "alpha", limit=3)
        above = await graph_hubs(engine, "alpha", above=50)
        none_above = await graph_hubs(engine, "alpha", above=90)
        one = await graph_hubs(engine, "alpha", limit=1)
        empty = await graph_hubs(engine, "beta")
    finally:
        await engine.close()
    assert [hub["label"] for hub in found.hubs][:2] == ["Acme Robotics", "Lisbon"], found.hubs
    assert found.hubs[0]["degree"] == 4 and found.hubs[0]["weight"] >= 4 and found.hubs[0]["community"]
    assert found.hubs[0]["pagerank"] > 0 and found.totals["shown"] == 3 and found.totals["own"] == 5
    assert "1. Acme Robotics" in found.text and "4 neighbours" in found.text and "hubs: space alpha" in found.text
    # Degrees 1, 2, 2, 3, 4: the 50th percentile (nearest rank) is 2, and two hubs are above it.
    assert [hub["label"] for hub in above.hubs] == ["Acme Robotics", "Lisbon"], "above the median degree, two hubs"
    assert above.totals["above"] == 50.0 and above.totals["above_percentile"] == 2 and "50th percentile" in above.text
    assert none_above.hubs == () and "result: no hub above the 90th percentile" in none_above.text, \
        "five entities have nothing above their 90th percentile; the answer says so rather than showing the top"
    assert one.totals["shown"] == 1 and "hubs_shown 1 of 5" in one.coverage["reasons"]
    assert empty.hubs == () and "result: no hub" in empty.text
    record = found.record("alpha", status="current", as_of="2025-06-01T00:00:00.000Z")
    assert record["hubs"][0]["label"] == "Acme Robotics" and record["totals"]["entities"] == 5


async def test_bounds_are_refused_before_anything_is_read():
    engine = await engine_with(*PEOPLE)
    try:
        with pytest.raises(StatsError, match=f"1 to {MAX_HUBS}"):
            await graph_hubs(engine, "alpha", limit=0)
        with pytest.raises(StatsError, match="percentile"):
            await graph_hubs(engine, "alpha", above=10)
        with pytest.raises(StatsError, match="max_bytes"):
            await graph_stats(engine, "alpha", max_bytes=10)
        with pytest.raises(StatsError, match="max_bytes"):
            await graph_hubs(engine, "alpha", max_bytes=10)
    finally:
        await engine.close()


async def test_a_capped_walk_of_implied_relations_is_said_in_the_coverage(monkeypatch):
    from scone_memory.entities import project as projecting
    from scone_memory.entities.meanings import RelationMeanings

    monkeypatch.setattr(projecting, "MAX_WALKED", 1)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"),
                                relation_meanings=RelationMeanings(transitive=["part_of"])).open()
    for n in range(4):
        await engine.assert_fact("alpha", f"Box {n}", "part_of", f"Box {n + 1}", valid_from=DAY)
    try:
        found = await graph_stats(engine, "alpha")
    finally:
        await engine.close()
    assert "implied_capped" in found.coverage["reasons"] and "coverage: limited: " in found.text, found.text
    assert found.totals["relations"] == 4, "the stated relations are counted whole; only the implied walk was cut"
