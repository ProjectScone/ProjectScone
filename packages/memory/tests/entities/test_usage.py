"""How much of the graph is actually used: recall counts on the map.

Every recall records the facts it returned. Counted per entity and per
relation over the recent recalls, that shows which parts of memory answer
questions and which sit unread, beside the graph itself. Only counts leave
the event log; queries never do.
"""

from __future__ import annotations

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities import usage as usage_module
from scone_memory.entities.read import load_projection
from scone_memory.entities.usage import recall_usage
from scone_memory.entities.view import knowledge_view
from scone_memory.observability.events import InMemoryEventLog
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def recalled(events=True) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog() if events else None, clock=Clock()).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "bob stone", "lives_in", "Porto", valid_from=DAY)
    await engine.assert_fact("alpha", "carol diaz", "lives_in", "Lisbon", valid_from=DAY)
    for question in ("where does alice chen work", "alice chen", "porto"):
        await engine.recall("alpha", question)
    return engine


async def view(engine, **options):
    projection, coverage = await load_projection(engine, "alpha", mode="current")
    found = await recall_usage(engine, "alpha", **options)
    return knowledge_view(projection, mode="current", as_of=engine.clock(), limit=50, attribute_limit=50,
                          coverage=coverage, usage=found)


async def test_each_entity_and_relation_says_how_many_recalls_returned_it():
    shown = await view(await recalled())
    by_key = {entity["key"]: entity["recalled"] for entity in shown["entities"]}
    assert by_key == {"alice chen": 2, "acme robotics": 2, "bob stone": 1, "porto": 1, "carol diaz": 0, "lisbon": 0}
    works = next(relation for relation in shown["relations"] if relation["predicate"] == "works_at")
    assert works["recalled"] == 2
    assert shown["coverage"]["usage"] == {"available": True, "recalls_read": 3, "truncated": False, "since": None}


async def test_usage_counts_only_the_recalls_since_a_moment():
    engine = await recalled()
    engine.clock.now = moment = "2025-02-01T00:00:00.000Z"
    await engine.recall("alpha", "porto")
    shown = await view(engine, since=moment)
    by_key = {entity["key"]: entity["recalled"] for entity in shown["entities"]}
    assert by_key["porto"] == 1 and by_key["alice chen"] == 0 and shown["coverage"]["usage"]["since"] == moment


async def test_a_capped_read_of_recalls_says_so(monkeypatch):
    monkeypatch.setattr(usage_module, "MAX_RECALLS", 2)
    shown = await view(await recalled())
    assert shown["coverage"]["usage"]["recalls_read"] == 2 and shown["coverage"]["usage"]["truncated"] is True


async def test_an_engine_that_keeps_no_events_says_usage_is_unavailable():
    shown = await view(await recalled(events=False))
    assert shown["coverage"]["usage"]["available"] is False
    assert all(entity["recalled"] is None for entity in shown["entities"])


async def test_without_usage_the_view_is_as_before():
    engine = await recalled()
    projection, coverage = await load_projection(engine, "alpha", mode="current")
    shown = knowledge_view(projection, mode="current", as_of=engine.clock(), limit=50, attribute_limit=50,
                           coverage=coverage)
    assert "usage" not in shown["coverage"] and all("recalled" not in entity for entity in shown["entities"])


async def test_the_report_says_what_recall_uses_and_what_it_never_reaches():
    from scone_memory.entities.report import render_markdown, report_record

    engine = await recalled()
    report = await report_record(engine, "alpha", usage=True)
    uses = report["recall_usage"]
    assert uses["recalls_read"] == 3 and uses["available"] is True
    assert [(item["label"].lower(), item["recalled"]) for item in uses["most_recalled"]][:2] == [
        ("acme robotics", 2), ("alice chen", 2)]
    assert {item["label"].lower() for item in uses["central_unrecalled"]} >= {"carol diaz", "lisbon"}
    text = render_markdown(report)
    assert "## What recall uses" in text and "Central but never recalled" in text


async def test_a_report_without_events_says_recall_use_is_unknown():
    from scone_memory.entities.report import render_markdown, report_record

    report = await report_record(await recalled(events=False), "alpha", usage=True)
    assert report["recall_usage"]["available"] is False
    assert "central_unrecalled" not in report["recall_usage"], "unknown is not never"
    assert "keeps no events" in render_markdown(report)


async def test_a_report_without_usage_is_as_before():
    from scone_memory.entities.report import render_markdown, report_record

    report = await report_record(await recalled(), "alpha")
    assert "recall_usage" not in report and "## What recall uses" not in render_markdown(report)
