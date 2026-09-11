"""How much of the graph is actually used: recall counts on the map.

Every recall records the facts it returned. Counted per entity and per
relation over the recent recalls, that shows which parts of memory answer
questions and which sit unread, beside the graph itself. Only counts leave
the event log; queries never do.
"""

from __future__ import annotations

import pytest

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
    assert shown["coverage"]["usage"] == {
        "available": True, "recalls_read": 3, "truncated": False, "since": None, "oldest": "2025-01-01T00:00:00.000Z",
        "unsupported": 0, "malformed": 0, "history_unrecorded": 0, "retention": {"max_events": 10_000}}


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


async def test_the_report_says_what_recall_uses_and_what_the_recalls_read_never_reached():
    from scone_memory.entities.report import render_markdown, report_record

    engine = await recalled()
    report = await report_record(engine, "alpha", usage=True)
    uses = report["recall_usage"]
    assert uses["recalls_read"] == 3 and uses["available"] is True
    assert [(item["label"].lower(), item["recalled"]) for item in uses["most_recalled"]][:2] == [
        ("acme robotics", 2), ("alice chen", 2)]
    assert {item["label"].lower() for item in uses["central_unrecalled"]} >= {"carol diaz", "lisbon"}
    text = render_markdown(report)
    assert "## What recall uses" in text and "Over the 3 recalls the event log keeps" in text
    assert "- Central, and returned by none of them: " in text and " never " not in text
    assert text.split("- Most recalled: ")[1].split("\n")[0].lower().startswith("acme robotics (2), alice chen (2)")


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


async def test_a_fact_returned_as_history_counts_on_the_history_map():
    """Asked with history, a recall returns what held before too; the map of
    history counts it like any fact returned."""
    from scone_memory.core.ports import NewEvent  # noqa: F401 - kept beside the other event tests

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(), clock=Clock()).open()
    before = await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Globex Robotics", valid_from="2023-01-01T00:00:00Z")
    returned = await engine.recall("alpha", "alice chen", history=True)
    assert before.fact_id in {fact.fact_id for fact in returned.history}
    projection, coverage = await load_projection(engine, "alpha", mode="history")
    shown = knowledge_view(projection, mode="history", as_of=engine.clock(), limit=50, attribute_limit=50,
                           coverage=coverage, usage=await recall_usage(engine, "alpha"))
    assert {entity["key"]: entity["recalled"] for entity in shown["entities"]}["acme robotics"] == 1
    assert shown["coverage"]["usage"]["history_unrecorded"] == 0


async def append_recall(engine, payload, version=1):
    from scone_memory.core.ports import NewEvent

    await engine.events.append(NewEvent(ts=engine.clock(), space="alpha", kind="recall", payload=payload,
                                        schema_version=version))


async def test_a_recall_event_of_a_version_this_reader_does_not_know_is_not_read():
    engine = await recalled()
    acme = next(fact for fact in await engine.facts("alpha") if fact.object == "Acme Robotics")
    await append_recall(engine, {"fact_ids": [acme.fact_id], "history_fact_ids": []}, version=2)
    shown = await view(engine)
    assert {entity["key"]: entity["recalled"] for entity in shown["entities"]}["acme robotics"] == 2
    assert (shown["coverage"]["usage"]["recalls_read"], shown["coverage"]["usage"]["unsupported"]) == (3, 1)


@pytest.mark.parametrize("ids", [[None], [True], [1.9], ["n/a"], "1", {1: "fact"}])
@pytest.mark.parametrize("field", ["fact_ids", "history_fact_ids"])
async def test_a_recall_event_whose_ids_are_not_whole_numbers_is_counted_as_malformed(ids, field):
    """Nothing is guessed from a payload that does not hold ids: True is not
    fact 1, nor is 1.9."""
    engine = await recalled()
    await append_recall(engine, {"fact_ids": [], "history_fact_ids": [], field: ids})
    shown = await view(engine)
    assert {entity["key"]: entity["recalled"] for entity in shown["entities"]}["alice chen"] == 2
    assert (shown["coverage"]["usage"]["recalls_read"], shown["coverage"]["usage"]["malformed"]) == (3, 1)


async def test_a_recall_recorded_before_its_history_was_is_said():
    engine = await recalled()
    await append_recall(engine, {"fact_ids": []})
    usage = await recall_usage(engine, "alpha")
    assert (usage.recalls_read, usage.history_unrecorded) == (4, 1)


async def test_a_failed_recall_says_it_returned_nothing():
    engine = await recalled()

    async def broken(*args, **kwargs):
        raise RuntimeError("the lane is down")

    engine.vectors.search = broken
    engine.documents.search_text = broken
    with pytest.raises(RuntimeError):
        await engine.recall("alpha", "alice chen", history=True)
    usage = await recall_usage(engine, "alpha")
    assert (usage.recalls_read, usage.history_unrecorded, usage.malformed) == (4, 0, 0)


async def test_usage_says_what_the_log_keeps_and_the_oldest_recall_read():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(max_events=3), clock=Clock()).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.recall("alpha", "alice chen")
    for _ in range(3):
        await engine.record("alpha", "job", {"job_id": "fixture", "name": "fixture", "status": "running"})
    usage = await recall_usage(engine, "alpha")
    assert (usage.recalls_read, usage.oldest, usage.retention) == (0, None, {"max_events": 3})


async def test_a_report_over_a_window_never_says_never():
    """Recalls outside the window, or dropped by the log, may have reached
    what the window's did not: the report says which recalls it read."""
    from scone_memory.entities.report import render_markdown, report_record

    engine = await recalled()
    engine.clock.now = moment = "2025-02-01T00:00:00.000Z"
    await engine.recall("alpha", "porto")
    text = render_markdown(await report_record(engine, "alpha", usage=True, usage_since=moment))
    uses = text[text.index("## What recall uses"):text.index("## Coverage")]
    assert "Over the 1 recall the event log keeps since 2025-02-01T00:00:00.000Z" in uses
    assert "Acme Robotics" in uses.split("returned by none of them: ")[1]
    assert "never" not in uses and "so far" not in uses
    assert "Recalls before that, or ones the log no longer keeps, may have returned them." in uses


async def test_a_report_with_no_recalls_kept_names_nothing_as_unreached():
    from scone_memory.entities.report import render_markdown, report_record

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(max_events=3), clock=Clock()).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.recall("alpha", "alice chen")
    for _ in range(3):
        await engine.record("alpha", "job", {"job_id": "fixture", "name": "fixture", "status": "running"})
    report = await report_record(engine, "alpha", usage=True)
    assert "central_unrecalled" not in report["recall_usage"] and "most_recalled" not in report["recall_usage"]
    text = render_markdown(report)
    assert "The event log keeps no recalls, so which knowledge recall returns is unknown." in text
    assert "It keeps at most 3 events." in text


async def test_a_report_says_which_recall_events_it_could_not_count():
    from scone_memory.entities.report import render_markdown, report_record

    engine = await recalled()
    await append_recall(engine, {"fact_ids": [True]})
    await append_recall(engine, {"fact_ids": [1]}, version=2)
    await append_recall(engine, {"fact_ids": []})
    text = render_markdown(await report_record(engine, "alpha", usage=True))
    assert "- 2 recall events were not counted: 1 of another version, 1 whose fact ids are not whole numbers." in text
    assert "- 1 of the recalls read was recorded before recalls kept the history they returned" in text


def test_a_log_kept_by_age_is_said_to_be():
    from scone_memory.entities.report import _usage_lines

    lines = _usage_lines({"available": True, "recalls_read": 0, "truncated": False, "since": None,
                          "retention": {"max_age_days": 30}})
    assert lines == ["The event log keeps no recalls, so which knowledge recall returns is unknown.",
                     "It keeps events for 30 days."]
