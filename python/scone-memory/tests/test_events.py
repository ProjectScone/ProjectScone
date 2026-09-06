"""Evidence: every operation leaves one event, latencies are measured,
feedback refers to a real recall, and the sinks agree."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import (
    HashEmbedder,
    InMemoryDocumentStore,
    InMemoryEventLog,
    InMemoryVectorIndex,
    InvalidInput,
    MemoryEngine,
    NotFound,
    SqliteEventLog,
)
from scone_memory.api import create_app
from scone_memory.ports import NewEvent
from scone_memory.testing import Clock


@pytest.fixture(params=["memory", "sqlite"])
def sink(request, tmp_path):
    clock = Clock()
    if request.param == "memory":
        log = InMemoryEventLog(max_events=5)
    else:
        log = SqliteEventLog(tmp_path / "events.db", max_age_days=30, clock=clock)
    log.test_clock = clock
    return log


def ev(space, kind, ts="2025-01-01T00:00:00.000Z", **payload):
    return NewEvent(ts=ts, space=space, kind=kind, payload=payload)


async def test_sink_returns_newest_first_filters_and_isolates_spaces(sink):
    a = await sink.append(ev("alpha", "recall", "2025-01-01T00:00:00.000Z", query="one", items=[{"chunk_id": 1}]))
    b = await sink.append(ev("alpha", "remember", "2025-01-02T00:00:00.000Z", fresh=1))
    c = await sink.append(ev("alpha", "recall", "2025-01-03T00:00:00.000Z", query="two", nested={"k": ["v", 1]}))
    await sink.append(ev("beta", "recall", "2025-01-04T00:00:00.000Z", query="other space"))
    assert [e.event_id for e in await sink.query("alpha")] == [c.event_id, b.event_id, a.event_id]
    assert [e.event_id for e in await sink.query("alpha", kind="recall")] == [c.event_id, a.event_id]
    assert [e.event_id for e in await sink.query("alpha", since="2025-01-02T00:00:00.000Z")] == [c.event_id, b.event_id]
    assert [e.event_id for e in await sink.query("alpha", limit=1)] == [c.event_id]
    assert (await sink.get("alpha", c.event_id)).payload == {"query": "two", "nested": {"k": ["v", 1]}}
    assert await sink.get("beta", c.event_id) is None
    assert (await sink.get("alpha", a.event_id)).schema_version == 1


async def test_retention_is_a_stated_policy(sink):
    if sink.name == "memory":
        for i in range(7):
            await sink.append(ev("alpha", "recall", query=str(i)))
        assert [e.payload["query"] for e in await sink.query("alpha", limit=10)] == ["6", "5", "4", "3", "2"]
    else:
        await sink.append(ev("alpha", "recall", "2024-11-01T00:00:00.000Z", query="old"))
        await sink.append(ev("alpha", "recall", "2024-12-20T00:00:00.000Z", query="recent"))
        sink.test_clock.now = "2025-01-01T00:00:00.000Z"
        assert sink.sweep() == 1
        assert [e.payload["query"] for e in await sink.query("alpha")] == ["recent"]


async def fresh_engine(record_queries=True):
    # Tests opt in to plain-text queries; the engine default is hashing.
    return await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock(),
        events=InMemoryEventLog(), record_queries=record_queries,
    ).open()


async def test_every_operation_leaves_one_event_with_measured_latency():
    engine = await fresh_engine()
    added = await engine.remember("default", "the lighthouse keeper logs the tide")
    result = await engine.recall("default", "lighthouse tide", limit=3)
    fact = await engine.assert_fact("default", "keeper", "logs", "the tide", valid_from="2024-01-01")
    await engine.assert_fact("default", "keeper", "logs", "the tide", valid_from="2024-01-01")
    newer = await engine.assert_fact("default", "keeper", "logs", "the weather", valid_from="2024-06-01")
    await engine.close_fact("default", newer.fact_id, "closed by hand")
    await engine.forget("default", added.episode_id)

    events = list(reversed(await engine.events.query("default", limit=100)))
    assert [e.kind for e in events] == [
        "remember", "recall", "fact_assert", "fact_assert", "fact_assert", "fact_close", "forget",
    ]
    remember, recall, first, restated, superseding, close, forget = events
    assert (remember.payload["fresh"], remember.payload["chunks"]) == (1, 1)
    assert remember.payload["latency_ms"] > 0
    assert recall.payload["query"] == "lighthouse tide" and recall.payload["query_hashed"] is False
    assert set(recall.payload["latency_ms"]) == {"embed", "vector", "text", "total"}
    assert recall.payload["latency_ms"]["total"] >= recall.payload["latency_ms"]["vector"]
    assert recall.payload["items"][0]["chunk_id"] == result.items[0].chunk_id
    assert recall.payload["items"][0]["lanes"] == {"vector": 1, "text": 1}
    assert recall.payload["items_coverage"] == "returned"
    assert recall.payload["lane_candidates"] == {"vector": 1, "text": 1}
    assert result.event_id == recall.event_id
    assert first.payload["outcome"] == "new_active"
    assert restated.payload["outcome"] == "restated" and restated.payload["fact_id"] == fact.fact_id
    assert superseding.payload["outcome"] == "new_active" and superseding.payload["superseded"] == [fact.fact_id]
    assert close.payload == {"fact_id": newer.fact_id, "reason_kind": "manual"}
    assert forget.payload["chunks_removed"] == 1


async def test_failures_are_evidence_too():
    engine = await fresh_engine()
    await engine.remember("default", "one note")

    async def explode(*_a, **_k):
        raise ConnectionError("down")

    engine.vectors.search = explode
    engine.documents.search_text = explode
    with pytest.raises(RuntimeError):
        await engine.recall("default", "anything")
    with pytest.raises(NotFound):
        await engine.forget("default", 99)
    kinds = [(e.kind, e.payload.get("error")) for e in await engine.events.query("default", limit=2)]
    assert kinds == [("forget", "NotFound"), ("recall", "both lanes failed")]


async def test_queries_are_hashed_unless_asked_otherwise():
    engine = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()
    ).open()
    await engine.remember("default", "a private note about the harbour")
    await engine.recall("default", "harbour")
    [recall] = await engine.events.query("default", kind="recall")
    assert recall.payload["query_hashed"] is True
    assert "harbour" not in recall.payload["query"] and len(recall.payload["query"]) == 16


async def test_feedback_must_name_a_returned_item():
    engine = await fresh_engine()
    await engine.remember("default", "the harbour crane was repainted")
    result = await engine.recall("default", "harbour crane")
    chunk = result.items[0].chunk_id
    event = await engine.feedback("default", result.event_id, chunk, useful=True, note="exactly it")
    assert event.kind == "feedback" and event.payload["useful"] is True
    with pytest.raises(InvalidInput):
        await engine.feedback("default", result.event_id, chunk + 1000, useful=False)
    with pytest.raises(NotFound):
        await engine.feedback("default", result.event_id + 50, chunk, useful=False)
    with pytest.raises(NotFound):
        await engine.feedback("other", result.event_id, chunk, useful=False)


async def test_without_a_log_there_is_no_evidence_and_no_feedback():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "quiet")
    result = await engine.recall("default", "quiet")
    assert result.event_id is None
    with pytest.raises(InvalidInput):
        await engine.feedback("default", 1, 1, useful=True)


def test_events_and_feedback_over_http():
    import asyncio

    engine = asyncio.run(fresh_engine())
    with TestClient(create_app(engine, {"k": "default"})) as c:
        h = {"authorization": "Bearer k"}
        c.post("/v1/episodes", json={"content": "the harbour crane was repainted"}, headers=h)
        recall = c.get("/v1/recall", params={"q": "harbour crane"}, headers=h).json()
        assert recall["event_id"] is not None
        chunk = recall["items"][0]["chunk_id"]
        r = c.post("/v1/feedback", json={"recall_event_id": recall["event_id"], "chunk_id": chunk, "useful": True}, headers=h)
        assert r.status_code == 200 and "recorded" in r.json()
        events = c.get("/v1/events", headers=h).json()
        assert [e["kind"] for e in events["events"]] == ["feedback", "recall", "remember"]
        assert events["evidence"] == "memory" and events["queries_recorded"] == "text"  # fresh_engine opts in
        only = c.get("/v1/events", params={"kind": "recall", "limit": 1}, headers=h).json()["events"]
        assert [e["payload"]["query"] for e in only] == ["harbour crane"]
        bad = c.post("/v1/feedback", json={"recall_event_id": recall["event_id"], "chunk_id": 9999, "useful": True}, headers=h)
        assert bad.status_code == 422


async def test_outside_processes_can_report_jobs_but_not_forge_engine_events():
    engine = await fresh_engine()
    first = await engine.record("default", "job", {"job_id": "e31", "name": "vector baseline", "status": "running",
                                                   "progress": {"done": 1, "total": 4}, "adapter": "qdrant-local"})
    done = await engine.record("default", "job", {"job_id": "e31", "name": "vector baseline", "status": "completed",
                                                  "progress": {"done": 4, "total": 4}})
    assert first.kind == "job" and done.event_id > first.event_id
    for bad in (
        ("recall", {"job_id": "x", "name": "n", "status": "running"}),  # forged engine kind
        ("job", {"job_id": "x", "name": "n", "status": "done"}),  # unknown status
        ("job", {"job_id": "x", "name": "n", "status": "running", "progress": {"done": 5, "total": 4}}),
        ("job", {"job_id": "", "name": "n", "status": "running"}),
        ("job", {"job_id": "x", "name": "n", "status": "failed", "error": "e" * 501}),
        ("job", {"job_id": "x", "name": "n", "status": "running", "detail": "d" * 5000}),
    ):
        with pytest.raises(InvalidInput):
            await engine.record("default", *bad)
    assert [e.kind for e in await engine.events.query("default")] == ["job", "job"]


def test_job_events_over_http():
    import asyncio

    engine = asyncio.run(fresh_engine())
    with TestClient(create_app(engine, {"k": "default"})) as c:
        h = {"authorization": "Bearer k"}
        ok = c.post("/v1/events", json={"kind": "job", "payload": {"job_id": "j1", "name": "smoke", "status": "running"}}, headers=h)
        assert ok.status_code == 200 and "recorded" in ok.json()
        forged = c.post("/v1/events", json={"kind": "recall", "payload": {"job_id": "j1", "name": "x", "status": "running"}}, headers=h)
        assert forged.status_code == 422
        assert [e["kind"] for e in c.get("/v1/events", headers=h).json()["events"]] == ["job"]
