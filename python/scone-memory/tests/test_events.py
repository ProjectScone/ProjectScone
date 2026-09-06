"""Evidence: every operation leaves one event, latencies are measured,
feedback refers to a real recall, and the sinks agree."""

from __future__ import annotations

import json

import os
import uuid

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
from scone_memory.testing.events_contract import *  # noqa: F401,F403
from scone_memory.testing.events_contract import ev


def sink_params():
    yield pytest.param("memory", id="memory")
    yield pytest.param("sqlite", id="sqlite")
    if os.environ.get("SCONE_TEST_MONGO_URL"):
        yield pytest.param("mongo", id="mongo", marks=pytest.mark.mongo)


@pytest.fixture(params=list(sink_params()))
async def sink(request, tmp_path):
    clock = Clock()
    if request.param == "memory":
        log = InMemoryEventLog(max_events=5)
    elif request.param == "sqlite":
        log = SqliteEventLog(tmp_path / "events.db", max_age_days=30, clock=clock)
    else:
        from scone_memory.events import MongoEventLog

        log = await MongoEventLog(os.environ["SCONE_TEST_MONGO_URL"], f"scone_test_ev_{uuid.uuid4().hex[:8]}", max_age_days=7).open()
    log.test_clock = clock
    yield log
    if hasattr(log, "drop"):
        await log.drop()
    if hasattr(log, "close"):
        await log.close()


async def test_retention_is_a_stated_policy(sink):
    if sink.name == "memory":
        for i in range(7):
            await sink.append(ev("alpha", "recall", query=str(i)))
        assert [e.payload["query"] for e in await sink.query("alpha", limit=10)] == ["6", "5", "4", "3", "2"]
    elif sink.name == "mongo":
        # Mongo expires through a TTL index; the sweep is the server's, so
        # the check is that the index exists with the configured age.
        indexes = await sink.events.index_information()
        ttl = [i for i in indexes.values() if i.get("expireAfterSeconds") is not None]
        assert ttl and ttl[0]["expireAfterSeconds"] == 7 * 86400
        e = await sink.append(ev("alpha", "recall", query="dated"))
        doc = await sink.events.find_one({"_id": e.event_id})
        assert doc["ts_date"].year == 2025
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
    assert close.payload == {"fact_id": newer.fact_id, "reason_kind": "manual", "actor": None}
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


async def test_agent_events_are_validated_scrubbed_and_idempotent():
    engine = await fresh_engine()
    ep = await engine.remember("default", "the user asked about the harbour crane")
    first = await engine.record("default", "agent", {
        "agent": "claude-code", "session_id": "s-1", "project": "scone", "event": "prompt",
        "text": "please rotate token sk-live-ABCDEFGHIJKLMNOPQRSTUVWXYZ and see https://user:hunter2pass@host/db",
        "episode_id": ep.episode_id, "source_event_id": "p-1",
    })
    assert first.kind == "agent"
    assert "sk-live-" not in first.payload["text"] and "hunter2pass" not in first.payload["text"]
    assert "[redacted]" in first.payload["text"]
    same = {"agent": "claude-code", "session_id": "s-1", "project": "scone", "event": "prompt",
            "text": "please rotate token sk-live-ABCDEFGHIJKLMNOPQRSTUVWXYZ and see https://user:hunter2pass@host/db",
            "episode_id": ep.episode_id, "source_event_id": "p-1"}
    again = await engine.record("default", "agent", same)
    assert again.event_id == first.event_id, "a retried connector event is not duplicated"
    with pytest.raises(InvalidInput, match="different payload"):
        await engine.record("default", "agent", {**same, "text": "something else entirely"})
    for bad in (
        {"agent": "hal", "session_id": "s", "event": "prompt"},
        {"agent": "codex", "session_id": "s", "event": "thinking"},
        {"agent": "codex", "session_id": "s", "event": "prompt", "episode_id": 999},  # not in this space
        {"agent": "codex", "session_id": "s", "event": "prompt", "reasoning": "hidden"},  # unknown field
        {"agent": "codex", "session_id": "s", "event": "tool_use", "text": "x" * 70000},
    ):
        with pytest.raises(InvalidInput):
            await engine.record("default", "agent", bad)


async def test_graph_has_only_recorded_edges():
    engine = await fresh_engine()
    ep = await engine.remember("default", "Ana moved to Lisbon in March 2024 for the harbour job.", created_at="2024-03-02")
    other = await engine.remember("default", "Quarterly revenue exceeded expectations.")
    turn = await engine.record("default", "agent", {"agent": "claude-code", "session_id": "s-9", "event": "prompt",
                                                    "text": "where did ana move", "episode_id": ep.episode_id})
    tool = await engine.record("default", "agent", {"agent": "claude-code", "session_id": "s-9", "event": "tool_use", "tool_name": "memory_recall", "ok": True})
    austin = await engine.assert_fact("default", "ana", "lives_in", "Austin", valid_from="2022-01-01")
    lisbon = await engine.assert_fact("default", "ana", "lives_in", "Lisbon", valid_from="2024-03-02", origin="extracted", source_episode_id=ep.episode_id)
    unrelated = await engine.assert_fact("default", "bob", "drinks", "tea")
    result = await engine.recall("default", "where did ana move", limit=2)
    await engine.feedback("default", result.event_id, result.items[0].chunk_id, useful=True)

    g = await engine.graph("default", session_id="s-9")
    ids = set(g.nodes)
    assert f"session:claude-code:s-9" in ids and f"turn:{turn.event_id}" in ids and f"tool:{tool.event_id}" in ids
    assert f"episode:{ep.episode_id}" in ids and f"episode:{other.episode_id}" not in ids
    assert f"claim:{lisbon.fact_id}" in ids and f"claim:{unrelated.fact_id}" not in ids
    kinds = {(e.source, e.target, e.kind) for e in g.edges}
    assert (f"turn:{turn.event_id}", f"episode:{ep.episode_id}", "captured_as") in kinds
    assert (f"episode:{ep.episode_id}", f"claim:{lisbon.fact_id}", "source_of") in kinds
    assert (f"session:claude-code:s-9", f"tool:{tool.event_id}", "invoked") in kinds
    returned = [e for e in g.edges if e.kind == "returned"]
    assert returned and all(e.label.startswith("retrieval evidence") for e in returned)
    assert any(e.kind == "judged" for e in g.edges)
    # No edge exists between two chunks, and no similarity is ever an edge kind.
    assert not any(e.source.startswith("chunk:") and e.target.startswith("chunk:") for e in g.edges)
    assert "similar" not in {e.kind for e in g.edges}

    whole = await engine.graph("default")
    assert (f"claim:{austin.fact_id}", f"claim:{lisbon.fact_id}", "superseded_by") in {(e.source, e.target, e.kind) for e in whole.edges}
    assert whole.as_dict()["counts"]["claim"] == 3


async def test_every_judgement_records_who_made_it():
    engine = await fresh_engine()
    p1 = await engine.assert_fact("default", "mark", "lives_in", "Lisbon", origin="extracted", proposed=True)
    p2 = await engine.assert_fact("default", "mark", "drinks", "tea", origin="extracted", proposed=True)
    held = await engine.assert_fact("default", "mark", "uses", "neovim")
    await engine.approve("default", p1.fact_id, actor="key:abc123 console")
    await engine.decline("default", p2.fact_id, "not about mark", actor="cli:mark")
    await engine.exclude("default", held.fact_id, "private", actor="key:abc123")
    await engine.include("default", held.fact_id, actor="key:abc123")
    await engine.close_fact("default", held.fact_id, "stopped", actor="cli:mark")
    events = [e for e in reversed(await engine.events.query("default", limit=100)) if e.kind in ("fact_review", "fact_exclude", "fact_close")]
    assert [(e.kind, e.payload.get("decision") or e.payload.get("action") or e.payload.get("reason_kind"), e.payload["actor"]) for e in events] == [
        ("fact_review", "approved", "key:abc123 console"),
        ("fact_review", "declined", "cli:mark"),
        ("fact_exclude", "exclude", "key:abc123"),
        ("fact_exclude", "include", "key:abc123"),
        ("fact_close", "manual", "cli:mark"),
    ]
    unlabelled = await engine.assert_fact("default", "mark", "likes", "rain", origin="extracted", proposed=True)
    await engine.approve("default", unlabelled.fact_id)
    assert (await engine.events.query("default", kind="fact_review", limit=1))[0].payload["actor"] is None, "no actor is recorded as None, never invented"


def test_http_judgements_carry_a_key_fingerprint_and_optional_label():
    import asyncio
    import hashlib

    engine = asyncio.run(fresh_engine())
    with TestClient(create_app(engine, {"reviewer-key": "default"})) as c:
        h = {"authorization": "Bearer reviewer-key"}
        p = c.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon", "origin": "extracted", "proposed": True}, headers=h).json()
        c.post(f"/v1/facts/{p['fact_id']}/approve", headers={**h, "x-scone-actor": "console"})
        q = c.post("/v1/facts", json={"subject": "mark", "predicate": "drinks", "object": "tea", "origin": "extracted", "proposed": True}, headers=h).json()
        c.post(f"/v1/facts/{q['fact_id']}/decline", json={"reason": "no"}, headers=h)
        events = c.get("/v1/events", params={"kind": "fact_review"}, headers=h).json()["events"]
        fp = hashlib.sha256(b"reviewer-key").hexdigest()[:12]
        assert [e["payload"]["actor"] for e in events] == [f"key:{fp}", f"key:{fp} console"]
        assert "reviewer-key" not in json.dumps(events), "the key itself never appears"


async def test_episode_focus_reaches_its_own_turn_past_the_window_and_excludes_neighbours():
    """Seen on the live store: focusing on episode 11 returned 373 tool calls
    from other sessions and no edge to episode 11, because the linking event
    was older than the 400-event window and agent events were only filtered
    by session. A focused graph follows its subject, not the clock."""
    engine = await fresh_engine()
    ep = await engine.remember("default", "resume", created_at="2026-09-06")
    turn = await engine.record("default", "agent", {"agent": "codex", "session_id": "c-1", "event": "prompt", "text": "resume", "episode_id": ep.episode_id, "source_event_id": "p1"})
    for i in range(60):  # newer, unrelated activity from another session pushes the turn out of a small window
        await engine.record("default", "agent", {"agent": "claude-code", "session_id": "other", "event": "tool_use", "tool_name": "Bash", "source_event_id": f"t{i}"})
    g = await engine.graph("default", episode_id=ep.episode_id, limit=20)
    ids = set(g.nodes)
    assert f"episode:{ep.episode_id}" in ids and f"turn:{turn.event_id}" in ids and "session:codex:c-1" in ids
    assert not any(n.kind == "tool_call" for n in g.nodes.values()), "neighbouring sessions' tool calls are not part of this episode's graph"
    assert (f"turn:{turn.event_id}", f"episode:{ep.episode_id}", "captured_as") in {(e.source, e.target, e.kind) for e in g.edges}
    by_session = await engine.graph("default", session_id="other", limit=20)
    assert by_session.as_dict()["counts"].get("tool_call") == 60, "a session focus reaches all of that session's events, not only the window"
