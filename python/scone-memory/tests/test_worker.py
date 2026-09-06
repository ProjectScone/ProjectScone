"""Consolidation as a worker: every pass leaves evidence, never dies."""

from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from scone_memory import FakeChat, HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.config import Settings, build_worker
from scone_memory.distill import Distiller
from scone_memory.testing import Clock
from scone_memory.worker import ConsolidationWorker

LISBON = json.dumps([{"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9}])
COFFEE = json.dumps([{"subject": "Carol", "predicate": "drinks", "object": "black coffee", "confidence": 0.6}])


async def engine_with(*episodes):
    e = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock(), events=InMemoryEventLog()).open()
    for text in episodes:
        await e.remember("default", text, created_at="2024-03-02")
    return e


async def test_a_pass_proposes_claims_and_records_a_distill_event():
    engine = await engine_with("Ana moved to Lisbon.", "Carol takes her coffee black.")
    worker = ConsolidationWorker(engine, Distiller(engine, FakeChat([LISBON, COFFEE])), ["default"], interval_s=999)
    report = await worker.run_once("default")
    assert (report.episodes, report.proposed, report.accepted, report.error) == (2, 2, 0, None)
    assert await engine.pending_distillation("default") == 0
    assert [f.status for f in await engine.facts("default", status="proposed")] == ["proposed", "proposed"]
    [event] = await engine.events.query("default", kind="distill")
    assert event.payload["proposed"] == 2 and event.payload["episodes"] == 2 and event.payload["latency_ms"] >= 0
    assert worker.passes == 1 and worker.last["default"].proposed == 2


async def test_accept_at_admits_confident_extractions_directly():
    engine = await engine_with("Ana moved to Lisbon.", "Carol takes her coffee black.")
    worker = ConsolidationWorker(engine, Distiller(engine, FakeChat([LISBON, COFFEE]), accept_at=0.8), ["default"])
    report = await worker.run_once("default")
    assert (report.proposed, report.accepted) == (1, 1)  # 0.9 accepted, 0.6 proposed
    assert [f.object for f in await engine.facts("default")] == ["Lisbon"]


async def test_a_failing_model_is_recorded_not_fatal_and_parks_after_three():
    engine = await engine_with("Ana moved to Lisbon.")
    chat = FakeChat(["not json"] * 5)
    worker = ConsolidationWorker(engine, Distiller(engine, chat, max_attempts=3), ["default"])
    reports = [await worker.run_once("default") for _ in range(4)]
    assert all(r.error == "DistillError: 1 episode(s) failed" for r in reports[:3])
    assert reports[3].parked == 1 and reports[3].error is None, "parked episodes are reported, not retried"
    assert len(chat.calls) == 3, "no fourth model call"
    events = await engine.events.query("default", kind="distill")
    assert len(events) == 4 and events[0].payload["parked"] == 1 and events[-1].payload["error"].startswith("DistillError")


async def test_the_loop_runs_on_its_interval_and_stops_cleanly():
    engine = await engine_with("Ana moved to Lisbon.")
    worker = ConsolidationWorker(engine, Distiller(engine, FakeChat([LISBON, "[]", "[]", "[]"])), ["default"], interval_s=0.05)
    worker.start()
    assert worker.running
    await asyncio.sleep(0.2)
    await worker.stop()
    assert not worker.running
    assert worker.passes >= 2
    assert [f.object for f in await engine.facts("default", status="proposed")] == ["Lisbon"]


def test_build_worker_needs_both_chat_settings_and_serve_reports_the_lane():
    import pytest

    from scone_memory import InvalidInput

    engine = asyncio.run(engine_with())
    assert build_worker(engine, Settings(), ["default"]) is None
    with pytest.raises(InvalidInput):
        build_worker(engine, Settings(chat_url="http://x"), ["default"])
    with pytest.raises(InvalidInput):
        build_worker(engine, Settings(chat_url="http://x", chat_model="m", distill_accept_at=1.5), ["default"])
    worker = build_worker(engine, Settings(chat_url="http://127.0.0.1:1", chat_model="m", distill_interval_s=7, distill_batch=3, distill_accept_at=0.75), ["b", "a", "a"])
    assert (worker.spaces, worker.interval_s, worker.batch) == (["a", "b"], 7.0, 3)
    assert worker.distiller.accept_at == 0.75, "the acceptance threshold reaches the distiller"

    with TestClient(create_app(engine, {"k": "default"})) as c:
        s = c.get("/v1/status", headers={"authorization": "Bearer k"}).json()
        assert (s["semantic_lane"], s["pending_distill"], s["last_distill"]) == ("manual", 0, None)
    engine2 = asyncio.run(engine_with("Ana moved to Lisbon."))
    w2 = ConsolidationWorker(engine2, Distiller(engine2, FakeChat([LISBON, "[]", "[]", "[]", "[]"])), ["default"], interval_s=999)
    with TestClient(create_app(engine2, {"k": "default"}, worker=w2)) as c:
        h = {"authorization": "Bearer k"}
        for _ in range(20):
            s = c.get("/v1/status", headers=h).json()
            if s["last_distill"]:
                break
        assert s["semantic_lane"] == "active" and s["pending_distill"] == 0 and s["last_distill"]["proposed"] == 1
        assert w2.running
    assert w2._task is None, "stopped with the app, not merely finished"
