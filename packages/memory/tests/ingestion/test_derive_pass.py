"""The derivation pass by hand and in the worker: the worker runs it after
extraction when it has a deriver, the flag SCONE_DERIVE=1 gives it one,
`scone-memory derive` runs one pass by hand, status counts pending
groups, and POST /v1/consolidate {scope: derive} runs a pass over HTTP."""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import FakeChat, HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.derive import Deriver
from scone_memory.ingestion.worker import ConsolidationWorker
from scone_memory.runtime import config
from scone_memory.runtime.cli import main
from scone_memory.runtime.config import Settings, build_worker

INFERENCE = json.dumps([{"subject": "mark", "predicate": "works_in", "object": "lisbon", "premises": [1, 2], "confidence": 0.7}])


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


async def seeded():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()
    await engine.assert_fact("default", "mark", "works_at", "acme")
    await engine.assert_fact("default", "acme", "based_in", "lisbon")
    return engine


async def test_the_worker_runs_the_derivation_pass_when_it_has_a_deriver():
    engine = await seeded()
    worker = ConsolidationWorker(engine, None, ["default"], interval_s=999, deriver=Deriver(engine, FakeChat([INFERENCE])))
    report = await worker.run_once("default")
    assert (report.derived_sent, report.derived_proposed, report.derived_restated, report.error) == (1, 1, 0, None)
    assert (await worker.run_once("default")).derived_sent == 0, "an unchanged group is not sent again"
    newest = (await engine.events.query("default", kind="distill"))[0].payload
    assert newest["derived_sent"] == 0 and newest["derived_proposed"] == 0, "the pass report carries the derivation counts"


def test_the_flag_gives_the_worker_a_deriver_and_only_the_flag(monkeypatch):
    monkeypatch.setattr(config, "build_chat", lambda settings: FakeChat([]))
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    base = {"SCONE_CHAT_URL": "http://x", "SCONE_CHAT_MODEL": "m"}
    assert build_worker(engine, Settings.from_env(base), ["default"]).deriver is None, "off unless asked"
    assert isinstance(build_worker(engine, Settings.from_env({**base, "SCONE_DERIVE": "1"}), ["default"]).deriver, Deriver)
    assert Settings.from_env({**base, "SCONE_DERIVE": "true"}).derive is True
    with pytest.raises(InvalidInput):
        Settings.from_env({**base, "SCONE_DERIVE": "maybe"})


def test_the_cli_runs_one_derivation_pass_by_hand(monkeypatch, tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_EMBEDDER": "hash"}
    assert main(["--json", "derive"], env=env, out=io.StringIO()) == 2, "no model configured"
    for args in (["assert", "mark", "works_at", "acme"], ["assert", "acme", "based_in", "lisbon"]):
        assert main(args, env=env, out=io.StringIO()) == 0
    monkeypatch.setattr(config, "build_chat", lambda settings: FakeChat([INFERENCE]))
    out = io.StringIO()
    assert main(["--json", "derive"], env={**env, "SCONE_CHAT_URL": "http://x", "SCONE_CHAT_MODEL": "m"}, out=out) == 0
    payload = json.loads(out.getvalue())
    assert (payload["space"], payload["sent"], payload["proposed"], payload["restated"]) == ("default", 1, 1, 0)


def test_status_counts_pending_groups_and_consolidate_runs_the_pass_over_http():
    engine = asyncio.run(seeded())
    # No scheduled spaces: the app starts the worker's loop, and only the
    # by-hand route should spend the scripted reply.
    worker = ConsolidationWorker(engine, None, [], interval_s=999, deriver=Deriver(engine, FakeChat([INFERENCE])))
    keys, roles = {"k": "default", "r": "default"}, {"r": "read"}
    with TestClient(create_app(engine, keys, worker=worker, roles=roles)) as c:
        features = c.get("/v1/capabilities", headers=bearer("k")).json()["features"]
        assert features.get("processing.distill", False) is False
        assert features.get("processing.derive", False) is False
        status = c.get("/v1/status", headers=bearer("k")).json()
        assert (status["pending_derivation"], status["derivation"]) == (1, "on")
        assert c.post("/v1/consolidate", json={"scope": "derive"}, headers=bearer("r")).status_code == 403, "a pass writes proposals"
        assert c.post("/v1/consolidate", json={"scope": "dream"}, headers=bearer("k")).status_code == 422
        run = c.post("/v1/consolidate", json={"scope": "derive"}, headers=bearer("k"))
        assert run.status_code == 200 and (run.json()["scope"], run.json()["proposed"]) == ("derive", 1), run.text
        assert c.get("/v1/status", headers=bearer("k")).json()["pending_derivation"] == 0
        by_hand = c.post("/v1/consolidate", json={"scope": "distill"}, headers=bearer("k"))
        assert by_hand.status_code == 200 and by_hand.json()["scope"] == "distill" and by_hand.json()["derived_sent"] == 0
    with TestClient(create_app(asyncio.run(seeded()), {"k": "default"})) as c:
        features = c.get("/v1/capabilities", headers=bearer("k")).json()["features"]
        assert features.get("processing.distill", False) is False
        assert features.get("processing.derive", False) is False
        assert c.get("/v1/status", headers=bearer("k")).json()["derivation"] == "off"
        assert c.post("/v1/consolidate", json={"scope": "derive"}, headers=bearer("k")).status_code == 501, "no model, no pass"


async def test_a_derivation_already_held_under_another_spelling_is_a_restatement():
    """Stored subjects are keys; the model writes names as people do. The
    already-held check looked the model's spelling up as written, missed the
    stored claim and proposed the same inference twice."""
    engine = await seeded()
    first = await Deriver(engine, FakeChat([INFERENCE])).derive("default")
    assert len(first.proposed) == 1
    engine._derive_seen.clear()
    shouted = json.dumps([{"subject": "Mark", "predicate": "works_in", "object": "lisbon",
                           "premises": [1, 2], "confidence": 0.7}])
    second = await Deriver(engine, FakeChat([shouted])).derive("default")
    assert (len(second.proposed), second.restated) == (0, 1)
