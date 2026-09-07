"""Retention: episodes older than a per-kind policy are forgotten by the
worker on its interval, receipts and tombstones written; facts never
expire; nothing moves unless a policy is set; a frozen clock decides
what "older" means."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import __main__ as serve
from scone_memory.api import create_app
from scone_memory.core.errors import Gone, InvalidInput
from scone_memory.ingestion.worker import ConsolidationWorker
from scone_memory.observability.events import InMemoryEventLog
from scone_memory.runtime.config import Settings, build_worker

NOW = "2026-09-07T12:00:00.000Z"


async def engine_at(now=NOW):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              events=InMemoryEventLog(), clock=lambda: now).open()


async def seed(engine):
    old_chat = await engine.remember("default", "an old chat turn", kind="conversation", created_at="2026-07-01T00:00:00Z")
    new_chat = await engine.remember("default", "a recent chat turn", kind="conversation", created_at="2026-09-01T00:00:00Z")
    old_note = await engine.remember("default", "an old note", kind="note", created_at="2026-01-01T00:00:00Z")
    return old_chat, new_chat, old_note


async def test_a_policy_forgets_only_what_it_names_and_only_when_old_enough():
    engine = await engine_at()
    old_chat, new_chat, old_note = await seed(engine)
    cited = await engine.assert_fact("default", "mark", "said", "hello", source_episode_id=old_chat.episode_id, quote="old chat")

    report = await engine.expire("default", {"conversation": 30})
    assert report.forgotten == [old_chat.episode_id] and report.remaining == 0
    assert [r.episode_id for r in report.receipts] == [old_chat.episode_id] and report.receipts[0].forgotten_at == NOW
    with pytest.raises(Gone):
        await engine.episode("default", old_chat.episode_id)
    assert (await engine.episode("default", new_chat.episode_id)).content == "a recent chat turn", "younger than the policy"
    assert (await engine.episode("default", old_note.episode_id)).content == "an old note", "another kind, another policy"
    standing = await engine.fact("default", cited.fact_id)
    assert standing.status == "active" and standing.source_episode_id == old_chat.episode_id, "facts never expire"
    [event] = await engine.events.query("default", kind="expire")
    assert event.payload["forgotten"] == 1 and event.payload["policy"] == {"conversation": 30.0}

    again = await engine.expire("default", {"conversation": 30})
    assert again.forgotten == [] and again.remaining == 0, "a second pass finds nothing"
    with pytest.raises(InvalidInput):
        await engine.expire("default", {"poem": 30})
    with pytest.raises(InvalidInput):
        await engine.expire("default", {"note": 0})


async def test_a_pass_is_bounded_and_says_what_it_left():
    engine = await engine_at()
    for day in range(1, 6):
        await engine.remember("default", f"chat turn {day}", kind="conversation", created_at=f"2026-01-0{day}T00:00:00Z")
    first = await engine.expire("default", {"conversation": 30}, limit=2)
    assert len(first.forgotten) == 2 and first.remaining == 3, "oldest first, and honest about the rest"
    assert first.forgotten == sorted(first.forgotten)
    preview = await engine.expire("default", {"conversation": 30}, dry_run=True)
    assert preview.forgotten == [] and preview.remaining == 3 and preview.receipts == [], "a dry run forgets nothing"
    assert (await engine.status("default")).episodes == 3


async def test_the_clock_decides_what_older_means():
    engine = await engine_at(now="2026-07-15T00:00:00.000Z")
    old_chat, *_ = await seed(engine)
    report = await engine.expire("default", {"conversation": 30})
    assert report.forgotten == [], "on 15 July a 1 July turn is fourteen days old"
    engine.clock = lambda: "2026-08-15T00:00:00.000Z"
    assert (await engine.expire("default", {"conversation": 30})).forgotten == [old_chat.episode_id]


async def test_the_worker_runs_retention_without_a_model_and_status_says_so():
    engine = await engine_at()
    old_chat, new_chat, old_note = await seed(engine)
    worker = ConsolidationWorker(engine, None, ["default"], retention={"conversation": 30})
    report = await worker.run_once("default")
    assert report.expired == 1 and report.error is None and report.episodes == 0
    with pytest.raises(Gone):
        await engine.episode("default", old_chat.episode_id)
    with TestClient(create_app(engine, {"k": "default"}, console=False, worker=worker)) as c:
        status = c.get("/v1/status", headers={"Authorization": "Bearer k"}).json()
        assert status["semantic_lane"] == "manual", "no model, so no consolidation lane, whatever else the worker does"
        assert status["retention"] == {"conversation": 30.0}


def test_retention_is_a_setting_that_builds_a_worker_on_its_own(tmp_path):
    settings = Settings.from_env({"SCONE_API_KEY": "k", "SCONE_RETAIN": "conversation=30,note=365.5"})
    assert settings.retention == {"conversation": 30.0, "note": 365.5}
    assert Settings.from_env({"SCONE_API_KEY": "k"}).retention == {}
    with pytest.raises(InvalidInput):
        Settings.from_env({"SCONE_API_KEY": "k", "SCONE_RETAIN": "poem=30"})
    engine = asyncio.run(engine_at())
    worker = build_worker(engine, settings, ["default"])
    assert worker is not None and worker.distiller is None and worker.retention == settings.retention
    assert build_worker(engine, Settings.from_env({"SCONE_API_KEY": "k"}), ["default"]) is None, "nothing to run, no worker"
    app = serve.build_app(settings, engine)
    with TestClient(app) as c:
        assert c.get("/v1/status", headers={"Authorization": "Bearer k"}).json()["retention"] == {"conversation": 30.0, "note": 365.5}
