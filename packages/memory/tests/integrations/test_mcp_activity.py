"""Recorded activity is bounded, scoped, and never resurrects forgotten text."""

from __future__ import annotations

import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.bearer_keys import KeyHolder
from scone_memory.core.ports import NewEvent
from scone_memory.observability.events import InMemoryEventLog, SqliteEventLog
from scone_memory.runtime.mcp import create_server, http_app
from scone_memory.testing import Clock

from .test_mcp import call
from .test_mcp_http import KEYS, ROLES, opened, served


@pytest.fixture(params=["memory", "sqlite"])
async def activity(request, tmp_path):
    clock = Clock("2026-10-04T12:00:00.000Z")
    log = InMemoryEventLog() if request.param == "memory" else SqliteEventLog(tmp_path / "activity.db")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=clock, events=log).open()
    yield engine, clock, create_server(engine, "alpha")
    await engine.close()


async def record(engine, ts="2026-10-04T11:00:00.000Z", *, space="alpha", **payload):
    return await engine.events.append(NewEvent(ts=ts, space=space, kind="agent", payload={
        "agent": "other", "session_id": "session-a", "project": "orchard", "event": "tool_result",
        "tool_name": "tests", "ok": True, **payload,
    }))


async def read(server, **kwargs):
    error, text = await call(server, "memory_activity", **kwargs)
    assert not error, text
    return json.loads(text)


async def test_default_two_hours_and_48_hours_include_boundaries_without_future(activity):
    engine, _, server = activity
    for stamp in ["2026-10-02T11:59:59.999Z", "2026-10-02T12:00:00.000Z",
                  "2026-10-04T09:59:59.999Z", "2026-10-04T10:00:00.000Z",
                  "2026-10-04T12:00:00.000Z", "2026-10-04T12:00:00.001Z"]:
        await record(engine, stamp)
    first = await read(server)
    assert [e["event_id"] for e in first["events"]] == [4, 5]
    assert first["window"] == {"since": "2026-10-04T10:00:00.000Z", "until": "2026-10-04T12:00:00.000Z", "hours": 2}
    assert first["time_basis"] == "server_receipt_time"
    assert first["coverage"]["capture_complete"] is False
    assert first["next_cursor"] is None
    extended = await read(server, hours=48)
    assert [e["event_id"] for e in extended["events"]] == [2, 3, 4, 5]


async def test_cursor_freezes_window_and_high_water_mark_across_new_receipts(activity):
    engine, clock, server = activity
    for _ in range(4):
        await record(engine)
    first = await read(server, limit=2)
    assert [e["event_id"] for e in first["events"]] == [1, 2]
    await record(engine)  # A later receipt with a timestamp inside the original window.
    clock.now = "2026-10-04T15:00:00.000Z"
    second = await read(server, cursor=first["next_cursor"], limit=2)
    assert [e["event_id"] for e in second["events"]] == [3, 4]
    assert second["window"] == first["window"]
    assert second["next_cursor"] is None


async def test_scan_budget_progresses_past_other_projects_without_claiming_empty_history(activity):
    engine, _, server = activity
    for _ in range(205):
        await record(engine, project="elsewhere")
    wanted = await record(engine, session_id="chosen")
    first = await read(server, project="orchard", session_id="chosen")
    assert first["events"] == []
    assert first["scanned"] == 200 and first["coverage"]["scan_limited"] is True
    assert first["next_cursor"]
    second = await read(server, project="orchard", session_id="chosen", cursor=first["next_cursor"])
    assert [e["event_id"] for e in second["events"]] == [wanted.event_id]
    assert second["scanned"] == 6 and second["next_cursor"] is None


async def test_retained_evidence_and_metadata_only_never_use_cached_payload_text(activity, monkeypatch):
    engine, _, server = activity
    added = await engine.remember("alpha", "Retained source evidence", source="agent://session-a")
    linked = await record(engine, episode_id=added.episode_id, text="STALE CACHED TEXT")
    await record(engine, text="UNLINKED PRIVATE TEXT", event="response")
    await record(engine, event="session_start", tool_name=None)

    async def forbidden(*args, **kwargs):
        pytest.fail("activity must not call recall, profile, or embedding")

    monkeypatch.setattr(engine, "profile", forbidden)
    monkeypatch.setattr(engine, "recall", forbidden)
    monkeypatch.setattr(engine.embedder, "embed", forbidden)
    result = await read(server)
    assert [e["evidence_status"] for e in result["events"]] == ["retained", "unlinked_text_omitted", "metadata_only"]
    assert result["events"][0]["event_id"] == linked.event_id
    evidence = result["events"][0]["evidence"]
    assert evidence["text"] == "Retained source evidence"
    assert evidence["episode_id"] == added.episode_id and evidence["source"] == "agent://session-a"
    assert evidence["chunk_id"] > 0 and evidence["link_provenance"] == "connector_reported"
    assert "CACHED" not in json.dumps(result) and "PRIVATE" not in json.dumps(result)


async def test_deleted_and_expired_sources_never_reappear_from_event_text(activity):
    engine, clock, server = activity
    deleted = await engine.remember("alpha", "Forget this evidence")
    expiry = await engine.remember("alpha", "Expiring evidence", forget_after="1h")
    await record(engine, episode_id=deleted.episode_id, text="Forget this evidence")
    await record(engine, episode_id=expiry.episode_id, text="Expiring evidence")
    await engine.forget("alpha", deleted.episode_id)
    clock.now = "2026-10-04T13:00:00.000Z"
    result = await read(server)
    assert [e["evidence_status"] for e in result["events"]] == ["unavailable", "expired"]
    assert all(e["evidence"] is None for e in result["events"])
    assert "Forget this" not in json.dumps(result) and "Expiring" not in json.dumps(result)


async def test_foreign_episode_and_wrong_chunk_identity_are_never_evidence(activity, monkeypatch):
    engine, _, server = activity
    foreign = await engine.remember("beta", "Foreign private evidence")
    own = await engine.remember("alpha", "Owned evidence")
    await record(engine, episode_id=foreign.episode_id)
    await record(engine, episode_id=own.episode_id)
    chunks = await engine.documents.chunks_of("alpha", own.episode_id)

    async def wrong_chunks(*args):
        return [chunks[0].model_copy(update={"space": "beta"}),
                chunks[0].model_copy(update={"episode_id": foreign.episode_id})]

    monkeypatch.setattr(engine.documents, "chunks_of", wrong_chunks)
    result = await read(server)
    assert all(e["evidence"] is None for e in result["events"])
    assert "private" not in json.dumps(result) and "Owned evidence" not in json.dumps(result)


async def test_forget_during_chunk_read_is_rechecked(activity, monkeypatch):
    engine, _, server = activity
    added = await engine.remember("alpha", "Concurrent forgotten evidence")
    await record(engine, episode_id=added.episode_id)
    native = engine.documents.chunks_of

    async def forget_after_read(space, episode_id):
        chunks = await native(space, episode_id)
        monkeypatch.setattr(engine.documents, "chunks_of", native)
        await engine.forget(space, episode_id)
        return chunks

    monkeypatch.setattr(engine.documents, "chunks_of", forget_after_read)
    result = await read(server)
    assert result["events"][0]["evidence"] is None


async def test_cursor_tampering_filter_changes_and_cross_space_are_refused(activity):
    engine, _, server = activity
    await record(engine)
    await record(engine)
    first = await read(server, limit=1)
    token = first["next_cursor"]
    for options in [{"cursor": token[:-2] + "xx"}, {"cursor": token, "hours": 48},
                    {"cursor": token, "project": "other"}, {"cursor": token, "space": "beta"}]:
        error, text = await call(server, "memory_activity", **options)
        assert error and "cursor" in text.lower()


@pytest.mark.parametrize("options", [
    {"hours": 0}, {"hours": 721}, {"hours": True}, {"hours": 2.5}, {"hours": "2"},
    {"limit": 0}, {"limit": 51}, {"limit": False}, {"limit": "20"},
    {"project": ""}, {"project": "p" * 121}, {"session_id": ""}, {"session_id": "s" * 129},
    {"cursor": ""}, {"cursor": "x" * 4097},
])
async def test_activity_rejects_invalid_inputs(activity, options):
    _, _, server = activity
    try:
        error, _ = await call(server, "memory_activity", **options)
        assert error
    except ToolError as error:
        assert "validation error" in str(error).lower()


async def test_unavailable_event_log_is_distinct_from_empty_history(activity):
    engine, _, server = activity
    assert (await read(server))["status"] == "available"
    engine.events = None
    result = await read(server)
    assert result["status"] == "unavailable" and result["events"] == []
    assert result["coverage"]["capture_complete"] is False


@pytest.mark.parametrize("role", ["read", "review", "write", "admin"])
async def test_read_roles_can_query_only_authorized_space(activity, role):
    engine, _, _ = activity
    await record(engine)
    await record(engine, space="beta", session_id="other-tenant")
    server = create_server(engine, "alpha", holder=KeyHolder("alpha", role))
    result = await read(server)
    assert len(result["events"]) == 1 and result["space"] == "alpha"
    error, text = await call(server, "memory_activity", space="beta")
    assert error and "not this key" in text


async def test_http_reader_receives_scoped_activity_with_no_write_authority(activity):
    engine, _, _ = activity
    await record(engine)
    async with served(http_app(engine, "default", KEYS, ROLES)) as client:
        session = await opened(client, "reader-key")
        error, text = await session.call("memory_activity")
        assert not error and json.loads(text)["space"] == "alpha"
        error, _ = await session.call("memory_activity", space="beta")
        assert error
        error, _ = await session.call("memory_store", content="Unauthorized write")
        assert error


async def test_excerpt_has_a_utf8_byte_bound_and_declares_truncation(activity):
    engine, _, server = activity
    added = await engine.remember("alpha", "界" * 1000)
    await record(engine, episode_id=added.episode_id)
    result = await read(server)
    evidence = result["events"][0]["evidence"]
    assert evidence["text"] and len(evidence["text"].encode("utf-8")) <= 1024
    assert evidence["truncated"] is True


async def test_backend_errors_are_not_empty_success_or_private_exception_details(activity, monkeypatch):
    engine, _, server = activity

    async def broken(*args, **kwargs):
        raise RuntimeError("private connection credential or path")

    monkeypatch.setattr(engine.events, "query", broken)
    error, text = await call(server, "memory_activity")
    assert error and "unavailable" in text
    assert "private connection" not in text


async def test_continuation_rechecks_expiry_against_new_now_not_frozen_window(activity):
    engine, clock, server = activity
    await record(engine)
    added = await engine.remember("alpha", "Future expired evidence", forget_after="1h")
    await record(engine, episode_id=added.episode_id)
    first = await read(server, limit=1)
    clock.now = "2026-10-04T13:00:00.000Z"
    second = await read(server, cursor=first["next_cursor"])
    assert second["window"] == first["window"]
    assert second["events"][0]["evidence_status"] == "expired"
    assert second["events"][0]["evidence"] is None
