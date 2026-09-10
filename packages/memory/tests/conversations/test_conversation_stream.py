"""Bounded provisional text; durable receipt remains a different surface."""

import asyncio
import json

import pytest

from scone_memory.api.text_stream import TextWindow, sse
from scone_memory.api.conversations import create_conversation_app
from ..api.test_conversations_api import engine, client_for, create, ControlledConversation


def test_utf8_byte_budget_evicts_whole_chunks_and_reports_gap():
    window = TextWindow()
    window.append("é" * 32768)
    assert window.next_after(0) == (None, (1, "é" * 32768))
    window.append("x")
    assert window.next_after(0) == (2, None)
    assert window.next_after(1) == (None, (2, "x"))


def test_count_budget_and_cursor_boundaries():
    window = TextWindow()
    for _ in range(257):
        window.append("a")
    assert window.next_after(0) == (2, None)
    assert window.next_after(256) == (None, (257, "a"))
    assert window.next_after(257) == (None, None)
    with pytest.raises(ValueError):
        window.next_after(258)


def test_finish_erases_provisional_text_and_rejects_late_writes():
    window = TextWindow()
    window.append("private provisional")
    window.changed.clear()
    window.finish()
    assert window.changed.is_set()
    assert window.next_after(0) == (None, None)
    assert window.last_sequence == 1
    with pytest.raises(RuntimeError):
        window.append("late result")


@pytest.mark.parametrize("text", [None, 4, "x" * 65537, "é" * 32769], ids=["null", "number", "ascii-limit", "utf8-limit"])
def test_invalid_or_oversized_chunk_cannot_enter_window(text):
    window = TextWindow()
    with pytest.raises(ValueError):
        window.append(text)
    assert window.last_sequence == 0


def test_empty_chunk_does_not_create_a_sequence():
    window = TextWindow()
    window.append("")
    assert window.last_sequence == 0
    assert window.next_after(0) == (None, None)


async def test_wait_checks_sequence_even_if_another_reader_cleared_notification():
    window = TextWindow()
    window.append("already available")
    window.changed.clear()
    await asyncio.wait_for(window.wait_after(0), 0.2)
    window.finish()
    window.changed.clear()
    await asyncio.wait_for(window.wait_after(1), 0.2)


async def test_multiple_readers_observe_the_same_append_without_private_queues():
    window = TextWindow()
    readers = [asyncio.create_task(window.wait_after(0)) for _ in range(3)]
    window.append("shared")
    await asyncio.wait_for(asyncio.gather(*readers), 0.2)
    assert window.next_after(0) == (None, (1, "shared"))


def test_sse_json_preserves_literal_newlines_without_injecting_events():
    framed = sse("text", {"text": "🌍\n\nid: 99\nevent: terminal", "provisional": True}, sequence=1)
    lines = framed.splitlines()
    assert lines[:2] == ["event: text", "id: 1"]
    assert len(lines) == 4 and lines[3] == ""
    assert json.loads(lines[2][6:]) == {"text": "🌍\n\nid: 99\nevent: terminal", "provisional": True}


class StreamingConversation(ControlledConversation):
    async def reply(self, text, *, on_text):
        self.calls += 1
        await on_text("Hello ")
        await on_text("🌍")
        self.started.set()
        await self.release.wait()
        item = await self.engine.remember(self.space, "Hello 🌍", metadata={"session_id": self.sid})
        return {"text": "Hello 🌍", "assistant_episode_id": item.episode_id}


def stream_app(engine, path, **kwargs):
    runtimes = []
    def factory(space, sid):
        runtime = StreamingConversation(engine, space, sid, blocked=True)
        runtimes.append(runtime)
        return runtime
    return create_conversation_app(engine, {"alpha-key": "alpha", "beta-key": "beta"}, path, factory,
                                   public_text_streaming=True, **kwargs), runtimes


async def submitted(client, runtimes):
    session = await create(client)
    url = "/v1/conversations/" + session["session_id"]
    body = {"request_id": "first", "text": "question", "expected_revision": session["revision"]}
    assert (await client.post(url + "/turns", json=body)).status_code == 202
    await asyncio.wait_for(runtimes[-1].started.wait(), 3)
    return url, body


async def test_stream_capability_is_explicit_and_legacy_factory_unchanged(engine, tmp_path):
    legacy = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "old.db",
                                     lambda space, sid: ControlledConversation(engine, space, sid))
    async with client_for(legacy) as client:
        assert (await client.get("/v1/conversations/capabilities")).json()["streaming"] is False
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "first", "text": "question", "expected_revision": session["revision"]}
        assert (await client.post(url + "/turns", json=body)).status_code == 202
        assert (await client.get(url + "/turns/first/stream")).status_code == 501
        async with asyncio.timeout(3):
            while (await client.get(url + "/turns/first")).json()["status"] == "pending":
                await asyncio.sleep(0)
        assert (await client.get(url + "/turns/first")).json()["status"] == "completed"


@pytest.mark.parametrize("options", [{"public_text_streaming": "yes"}, {"public_text_streaming": True}])
def test_bad_stream_configuration_rejected_before_opening_journal(engine, tmp_path, options):
    with pytest.raises(ValueError):
        create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "bad.db", None, **options)
    assert not (tmp_path / "bad.db").exists()


async def test_stream_auth_scope_and_cursor_validation(engine, tmp_path):
    app, runtimes = stream_app(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        cap = (await client.get("/v1/conversations/capabilities")).json()
        assert cap["streaming"] is True and cap["reply_transport"] == "poll"
        assert cap["text_stream"] == {"transport": "sse", "replay": "active_window", "max_bytes": 65536, "max_chunks": 256}
        url, _ = await submitted(client, runtimes)
        stream = url + "/turns/first/stream"
        assert (await client.get(stream, headers={"Authorization": ""})).status_code == 401
        assert (await client.get(stream, headers={"Authorization": "Bearer beta-key"})).status_code == 404
        assert (await client.get(url + "/turns/missing/stream")).status_code == 404
        for query in ["after=-1", "after=3.2", "after=true", "after=9223372036854775808", "unknown=x", "after=0&after=1"]:
            assert (await client.get(stream + "?" + query)).status_code == 422, query
        assert (await client.get(stream + "?after=3")).status_code == 409
        assert (await client.get(stream, headers={"Last-Event-ID": "invalid"})).status_code == 422
        assert (await client.get(stream + "?after=0", headers={"Last-Event-ID": "1"})).status_code == 422
        assert (await client.post(url + "/turns/first/cancel")).json()["status"] == "cancelled"


async def test_terminal_stream_never_replays_provisional_or_forgotten_text(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, runtimes = stream_app(engine, path)
    async with client_for(app) as client:
        url, body = await submitted(client, runtimes)
        runtimes[0].release.set()
        async with asyncio.timeout(3):
            while (await client.get(url + "/turns/first")).json()["status"] == "pending":
                await asyncio.sleep(0)
        receipt = (await client.get(url + "/turns/first")).json()
        assert receipt["status"] == "completed"
        assert (await client.post(url + "/turns", json=body)).json() == receipt
        assert runtimes[0].calls == 1
        response = await client.get(url + "/turns/first/stream")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-accel-buffering"] == "no"
        assert "event: terminal" in response.text and '"status": "completed"' in response.text
        assert "Hello" not in response.text and "event: text" not in response.text
        await engine.forget("alpha", receipt["result"]["assistant_episode_id"])
        assert "Hello" not in (await client.get(url + "/turns/first/stream")).text
        assert (await client.get(url + "/turns/first")).json()["result_state"] == "forgotten"
    reopened, _ = stream_app(engine, path)
    async with client_for(reopened) as client:
        response = await client.get(url + "/turns/first/stream?after=2")
        assert "event: terminal" in response.text and "event: text" not in response.text
        assert (await client.delete(url)).status_code == 204
        assert (await client.get(url + "/turns/first/stream")).status_code == 404


async def test_cancelled_stream_has_terminal_receipt_not_partial_text(engine, tmp_path):
    app, runtimes = stream_app(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        url, body = await submitted(client, runtimes)
        assert (await client.post(url + "/turns/first/cancel")).json()["status"] == "cancelled"
        response = await client.get(url + "/turns/first/stream")
        assert '"status": "cancelled"' in response.text and "Hello" not in response.text
        assert (await client.post(url + "/turns", json=body)).json()["status"] == "cancelled"
        assert runtimes[0].calls == 1


@pytest.mark.parametrize("chunk", [None, "x" * 65537, "\ud800"], ids=["non-text", "oversized", "invalid-unicode"])
async def test_runtime_cannot_hide_rejected_observation_and_claim_completion(engine, tmp_path, chunk):
    class IgnoresObserverError(ControlledConversation):
        async def reply(self, text, *, on_text):
            for value in ["prefix", chunk, "late"]:
                try:
                    await on_text(value)
                except (ValueError, RuntimeError):
                    pass
            return await super().reply(text)
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "journal.db",
                                  lambda space, sid: IgnoresObserverError(engine, space, sid), public_text_streaming=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "bad", "text": "question", "expected_revision": session["revision"]}
        await client.post(url + "/turns", json=body)
        async with asyncio.timeout(3):
            while (await client.get(url + "/turns/bad")).json()["status"] == "pending":
                await asyncio.sleep(0)
        receipt = (await client.get(url + "/turns/bad")).json()
        assert receipt["status"] == "failed" and receipt["result"] is None
        assert (await client.post(url + "/turns", json=body)).json()["status"] == "failed"
        assert "prefix" not in (await client.get(url + "/turns/bad/stream")).text


async def test_shutdown_gate_rejects_new_work_and_interrupts_late_results(engine, tmp_path):
    app, runtimes = stream_app(engine, tmp_path / "journal.db")
    async with client_for(app) as client:
        url, body = await submitted(client, runtimes)
        app.state.begin_conversation_shutdown()
        assert (await client.post("/v1/conversations", json={"request_id": "late", "capture": True})).status_code == 503
        assert (await client.post(url + "/turns", json={**body, "request_id": "late"})).status_code == 503
        runtimes[0].release.set()
        async with asyncio.timeout(3):
            while (await client.get(url + "/turns/first")).json()["status"] == "pending":
                await asyncio.sleep(0)
        assert (await client.get(url + "/turns/first")).json()["status"] == "interrupted"
        assert len(runtimes) == 1
