"""Native text turns preserve safe diagnostic correlation across background tasks."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.providers.llm import OpenAICompatibleTextModel
from scone_memory.realtime.context import MemoryContext
from scone_memory.realtime.text import TextConversation
from scone_memory.runtime.diagnostics import configure_diagnostics, context, install_http_diagnostics


async def wait_for_turn(client, session_id, request_id):
    async with asyncio.timeout(3):
        while True:
            response = await client.get(f"/v1/conversations/{session_id}/turns/{request_id}")
            assert response.status_code == 200
            receipt = response.json()
            if receipt["status"] != "pending":
                return receipt
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("failed", [False, True])
async def test_native_turn_logs_correlate_capture_recall_and_model_without_private_content(tmp_path, failed):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("alpha", "PRIVATE-SOURCE says Polaris is north.")

    def respond(request):
        if failed:
            raise httpx.ReadTimeout("PRIVATE-PROVIDER-DETAIL", request=request)
        chunk = json.dumps({"choices": [{"delta": {"content": "PRIVATE-REPLY"}}]})
        return httpx.Response(200, content=f"data: {chunk}\n\ndata: [DONE]\n\n".encode(),
                              headers={"content-type": "text/event-stream"})

    def runtime(space, session_id):
        return TextConversation(engine, space, session_id, lambda: OpenAICompatibleTextModel(
            "http://private-username:private-password@llm.local/v1", "test-local-model",
            api_key="PRIVATE-PROVIDER-KEY", transport=httpx.MockTransport(respond),
        ), system_prompt="PRIVATE-SYSTEM")

    app = create_conversation_app(engine, {"PRIVATE-MEMORY-KEY": "alpha"}, tmp_path / "journal.db", runtime,
                                  public_text_streaming=True)
    install_http_diagnostics(app)
    path = tmp_path / "diagnostics.jsonl"
    logger = logging.getLogger("scone_memory")
    previous_level = logger.level
    configure_diagnostics(str(path))
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test",
                                         headers={"Authorization": "Bearer PRIVATE-MEMORY-KEY"}) as client:
                created = await client.post("/v1/conversations", json={"request_id": "create-one", "capture": True})
                assert created.status_code == 200
                session_id = created.json()["session_id"]
                started = await client.post(f"/v1/conversations/{session_id}/turns", json={
                    "request_id": "turn-one", "expected_revision": created.json()["revision"],
                    "text": "PRIVATE-QUESTION asks whether Polaris is north.",
                })
                assert started.status_code == 202
                receipt = await wait_for_turn(client, session_id, "turn-one")
                assert receipt["status"] == ("failed" if failed else "completed")
        contents = path.read_text()
    finally:
        configure_diagnostics(None)
        logger.setLevel(previous_level)

    records = [json.loads(line) for line in contents.splitlines()]
    turn_logs = [record for record in records if record["event"].startswith(("conversation.", "capture.", "recall.", "model_call."))]
    assert all(record["session_id"] == session_id and record["request_id"] == "turn-one" for record in turn_logs)
    assert {record["event"] for record in turn_logs} >= {
        "conversation.started", "capture.finished", "recall.finished", "model_call.started",
        "model_call.finished", "conversation.finished",
    }
    [model_result] = [record for record in turn_logs if record["event"] == "model_call.finished"]
    assert model_result["outcome"] == ("failed" if failed else "completed")
    assert [record["role"] for record in turn_logs if record["event"] == "capture.finished"] == (
        ["user"] if failed else ["user", "assistant"]
    )
    if failed:
        assert model_result["exception_type"] == "ReadTimeout"
        assert any(record["event"] == "conversation.failed" for record in turn_logs)
    else:
        assert 0 <= model_result["first_token_ms"] <= model_result["elapsed_ms"]
    assert "PRIVATE" not in contents and "private-" not in contents
    assert context.get() == {}, "background turn correlation must not leak to the request caller"


async def test_initial_capture_timeout_does_not_claim_the_user_message_was_saved(tmp_path, monkeypatch):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    async def delayed_capture(*args, **kwargs):
        await asyncio.Event().wait()

    def unexpected_model() -> OpenAICompatibleTextModel:
        raise AssertionError("the model must not start before capture finishes")

    monkeypatch.setattr(engine, "remember_many", delayed_capture)
    app = create_conversation_app(engine, {"key": "alpha"}, tmp_path / "journal.db",
        lambda space, session_id: TextConversation(engine, space, session_id, unexpected_model, turn_timeout=0.02))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test",
                                     headers={"Authorization": "Bearer key"}) as client:
            created = (await client.post("/v1/conversations", json={"request_id": "create", "capture": True})).json()
            session_id = created["session_id"]
            response = await client.post(f"/v1/conversations/{session_id}/turns", json={
                "request_id": "turn", "expected_revision": created["revision"], "text": "Not yet saved",
            })
            assert response.status_code == 202
            receipt = await wait_for_turn(client, session_id, "turn")
            assert receipt["status"] == "failed" and "timed out" in receipt["error"]
            assert "Your message was saved" not in receipt["error"]
            assert await engine.episodes("alpha", {"session_id": session_id}) == []


async def test_cancelled_recall_logs_its_stage_and_reraises_cancellation(tmp_path, monkeypatch):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    entered = asyncio.Event()

    async def blocked_recall(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(engine, "recall", blocked_recall)
    recall = MemoryContext(engine, "alpha", "cancelled-session")
    path = tmp_path / "cancelled.jsonl"
    logger = logging.getLogger("scone_memory")
    previous_level = logger.level
    configure_diagnostics(str(path))
    token = context.set({"session_id": "cancelled-session", "request_id": "cancelled-turn"})
    task = asyncio.create_task(recall.prepare([{"role": "user", "content": "PRIVATE-CANCELLED-QUERY"}]))
    context.reset(token)
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        contents = path.read_text()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        configure_diagnostics(None)
        logger.setLevel(previous_level)

    records = [json.loads(line) for line in contents.splitlines()]
    [started] = [record for record in records if record["event"] == "recall.started"]
    [finished] = [record for record in records if record["event"] == "recall.finished"]
    assert started["request_id"] == finished["request_id"] == "cancelled-turn"
    assert started["session_id"] == finished["session_id"] == "cancelled-session"
    assert finished["outcome"] == "cancelled" and finished["exception_type"] == "CancelledError"
    assert finished["elapsed_ms"] >= 0 and finished["context_bytes"] == finished["reference_count"] == 0
    assert "PRIVATE" not in contents and context.get() == {}
