"""Provider diagnostics explain timing and failures without recording memory text."""

import json
import logging

import httpx
import pytest

from scone_memory.providers.llm import ChatError, OpenAICompatibleChat, OpenAICompatibleTextModel


async def test_chat_diagnostics_record_timing_and_model_without_request_or_response_text(caplog):
    caplog.set_level(logging.INFO, logger="scone_memory.providers.llm")
    response = httpx.Response(200, json={"choices": [{"message": {"content": "private response"}}]})
    chat = OpenAICompatibleChat(
        "http://username:password@llm.local/v1", "local-model", api_key="private-api-key",
        transport=httpx.MockTransport(lambda _: response),
    )
    assert await chat.complete("private system", "private user") == "private response"

    finished = [record for record in caplog.records if getattr(record, "outcome", None) == "completed"]
    assert len(finished) == 1 and finished[0].model_name == "local-model"
    assert finished[0].elapsed_ms >= 0
    assert finished[0].call_id
    assert finished[0].event == "model_call.finished"
    [started] = [record for record in caplog.records if getattr(record, "event", None) == "model_call.started"]
    assert started.call_id == finished[0].call_id
    assert started.model_name == "local-model" and started.mode == "chat" and started.timeout_s > 0
    for secret in ("private", "username", "password", "llm.local"):
        assert secret not in caplog.text


async def test_chat_transport_failure_logs_exception_type_without_exception_body(caplog):
    caplog.set_level(logging.INFO, logger="scone_memory.providers.llm")

    def timeout(request):
        raise httpx.ReadTimeout("private timeout details", request=request)

    chat = OpenAICompatibleChat("http://llm.local/v1", "local-model", transport=httpx.MockTransport(timeout))
    with pytest.raises(ChatError):
        await chat.complete("private system", "private user")
    [failure] = [record for record in caplog.records if getattr(record, "outcome", None) == "failed"]
    assert failure.exception_type == "ReadTimeout" and failure.elapsed_ms >= 0
    assert "private" not in caplog.text


async def test_stream_diagnostics_include_first_token_and_completion(caplog):
    caplog.set_level(logging.INFO, logger="scone_memory.providers.llm")
    chunk = json.dumps({"choices": [{"delta": {"content": "private response"}}]})
    response = httpx.Response(200, content=f"data: {chunk}\n\ndata: [DONE]\n\n".encode(),
                              headers={"content-type": "text/event-stream"})
    model = OpenAICompatibleTextModel("http://llm.local/v1", "local-model",
                                      transport=httpx.MockTransport(lambda _: response))
    try:
        _ = [event async for event in model.respond([{"role": "user", "content": "private user"}])]
    finally:
        await model.aclose()
    [finished] = [record for record in caplog.records if getattr(record, "outcome", None) == "completed"]
    assert finished.mode == "stream" and 0 <= finished.first_token_ms <= finished.elapsed_ms
    [started] = [record for record in caplog.records if getattr(record, "event", None) == "model_call.started"]
    [first] = [record for record in caplog.records if getattr(record, "event", None) == "model_call.first_token"]
    assert started.call_id == first.call_id == finished.call_id
    assert first.model_name == "local-model" and first.mode == "stream"
    assert first.first_token_ms == finished.first_token_ms
    assert "private" not in caplog.text
