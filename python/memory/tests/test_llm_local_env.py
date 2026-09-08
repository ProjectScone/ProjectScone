"""Local chat adapters can ignore ambient HTTP proxy and TLS settings."""

from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory.providers.llm import OpenAICompatibleChat, OpenAICompatibleTextModel
from scone_memory.realtime.events import ReplyCompleted, TextDelta


@pytest.mark.parametrize("streaming", [False, True])
async def test_local_chat_bypasses_ambient_proxy_and_tls_settings(monkeypatch, tmp_path, streaming):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-cert.pem"))

    async def respond(reader, writer):
        headers = await reader.readuntil(b"\r\n\r\n")
        length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                      if line.lower().startswith(b"content-length:"))
        await reader.readexactly(length)
        body = json.dumps({"choices": [{"message": {"content": "Local reply"}}]}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        if not streaming:
            chat = OpenAICompatibleChat(url, "local-model", trust_env=False, timeout=2)
            assert await chat.complete("System", "Hello") == "Local reply"
            return
        model = OpenAICompatibleTextModel(url, "local-model", trust_env=False, timeout=2)
        try:
            events = [event async for event in model.respond([{"role": "user", "content": "Hello"}])]
            assert [event.text for event in events if isinstance(event, TextDelta)] == ["Local reply"]
            assert isinstance(events[-1], ReplyCompleted)
        finally:
            await model.aclose()
