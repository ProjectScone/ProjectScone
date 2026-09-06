"""Real TCP and real Pipecat scheduling; no paid model or live database."""

import asyncio
from contextlib import asynccontextmanager
import json
import socket

import httpx
import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import LLMContextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame, LLMThoughtTextFrame
from pipecat.processors.frame_processor import FrameProcessor

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.api.conversation_server import create_server
from scone_memory.integrations.pipecat_text import PipecatTextConversation


@asynccontextmanager
async def serving(app, *, servers=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = create_server(app, host="127.0.0.1", port=port)
        if servers is not None:
            servers.append(server)
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise AssertionError("fixture server stopped before startup")
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": "Bearer alpha-key"}, timeout=5) as client:
                yield client
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)


async def next_event(lines):
    values = {}
    async with asyncio.timeout(5):
        async for line in lines:
            if not line and values:
                return values["event"], json.loads(values["data"]), values.get("id")
            if line.startswith(":"):
                continue
            if ": " in line:
                key, value = line.split(": ", 1)
                values[key] = value
    raise AssertionError("stream ended before the next event")


class LiveModel(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.next_chunk, self.end_response = asyncio.Event(), asyncio.Event()
        self.ended = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMThoughtTextFrame("never public"))
        await self.push_frame(LLMTextFrame("First "))
        await self.next_chunk.wait()
        await self.push_frame(LLMTextFrame("🌍\nsecond"))
        await self.end_response.wait()
        self.ended = True
        await self.push_frame(LLMFullResponseEndFrame())


@pytest.mark.parametrize("outcome", ["complete", "cancel", "stop"])
async def test_native_http_live_chunks_reconnect_and_terminal_evidence(tmp_path, outcome):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    models = []
    def factory():
        model = LiveModel()
        models.append(model)
        return model
    app = create_conversation_app(memory, {"alpha-key": "alpha", "beta-key": "beta"}, tmp_path / "journal.db",
                                  lambda space, sid: PipecatTextConversation(memory, space, sid, factory),
                                  public_text_streaming=True)
    async with serving(app) as client:
        session = (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).json()
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "turn", "text": "Hello", "expected_revision": session["revision"]}
        assert (await client.post(url + "/turns", json=body)).status_code == 202
        route = url + "/turns/turn/stream"
        async with client.stream("GET", route) as response:
            assert response.status_code == 200
            assert await next_event(response.aiter_lines()) == ("text", {"sequence": 1, "text": "First ", "provisional": True}, "1")
            assert not models[0].ended
            transcript = (await client.get(url + "/transcript")).json()["episodes"]
            assert [e["content"] for e in transcript] == ["Hello"]
        # Closing only the read connection must not cancel or resubmit the turn.
        assert (await client.post(url + "/turns", json=body)).json()["status"] == "pending"
        assert (await client.get(route, headers={"Authorization": "Bearer beta-key"})).status_code == 404
        async with client.stream("GET", route, headers={"Last-Event-ID": "1"}) as response:
            lines = response.aiter_lines()
            models[0].next_chunk.set()
            assert await next_event(lines) == ("text", {"sequence": 2, "text": "🌍\nsecond", "provisional": True}, "2")
            assert not models[0].ended
            if outcome == "complete":
                models[0].end_response.set()
                assert await next_event(lines) == ("terminal", {"request_id": "turn", "status": "completed", "read_receipt": True}, None)
                receipt = (await client.get(url + "/turns/turn")).json()
                assert receipt["result"]["text"] == "First 🌍\nsecond"
                assert receipt["result"]["provider_completion"] == "unverified"
            elif outcome == "cancel":
                assert (await client.post(url + "/turns/turn/cancel")).json()["status"] == "cancelled"
                assert await next_event(lines) == ("terminal", {"request_id": "turn", "status": "cancelled", "read_receipt": True}, None)
            else:
                stopped = await client.post(url + "/stop", json={"request_id": "stop", "expected_revision": session["revision"]})
                assert stopped.json()["state"] == "ended"
                event, data, _ = await next_event(lines)
                assert (event == "end" and data["reason"] == "window_unavailable") or (event == "terminal" and data["status"] == "interrupted")
            # No additional chunks, private thought or late result after terminal.
            assert [line async for line in lines] == []
        assert len(models) == 1
        transcript = (await client.get(url + "/transcript")).json()["episodes"]
        assert [e["content"] for e in transcript] == (["Hello", "First 🌍\nsecond"] if outcome == "complete" else ["Hello"])
        assert "never public" not in repr(transcript)


async def test_native_http_reports_evicted_prefix_without_fabricating_chunks(tmp_path):
    from test_conversation_stream import StreamingConversation
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    ready, release = asyncio.Event(), asyncio.Event()
    class ManyChunks(StreamingConversation):
        async def reply(self, text, *, on_text):
            for index in range(257):
                await on_text(str(index))
            ready.set()
            await release.wait()
    app = create_conversation_app(memory, {"alpha-key": "alpha"}, tmp_path / "journal.db",
                                  lambda space, sid: ManyChunks(memory, space, sid), public_text_streaming=True)
    async with serving(app) as client:
        session = (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).json()
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "many", "text": "Hi", "expected_revision": session["revision"]})
        await asyncio.wait_for(ready.wait(), 3)
        async with client.stream("GET", url + "/turns/many/stream") as response:
            lines = response.aiter_lines()
            assert await next_event(lines) == ("gap", {"after": 0, "next_sequence": 2}, None)
            assert await next_event(lines) == ("text", {"sequence": 2, "text": "1", "provisional": True}, "2")
        assert (await client.post(url + "/turns/many/cancel")).json()["status"] == "cancelled"


async def test_server_shutdown_closes_idle_stream_before_waiting_for_http_tasks(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    model = LiveModel()
    app = create_conversation_app(memory, {"alpha-key": "alpha"}, tmp_path / "journal.db",
                                  lambda space, sid: PipecatTextConversation(memory, space, sid, lambda: model),
                                  public_text_streaming=True)
    servers = []
    async with serving(app, servers=servers) as client:
        session = (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).json()
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "live", "text": "Hi", "expected_revision": session["revision"]})
        async with client.stream("GET", url + "/turns/live/stream") as response:
            lines = response.aiter_lines()
            assert (await next_event(lines))[0] == "text"
            servers[0].should_exit = True
            assert await next_event(lines) == ("end", {"request_id": "live", "reason": "service_shutdown", "read_receipt": True}, None)
            assert [line async for line in lines] == []
    assert not model.ended
    assert [e.content for e in await memory.episodes("alpha", {"session_id": session["session_id"]})] == ["Hi"]
