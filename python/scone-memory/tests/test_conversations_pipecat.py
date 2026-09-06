"""API → real Pipecat scheduling → scoped memory; scripted model, no network."""

import asyncio
import copy

import httpx
import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import (
    LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
    LLMTextFrame, LLMThoughtTextFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.integrations.pipecat_text import PipecatTextConversation


async def test_http_sessions_use_pipecat_context_history_and_public_capture(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Juniper is calibrated with Polaris.", metadata={"collection": "manuals"})
    requests = []

    class ScriptedModel(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if not isinstance(frame, LLMContextFrame):
                await self.push_frame(frame, direction)
                return
            requests.append(copy.deepcopy(frame.context.get_messages()))
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMThoughtTextFrame("not public"))
            await self.push_frame(LLMTextFrame("Scripted Polaris answer."))
            await self.push_frame(LLMFullResponseEndFrame())

    def factory(space, sid):
        return PipecatTextConversation(memory, space, sid, ScriptedModel, where={"collection": "manuals"})

    app = create_conversation_app(memory, {"alpha-key": "alpha", "beta-key": "beta"}, tmp_path / "sessions.db", factory)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers={"Authorization": "Bearer alpha-key"}) as client:
            session = (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).json()
            url = "/v1/conversations/" + session["session_id"]
            results = []
            for index, message in enumerate(["How is Juniper calibrated?", "What was my question?"]):
                body = {"request_id": f"turn-{index}", "text": message, "expected_revision": session["revision"]}
                response = await client.post(url + "/turns", json=body)
                assert response.status_code == 202
                async def completed():
                    while True:
                        receipt = (await client.get(url + f"/turns/turn-{index}")).json()
                        if receipt["status"] != "pending":
                            return receipt
                        await asyncio.sleep(0.01)
                receipt = await asyncio.wait_for(completed(), 5)
                assert receipt["status"] == "completed", receipt
                assert receipt["result"]["text"] == "Scripted Polaris answer."
                assert receipt["result"]["provider_completion"] == "unverified"
                results.append(receipt["result"])
                assert (await client.post(url + "/turns", json=body)).json() == receipt
            assert len(requests) == 2  # matching HTTP retries did not invoke the model
            assert any("Juniper is calibrated with Polaris." in m["content"] for m in requests[0])
            assert {"role": "user", "content": "How is Juniper calibrated?"} in requests[1]
            assert {"role": "assistant", "content": "Scripted Polaris answer."} in requests[1]
            episodes = await memory.episodes("alpha", {"session_id": session["session_id"]})
            assert [e.content for e in episodes] == [
                "How is Juniper calibrated?", "Scripted Polaris answer.",
                "What was my question?", "Scripted Polaris answer.",
            ]
            assert [e.episode_id for e in episodes] == [
                results[0]["user_episode_id"], results[0]["assistant_episode_id"],
                results[1]["user_episode_id"], results[1]["assistant_episode_id"],
            ]
            transcript = await client.get(url + "/transcript")
            assert transcript.status_code == 200
            assert [e["content"] for e in transcript.json()["episodes"]] == [e.content for e in episodes]
            assert transcript.json()["has_more"] is False
            assert (await client.get(url + "/transcript", headers={"Authorization": "Bearer beta-key"})).status_code == 404
            assert (await client.get(url, headers={"Authorization": "Bearer beta-key"})).status_code == 404
            assert not await memory.episodes("beta", {"session_id": session["session_id"]})
            assert (await client.post(url + "/stop", json={"request_id": "stop", "expected_revision": session["revision"]})).json()["state"] == "ended"
