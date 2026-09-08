"""Receipt graphs are current retained evidence, never cached source copies."""

import asyncio
import json

import pytest

from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.text import TextConversation
from test_conversations_api import client_for, create, engine  # noqa: F401
from test_text_conversation import ScriptedModel


async def completed(client, session):
    url = f"/v1/conversations/{session['session_id']}/turns"
    accepted = await client.post(url, json={"request_id": "question", "expected_revision": session["revision"],
                                            "text": "What have we been working on?"})
    assert accepted.status_code == 202
    async with asyncio.timeout(5):
        while True:
            receipt = (await client.get(url + "/question")).json()
            if receipt["status"] != "pending":
                assert receipt["status"] == "completed", receipt
                return url + "/question", receipt
            await asyncio.sleep(.01)


@pytest.mark.parametrize("change", ["forget_source", "change_chunk", "failed_read", "failed_graph_read", "timeout", "forget_query"])
async def test_graph_receipt_never_replays_unretained_evidence(engine, tmp_path, monkeypatch, change):
    secret = "The retained project evidence contains violet-marker and confirms telescope calibration finished successfully."
    added = await engine.remember("alpha", secret, source="project/notes")
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db",
        lambda space, sid: TextConversation(engine, space, sid, lambda: ScriptedModel("The work is complete.")))
    async with client_for(app) as client:
        session = await create(client)
        route, original = await completed(client, session)
        context = original["result"]["memory_context"]
        assert context["evidence_graph_status"] == "prepared"
        assert secret in json.dumps(context["evidence_graph"])
        if change == "forget_source":
            await engine.forget("alpha", added.episode_id)
        elif change == "forget_query":
            await engine.forget("alpha", original["result"]["user_episode_id"])
        else:
            get_chunks = engine.documents.get_chunks
            reads = 0

            async def changed(space, ids):
                nonlocal reads
                reads += 1
                if change == "timeout":
                    await asyncio.Future()
                if change == "failed_read" or (change == "failed_graph_read" and reads == 2):
                    raise RuntimeError("private-read-failure")
                if change == "failed_graph_read":
                    return await get_chunks(space, ids)
                return [chunk.model_copy(update={"text": "replacement-private-marker"})
                        for chunk in await get_chunks(space, ids)]

            monkeypatch.setattr(engine.documents, "get_chunks", changed)
        subsequent = (await client.get(route)).json()
        assert subsequent["status"] == "completed"
        assert subsequent["result"]["text"] == "The work is complete."
        assert subsequent["result"]["memory_context"]["references"] == context["references"]
        assert "violet-marker" not in json.dumps(subsequent)
        assert "replacement-private-marker" not in json.dumps(subsequent)
        assert "private-read-failure" not in json.dumps(subsequent)
        if change in {"failed_read", "failed_graph_read", "timeout", "forget_query"}:
            assert subsequent["result"]["memory_context"]["evidence_graph_status"] == "unavailable"


async def test_rebuilt_graph_preserves_scope_and_omits_reconstructed_rank(engine, tmp_path):
    allowed = await engine.remember("alpha", "We finished the scoped telescope alignment procedure and validated calibration results.",
                                    metadata={"project": "allowed"})
    await engine.remember("alpha", "excluded-project-marker shows the unrelated private project's calibration details.",
                          metadata={"project": "excluded"})
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", None,
        scoped_runtime_factory=lambda space, sid, scope: TextConversation(engine, space, sid,
            lambda: ScriptedModel("The work is complete."), **scope.kwargs()))
    async with client_for(app) as client:
        response = await client.post("/v1/conversations", json={"request_id": "new", "capture": True,
                                                                 "recall_scope": {"where": {"project": "allowed"}}})
        _, receipt = await completed(client, response.json())
        graph = receipt["result"]["memory_context"]["evidence_graph"]
        assert "excluded-project-marker" not in json.dumps(graph)
        chunks = [node for node in graph["nodes"] if node["kind"] == "chunk"]
        assert len(chunks) == 1 and chunks[0]["data"]["episode_id"] == allowed.episode_id
        assert not {"score", "similarity", "lanes"} & chunks[0]["data"].keys()
