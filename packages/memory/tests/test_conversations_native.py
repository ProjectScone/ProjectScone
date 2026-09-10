"""API → real Scone scheduling → scoped memory; scripted model, no network."""

import asyncio
import copy

import httpx
import pytest


from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.text import TextConversation
from scone_memory.realtime.events import TextDelta, ReplyCompleted


@pytest.mark.parametrize("empty_scope", [False, True])
async def test_session_scope_controls_real_native_context_across_turns(tmp_path, empty_scope):
    from test_text_conversation import ScriptedModel
    from test_conversations_api import client_for

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    allowed = await memory.remember("alpha", "Juniper calibration: allowed-manual uses Polaris.", kind="file",
                                    source="docs/public/one", created_at="2026-09-02", metadata={"collection": "manuals"})
    excluded = [
        ("alpha", "wrong-kind", "note", "docs/public/note", "2026-09-02", "manuals"),
        ("alpha", "wrong-source", "file", "private/one", "2026-09-02", "manuals"),
        ("alpha", "too-old", "file", "docs/public/old", "2026-08-01", "manuals"),
        ("alpha", "too-new", "file", "docs/public/new", "2026-10-01", "manuals"),
        ("alpha", "wrong-metadata", "file", "docs/public/other", "2026-09-02", "other"),
        ("beta", "wrong-space", "file", "docs/public/tenant", "2026-09-02", "manuals"),
    ]
    for space, label, kind, source, created, collection in excluded:
        await memory.remember(space, f"Juniper calibration: {label} is not eligible.", kind=kind, source=source,
                              created_at=created, metadata={"collection": collection})
    models = []
    def model_factory():
        model = ScriptedModel("A scoped answer")
        models.append(model)
        return model
    def scoped(space, sid, scope):
        return TextConversation(memory, space, sid, model_factory, **scope.kwargs())
    app = create_conversation_app(memory, {"alpha-key": "alpha"}, tmp_path / "sessions.db", None,
                                  scoped_runtime_factory=scoped)
    scope = {"kind": "file", "where": {"collection": "missing" if empty_scope else "manuals"},
             "source_prefix": "docs/public/", "since": "2026-09-01", "until": "2026-09-03"}
    async with client_for(app) as client:
        created = await client.post("/v1/conversations", json={"request_id": "scoped", "capture": True, "recall_scope": scope})
        assert created.status_code == 200, created.text
        session = created.json()
        url = "/v1/conversations/" + session["session_id"]
        for index in range(2):
            route = url + "/turns/turn-" + str(index)
            accepted = await client.post(url + "/turns", json={"request_id": f"turn-{index}",
                                          "expected_revision": session["revision"], "text": "How is Juniper calibrated?"})
            assert accepted.status_code == 202
            async with asyncio.timeout(5):
                while True:
                    receipt = (await client.get(route)).json()
                    if receipt["status"] != "pending":
                        break
                    await asyncio.sleep(0.01)
            assert receipt["status"] == "completed", receipt
            references = receipt["result"]["memory_context"]["references"]
            assert {ref["episode_id"] for ref in references} == (set() if empty_scope else {allowed.episode_id})
        assert len(models) == 2
        for model in models:
            request = repr(model.requests)
            assert ("allowed-manual" in request) is not empty_scope
            for _, label, *_ in excluded:
                assert label not in request
        assert len((await client.get(url + "/transcript")).json()["episodes"]) == 4


async def test_cancelled_native_turn_can_be_followed_by_a_completed_reply(tmp_path):
    from test_text_conversation import ScriptedModel
    from test_conversations_api import client_for, create

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    waiting = ScriptedModel(mode="wait")
    following = ScriptedModel("A new answer")
    models = iter([waiting, following])
    app = create_conversation_app(memory, {"alpha-key": "alpha"}, tmp_path / "sessions.db",
                                 lambda space, sid: TextConversation(memory, space, sid, lambda: next(models)))
    async with client_for(app) as client:
        created = await create(client)
        url = "/v1/conversations/" + created["session_id"]
        def body(key, text):
            return {"request_id": key, "text": text, "expected_revision": created["revision"]}
        original = body("cancel-this", "Regretted question")
        assert (await client.post(url + "/turns", json=original)).status_code == 202
        await asyncio.wait_for(waiting.entered.wait(), 3)
        assert (await client.post(url + "/turns/cancel-this/cancel")).json()["status"] == "cancelled"
        assert (await client.get(url)).json()["state"] == "running"
        assert (await client.post(url + "/turns", json=original)).json()["status"] == "cancelled"
        assert (await client.post(url + "/turns", json=body("new-turn", "A different question"))).status_code == 202
        async def settled():
            while True:
                receipt = (await client.get(url + "/turns/new-turn")).json()
                if receipt["status"] != "pending":
                    return receipt
                await asyncio.sleep(0.01)
        receipt = await asyncio.wait_for(settled(), 5)
        assert receipt["status"] == "completed", receipt
        assert receipt["result"]["text"] == "A new answer"
        transcript = (await client.get(url + "/transcript")).json()["episodes"]
        assert [e["content"] for e in transcript] == ["Regretted question", "A different question", "A new answer"]
        assert len(waiting.requests) == len(following.requests) == 1
        assert {"role": "user", "content": "Regretted question"} not in following.requests[0]
        assert (await client.get("/v1/conversations/capabilities")).json()["turn_cancellation"] is True


@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_cancellation_during_capture_does_not_leave_a_closed_runtime_running(tmp_path, monkeypatch, role):
    from test_text_conversation import ScriptedModel
    from test_conversations_api import client_for, create

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    recording = asyncio.Event()
    remember = memory.remember_many
    async def pause_capture(space, records):
        records = list(records)
        added = await remember(space, records)
        if records[0].metadata.get("role") == role:
            recording.set()
            await asyncio.Event().wait()  # write happened; acknowledgment is interrupted
        return added
    monkeypatch.setattr(memory, "remember_many", pause_capture)
    runtime = TextConversation(memory, "alpha", "capture-session", ScriptedModel)
    app = create_conversation_app(memory, {"alpha-key": "alpha"}, tmp_path / "sessions.db", lambda space, sid: runtime)
    async with client_for(app) as client:
        created = await create(client)
        url = "/v1/conversations/" + created["session_id"]
        body = {"request_id": "capture-turn", "text": "Question", "expected_revision": created["revision"]}
        assert (await client.post(url + "/turns", json=body)).status_code == 202
        await asyncio.wait_for(recording.wait(), 3)
        assert (await client.post(url + "/turns/capture-turn/cancel")).json()["status"] == "cancelled"
        assert (await client.get(url)).json()["state"] == "interrupted"
        assert (await client.post(url + "/turns", json={**body, "request_id": "do-not-run"})).status_code == 409
        assert (await client.get(url + "/turns/capture-turn")).json()["status"] == "cancelled"


async def test_identical_public_messages_remain_distinct_and_delete_only_their_session(tmp_path):
    """Real capture keys must include session, turn and speaker, not just text.

    Unlike the content-deduplicating custom API fixture, Scone preserves
    each occurrence. Deletion must remove all four occurrences in one session
    without removing the other's identical text or independent knowledge.
    """
    from test_text_conversation import ScriptedModel
    from test_conversations_api import client_for, create

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    manual = await memory.remember("alpha", "Same words", metadata={"collection": "manuals"})
    def factory(space, sid):
        return TextConversation(memory, space, sid, lambda: ScriptedModel("Same words"))
    app = create_conversation_app(memory, {"alpha-key": "alpha", "beta-key": "beta"}, tmp_path / "sessions.db", factory)
    async with client_for(app) as client:
        sessions = [await create(client, "first"), await create(client, "second")]
        transcripts = []
        for value in sessions:
            url = "/v1/conversations/" + value["session_id"]
            for index in range(2):
                body = {"request_id": f"turn-{index}", "text": "Same words", "expected_revision": value["revision"]}
                assert (await client.post(url + "/turns", json=body)).status_code == 202
                async def settled():
                    while True:
                        receipt = (await client.get(url + f"/turns/turn-{index}")).json()
                        if receipt["status"] != "pending":
                            return receipt
                        await asyncio.sleep(0.01)
                receipt = await asyncio.wait_for(settled(), 5)
                assert receipt["status"] == "completed", receipt
                assert (await client.post(url + "/turns", json=body)).json() == receipt
            transcript = (await client.get(url + "/transcript")).json()["episodes"]
            assert [e["content"] for e in transcript] == ["Same words"] * 4
            assert [e["metadata"]["role"] for e in transcript] == ["user", "assistant", "user", "assistant"]
            assert {e["metadata"]["session_id"] for e in transcript} == {value["session_id"]}
            transcripts.append(transcript)
            assert (await client.post(url + "/stop", json={"request_id": "stop", "expected_revision": value["revision"]})).json()["state"] == "ended"
        assert len({e["episode_id"] for transcript in transcripts for e in transcript}) == 8
        first = "/v1/conversations/" + sessions[0]["session_id"]
        second = "/v1/conversations/" + sessions[1]["session_id"]
        assert (await client.delete(first)).status_code == 204
        assert (await client.get(first)).status_code == 404
        for removed in transcripts[0]:
            gone = await client.get(f"/v1/episodes/{removed['episode_id']}")
            assert gone.status_code == 410 and gone.json()["forgotten_at"], "deleted with its session: forgotten on purpose, not unknown"
        assert (await client.get(second + "/transcript")).json()["episodes"] == transcripts[1]
        assert (await memory.episode("alpha", manual.episode_id)).content == "Same words"
        assert (await client.get("/v1/conversations/capabilities")).json()["session_deletion"] is True


async def test_http_sessions_use_native_context_history_and_public_capture(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Juniper is calibrated with Polaris.", metadata={"collection": "manuals"})
    requests = []

    class ScriptedModel:
        async def aclose(self):
            pass

        async def respond(self, messages):
            requests.append(copy.deepcopy(messages))
            yield TextDelta("Scripted Polaris answer.")
            yield ReplyCompleted()

    def factory(space, sid):
        return TextConversation(memory, space, sid, ScriptedModel, where={"collection": "manuals"})

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
