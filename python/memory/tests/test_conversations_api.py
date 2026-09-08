"""Authenticated service plumbing with scripted replies and isolated journals."""

import asyncio
from contextlib import asynccontextmanager
import sqlite3
import os

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.session_journal import SessionJournal


class ControlledConversation:
    def __init__(self, engine, space, sid, *, blocked=False):
        self.engine, self.space, self.sid = engine, space, sid
        self.started, self.release = asyncio.Event(), asyncio.Event()
        if not blocked:
            self.release.set()
        self.calls = 0
        self.closed = False

    async def reply(self, text):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        item = await self.engine.remember(self.space, "Scripted response to " + text,
                                          metadata={"session_id": self.sid})
        return {"text": "Scripted response to " + text, "assistant_episode_id": item.episode_id,
                "provider_completion": "unverified"}

    async def close(self):
        self.closed = True


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers={"Authorization": "Bearer alpha-key"}) as client:
            yield client


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


def configured(engine, path, *, blocked=False, **kwargs):
    runtimes = []
    def factory(space, sid):
        runtime = ControlledConversation(engine, space, sid, blocked=blocked)
        runtimes.append(runtime)
        return runtime
    return create_conversation_app(engine, {"alpha-key": "alpha", "beta-key": "beta"}, path, factory, **kwargs), runtimes


async def create(client, request_id="new"):
    response = await client.post("/v1/conversations", json={"request_id": request_id, "capture": True})
    assert response.status_code == 200, response.text
    return response.json()


async def test_timeout_receipt_and_logs_name_failure_without_echoing_private_text(engine, tmp_path, caplog):
    app, runtimes = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)

        async def timed_out(text):
            raise TimeoutError("private-provider-content")

        runtimes[0].reply = timed_out
        url = "/v1/conversations/" + session["session_id"]
        response = await client.post(url + "/turns", json={"request_id": "timeout", "text": "private-user-content", "expected_revision": session["revision"]})
        assert response.status_code == 202
        for _ in range(100):
            receipt = (await client.get(url + "/turns/timeout")).json()
            if receipt["status"] == "failed":
                break
            await asyncio.sleep(.01)
        assert "timed out" in receipt["error"]
        assert "TimeoutError" in caplog.text
        assert "private-user-content" not in caplog.text
        assert "private-provider-content" not in caplog.text


async def test_scoped_factory_receives_immutable_validated_scope_and_retries_cannot_change_it(engine, tmp_path):
    seen = []
    def scoped(space, sid, scope):
        seen.append((space, sid, scope))
        return ControlledConversation(engine, space, sid)
    path = tmp_path / "sessions.db"
    app = create_conversation_app(engine, {"alpha-key": "alpha", "beta-key": "beta"}, path, None,
                                  scoped_runtime_factory=scoped)
    expected = {"kind": "file", "where": {"collection": "manuals"}, "since": "2026-09-01T00:00:00.000Z"}
    body = {"request_id": "new", "capture": True, "recall_scope": {**expected, "since": "2026-09-01T01:00:00+01:00"}}
    async with client_for(app) as client:
        cap = (await client.get("/v1/conversations/capabilities")).json()
        assert cap["text_configured"] is True and cap["recall_scope"] is True
        response = await client.post("/v1/conversations", json=body)
        assert response.status_code == 200, response.text
        saved = response.json()
        assert saved["recall_scope"] == expected
        assert seen[0][0:2] == ("alpha", saved["session_id"])
        # The runtime can obtain kwargs, but cannot mutate the frozen policy.
        kwargs = seen[0][2].kwargs()
        kwargs["where"]["collection"] = "private"
        assert seen[0][2].as_dict() == expected
        with pytest.raises(AttributeError):
            seen[0][2].kind = "note"
        assert (await client.post("/v1/conversations", json={**body, "recall_scope": expected})).json() == saved
        assert (await client.post("/v1/conversations", json={**body, "recall_scope": {}})).status_code == 409
        assert len(seen) == 1
        assert (await client.get("/v1/conversations")).json()["items"][0]["recall_scope"] == expected
    # Recreated service exposes the same constraints without starting a provider.
    async with client_for(create_conversation_app(engine, {"alpha-key": "alpha"}, path, None)) as client:
        assert (await client.get("/v1/conversations/" + saved["session_id"])).json()["recall_scope"] == expected
        assert len(seen) == 1


@pytest.mark.parametrize("scope", [{"kind": "file"}, {"source_prefix": ""}])
async def test_legacy_factories_reject_narrowing_instead_of_silently_ignoring_it(engine, tmp_path, scope):
    app, runtimes = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        assert (await client.get("/v1/conversations/capabilities")).json()["recall_scope"] is False
        rejected = await client.post("/v1/conversations", json={"request_id": "new", "capture": True, "recall_scope": scope})
        assert rejected.status_code == 422
        assert runtimes == []
        assert (await client.get("/v1/conversations")).json()["items"] == []
        accepted = await client.post("/v1/conversations", json={"request_id": "new", "capture": True, "recall_scope": {}})
        assert accepted.status_code == 200 and len(runtimes) == 1


@pytest.mark.parametrize("scope", [{"space": "beta"}, {"where": {"team": 1}}, {"since": "tomorrow"},
                                    {"since": "2026-09-02", "until": "2026-09-01"}, None, []])
async def test_api_invalid_scopes_never_start_a_runtime(engine, tmp_path, scope):
    started = []
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", None,
                                  scoped_runtime_factory=lambda *args: started.append(args))
    async with client_for(app) as client:
        result = await client.post("/v1/conversations", json={"request_id": "new", "capture": True, "recall_scope": scope})
        assert result.status_code == 422
        assert started == []
        assert (await client.get("/v1/conversations")).json()["items"] == []


async def test_auth_scope_strict_bodies_and_existing_memory_surface(engine, tmp_path):
    app, runtimes = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        assert (await client.get("/v1/conversations", headers={"Authorization": ""})).status_code == 401
        assert (await client.get("/v1/conversations?key=alpha-key", headers={"Authorization": ""})).status_code == 401
        assert (await client.post("/v1/conversations", json={"request_id": "new", "capture": True, "space": "beta"})).status_code == 422
        assert (await client.post("/v1/conversations", json={"request_id": "new", "capture": False})).status_code == 422
        session = await create(client)
        assert session["space"] == "alpha" and session["state"] == "running"
        assert (await client.get("/v1/status")).json()["space"] == "alpha"
        path = "/v1/conversations/" + session["session_id"]
        wrong = {"Authorization": "Bearer beta-key"}
        assert (await client.get(path, headers=wrong)).status_code == 404
        assert (await client.get(path + "/events", headers=wrong)).status_code == 404
        assert (await client.get("/v1/conversations", headers=wrong)).json()["items"] == []
        assert len(runtimes) == 1


async def test_workspace_is_opt_in_and_deep_links_never_disclose_keys(engine, tmp_path):
    """Catch absent SPA deep links, an over-broad fallback, or key injection."""
    disabled, _ = configured(engine, tmp_path / "disabled.db")
    async with client_for(disabled) as client:
        for path in ("/", "/memory", "/playground", "/conversations", "/conversations/session-one",
                     "/memory/sources/42", "/learn", "/learn/how-it-works", "/learn/graph-memory",
                     "/learn/quickstart", "/learn/sources", "/learn/search", "/learn/review",
                     "/learn/profiles", "/learn/conversations", "/learn/spaces", "/learn/api"):
            assert (await client.get(path)).status_code == 404

    enabled, _ = configured(engine, tmp_path / "enabled.db", console=True)
    async with client_for(enabled) as client:
        for path in ("/", "/memory", "/playground", "/conversations", "/conversations/session-one",
                     "/memory/sources/42", "/learn", "/learn/how-it-works", "/learn/graph-memory",
                     "/learn/quickstart", "/learn/sources", "/learn/search", "/learn/review",
                     "/learn/profiles", "/learn/conversations", "/learn/spaces", "/learn/api"):
            response = await client.get(path, headers={"Authorization": ""})
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert 'id="root"' in response.text
            assert "alpha-key" not in response.text and "beta-key" not in response.text
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            head = await client.head(path)
            assert head.status_code == 200 and head.content == b""
            assert head.headers["content-length"] == response.headers["content-length"]
        assert (await client.get("/v1/unknown")).status_code == 404
        assert (await client.get("/conversations/session-one/not-a-route")).status_code == 404
        assert (await client.get("/learn/not-a-page")).status_code == 404, "no catch-all"
        # The mounted memory router can turn a method mismatch into 404; neither
        # may serve the SPA for a write or turn it into a session command.
        assert (await client.post("/conversations")).status_code in {404, 405}
        assert (await client.get("/v1/conversations", headers={"Authorization": ""})).status_code == 401
        assert (await client.get("/v1/status", headers={"Authorization": ""})).status_code == 401


async def test_turn_reply_capture_and_request_retry_execute_once(engine, tmp_path):
    app, runtimes = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        assert (await create(client))["session_id"] == session["session_id"]
        assert len(runtimes) == 1
        path = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "turn-1", "text": "Juniper", "expected_revision": session["revision"]}
        assert (await client.post(path + "/turns", json=body)).status_code == 202
        await asyncio.wait_for(runtimes[0].started.wait(), 1)
        # Yield through ASGI until the owned task has written its receipt.
        for _ in range(30):
            response = await client.get(path + "/turns/turn-1")
            if response.json()["status"] != "pending":
                break
            await asyncio.sleep(0)
        assert response.json()["status"] == "completed"
        assert response.json()["result"]["text"] == "Scripted response to Juniper"
        assert response.headers["cache-control"] == "no-store"
        assert (await client.post(path + "/turns", json=body)).json()["status"] == "completed"
        assert (await client.post(path + "/turns", json={**body, "text": "changed"})).status_code == 409
        assert runtimes[0].calls == 1
        assert [item.text for item in (await engine.recall("alpha", "Juniper")).items] == ["Scripted response to Juniper"]
        stop = {"request_id": "stop-1", "expected_revision": session["revision"]}
        assert (await client.post(path + "/stop", json=stop)).json()["state"] == "ended"
        assert (await client.post(path + "/stop", json=stop)).json()["state"] == "ended"
        assert runtimes[0].closed


async def test_busy_stop_and_restart_never_resubmit_model_work(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, runtimes = configured(engine, path, blocked=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "turn", "text": "pending", "expected_revision": session["revision"]}
        await client.post(url + "/turns", json=body)
        await asyncio.wait_for(runtimes[0].started.wait(), 1)
        assert (await client.post(url + "/turns", json={**body, "request_id": "other"})).status_code == 409
        assert (await client.post(url + "/stop", json={"request_id": "stop", "expected_revision": session["revision"]})).json()["state"] == "ended"
        assert (await client.get(url + "/turns/turn")).json()["status"] == "interrupted"
        runtimes[0].release.set()
        assert not (await engine.recall("alpha", "pending")).items
    restarted, new_runtimes = configured(engine, path)
    async with client_for(restarted) as client:
        assert (await client.get(url)).json()["state"] == "ended"
        assert (await client.post(url + "/turns", json=body)).status_code == 409
        # Agreed with Opus in memory/COORDINATION-FABLE.md (14:40, accepted
        # 14:24): a turn this process never ran is answered from the journal
        # rather than 404'd, because 404 conflated "nobody sent that" with
        # "the process that ran it is gone". 404 still means the first one.
        recovered = (await client.get(url + "/turns/turn")).json()
        assert recovered["status"] == "interrupted" and recovered["result"] is None
        assert (await client.get(url + "/turns/never-sent")).status_code == 404
        assert not new_runtimes


async def test_exclusive_owner_and_startup_recovery(engine, tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = journal.create("alpha", "crashed")["session_id"]
        journal.transition("alpha", sid, "start", "start", 1)
    app, runtimes = configured(engine, path)
    other, _ = configured(engine, path)
    async with client_for(app) as client:
        with pytest.raises(RuntimeError, match="owned"):
            async with other.router.lifespan_context(other):
                pass
        assert (await client.get("/v1/conversations/" + sid)).json()["state"] == "interrupted"
        assert not runtimes


async def test_unrelated_database_remains_unchanged(engine, tmp_path):
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE VIEW important AS SELECT 42")
    before = path.read_bytes()
    app, _ = configured(engine, path)
    with pytest.raises(Exception, match="refusing"):
        async with app.router.lifespan_context(app):
            pass
    assert path.read_bytes() == before


async def test_failed_create_attempts_are_bounded(engine, tmp_path):
    def broken(_space, _sid):
        raise RuntimeError("private credential detail")
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", broken, max_sessions=2)
    async with client_for(app) as client:
        for key in ["one", "two"]:
            response = await client.post("/v1/conversations", json={"request_id": key, "capture": True})
            assert response.status_code == 503
            assert "private credential" not in response.text
        assert (await client.post("/v1/conversations", json={"request_id": "three", "capture": True})).status_code == 429
        assert len((await client.get("/v1/conversations")).json()["items"]) == 2


async def test_completed_turns_count_toward_limit_but_matching_retry_still_works(engine, tmp_path):
    app, runtimes = configured(engine, tmp_path / "sessions.db", max_turns=1)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "one", "text": "Juniper", "expected_revision": session["revision"]}
        assert (await client.post(url + "/turns", json=body)).status_code == 202
        for _ in range(30):
            result = (await client.get(url + "/turns/one")).json()
            if result["status"] != "pending":
                break
            await asyncio.sleep(0)
        assert result["status"] == "completed"
        assert (await client.post(url + "/turns", json={**body, "request_id": "two"})).status_code == 429
        assert (await client.post(url + "/turns", json=body)).json() == result
        assert runtimes[0].calls == 1


async def test_shutdown_attempts_every_runtime_even_if_one_close_fails(engine, tmp_path):
    runtimes = []
    class BrokenClose(ControlledConversation):
        async def close(self):
            self.closed = True
            raise RuntimeError("private close error")
    def factory(space, sid):
        runtime = (BrokenClose if not runtimes else ControlledConversation)(engine, space, sid)
        runtimes.append(runtime)
        return runtime
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", factory)
    with pytest.raises(RuntimeError):
        async with client_for(app) as client:
            await create(client, "one")
            await create(client, "two")
    assert all(runtime.closed for runtime in runtimes)


async def test_cancelled_stop_request_does_not_abandon_owned_stop(engine, tmp_path):
    closing, release = asyncio.Event(), asyncio.Event()
    closes = []
    class SlowClose(ControlledConversation):
        async def close(self):
            closes.append(self)
            closing.set()
            await release.wait()
            self.closed = True
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", lambda space, sid: SlowClose(engine, space, sid))
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        pending = asyncio.create_task(client.post(url + "/stop", json={"request_id": "stop", "expected_revision": session["revision"]}))
        await asyncio.wait_for(closing.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        retry = await client.post(url + "/stop", json={"request_id": "stop", "expected_revision": session["revision"]})
        assert retry.json()["state"] == "stopping"
        release.set()
        for _ in range(30):
            response = await client.get(url)
            if response.json()["state"] != "stopping":
                break
            await asyncio.sleep(0)
        assert response.json()["state"] == "ended"
    assert len(closes) == 1  # neither retry nor lifespan shutdown closes twice


async def test_no_provider_body_limit_and_no_query_authorization(engine, tmp_path):
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", None)
    async with client_for(app) as client:
        cap = (await client.get("/v1/conversations/capabilities")).json()
        assert cap["text_configured"] is False and cap["streaming"] is False
        assert (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).status_code == 503
        assert (await client.post("/v1/conversations", content=b"x" * 40001)).status_code == 413


@pytest.mark.parametrize("cancelled", [False, True], ids=["failure", "internal-cancellation"])
async def test_failed_turn_closes_runtime_without_waiting_for_service_shutdown(engine, tmp_path, cancelled):
    class FailedReply(ControlledConversation):
        async def reply(self, text):
            self.calls += 1
            if cancelled:
                raise asyncio.CancelledError()
            raise RuntimeError("private provider credential")
    runtime = FailedReply(engine, "alpha", "placeholder")
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "sessions.db", lambda _space, _sid: runtime)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "turn", "text": "question", "expected_revision": session["revision"]}
        await client.post(url + "/turns", json=body)
        for _ in range(30):
            response = await client.get(url + "/turns/turn")
            if runtime.closed:
                break
            await asyncio.sleep(0)
        assert runtime.closed
        expected = "interrupted" if cancelled else "failed"
        assert response.json()["status"] == expected
        assert "private provider" not in response.text
        assert (await client.get(url)).json()["state"] == expected
        assert (await client.post(url + "/turns", json=body)).json()["status"] == expected
        assert (await client.post(url + "/turns", json={**body, "request_id": "another"})).status_code == 409
        assert runtime.calls == 1


async def test_symlink_alias_cannot_get_a_second_owner(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app):
        alias = tmp_path / "alias.db"
        alias.symlink_to(path)
        other, _ = configured(engine, alias)
        with pytest.raises(RuntimeError, match="owned"):
            async with client_for(other):
                pass
    link = tmp_path / "hardlink.db"
    os.link(path, link)
    linked, _ = configured(engine, link)
    with pytest.raises(Exception, match="hard-linked"):
        async with client_for(linked):
            pass
