"""A key names a space and a role. read only reads; write adds and forgets
but never decides; review decides but never adds; full does everything.
The same rule holds on the conversation service."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import __main__ as serve
from scone_memory.api import create_app
from scone_memory.api.conversations import create_conversation_app
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings

KEYS = {"reader": "default", "writer": "default", "reviewer": "default", "boss": "default"}
ROLES = {"reader": "read", "writer": "write", "reviewer": "review", "boss": "full"}


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_roles_are_parsed_from_the_keys_setting_and_default_to_full():
    settings = Settings.from_env({"SCONE_API_KEYS": "r:alpha:read,w:alpha:write,v:alpha:review,f:beta:full,plain:beta"})
    assert settings.keys == {"r": "alpha", "w": "alpha", "v": "alpha", "f": "beta", "plain": "beta"}
    assert settings.roles == {"r": "read", "w": "write", "v": "review", "f": "full", "plain": "full"}
    assert Settings.from_env({"SCONE_API_KEY": "k"}).roles == {"k": "full"}
    with pytest.raises(InvalidInput):
        Settings.from_env({"SCONE_API_KEYS": "x:alpha:owner"})


def test_each_role_can_do_what_it_says_and_nothing_more():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, KEYS, console=False, roles=ROLES)) as c:
        episode = c.post("/v1/episodes", json={"content": "written by the writer"}, headers=bearer("writer"))
        assert episode.status_code == 200
        proposed = c.post("/v1/facts", json={"subject": "mark", "predicate": "likes", "object": "tea", "proposed": True}, headers=bearer("writer")).json()

        denied = c.post("/v1/episodes", json={"content": "a reader cannot write"}, headers=bearer("reader"))
        assert denied.status_code == 403 and "read" in denied.json()["error"]
        assert c.get("/v1/status", headers=bearer("reader")).status_code == 200
        assert c.get(f"/v1/episodes/{episode.json()['episode_id']}", headers=bearer("reader")).status_code == 200
        assert c.delete(f"/v1/episodes/{episode.json()['episode_id']}", headers=bearer("reader")).status_code == 403

        assert c.post(f"/v1/facts/{proposed['fact_id']}/approve", headers=bearer("writer")).status_code == 403, "write does not decide"
        assert c.post("/v1/facts/decide", json={"decision": "approve", "fact_ids": [proposed["fact_id"]], "expect_revision": 1}, headers=bearer("writer")).status_code == 403
        assert c.post("/v1/episodes", json={"content": "a reviewer cannot add"}, headers=bearer("reviewer")).status_code == 403
        approved = c.post(f"/v1/facts/{proposed['fact_id']}/approve", headers=bearer("reviewer"))
        assert approved.status_code == 200 and approved.json()["status"] == "active"

        assert c.post("/v1/episodes", json={"content": "the boss writes"}, headers=bearer("boss")).status_code == 200
        assert c.post(f"/v1/facts/{proposed['fact_id']}/close", json={"reason": "done"}, headers=bearer("boss")).status_code == 200
        assert c.get("/v1/status", headers=bearer("nobody")).status_code == 401, "an unknown key is still unknown, not a role problem"
    with TestClient(create_app(engine, {"k": "default"}, console=False)) as c:
        assert c.post("/v1/episodes", json={"content": "no roles given means full"}, headers=bearer("k")).status_code == 200


async def test_the_conversation_service_holds_the_same_rule(tmp_path):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_conversation_app(engine, KEYS, tmp_path / "j.db", None, roles=ROLES)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test") as client:
            assert (await client.get("/v1/conversations", headers=bearer("reader"))).status_code == 200
            denied = await client.post("/v1/conversations", json={"request_id": "a", "capture": True}, headers=bearer("reader"))
            assert denied.status_code == 403
            assert (await client.post("/v1/episodes", json={"content": "x"}, headers=bearer("reader"))).status_code == 403, "the mounted memory app carries the roles too"


def test_serve_passes_roles_to_both_hosts(tmp_path):
    settings = Settings.from_env({"SCONE_API_KEYS": "r:default:read,w:default:write", "SCONE_SQLITE_PATH": str(tmp_path / "m.db")})
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(serve.build_app(settings, engine)) as c:
        assert c.post("/v1/episodes", json={"content": "x"}, headers=bearer("r")).status_code == 403
        assert c.post("/v1/episodes", json={"content": "x"}, headers=bearer("w")).status_code == 200
    composed = Settings.from_env({"SCONE_API_KEYS": "r:default:read,w:default:write", "SCONE_SQLITE_PATH": str(tmp_path / "m.db"),
                                  "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db")})
    with TestClient(serve.build_app(composed, engine)) as c:
        assert c.post("/v1/episodes", json={"content": "y"}, headers=bearer("r")).status_code == 403
        assert c.post("/v1/conversations", json={"request_id": "a", "capture": True}, headers=bearer("r")).status_code == 403


def test_the_audio_socket_refuses_a_reader_at_hello(tmp_path):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    app = create_conversation_app(engine, KEYS, tmp_path / "j.db", None, roles=ROLES)

    def hello(client, key):
        with client.websocket_connect("/v1/conversations/none/audio") as socket:
            socket.send_text(json.dumps({"type": "hello", "key": key, "sample_rate": 16000, "channels": 1}))
            return json.loads(socket.receive_text())["reason"]

    with TestClient(app) as client:
        refused = hello(client, "reader")
        assert "read" in refused and "session" in refused, "the role refuses before any session is looked up"
        assert hello(client, "boss") == "unknown session", "a full key reaches the session lookup"
