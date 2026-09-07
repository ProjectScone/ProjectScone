"""Deleting a space over HTTP and from the CLI: a preview first, a
confirmed deed, then 404 on every route for the key that named it. Only
the full role may do it; write can forget one episode but not a space."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.runtime.cli import main

FIXTURE = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/space-receipt.json").read_text())

KEYS = {"alpha-full": "alpha", "alpha-write": "alpha", "beta": "beta"}
ROLES = {"alpha-full": "full", "alpha-write": "write", "beta": "full"}


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_a_space_is_deleted_by_its_own_full_key_after_a_preview_and_confirmation():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, KEYS, console=False, roles=ROLES)) as c:
        for text in ("first note in alpha", "second note in alpha"):
            assert c.post("/v1/episodes", json={"content": text}, headers=bearer("alpha-full")).status_code == 200
        assert c.post("/v1/episodes", json={"content": "beta keeps this"}, headers=bearer("beta")).status_code == 200
        preview = c.get("/v1/spaces/alpha/impact", headers=bearer("alpha-full"))
        assert preview.status_code == 200 and preview.json()["episodes"] == 2 and preview.json()["deleted_at"] is None
        assert c.get("/v1/spaces/beta/impact", headers=bearer("alpha-full")).status_code == 404, "no cross-space preview"
        assert c.delete("/v1/spaces/alpha", headers=bearer("alpha-full")).status_code == 422, "confirm is required"
        assert c.delete("/v1/spaces/alpha", params={"confirm": "beta"}, headers=bearer("alpha-full")).status_code == 422
        denied = c.delete("/v1/spaces/alpha", params={"confirm": "alpha"}, headers=bearer("alpha-write"))
        assert denied.status_code == 403 and "write" in denied.json()["error"], "only full deletes a space"
        assert c.get("/v1/status", headers=bearer("alpha-full")).json()["episodes"] == 2, "nothing removed yet"
        done = c.delete("/v1/spaces/alpha", params={"confirm": "alpha"}, headers=bearer("alpha-full"))
        assert done.status_code == 200, done.text
        want = sorted(FIXTURE["receipt"])
        assert sorted(preview.json()) == want and sorted(k for k in done.json() if k != "deleted") == want, "the shared receipt keys"
        assert done.json()["deleted"] == "alpha" and done.json()["episodes"] == 2 and done.json()["deleted_at"]
        for path in ("/v1/status", "/v1/profile", "/v1/spaces/alpha/impact"):
            assert c.get(path, headers=bearer("alpha-full")).status_code == 404, f"{path} answers 404 for a deleted space"
        assert c.post("/v1/episodes", json={"content": "back?"}, headers=bearer("alpha-full")).status_code == 404
        assert c.delete("/v1/spaces/alpha", params={"confirm": "alpha"}, headers=bearer("alpha-full")).status_code == 404
        assert c.get("/v1/status", headers=bearer("beta")).json()["episodes"] == 1, "the neighbour is untouched"


def test_the_cli_deletes_a_space_only_with_its_name_confirmed(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), "SCONE_EMBEDDER": "hash"}
    out = io.StringIO()
    assert main(["--space", "alpha", "--json", "remember"], env=env, stdin=io.StringIO("a note"), out=out) == 0
    out = io.StringIO()
    assert main(["--space", "alpha", "--json", "delete-space", "--dry-run"], env=env, out=out) == 0
    assert json.loads(out.getvalue())["episodes"] == 1
    assert main(["--space", "alpha", "delete-space", "--confirm", "beta"], env=env, out=io.StringIO()) != 0, "the name must match"
    out = io.StringIO()
    assert main(["--space", "alpha", "--json", "delete-space", "--confirm", "alpha"], env=env, out=out) == 0
    receipt = json.loads(out.getvalue())
    assert receipt["deleted"] == "alpha" and receipt["episodes"] == 1 and receipt["deleted_at"]
    assert main(["--space", "alpha", "--json", "remember"], env=env, stdin=io.StringIO("again"), out=io.StringIO()) != 0, "a deleted space takes no writes"


async def test_the_conversation_service_answers_404_for_a_deleted_space(tmp_path):
    import httpx
    from scone_memory.api.conversations import create_conversation_app

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_conversation_app(engine, {"k": "alpha", "other": "beta"}, tmp_path / "j.db", None)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test") as client:
            assert (await client.get("/v1/conversations", headers=bearer("k"))).status_code == 200
            await engine.delete_space("alpha")
            assert (await client.get("/v1/conversations", headers=bearer("k"))).status_code == 404
            started = await client.post("/v1/conversations", json={"request_id": "a", "capture": True}, headers=bearer("k"))
            assert started.status_code == 404, started.text
            assert (await client.post("/v1/episodes", json={"content": "x"}, headers=bearer("k"))).status_code == 404
            assert (await client.get("/v1/conversations", headers=bearer("other"))).status_code == 200, "the neighbour is untouched"


def test_the_audio_socket_refuses_a_deleted_space_at_hello(tmp_path):
    from scone_memory.api.conversations import create_conversation_app

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    asyncio.run(engine.delete_space("alpha"))
    app = create_conversation_app(engine, {"k": "alpha", "other": "beta"}, tmp_path / "j.db", None)

    def hello(client, key):
        with client.websocket_connect("/v1/conversations/none/audio") as socket:
            socket.send_text(json.dumps({"type": "hello", "key": key, "sample_rate": 16000, "channels": 1}))
            return json.loads(socket.receive_text())["reason"]

    with TestClient(app) as client:
        assert "deleted" in hello(client, "k"), "a deleted space's key is refused before any session is looked up"
        assert hello(client, "other") == "unknown session"
