"""Personas over HTTP: a catalog a client can read, a choice fixed at session
creation and echoed in every receipt, and nothing chosen implicitly."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import sys
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import __main__ as serve
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.catalog import bind_catalog
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.persona import Persona
from scone_memory.realtime.providers import ProviderRegistry
from scone_memory.runtime.config import Settings

HELPER = {"schema_version": 1, "id": "helper", "name": "Helper", "instructions": "Answer briefly.",
          "reply": {"provider": "stub", "model": "echo"}, "transcription": {"provider": "stub", "model": "ears"},
          "speech": {"provider": "stub", "model": "mouth", "voice": "alto"}}
KEYS = {"alpha-key": "alpha"}
AUTH = {"Authorization": "Bearer alpha-key"}


class ScriptedModel:
    seen: list = []  # shared, so a plain zero-argument factory can be the registry's

    async def aclose(self):
        pass

    async def respond(self, messages):
        ScriptedModel.seen.append(messages)
        yield TextDelta("Scripted reply")
        yield ReplyCompleted()


def registry():
    return ProviderRegistry(reply={("stub", "echo"): ScriptedModel},
                            transcription={("stub", "ears"): lambda: None},
                            speech={("stub", "mouth", "alto"): lambda: None})


def catalog(*documents):
    return bind_catalog([Persona.model_validate(document) for document in documents], registry())


class BareRuntime:
    def __init__(self, engine, space, sid):
        self.engine, self.space, self.sid = engine, space, sid

    async def reply(self, text):
        item = await self.engine.remember(self.space, "Bare reply to " + text, metadata={"session_id": self.sid})
        return {"text": "Bare reply to " + text, "assistant_episode_id": item.episode_id, "provider_completion": "unverified"}

    async def close(self):
        pass


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
            yield client


async def engine_for():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


async def settled(client, sid, request_id):
    async with asyncio.timeout(5):
        while True:
            receipt = (await client.get(f"/v1/conversations/{sid}/turns/{request_id}")).json()
            if receipt["status"] != "pending":
                return receipt
            await asyncio.sleep(0.02)


async def test_the_catalog_is_readable_and_counted_but_never_shows_instructions(tmp_path):
    app = create_conversation_app(await engine_for(), KEYS, tmp_path / "j.db", None, catalog=catalog(HELPER), public_text_streaming=True)
    async with client_for(app) as client:
        ready = (await client.get("/v1/conversations/capabilities")).json()
        assert ready["personas"] == 1 and ready["text_configured"] is True
        listing = (await client.get("/v1/conversations/personas")).json()
        assert listing["schema_version"] == 1 and [p["id"] for p in listing["personas"]] == ["helper"]
        assert listing["personas"][0]["speech"] == {"provider": "stub", "model": "mouth", "voice": "alto"}
        assert listing["personas"][0]["voice_ready"] is True, "this host carries audio; the registry admitted the choices"
        assert "Answer briefly" not in json.dumps(listing)
        assert (await client.get("/v1/conversations/personas", headers={"Authorization": ""})).status_code == 401
    bare = create_conversation_app(await engine_for(), KEYS, tmp_path / "k.db", None)
    async with client_for(bare) as client:
        ready = (await client.get("/v1/conversations/capabilities")).json()
        assert ready["personas"] == 0 and ready["text_configured"] is False
        empty = (await client.get("/v1/conversations/personas")).json()
        assert empty["personas"] == [] and len(empty["revision"]) == 16


async def test_a_persona_is_fixed_at_creation_echoed_in_receipts_and_drives_the_reply(tmp_path):
    ScriptedModel.seen.clear()
    engine = await engine_for()
    app = create_conversation_app(engine, KEYS, tmp_path / "j.db", None, catalog=catalog(HELPER), public_text_streaming=True)
    async with client_for(app) as client:
        missing = await client.post("/v1/conversations", json={"request_id": "a", "capture": True, "persona": "nobody"})
        assert missing.status_code == 422 and "nobody" in missing.text
        unchosen = await client.post("/v1/conversations", json={"request_id": "b", "capture": True})
        assert unchosen.status_code == 422 and "helper" in unchosen.text, "nothing is chosen implicitly"
        created = await client.post("/v1/conversations", json={"request_id": "c", "capture": True, "persona": "helper"})
        identity = {"id": "helper", "name": "Helper", "fingerprint": created.json()["persona"]["fingerprint"], "current": True}
        assert created.status_code == 200 and created.json()["persona"] == identity
        sid = created.json()["session_id"]
        assert (await client.get(f"/v1/conversations/{sid}")).json()["persona"] == identity
        assert [s["persona"] for s in (await client.get("/v1/conversations")).json()["items"]] == [identity]
        retry = await client.post("/v1/conversations", json={"request_id": "c", "capture": True, "persona": "helper"})
        assert retry.json()["session_id"] == sid and retry.json()["persona"]["id"] == "helper"
        body = {"request_id": "t", "text": "Hello", "expected_revision": created.json()["revision"]}
        assert (await client.post(f"/v1/conversations/{sid}/turns", json=body)).status_code == 202
        assert (await settled(client, sid, "t"))["status"] == "completed"
        assert ScriptedModel.seen[0][0] == {"role": "system", "content": "Answer briefly."}
    # A catalog is fixed for a process's life; a later process without this
    # persona still reports the id, with no name to give it.
    again = create_conversation_app(engine, KEYS, tmp_path / "j.db", None)
    async with client_for(again) as client:
        gone = (await client.get(f"/v1/conversations/{sid}")).json()["persona"]
        assert gone == {"id": "helper", "name": None, "fingerprint": identity["fingerprint"], "current": False}


async def test_a_bare_runtime_still_serves_sessions_created_without_a_persona(tmp_path):
    engine = await engine_for()
    app = create_conversation_app(engine, KEYS, tmp_path / "j.db", lambda space, sid: BareRuntime(engine, space, sid), catalog=catalog(HELPER))
    async with client_for(app) as client:
        created = await client.post("/v1/conversations", json={"request_id": "a", "capture": True})
        assert created.status_code == 200 and created.json()["persona"] is None
        chosen = await client.post("/v1/conversations", json={"request_id": "b", "capture": True, "persona": "helper"})
        assert chosen.status_code == 200 and chosen.json()["persona"]["id"] == "helper"


def test_serve_binds_the_catalog_at_startup_or_refuses(tmp_path, monkeypatch, capsys):
    module = ModuleType("catalog_test_registry")
    module.registry = registry
    module.not_a_registry = lambda: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    personas = tmp_path / "personas.json"
    personas.write_text(json.dumps([HELPER]))
    base = {"SCONE_API_KEY": "alpha-key", "SCONE_SQLITE_PATH": str(tmp_path / "memory.db"),
            "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db"), "SCONE_CONVERSATIONS_PERSONAS": str(personas)}
    engine = asyncio.run(engine_for())
    with pytest.raises(ValueError, match="SCONE_CONVERSATIONS_REGISTRY"):
        serve.build_app(Settings.from_env(base), engine)
    with pytest.raises(ValueError, match="ProviderRegistry"):
        serve.build_app(Settings.from_env({**base, "SCONE_CONVERSATIONS_REGISTRY": "catalog_test_registry:not_a_registry"}), engine)
    app = serve.build_app(Settings.from_env({**base, "SCONE_CONVERSATIONS_REGISTRY": "catalog_test_registry:registry"}), engine)
    with TestClient(app) as c:
        assert [p["id"] for p in c.get("/v1/conversations/personas", headers=AUTH).json()["personas"]] == ["helper"]
        assert c.get("/v1/conversations/capabilities", headers=AUTH).json()["text_configured"] is True

    personas.write_text(json.dumps([{**HELPER, "id": "singer", "speech": {"provider": "stub", "model": "mouth", "voice": "tenor"}}]))

    async def fake_build(settings):
        return await engine_for()

    monkeypatch.setattr(serve, "build_engine", fake_build)
    refused = {**base, "SCONE_CONVERSATIONS_REGISTRY": "catalog_test_registry:registry",
               "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "never-opened.db")}
    with pytest.raises(SystemExit) as stop:
        serve.main(Settings.from_env(refused))
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "singer" in err and "speech" in err and "Answer briefly" not in err
    assert not (tmp_path / "never-opened.db").exists(), "refused before anything was served"


async def test_a_stale_persona_selection_is_refused_and_receipts_keep_the_recorded_identity(tmp_path):
    engine = await engine_for()
    app = create_conversation_app(engine, KEYS, tmp_path / "j.db", None, catalog=catalog(HELPER), public_text_streaming=True)
    async with client_for(app) as client:
        listing = (await client.get("/v1/conversations/personas")).json()
        current = listing["personas"][0]["fingerprint"]
        assert len(listing["revision"]) == 16 and len(current) == 16
        stale = await client.post("/v1/conversations", json={"request_id": "a", "capture": True, "persona": "helper",
                                                              "persona_fingerprint": "0000000000000000"})
        assert stale.status_code == 409 and "helper" in stale.text and current in stale.text
        assert (await client.get("/v1/conversations")).json()["items"] == [], "a stale selection creates nothing"
        created = await client.post("/v1/conversations", json={"request_id": "b", "capture": True, "persona": "helper",
                                                                "persona_fingerprint": current})
        assert created.status_code == 200
        assert created.json()["persona"] == {"id": "helper", "name": "Helper", "fingerprint": current, "current": True}
        sid = created.json()["session_id"]
        legacy = await client.post("/v1/conversations", json={"request_id": "c", "capture": True, "persona": "helper"})
        assert legacy.status_code == 200 and legacy.json()["persona"]["fingerprint"] == current, "omitting the fingerprint skips the check, not the record"
    # The operator changes the voice and restarts: the receipt keeps what was
    # chosen and says it is no longer the current configuration.
    louder = {**HELPER, "speech": {"provider": "stub", "model": "mouth", "voice": "tenor"}}
    changed = bind_catalog([Persona.model_validate(louder)], ProviderRegistry(
        reply={("stub", "echo"): ScriptedModel}, transcription={("stub", "ears"): lambda: None},
        speech={("stub", "mouth", "tenor"): lambda: None}))
    again = create_conversation_app(engine, KEYS, tmp_path / "j.db", None, catalog=changed, public_text_streaming=True)
    async with client_for(again) as client:
        receipt = (await client.get(f"/v1/conversations/{sid}")).json()["persona"]
        assert receipt == {"id": "helper", "name": "Helper", "fingerprint": current, "current": False}
        assert (await client.get("/v1/conversations/personas")).json()["personas"][0]["fingerprint"] != current
