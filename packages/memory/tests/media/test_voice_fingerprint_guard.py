"""The audio attachment checks the saved persona identity before starting voice."""

import asyncio
import json
import sqlite3

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app

from ..api.test_voice_http import AUTH, HELLO, KEYS, catalog, create_voice


def test_waiting_voice_with_a_stale_saved_fingerprint_is_refused_before_starting(tmp_path):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    journal = tmp_path / "voice.db"
    app = create_conversation_app(engine, KEYS, journal, None, catalog=catalog())
    with TestClient(app) as client:
        client.headers.update(AUTH)
        created = create_voice(client)
        assert created.status_code == 200
        receipt = created.json()
        session_id = receipt["session_id"]
        # Model a persisted selection that differs from this process's catalog
        # while preserving the waiting state; startup recovery is not involved.
        with sqlite3.connect(journal) as database:
            database.execute("UPDATE sessions SET persona_fingerprint = ? WHERE session_id = ?",
                             ("stale-selection", session_id))

        with client.websocket_connect(f"/v1/conversations/{session_id}/audio") as socket:
            socket.send_text(json.dumps(HELLO))
            refused = json.loads(socket.receive_text())
            assert refused["type"] == "error" and "model settings changed" in refused["reason"]

        after = client.get(f"/v1/conversations/{session_id}").json()
        assert after["state"] == "created" and after["revision"] == receipt["revision"]
