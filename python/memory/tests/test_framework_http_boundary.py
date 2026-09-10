"""The framework starts and authenticates without a Webapp installation."""

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.api.conversations import create_conversation_app


@pytest.mark.parametrize("conversations", [False, True])
async def test_framework_hosts_only_its_api_without_disclosing_browser_keys(tmp_path, conversations):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    key = "framework-boundary-private-key"
    try:
        app = (create_conversation_app(engine, {key: "alpha"}, tmp_path / "sessions.db", None)
               if conversations else create_app(engine, {key: "alpha"}))
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50200)) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/v1/status").status_code == 401
            headers = {"authorization": f"Bearer {key}"}
            assert client.get("/v1/status", headers=headers).json()["space"] == "alpha"
            for route in ("/", "/memory", "/playground", "/conversations", "/learn/quickstart",
                          "/memory/sources/1", "/v1/missing"):
                for method in (client.get, client.head):
                    response = method(route)
                    assert response.status_code == 404, route
                    assert "text/html" not in response.headers.get("content-type", "")
                    assert key not in response.text
            if conversations:
                assert client.get("/v1/conversations/capabilities", headers=headers).status_code == 200
    finally:
        await engine.close()
