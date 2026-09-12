"""Whatever the ledger accepts, the HTTP API can answer with.

The ledger takes any Python string, lone surrogates included, and UTF-8
cannot encode one; a plain JSON response then fails with a 500 on the
first stored name that holds one. Every route answers instead, writing
such a character as its JSON escape, which reads back as the same text.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

ODD = "al\ud800ice"


@pytest.fixture
async def client():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", ODD, "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "bob", "odd\ud800predicate", "value", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key": "alpha"})) as http:
        yield http


@pytest.mark.parametrize("path", ["/v1/graph/knowledge", "/v1/graph/report", "/v1/graph/schema", "/v1/entities",
                                  "/v1/recall?q=acme", "/v1/facts?subject=bob"])
def test_a_lone_surrogate_in_stored_text_is_answered_not_a_500(client, path):
    response = client.get(path, headers={"Authorization": "Bearer key"})
    assert response.status_code == 200 and b"\\ud800" in response.content


def test_the_escape_reads_back_as_the_stored_text(client):
    body = client.get("/v1/graph/schema", headers={"Authorization": "Bearer key"}).json()
    assert "odd\ud800predicate" in {entry["predicate"] for entry in body["predicates"]}


async def test_an_ambiguous_answer_carries_its_candidates_text_too():
    """The 409 listing candidates is written by the route itself, not by
    the app's default, and must not fail on a stored name either."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for who in ("zed one \ud800", "zed two \ud800"):
        await engine.assert_fact("alpha", who, "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key": "alpha"})) as http:
        response = http.get("/v1/graph/timeline", params={"entity": "zed"}, headers={"Authorization": "Bearer key"})
    assert response.status_code == 409 and len(response.json()["candidates"]) == 2
