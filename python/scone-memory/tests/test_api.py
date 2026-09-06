"""The HTTP surface, checked with the sibling ``scone-client`` models when
that package is on disk so a drift between the two stacks fails here."""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

REPO = pathlib.Path(__file__).resolve().parents[3]
#: The HTTP client lives at python/scone-client after the repository
#: cleanup; the older location is checked second during the move.
CLIENT_MODEL_PATHS = (
    REPO / "python" / "scone-client" / "scone" / "models.py",
    REPO / "clients" / "python" / "scone" / "models.py",
)


def load_client_models():
    found = next((path for path in CLIENT_MODEL_PATHS if path.exists()), None)
    if found is None:
        pytest.skip("scone-client not checked out beside this package")
    spec = importlib.util.spec_from_file_location("scone_client_models", found)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
async def client():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {"key-a": "alpha", "key-b": "beta"})
    with TestClient(app) as c:
        yield c


def auth(key: str = "key-a") -> dict:
    return {"authorization": f"Bearer {key}"}


def test_no_key_or_wrong_key_is_401(client):
    assert client.get("/v1/status").status_code == 401
    r = client.get("/v1/status", headers=auth("nope"))
    assert r.status_code == 401
    assert r.json() == {"error": "unknown key"}


def test_key_decides_the_space(client):
    client.post("/v1/episodes", json={"content": "alpha's secret garden"}, headers=auth("key-a"))
    seen_by_b = client.get("/v1/recall", params={"q": "secret garden"}, headers=auth("key-b")).json()
    assert seen_by_b["items"] == []
    assert client.get("/v1/status", headers=auth("key-b")).json()["space"] == "beta"


def test_unknown_field_is_refused(client):
    r = client.post("/v1/episodes", json={"content": "x", "colour": "red"}, headers=auth())
    assert r.status_code == 422
    assert "colour" in r.json()["error"]


def test_engine_errors_map_to_status_codes(client):
    assert client.post("/v1/episodes", json={"content": "   "}, headers=auth()).status_code == 422
    assert client.post("/v1/facts/42/close", json={"reason": "gone"}, headers=auth()).status_code == 404
    assert client.delete("/v1/episodes/42", headers=auth()).status_code == 404
    assert client.get("/v1/recall", params={"q": ""}, headers=auth()).status_code == 422


def test_round_trip_parses_with_the_shared_client(client):
    models = load_client_models()
    added = client.post(
        "/v1/episodes",
        json={"content": "Moved to Lisbon in March", "tags": ["life"], "created_at": "2024-03-02"},
        headers=auth(),
    )
    assert added.status_code == 200
    parsed_added = models.Added.from_json(added.json())
    assert parsed_added.episode_id == 1 and parsed_added.chunks == 1

    client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon"}, headers=auth())
    recall = client.get("/v1/recall", params={"q": "lisbon", "limit": "3"}, headers=auth())
    parsed = models.Recall.from_json(recall.json())
    assert [m.episode_id for m in parsed.items] == [1]
    assert parsed.items[0].created_at == "2024-03-02T00:00:00.000Z"
    assert [f.object for f in parsed.facts] == ["Lisbon"]
    assert parsed.context_reduction == 0.0

    facts = models.Fact  # every fact field the client reads is present
    for f in client.get("/v1/facts", headers=auth()).json()["facts"]:
        facts.from_json(f)
    closed = client.post("/v1/facts/1/close", json={"reason": "moved again"}, headers=auth()).json()
    assert closed == {"closed": 1, "reason": "moved again"}

    status = models.Status.from_json(client.get("/v1/status", headers=auth()).json())
    assert (status.space, status.episodes, status.chunks) == ("alpha", 1, 1)
    tags = [models.Tag.from_json(t) for t in client.get("/v1/tags", headers=auth()).json()["tags"]]
    assert [(t.name, t.count) for t in tags] == [("life", 1)]
    profile = models.Profile.from_json(client.get("/v1/profile", headers=auth()).json())
    assert profile.dynamic == ["Moved to Lisbon in March"]


def test_health_needs_no_key(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_where_filter_over_http(client):
    for user, seat in (("alice", "window"), ("bob", "aisle")):
        client.post(
            "/v1/episodes",
            json={"content": f"{user} prefers the {seat} seat", "metadata": {"user_id": user}},
            headers=auth(),
        )
    r = client.get("/v1/recall", params={"q": "seat", "where": "user_id:bob"}, headers=auth()).json()
    assert [i["metadata"]["user_id"] for i in r["items"]] == ["bob"]
    assert client.get("/v1/recall", params={"q": "seat", "where": "user_id"}, headers=auth()).status_code == 422


def test_console_is_served_with_the_key_baked_in():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        page = c.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert 'data-token="solo"' in page.text
    with TestClient(create_app(engine, {"a": "x", "b": "y"})) as c:
        assert 'data-token="' not in c.get("/").text  # more than one key: the page asks
    with TestClient(create_app(engine, {"a": "x"}, console=False)) as c:
        assert c.get("/").status_code == 404


def test_review_and_exclusion_over_http(client):
    h = auth()
    held = client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Austin", "valid_from": "2022-01-01"}, headers=h).json()
    proposed = client.post(
        "/v1/facts",
        json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon", "valid_from": "2024-03-02", "origin": "extracted", "proposed": True, "confidence": 0.7},
        headers=h,
    ).json()
    assert (proposed["status"], proposed["origin"]) == ("proposed", "extracted")
    assert [f["object"] for f in client.get("/v1/facts", headers=h).json()["facts"]] == ["Austin"]
    assert [f["fact_id"] for f in client.get("/v1/facts", params={"status": "proposed"}, headers=h).json()["facts"]] == [proposed["fact_id"]]
    assert client.get("/v1/status", headers=h).json()["pending_review"] == 1
    approved = client.post(f"/v1/facts/{proposed['fact_id']}/approve", headers=h).json()
    assert approved["status"] == "active"
    assert client.get("/v1/facts", params={"all": "true"}, headers=h).json()["facts"][0]["closed_reason"] == f"superseded by fact {proposed['fact_id']}"
    excluded = client.post(f"/v1/facts/{proposed['fact_id']}/exclude", json={"reason": "private"}, headers=h).json()
    assert excluded["excluded_reason"] == "private"
    assert client.get("/v1/facts", headers=h).json()["facts"] == []
    assert [f["fact_id"] for f in client.get("/v1/facts", params={"excluded": "true"}, headers=h).json()["facts"]] == [proposed["fact_id"]]
    assert client.post(f"/v1/facts/{proposed['fact_id']}/include", headers=h).json()["excluded_reason"] is None
    assert client.post(f"/v1/facts/{held['fact_id']}/decline", json={"reason": "x"}, headers=h).status_code == 422
    assert client.post("/v1/facts/999/approve", headers=h).status_code == 404


def test_an_episode_can_be_read_back_verbatim(client):
    h = auth()
    added = client.post("/v1/episodes", json={"content": "  Café ☕ verbatim\n", "created_at": "2024-01-02", "metadata": {"user_id": "ana"}}, headers=h).json()
    body = client.get(f"/v1/episodes/{added['episode_id']}", headers=h).json()
    assert (body["content"], body["kind"], body["created_at"], body["metadata"]) == ("  Café ☕ verbatim\n", "note", "2024-01-02T00:00:00.000Z", {"user_id": "ana"})
    assert client.get("/v1/episodes/999", headers=h).status_code == 404
    assert client.get(f"/v1/episodes/{added['episode_id']}", headers=auth("key-b")).status_code == 404


def test_playground_is_served_with_the_same_key_handling():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        page = c.get("/playground")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "__SCONE_TOKEN__" not in page.text and "solo" in page.text
    with TestClient(create_app(engine, {"a": "x", "b": "y"})) as c:
        assert "__SCONE_TOKEN__" in c.get("/playground").text  # several keys: the page asks
