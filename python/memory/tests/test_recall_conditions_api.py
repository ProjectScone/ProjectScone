"""Narrowing a search over HTTP by what was recorded about a memory."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def client():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for n in range(20):
        await engine.remember("alpha", f"quarterly planning note {n}", metadata={"status": "draft"})
    await engine.remember("alpha", "quarterly planning note, the one that shipped",
                          metadata={"status": "published", "team": "eng"})
    app = create_app(engine, {"key-a": "alpha"})
    with TestClient(app) as c:
        yield c


def auth() -> dict:
    return {"authorization": "Bearer key-a"}


def ask(client, conditions=None, q="quarterly planning note"):
    params = {"q": q, "limit": 5}
    if conditions is not None:
        params["conditions"] = json.dumps(conditions)
    return client.get("/v1/recall", params=params, headers=auth())


def test_a_narrowed_search_returns_only_what_it_asked_for(client):
    answer = ask(client, {"field": "status", "is": "published"})
    assert answer.status_code == 200
    texts = [item["text"] for item in answer.json()["items"]]
    assert texts == ["quarterly planning note, the one that shipped"]


def test_the_same_search_without_the_filter_is_full_of_drafts(client):
    """The filter is doing the work, not the query."""
    texts = [item["text"] for item in ask(client).json()["items"]]
    assert len(texts) == 5 and all("shipped" not in t for t in texts)


def test_conditions_nest_the_way_they_are_written(client):
    answer = ask(client, {"all": [{"field": "status", "is": "published"},
                                  {"any": [{"field": "team", "is": "eng"},
                                           {"field": "team", "is": "product"}]}]})
    assert [i["text"] for i in answer.json()["items"]] == ["quarterly planning note, the one that shipped"]


@pytest.mark.parametrize("conditions, complaint", [
    ("{not json", "conditions must be"),
    ('["a"]', "mapping"),
    ('{"field": "status"}', "exactly one test"),
    ('{"field": "Status", "is": "x"}', "metadata key"),
], ids=["malformed", "not a mapping", "no test", "bad key"])
def test_a_filter_that_cannot_mean_anything_is_refused_not_ignored(client, conditions, complaint):
    """Ignoring an unreadable filter answers from everything, which is
    the one outcome a caller narrowing a search must never get."""
    answer = client.get("/v1/recall", params={"q": "note", "conditions": conditions}, headers=auth())
    assert answer.status_code == 422
    assert complaint in answer.json()["error"]


def test_a_filter_too_long_to_be_a_question_is_refused(client):
    answer = client.get("/v1/recall", params={"q": "note", "conditions": "x" * 9000}, headers=auth())
    assert answer.status_code == 422
    assert "too long" in answer.json()["error"]


def test_the_manifest_says_a_search_can_be_narrowed_this_way(client):
    """A caller must not have to probe for it, or infer it from a failure."""
    features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    assert features["recall.conditions"] is True


def test_browsing_sources_can_be_narrowed_the_same_way(client):
    """The same question asked of the same memories must not depend on
    which screen it was asked from."""
    answer = client.get("/v1/sources", params={
        "limit": 25, "conditions": json.dumps({"field": "status", "is": "published"})},
        headers=auth())
    assert answer.status_code == 200
    items = answer.json()["items"]
    assert [i["preview"] for i in items] == ["quarterly planning note, the one that shipped"]


def test_an_unreadable_filter_stops_a_browse_too(client):
    answer = client.get("/v1/sources", params={"conditions": "{not json"}, headers=auth())
    assert answer.status_code == 422
    assert "conditions must be" in answer.json()["error"]
