"""Typed fact relations over HTTP: asserted with a claim, added on their own,
read beside the fact with its sources, and drawn on the graph."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.observability.events import InMemoryEventLog

AUTH = {"Authorization": "Bearer key-a"}
OTHER = {"Authorization": "Bearer key-b"}


@pytest.fixture
def client():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open())
    with TestClient(create_app(engine, {"key-a": "alpha", "key-b": "beta"}, console=False)) as c:
        yield c


def fact(client, subject, predicate, object, **extra):
    response = client.post("/v1/facts", json={"subject": subject, "predicate": predicate, "object": object, **extra}, headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_claim_can_extend_or_derive_from_others_and_reads_back_with_its_links(client):
    episode = client.post("/v1/episodes", json={"content": "Acme is headquartered in Lisbon, near the river."}, headers=AUTH).json()
    works = fact(client, "mark", "works_at", "Acme", source_episode_id=episode["episode_id"], quote="Acme")
    based = fact(client, "Acme", "based_in", "Lisbon", source_episode_id=episode["episode_id"], quote="headquartered in Lisbon")
    derived = fact(client, "mark", "works_in", "Lisbon", derived_from=[works["fact_id"], based["fact_id"]])
    assert derived["origin"] == "inferred"
    detail = fact(client, "Acme", "office_near", "the river", extends=based["fact_id"])

    read = client.get(f"/v1/facts/{derived['fact_id']}", headers=AUTH).json()
    assert read["fact"]["fact_id"] == derived["fact_id"] and read["fact"]["origin"] == "inferred"
    assert sorted((l["from_fact"], l["to_fact"], l["kind"]) for l in read["links"]) == sorted(
        [(derived["fact_id"], works["fact_id"], "derived_from"), (derived["fact_id"], based["fact_id"], "derived_from")])
    assert read["sources"] == [], "an inferred claim cites premises, not an episode"
    assert all({"link_id", "kind", "from_fact", "to_fact", "created_at", "source_episode_id", "quote"} <= set(l) for l in read["links"])

    based_read = client.get(f"/v1/facts/{based['fact_id']}", headers=AUTH).json()
    assert based_read["sources"] == [episode["episode_id"]], "source ids, not source objects"
    assert sorted(l["kind"] for l in based_read["links"]) == ["derived_from", "extends"]
    assert client.get(f"/v1/facts/{based['fact_id']}", headers=OTHER).status_code == 404, "another space sees nothing"
    assert client.get("/v1/facts/999999", headers=AUTH).status_code == 404
    assert client.get("/v1/facts/audit", headers=AUTH).status_code == 200, "a fixed address is never read as an id"
    assert client.get("/v1/capabilities", headers=AUTH).json()["features"]["facts.links"] is True


def test_links_are_added_on_their_own_and_refused_when_they_would_lie(client):
    a = fact(client, "a", "is", "1")
    b = fact(client, "b", "is", "2")
    c = fact(client, "c", "is", "3")
    episode = client.post("/v1/episodes", json={"content": "b follows from c, plainly."}, headers=AUTH).json()
    link = client.post(f"/v1/facts/{b['fact_id']}/links", headers=AUTH,
                       json={"to_fact": c["fact_id"], "kind": "derived_from", "source_episode_id": episode["episode_id"], "quote": "b follows from c"})
    assert link.status_code == 200 and link.json()["kind"] == "derived_from" and link.json()["quote"] == "b follows from c"
    again = client.post(f"/v1/facts/{b['fact_id']}/links", headers=AUTH, json={"to_fact": c["fact_id"], "kind": "derived_from"})
    assert again.status_code == 200 and again.json()["link_id"] == link.json()["link_id"], "the same link twice is one link"
    assert client.post(f"/v1/facts/{c['fact_id']}/links", headers=AUTH, json={"to_fact": a["fact_id"], "kind": "derived_from"}).status_code == 200
    cycle = client.post(f"/v1/facts/{a['fact_id']}/links", headers=AUTH, json={"to_fact": b["fact_id"], "kind": "derived_from"})
    assert cycle.status_code == 422 and "cycle" in cycle.text
    assert client.post(f"/v1/facts/{a['fact_id']}/links", headers=AUTH, json={"to_fact": a["fact_id"], "kind": "supports"}).status_code == 422
    assert client.post(f"/v1/facts/{a['fact_id']}/links", headers=AUTH, json={"to_fact": b["fact_id"], "kind": "resembles"}).status_code == 422
    assert client.post(f"/v1/facts/{a['fact_id']}/links", headers=AUTH, json={"to_fact": 999999, "kind": "supports"}).status_code == 404
    assert client.post(f"/v1/facts/{a['fact_id']}/links", headers=OTHER, json={"to_fact": b["fact_id"], "kind": "supports"}).status_code == 404
    bad_quote = client.post(f"/v1/facts/{a['fact_id']}/links", headers=AUTH,
                            json={"to_fact": b["fact_id"], "kind": "supports", "source_episode_id": episode["episode_id"], "quote": "not there"})
    assert bad_quote.status_code == 422
    assert client.post("/v1/facts", json={"subject": "x", "predicate": "y", "object": "z", "extends": 999999}, headers=AUTH).status_code == 404
    assert client.post("/v1/facts", json={"subject": "a", "predicate": "is", "object": "9", "extends": a["fact_id"]}, headers=AUTH).status_code == 422


def test_the_graph_draws_the_relations_between_claims(client):
    a = fact(client, "a", "is", "1")
    b = fact(client, "b", "is", "2", derived_from=[a["fact_id"]])
    client.post(f"/v1/facts/{a['fact_id']}/links", headers=AUTH, json={"to_fact": b["fact_id"], "kind": "contradicts"})
    graph = client.get("/v1/graph", headers=AUTH).json()
    edges = {(e["source"], e["target"], e["kind"]) for e in graph["edges"]}
    assert (f"claim:{b['fact_id']}", f"claim:{a['fact_id']}", "derived_from") in edges
    assert (f"claim:{a['fact_id']}", f"claim:{b['fact_id']}", "contradicts") in edges
    assert client.get("/v1/graph", headers=OTHER).json()["edges"] == []
