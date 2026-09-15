"""Entity merges over HTTP: a reviewer's decision, never a writer's.

Merging two entities changes what every graph view and analysis shows, so
it belongs to the review role beside approving and declining claims, and
the actor and reason go on the event. The knowledge map honours a merge the
moment it is recorded and lets the names part when it is closed.
"""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

KEYS = {"reader": "alpha", "writer": "alpha", "reviewer": "alpha", "boss": "alpha"}
ROLES = {"reader": "read", "writer": "write", "reviewer": "review", "boss": "full"}
DAY = "2024-01-01T00:00:00Z"


def bearer(key: str, actor: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", **({"X-Scone-Actor": actor} if actor else {})}


def engine_with_chen() -> MemoryEngine:
    async def build() -> MemoryEngine:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                    events=InMemoryEventLog()).open()
        await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
        await engine.assert_fact("alpha", "dr. alice chen", "leads", "Robotics Lab", valid_from=DAY)
        return engine
    return asyncio.run(build())


def entity_keys(client: TestClient) -> set[str]:
    view = client.get("/v1/graph/knowledge", headers=bearer("reader")).json()
    return {entity["key"] for entity in view["entities"]}


def test_a_reviewer_merges_two_entities_and_the_map_shows_one():
    engine = engine_with_chen()
    with TestClient(create_app(engine, KEYS, roles=ROLES)) as client:
        body = {"alias": "Dr. Alice Chen", "into": "Alice Chen", "reason": "same badge"}
        for key in ("reader", "writer"):
            assert client.post("/v1/entities/merges", json=body, headers=bearer(key)).status_code == 403
        recorded = client.post("/v1/entities/merges", json=body, headers=bearer("reviewer", "console"))
        assert recorded.status_code == 200, recorded.text
        decision = recorded.json()
        assert (decision["alias"], decision["into"], decision["status"]) == ("dr. alice chen", "Alice Chen", "active")
        assert "dr. alice chen" not in entity_keys(client)
        listed = client.get("/v1/entities/merges", headers=bearer("reader")).json()
        [merge] = listed["merges"]
        assert (merge["fact_id"], merge["alias_key"], merge["into_key"], merge["outcome"]) == (
            decision["fact_id"], "dr. alice chen", "alice chen", "applied")
        [event] = asyncio.run(engine.events.query("alpha", kind="entity_merge", limit=5))
        assert event.payload["reason"] == "same badge" and event.payload["actor"].endswith(" console")


def test_closing_a_merge_parts_the_names_and_a_missing_one_is_not_found():
    engine = engine_with_chen()
    with TestClient(create_app(engine, KEYS, roles=ROLES)) as client:
        body = {"alias": "dr. alice chen", "into": "alice chen", "reason": "same badge"}
        assert client.post("/v1/entities/merges", json=body, headers=bearer("reviewer")).status_code == 200
        close = {"alias": "Dr. Alice Chen", "reason": "two people"}
        assert client.post("/v1/entities/merges/close", json=close, headers=bearer("writer")).status_code == 403
        closed = client.post("/v1/entities/merges/close", json=close, headers=bearer("reviewer"))
        assert closed.status_code == 200 and closed.json()["status"] == "closed"
        assert "dr. alice chen" in entity_keys(client)
        assert client.get("/v1/entities/merges", headers=bearer("reader")).json()["merges"] == []
        again = client.post("/v1/entities/merges/close", json=close, headers=bearer("reviewer"))
        assert again.status_code == 404


def test_the_reserved_predicate_and_a_merge_of_one_name_are_refused():
    engine = engine_with_chen()
    with TestClient(create_app(engine, KEYS, roles=ROLES)) as client:
        stated = client.post("/v1/facts", json={"subject": "dr. alice chen", "predicate": "scone:same entity",
                                                "object": "alice chen"}, headers=bearer("writer"))
        assert stated.status_code == 422 and "merge" in stated.json()["error"]
        same = client.post("/v1/entities/merges", json={"alias": "Alice  Chen", "into": "alice chen", "reason": "x"},
                           headers=bearer("reviewer"))
        assert same.status_code == 422 and "already one name" in same.json()["error"]
        extra = client.post("/v1/entities/merges", json={"alias": "a", "into": "b", "reason": "x", "force": True},
                            headers=bearer("reviewer"))
        assert extra.status_code == 422 and "force" in extra.json()["error"]
