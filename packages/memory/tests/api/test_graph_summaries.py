"""Community and space summaries over HTTP: the model's sentences, each quoting a record; refused without a model."""

from __future__ import annotations

import asyncio
import json
import re

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


class CitingChat:
    async def complete(self, system: str, user: str) -> str:
        if "Notes:" in user:
            ids = re.findall(r"^\[(n\d+)\]", user, flags=re.M)
            return json.dumps({"summary": [{"sentence": "All of it, together.", "notes": ids}]})
        rows = [{"sentence": f"A note on {m.group(1)}.", "passage": m.group(1), "quote": " ".join(m.group(2).split()[:3])}
                for m in re.finditer(r"^\[((?:fact|quote|report):[^\]]+)\] (.+)$", user, flags=re.M)]
        return json.dumps({"notes": rows})


async def seeded() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    note = await engine.remember("alpha", "Alice Chen joined Acme Robotics.")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY,
                             source_episode_id=note.episode_id, quote="Alice Chen joined Acme Robotics")
    for person in ("bob stone", "carol diaz"):
        await engine.assert_fact("alpha", person, "works_at", "Acme Robotics", valid_from=DAY)
    for person in ("erin fox", "frank li"):
        await engine.assert_fact("alpha", person, "lives_in", "Porto", valid_from=DAY)
    return engine


def test_summaries_are_served_with_a_model_and_refused_without():
    engine = asyncio.run(seeded())
    auth = {"Authorization": "Bearer key-a"}
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        refused = client.get("/v1/graph/summary", headers=auth)
        assert refused.status_code == 422 and "needs a model" in refused.json()["error"]
    with TestClient(create_app(engine, {"key-a": "alpha"}, synthesis_factory=CitingChat)) as client:
        overview = client.get("/v1/graph/overview", headers=auth).json()
        community_id = next(c["id"] for c in overview["communities"] if "acme" in c["label"].lower())
        one = client.get(f"/v1/graph/communities/{community_id}/summary", headers=auth)
        assert one.status_code == 200, one.text
        record = one.json()
        assert record["community_id"] == community_id and record["verified_accuracy"] is False
        assert record["projection_digest"] and record["revision"] >= 1 and record["facts_cited"]
        cited = {c["passage"] for s in record["synthesis"]["sentences"] for c in s["citations"]}
        assert any(p.startswith("fact:") for p in cited) and any(p.startswith("quote:") for p in cited)
        whole = client.get("/v1/graph/summary", headers=auth, params={"reports": 5})
        assert whole.status_code == 200, whole.text
        assert whole.json()["communities_reported"] == 2 and whole.json()["text"]
        assert client.get("/v1/graph/communities/nope/summary", headers=auth).status_code == 422
        assert client.get("/v1/graph/summary", headers=auth, params={"reports": 1}).status_code == 422
