"""A profile's recent activity names its evidence: every ``dynamic``
excerpt comes with the episode it was cut from, in the shape both engines
share (tests/fixtures/profile-recent.json)."""

from __future__ import annotations

from ..paths import REPO_ROOT

import asyncio
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.runtime.cli import main

FIXTURE = json.loads((REPO_ROOT / "tests/fixtures/profile-recent.json").read_text())
AUTH = {"Authorization": "Bearer k"}


def test_recent_activity_carries_its_episode_and_matches_dynamic():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"k": "default"})) as c:
        older = c.post("/v1/episodes", json={"content": "older note", "created_at": "2024-01-01T00:00:00Z"}, headers=AUTH).json()["episode_id"]
        newer = c.post("/v1/episodes", json={"content": "n" * 300, "created_at": "2024-02-01T00:00:00Z"}, headers=AUTH).json()["episode_id"]
        body = c.get("/v1/profile", headers=AUTH).json()
        assert [sorted(item) for item in body["recent"]] == [sorted(FIXTURE["recent_item"])] * 2, "exactly the shared keys"
        assert [item["episode_id"] for item in body["recent"]] == [newer, older], "newest first"
        assert [item["excerpt"] for item in body["recent"]] == body["dynamic"], "recent is dynamic with its evidence"
        assert len(body["recent"][0]["excerpt"]) == FIXTURE["excerpt_max_chars"]
        for item in body["recent"]:
            episode = c.get(f"/v1/episodes/{item['episode_id']}", headers=AUTH).json()
            assert episode["created_at"] == item["created_at"]
            assert episode["content"].startswith(item["excerpt"])


def test_the_cli_profile_json_carries_recent_with_its_evidence(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), "SCONE_EMBEDDER": "hash"}
    out = io.StringIO()
    assert main(["--json", "remember"], env=env, stdin=io.StringIO("older note"), out=out) == 0
    episode_id = json.loads(out.getvalue())["episode_id"]
    out = io.StringIO()
    assert main(["--json", "profile"], env=env, out=out) == 0
    body = json.loads(out.getvalue())
    assert [sorted(item) for item in body["recent"]] == [sorted(FIXTURE["recent_item"])]
    assert body["recent"][0]["episode_id"] == episode_id and body["recent"][0]["excerpt"] == body["dynamic"][0] == "older note"
