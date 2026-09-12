"""The tree over HTTP: list, read, search, and write a note.

The same policy as everywhere else, and the same refusals, answered as
status codes: a path that cannot mean anything here is a 400, a write to
a read-only tree is a 403, and a write onto a note that moved is a 409,
which is what a conflict is for.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.api import create_app
from scone_memory.filesystem import FilesystemPolicy
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio


def auth(key: str = "key-a") -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
async def served():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"),
                                events=InMemoryEventLog()).open()
    note = await engine.remember("alpha", "The Lisbon office opened in March and holds forty desks.")
    await engine.assert_fact("alpha", "lisbon office", "opened_on", "March 2024",
                             valid_from="2024-01-01T00:00:00Z", source_episode_id=note.episode_id,
                             quote="The Lisbon office opened in March")
    app = create_app(engine, {"key-a": "alpha", "key-r": "alpha"}, roles={"key-r": "read"},
                     filesystem=FilesystemPolicy(writable=True))
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def read_only():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.remember("alpha", "a note")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        yield client


def test_the_tree_lists_and_reads_over_http(served):
    listing = served.get("/v1/fs", params={"path": "/"}, headers=auth()).json()
    assert [entry["path"] for entry in listing["entries"]] == ["/entities", "/episodes", "/facts", "/notes"]
    page = served.get("/v1/fs/read", params={"path": "/episodes/1.md"}, headers=auth()).json()
    assert "forty desks" in page["text"] and page["of"] == "episode"


def test_a_search_answers_in_paths_over_http(served):
    found = served.get("/v1/fs/search", params={"query": "desks"}, headers=auth()).json()
    assert found["hits"] and found["hits"][0]["path"].endswith(".md")


def test_a_note_is_written_and_read_back(served):
    written = served.post("/v1/fs/notes", json={"path": "/notes/plan.md", "text": "Ship on Friday."},
                          headers=auth())
    assert written.status_code == 200, written.text
    body = written.json()
    assert body["path"] == "/notes/plan.md" and body["version"] == body["episode_id"]
    page = served.get("/v1/fs/read", params={"path": "/notes/plan.md"}, headers=auth()).json()
    assert page["text"] == "Ship on Friday." and page["version"] == body["version"]


def test_a_write_onto_a_note_that_moved_is_a_conflict(served):
    served.post("/v1/fs/notes", json={"path": "/notes/plan.md", "text": "one"}, headers=auth())
    stale = served.get("/v1/fs/read", params={"path": "/notes/plan.md"}, headers=auth()).json()["version"]
    served.post("/v1/fs/notes", json={"path": "/notes/plan.md", "text": "two", "if_version": stale},
                headers=auth())
    again = served.post("/v1/fs/notes", json={"path": "/notes/plan.md", "text": "three",
                                              "if_version": stale}, headers=auth())
    assert again.status_code == 409, again.text
    assert "moved" in again.json()["error"] and again.json()["revision"] >= 1


def test_a_tree_nobody_made_writable_refuses_the_write(read_only):
    said = read_only.post("/v1/fs/notes", json={"path": "/notes/x.md", "text": "no"}, headers=auth())
    assert said.status_code == 403, said.text
    assert "read only" in said.json()["error"]


def test_a_read_only_key_may_not_write_a_note(served):
    said = served.post("/v1/fs/notes", json={"path": "/notes/x.md", "text": "no"}, headers=auth("key-r"))
    assert said.status_code == 403 and "role" in said.json()["error"]


def test_a_path_that_tries_to_leave_the_space_is_refused(served):
    said = served.get("/v1/fs/read", params={"path": "/episodes/../../etc/passwd"}, headers=auth())
    assert said.status_code == 422 and "leave" in said.json()["error"]


def test_the_tree_is_advertised_with_what_it_allows(served, read_only):
    said = served.get("/v1/capabilities", headers=auth()).json()["features"]
    assert said["filesystem.read"] is True and said["filesystem.write"] is True
    shut = read_only.get("/v1/capabilities", headers=auth()).json()["features"]
    assert shut["filesystem.read"] is True and shut["filesystem.write"] is False
