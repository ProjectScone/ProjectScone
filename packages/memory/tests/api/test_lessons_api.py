"""Lessons over HTTP and on the command line: asked for, shown beside, never reordering."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.testing import Clock


@pytest.fixture
async def engine():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2026-05-01T00:00:00.000Z"), events=InMemoryEventLog()).open()
    await engine.remember("alpha", "The crane survey was booked for the third of May.")
    await engine.remember("alpha", "The canteen serves soup on Tuesdays.")
    yield engine
    await engine.close()


def auth() -> dict:
    return {"authorization": "Bearer key-a"}


async def test_recall_carries_lessons_when_asked_and_the_space_lists_them(engine):
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        first = client.get("/v1/recall", params={"q": "crane survey booked"}, headers=auth()).json()
        chunk = first["items"][0]["chunk_id"]
        client.post("/v1/feedback", json={"recall_event_id": first["event_id"], "chunk_id": chunk, "useful": False},
                    headers=auth())
        plain = client.get("/v1/recall", params={"q": "crane survey booked"}, headers=auth()).json()
        shown = client.get("/v1/recall", params={"q": "crane survey booked", "lessons": "true"}, headers=auth()).json()
        listed = client.get("/v1/lessons", headers=auth())
        refused = client.get("/v1/lessons", params={"half_life_days": 0}, headers=auth())
        capabilities = client.get("/v1/capabilities", headers=auth()).json()
        merged = client.get("/v1/recall", params={"q": "crane survey booked", "lessons": "true", "merge": "true"},
                            headers=auth())
    assert "lessons" not in plain["items"][0]
    assert shown["items"][0]["lessons"]["state"] == "dead_end"
    assert [item["chunk_id"] for item in shown["items"]] == [item["chunk_id"] for item in plain["items"]]
    assert listed.status_code == 200 and listed.json()["lessons"][str(chunk)]["not_useful"] == 1
    assert listed.json()["events_cut"] is False and refused.status_code in (400, 422)
    flat = json.dumps(capabilities)
    assert '"recall.lessons": true' in flat
    assert merged.status_code in (400, 422) and "merge" in merged.text


async def test_the_command_line_shows_lessons(engine):
    from scone_memory.runtime.cli import build_parser, run

    first = await engine.recall("alpha", "crane survey booked")
    await engine.feedback("alpha", first.event_id, first.items[0].chunk_id, True)
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "alpha", "lessons", "--json"]), engine, io.StringIO(""), out)
    assert code == 0 and json.loads(out.getvalue())["lessons"][str(first.items[0].chunk_id)]["state"] == "tentative"
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "alpha", "recall", "crane survey booked", "--lessons"]),
                     engine, io.StringIO(""), out)
    assert code == 0 and "lesson tentative" in out.getvalue()
