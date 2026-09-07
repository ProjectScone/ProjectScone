"""The repair path end to end, as a person would actually run it.

Every piece of this is tested on its own: the audit finds claims their
source cannot support, the batch route settles a frozen list of ids, and
exclusion suppresses a fact from recall. Nobody had ever run them in
sequence, which is the only form in which they will be used against a
real ledger.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

SPACE = "alpha"
#: The live episode that produced "claude code / is_installed / nothing".
DENIAL = (
    "An authenticated memory API only proves the server accepts writes from a caller "
    "holding a valid token; it says nothing about whether the Claude Code or Codex "
    "hooks are actually installed."
)
PLAIN = "Ana moved to Lisbon in March and joined Farfetch."


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine, {"key-a": SPACE})) as c:
        yield c


def auth() -> dict:
    return {"authorization": "Bearer key-a"}


def seed(client) -> tuple[int, int]:
    """One claim its source denies, one the source supports."""
    denial = client.post("/v1/episodes", json={"content": DENIAL}, headers=auth()).json()["episode_id"]
    plain = client.post("/v1/episodes", json={"content": PLAIN}, headers=auth()).json()["episode_id"]
    wrong = client.post("/v1/facts", json={
        "subject": "claude code", "predicate": "is_installed", "object": "nothing",
        "source_episode_id": denial, "origin": "extracted", "confidence": 0.5,
    }, headers=auth()).json()["fact_id"]
    right = client.post("/v1/facts", json={
        "subject": "ana", "predicate": "moved_to", "object": "Lisbon",
        "source_episode_id": plain, "origin": "extracted",
        "quote": "Ana moved to Lisbon in March",
    }, headers=auth()).json()["fact_id"]
    return wrong, right


def test_a_person_can_audit_then_exclude_exactly_what_the_audit_flagged(client):
    wrong, right = seed(client)

    # Before anything: the wrong claim really is reachable through recall,
    # so the assertion after the repair is measuring its removal and not a
    # query that never matched it.
    before = client.get("/v1/recall", params={"q": "are the hooks installed"}, headers=auth()).json()
    assert wrong in [f["fact_id"] for f in before["facts"]]

    found = client.get("/v1/facts/audit", params={"flagged": "true"}, headers=auth()).json()
    assert [f["fact_id"] for f in found["findings"]] == [wrong]
    assert found["counts"]["grounded"] == 1

    settled = client.post("/v1/facts/decide", json={
        "decision": "exclude",
        "ids": [f["fact_id"] for f in found["findings"]],
        "reason": "audit: the object is the denial in its own source",
        "expect_revision": found["revision"],
    }, headers=auth())

    assert settled.status_code == 200
    assert [(r["fact_id"], r["outcome"]) for r in settled.json()["results"]] == [(wrong, "excluded")]
    assert settled.json()["applied"] == 1

    # The claim is gone from the ledger view and from recall, and the
    # grounded one is untouched by the repair.
    shown = [f["fact_id"] for f in client.get("/v1/facts", headers=auth()).json()["facts"]]
    assert shown == [right]
    recalled = client.get("/v1/recall", params={"q": "are the hooks installed"}, headers=auth()).json()
    assert [f["fact_id"] for f in recalled["facts"]] == []

    # And the record survives: excluded, with the reason, not deleted.
    kept = client.get("/v1/facts", params={"excluded": "true", "all": "true"}, headers=auth()).json()["facts"]
    excluded = next(f for f in kept if f["fact_id"] == wrong)
    assert excluded["excluded_reason"] == "audit: the object is the denial in its own source"
    assert excluded["status"] == "active"


def test_a_batch_built_from_a_stale_audit_is_refused_whole(client):
    """The audit read and the decision are separate requests, so the space
    can move between them. That is the case expect_revision exists for,
    and the repair is the place it matters: excluding a list that no
    longer matches what a person reviewed is exactly the harm."""
    wrong, _ = seed(client)
    found = client.get("/v1/facts/audit", params={"flagged": "true"}, headers=auth()).json()

    # Someone else distils another claim while the person is reading.
    client.post("/v1/facts", json={"subject": "ana", "predicate": "works_at", "object": "Farfetch"},
                headers=auth())

    refused = client.post("/v1/facts/decide", json={
        "decision": "exclude", "ids": [wrong], "reason": "audit",
        "expect_revision": found["revision"],
    }, headers=auth())

    assert refused.status_code == 409
    assert refused.json()["revision"] > found["revision"]
    still_there = [f["fact_id"] for f in client.get("/v1/facts", headers=auth()).json()["facts"]]
    assert wrong in still_there, "a refused batch must change nothing"


def test_the_repair_is_reversible(client):
    """Exclusion is the mechanism precisely because a person can be wrong
    about it. Including the fact again puts it back in recall."""
    wrong, _ = seed(client)
    found = client.get("/v1/facts/audit", params={"flagged": "true"}, headers=auth()).json()
    client.post("/v1/facts/decide", json={"decision": "exclude", "ids": [wrong], "reason": "audit"},
                headers=auth())

    assert client.post(f"/v1/facts/{wrong}/include", headers=auth()).status_code == 200

    shown = [f["fact_id"] for f in client.get("/v1/facts", headers=auth()).json()["facts"]]
    assert wrong in shown
