"""Deciding a reviewed batch in one request.

A review page freezes the list it renders and then acts on it. Sending N
single decisions from a browser leaves the outcome dependent on the order
the page happened to draw, cannot tell that the space moved underneath,
and leaves half a batch applied when a tab closes. One request settles
all three.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

SPACE = "alpha"


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


@pytest.fixture
def client(engine):
    app = create_app(engine, {"key-a": SPACE})
    with TestClient(app) as c:
        yield c


def auth() -> dict:
    return {"authorization": "Bearer key-a"}


def propose(client, object: str, valid_from: str) -> int:
    body = {"subject": "mark", "predicate": "lives_in", "object": object,
            "valid_from": valid_from, "proposed": True}
    return client.post("/v1/facts", json=body, headers=auth()).json()["fact_id"]


def test_a_batch_is_applied_oldest_first_whatever_order_it_arrives_in(client):
    """Approving inside one subject and predicate is order-sensitive where
    a proposal restates one already held: taken oldest first, the later
    statement of the same object is a duplicate that never held. Taken
    newest first, the same two rows become a supersession, and the earlier
    one is recorded as having held for a year it never held. The page's
    draw order must not decide which of those the ledger says."""
    first = propose(client, "Lisbon", "2021-01-01")
    restated = propose(client, "Lisbon", "2022-01-01")

    body = client.post("/v1/facts/decide",
                       json={"decision": "approve", "ids": [restated, first]},
                       headers=auth()).json()

    assert [(r["fact_id"], r["outcome"]) for r in body["results"]] == [
        (restated, "duplicate_of"), (first, "approved"),
    ]
    assert body["results"][0]["held_fact_id"] == first
    by_id = {f["fact_id"]: f for f in
             client.get("/v1/facts", params={"all": "true"}, headers=auth()).json()["facts"]}
    assert by_id[first]["status"] == "active" and by_id[first]["valid_until"] is None


def test_a_batch_refuses_whole_when_the_space_moved_under_the_page(client):
    """The frozen set is only honest if the server checks it. One check,
    before anything is applied, so a refusal leaves nothing half done."""
    austin = propose(client, "Austin", "2022-01-01")
    stale = client.get("/v1/facts", params={"status": "proposed"}, headers=auth()).json()["revision"]
    lisbon = propose(client, "Lisbon", "2023-01-01")  # the space moves

    refused = client.post("/v1/facts/decide",
                          json={"decision": "approve", "ids": [austin], "expect_revision": stale},
                          headers=auth())

    assert refused.status_code == 409
    assert refused.json()["revision"] > stale
    pending = client.get("/v1/facts", params={"status": "proposed"}, headers=auth()).json()["facts"]
    assert sorted(f["fact_id"] for f in pending) == sorted([austin, lisbon])


def test_one_id_that_cannot_be_decided_does_not_stop_the_others(client):
    """Per-item outcomes, keyed by the id sent, are the point: a page that
    stops at the first refusal leaves the person with no idea what landed."""
    austin = propose(client, "Austin", "2022-01-01")
    client.post(f"/v1/facts/{austin}/approve", headers=auth())
    lisbon = propose(client, "Lisbon", "2023-01-01")

    body = client.post("/v1/facts/decide",
                       json={"decision": "approve", "ids": [999, lisbon, austin]},
                       headers=auth()).json()

    assert [(r["fact_id"], r["outcome"]) for r in body["results"]] == [
        (999, "not_found"), (lisbon, "approved"), (austin, "refused"),
    ]
    assert "not proposed" in body["results"][2]["error"]
    assert body["applied"] == 1


def test_a_proposal_folded_into_a_fact_already_held_says_so(client):
    """approve on a duplicate marks the proposal declined and returns a
    DIFFERENT fact. Reporting that as approved would tell the page a row
    landed when what landed was already there."""
    lisbon = propose(client, "Lisbon", "2023-01-01")
    client.post(f"/v1/facts/{lisbon}/approve", headers=auth())
    again = propose(client, "Lisbon", "2023-06-01")

    body = client.post("/v1/facts/decide", json={"decision": "approve", "ids": [again]},
                       headers=auth()).json()

    assert body["results"] == [{
        "fact_id": again, "outcome": "duplicate_of", "held_fact_id": lisbon, "error": None,
    }]


def test_a_batch_can_exclude_ledger_facts_with_one_reason(client):
    """The audit's flagged list is a batch of exclusions, not approvals,
    so the route takes the decision as an argument rather than assuming."""
    austin = propose(client, "Austin", "2022-01-01")
    client.post(f"/v1/facts/{austin}/approve", headers=auth())

    body = client.post("/v1/facts/decide",
                       json={"decision": "exclude", "ids": [austin],
                             "reason": "audit: object appears only in a negated clause"},
                       headers=auth()).json()

    assert [(r["fact_id"], r["outcome"]) for r in body["results"]] == [(austin, "excluded")]
    shown = client.get("/v1/facts", headers=auth()).json()["facts"]
    assert [f["fact_id"] for f in shown] == []


def test_a_batch_is_bounded_and_names_its_decision(client):
    """A review queue, not a migration: the batch has a bound, and an
    unknown decision is refused rather than guessed at."""
    assert client.post("/v1/facts/decide", json={"decision": "approve", "ids": []},
                       headers=auth()).status_code == 422
    assert client.post("/v1/facts/decide", json={"decision": "approve", "ids": list(range(1, 502))},
                       headers=auth()).status_code == 422
    assert client.post("/v1/facts/decide", json={"decision": "delete", "ids": [1]},
                       headers=auth()).status_code == 422
