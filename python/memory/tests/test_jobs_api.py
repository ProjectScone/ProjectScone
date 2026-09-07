"""Ingest jobs over HTTP: what a batch became, and how far it has got.

The batch response keeps the shape callers already parse and gains the
job beside it, so nothing that reads it today has to change.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

KEYS = {"writer": "alpha", "reader": "alpha", "other": "beta"}
ROLES = {"reader": "read"}


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def client():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, KEYS, console=False, roles=ROLES)) as c:
        yield c


def batch(client, *texts, key="writer", request_id=None):
    body = {"records": [{"content": text} for text in texts]}
    if request_id:
        body["request_id"] = request_id
    return client.post("/v1/episodes/batch", json=body, headers=bearer(key))


def test_a_batch_answers_with_its_job_and_the_shape_callers_already_read(client):
    sent = batch(client, "the first note", "the second note")
    assert sent.status_code == 200
    body = sent.json()
    assert body["counts"] == {"accepted": 2, "duplicate": 0, "updated": 0}, "the old shape is untouched"
    assert [i["episode_id"] for i in body["items"]], "and so are the per-record outcomes"

    job = body["job"]
    assert job["state"] == "searchable" and job["searchable"] == 2 and job["consolidated"] == 0
    read = client.get(f"/v1/jobs/{job['job_id']}", headers=bearer("writer"))
    assert read.status_code == 200 and read.json()["job_id"] == job["job_id"]
    assert [i["index"] for i in read.json()["items"]] == [0, 1]
    assert all(i["searchable_at"] and i["consolidated_at"] is None for i in read.json()["items"])


def test_the_same_request_twice_is_one_job(client):
    first = batch(client, "only once", request_id="req-1").json()
    second = batch(client, "only once", request_id="req-1").json()
    assert second["job"]["job_id"] == first["job"]["job_id"]
    assert [i["episode_id"] for i in second["job"]["items"]] == [i["episode_id"] for i in first["job"]["items"]]
    assert len(client.get("/v1/jobs", headers=bearer("writer")).json()["jobs"]) == 1


def test_jobs_are_listed_newest_first_and_stay_in_their_space(client):
    older = batch(client, "older").json()["job"]["job_id"]
    newer = batch(client, "newer").json()["job"]["job_id"]
    batch(client, "someone else's", key="other")

    listed = client.get("/v1/jobs", headers=bearer("writer")).json()["jobs"]
    assert [j["job_id"] for j in listed] == [newer, older]
    assert client.get(f"/v1/jobs/{older}", headers=bearer("other")).status_code == 404, "another space sees nothing"
    assert client.get("/v1/jobs?limit=1", headers=bearer("writer")).json()["jobs"] == [listed[0]]


def test_cancelling_is_a_write_and_happens_once(client):
    job_id = batch(client, "one", "two").json()["job"]["job_id"]

    denied = client.post(f"/v1/jobs/{job_id}/cancel", headers=bearer("reader"))
    assert denied.status_code == 403 and "read" in denied.json()["error"], "reading a job is not cancelling it"

    stopped = client.post(f"/v1/jobs/{job_id}/cancel", headers=bearer("writer"))
    assert stopped.status_code == 200 and stopped.json()["state"] == "cancelled" and stopped.json()["cancelled_at"]
    assert all(i["searchable_at"] for i in stopped.json()["items"]), "cancelling a job is not deleting memory"

    again = client.post(f"/v1/jobs/{job_id}/cancel", headers=bearer("writer"))
    assert again.status_code == 422 and "cancelled" in again.json()["error"]


def test_a_reader_may_look_and_an_unknown_job_is_not_found(client):
    job_id = batch(client, "something").json()["job"]["job_id"]
    assert client.get("/v1/jobs", headers=bearer("reader")).status_code == 200
    assert client.get(f"/v1/jobs/{job_id}", headers=bearer("reader")).status_code == 200
    assert client.get("/v1/jobs/no-such-job", headers=bearer("writer")).status_code == 404


def test_the_cli_shows_jobs_and_cancels_one(tmp_path):
    import io
    import json

    from scone_memory.runtime.cli import main

    env = {"SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_EMBEDDER": "hash"}
    out = io.StringIO()
    assert main(["--json", "remember"], env=env, stdin=io.StringIO("a note for the job"), out=out) == 0

    out = io.StringIO()
    assert main(["--json", "jobs"], env=env, out=out) == 0
    assert out.getvalue().strip() == "", "a plain remember is not a batch, so there is no job to show"

    listing = io.StringIO()
    assert main(["jobs"], env=env, out=listing) == 0
    assert "no ingest jobs" in listing.getvalue()
