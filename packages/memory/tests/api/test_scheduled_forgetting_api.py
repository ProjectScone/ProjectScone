"""Scheduled forgetting over HTTP: the schedule on a write and a batch, refused
when past or unreadable, withheld from recall before any sweep, and swept by a
bounded route whose report says what went and why."""
import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.testing import Clock

AUTH = {"authorization": "Bearer k"}


@pytest.fixture
async def served():
    clock = Clock("2026-09-15T12:00:00.000Z")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock).open()
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {"k": "alpha", "r": "alpha"},
                                                                  roles={"r": "read"})),
                           base_url="http://fixture") as client:
        yield engine, client, clock
    await engine.close()


async def test_an_episode_takes_a_schedule_and_its_receipt_says_when(served):
    engine, client, _ = served
    response = await client.post("/v1/episodes", json={"content": "the gate code is 7788", "forget_after": "2d"}, headers=AUTH)
    assert response.status_code == 200 and response.json()["forget_after"] == "2026-09-17T12:00:00.000Z"
    stored = await engine.documents.get_episode("alpha", response.json()["episode_id"])
    assert stored.metadata["forget_after"] == "2026-09-17T12:00:00.000Z"


@pytest.mark.parametrize("bad", ["2026-09-14T00:00:00Z", "soon"], ids=["past", "malformed"])
async def test_a_past_or_unreadable_schedule_is_refused_and_nothing_stored(served, bad):
    engine, client, _ = served
    response = await client.post("/v1/episodes", json={"content": "refused", "forget_after": bad}, headers=AUTH)
    assert response.status_code == 422 and "forget_after" in response.text
    assert (await engine.status("alpha")).episodes == 0


async def test_a_batch_carries_each_record_schedule(served):
    engine, client, _ = served
    response = await client.post("/v1/episodes/batch", json={"records": [
        {"content": "one for an hour", "forget_after": "1h"}, {"content": "one kept"}]}, headers=AUTH)
    items = response.json()["items"]
    assert response.status_code == 200 and [i["forget_after"] for i in items] == ["2026-09-15T13:00:00.000Z", None]
    refused = await client.post("/v1/episodes/batch", json={"records": [
        {"content": "fine", "forget_after": "1h"}, {"content": "late", "forget_after": "2020-01-01"}]}, headers=AUTH)
    assert refused.status_code == 422 and "forget_after" in refused.text and (await engine.status("alpha")).episodes == 2


async def test_recall_withholds_what_is_due_and_says_so_before_the_sweep(served):
    _, client, clock = served
    due = (await client.post("/v1/episodes", json={"content": "the alarm code is 1234", "forget_after": "1h"}, headers=AUTH)).json()
    kept = (await client.post("/v1/episodes", json={"content": "the alarm panel is by the door"}, headers=AUTH)).json()
    clock.now = "2026-09-15T13:00:00.000Z"
    body = (await client.get("/v1/recall", params={"q": "alarm code"}, headers=AUTH)).json()
    assert [item["episode_id"] for item in body["items"]] == [kept["episode_id"]]
    assert body["past_forget_after"] == {"withheld": 1, "episode_ids": [due["episode_id"]], "at": "2026-09-15T13:00:00.000Z"}
    gone = await client.get(f"/v1/episodes/{due['episode_id']}", headers=AUTH)
    assert gone.status_code == 410


async def test_the_sweep_route_forgets_what_is_due_with_a_bounded_report(served):
    engine, client, clock = served
    for n in range(3):
        await client.post("/v1/episodes", json={"content": f"due note {n}", "forget_after": f"{n + 1}h"}, headers=AUTH)
    await client.post("/v1/episodes", json={"content": "a note kept"}, headers=AUTH)
    clock.now = "2026-09-16T00:00:00.000Z"
    earlier = (await client.post("/v1/episodes/forget-due", json={"dry_run": True, "now": "2026-09-15T12:30:00Z"}, headers=AUTH)).json()
    assert earlier["due"] == 0 and earlier["now"] == "2026-09-15T12:30:00.000Z"
    assert (await client.post("/v1/episodes/forget-due", json={"dry_run": True, "before": 2}, headers=AUTH)).json()["scanned"] == 1
    preview = (await client.post("/v1/episodes/forget-due", json={"dry_run": True}, headers=AUTH)).json()
    assert preview["due"] == 3 and preview["forgotten"] == [] and (await engine.status("alpha")).episodes == 4
    first = await client.post("/v1/episodes/forget-due", json={"limit": 2, "with_claims": "exclude"}, headers=AUTH)
    body = first.json()
    assert first.status_code == 200 and len(body["forgotten"]) == 2 and body["limited"] is True and body["remaining"] == 1
    assert body["with_claims"] == "exclude" and all(i["outcome"] == "forgotten" and "forget_after" in i["reason"]
                                                    for i in body["items"])
    rest = (await client.post("/v1/episodes/forget-due", json={}, headers=AUTH)).json()
    assert len(rest["forgotten"]) == 1 and rest["limited"] is False and (await engine.status("alpha")).episodes == 1


async def test_the_sweep_route_refuses_what_it_cannot_mean(served):
    _, client, _ = served
    for body in ({"limit": 0}, {"limit": 1001}, {"now": "2027-01-01T00:00:00Z"}, {"with_claims": "all"}, {"apply": True}):
        response = await client.post("/v1/episodes/forget-due", json=body, headers=AUTH)
        assert response.status_code == 422, (body, response.text)


async def test_a_read_only_key_cannot_sweep(served):
    _, client, _ = served
    response = await client.post("/v1/episodes/forget-due", json={}, headers={"authorization": "Bearer r"})
    assert response.status_code == 403


async def test_the_source_listing_leaves_an_overdue_memory_out(served):
    engine, client, clock = served
    kept = await client.post("/v1/episodes", json={"content": "the router lives in the hall"}, headers=AUTH)
    await client.post("/v1/episodes", json={"content": "the wifi password is hunter2", "forget_after": "1h"}, headers=AUTH)
    clock.now = "2026-09-15T13:00:00.000Z"
    listed = await client.get("/v1/sources", headers=AUTH)
    assert listed.status_code == 200 and "hunter2" not in listed.text
    assert [i["episode_id"] for i in listed.json()["items"]] == [kept.json()["episode_id"]]
    assert listed.json()["past_forget_after"] == 1
    clock.now = "2026-09-15T12:30:00.000Z"
    early = await client.get("/v1/sources", headers=AUTH)
    assert len(early.json()["items"]) == 2 and "past_forget_after" not in early.json()
