"""POST /v1/sync-runs takes forget_after: the run keeps the resolved instant and
the request as asked, a refused schedule is a 422 as on POST /v1/episodes, and a
run whose instant has passed is not resumed (409)."""
import asyncio

import httpx
import pytest

from scone_memory.api.app import create_app
from tests.ingestion.test_directory_service import service, settled
from tests.ingestion.test_directory_sync_forget_after import env, runner  # noqa: F401 - the clocked fixture


def auth(key="writer"):
    return {"authorization": "Bearer " + key}


@pytest.fixture
async def setup(env):
    memory, root, _ = env
    (root / "note.txt").write_text("A document to synchronize")
    sync = runner(env)
    entered, release = asyncio.Event(), asyncio.Event()
    parse = sync.parser.parse

    async def gated(*args):
        entered.set()
        await release.wait()
        return await parse(*args)

    sync.parser.parse = gated
    host = service(env, sync=sync)
    app = create_app(memory, {"writer": "alpha"}, roles={"writer": "write"}, directory_sync_service=host)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local.test") as client:
        try:
            yield client, host, entered, release, memory
        finally:
            release.set()
            await host.aclose()


async def test_a_run_takes_a_schedule_and_refuses_a_past_one(setup):
    client, host, entered, release, _ = setup
    body = {"run_id": "scan", "collection_id": "notes"}
    refused = await client.post("/v1/sync-runs", json={**body, "forget_after": "2020-01-01"}, headers=auth())
    assert refused.status_code == 422 and "forget_after" in refused.text
    assert (await client.get("/v1/sync-runs/scan", headers=auth())).status_code == 404
    admitted = await client.post("/v1/sync-runs", json={**body, "forget_after": "2d"}, headers=auth())
    assert admitted.status_code == 202, admitted.text
    spec = admitted.json()["record"]["spec"]
    assert (spec["forget_after"], spec["forget_after_asked"]) == ("2026-09-17T12:00:00.000Z", "2d")
    release.set()
    await settled(host)
    result = (await client.get("/v1/sync-runs/scan/result", headers=auth())).json()
    assert result["items"][0]["source"]["forget_after"] == "2026-09-17T12:00:00.000Z"


async def test_a_run_whose_instant_has_passed_is_not_resumed(setup):
    client, host, entered, release, memory = setup
    admitted = await client.post("/v1/sync-runs", json={"run_id": "scan", "collection_id": "notes", "forget_after": "1h"},
                                 headers=auth())
    await asyncio.wait_for(entered.wait(), 3)
    revision = admitted.json()["record"]["revision"]
    assert (await client.post("/v1/sync-runs/scan/cancel", json={"expected_revision": revision}, headers=auth())).status_code == 202
    status = await settled(host)
    memory.test_clock.now = "2026-09-15T13:30:00.000Z"
    resumed = await client.post("/v1/sync-runs/scan/resume", json={"expected_revision": status.record.revision},
                                headers=auth())
    assert resumed.status_code == 409 and resumed.json()["code"] == "sync_schedule_passed"
