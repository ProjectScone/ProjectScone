"""Cancelling one turn without ending the conversation.

Stopping a session already cancels what it was doing. A person who
regrets one message needs less than that: end this turn, keep the
conversation. Cancel is explicit, idempotent and terminal, and a retry of
a cancelled turn's request id never restarts it, because the provider may
already have been asked.
"""

import asyncio

import pytest

from scone_memory.realtime.session_journal import SessionJournal
from test_conversations_api import client_for, configured, create, engine  # noqa: F401


@pytest.mark.parametrize("outcome", ["return", "error", "unknown-state"])
async def test_a_runtime_that_returns_after_cancel_cannot_overwrite_the_cancelled_receipt(engine, tmp_path, outcome):
    from scone_memory.api.conversations import create_conversation_app
    started = asyncio.Event()
    class LateRuntime:
        closed = False
        async def reply(self, text):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if outcome == "error":
                    raise RuntimeError("cleanup failed")
                return {"text": "Late public answer"}
        async def close(self):
            self.closed = True
    if outcome == "unknown-state":
        del LateRuntime.closed
    app = create_conversation_app(engine, {"alpha-key": "alpha"}, tmp_path / "late.db", lambda space, sid: LateRuntime())
    async with client_for(app) as client:
        created = await create(client)
        url = "/v1/conversations/" + created["session_id"]
        body = {"request_id": "late", "text": "Question", "expected_revision": created["revision"]}
        await client.post(url + "/turns", json=body)
        await asyncio.wait_for(started.wait(), 1)
        receipt = (await client.post(url + "/turns/late/cancel")).json()
        assert receipt["status"] == "cancelled" and receipt["result"] is None
        assert (await client.get(url + "/turns/late")).json()["status"] == "cancelled"
        assert (await client.post(url + "/turns", json=body)).json()["status"] == "cancelled"
        assert (await client.get(url)).json()["state"] == ("running" if outcome == "return" else "interrupted")


async def test_cancelling_a_turn_ends_it_and_leaves_the_conversation_running(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, runtimes = configured(engine, path, blocked=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "regretted",
                                                "expected_revision": session["revision"]})
        await asyncio.wait_for(runtimes[0].started.wait(), 1)

        cancelled = await client.post(url + "/turns/turn-1/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert (await client.get(url)).json()["state"] == "running"

        # And the conversation still takes a new turn afterwards.
        runtimes[0].release.set()
        follow = await client.post(url + "/turns", json={"request_id": "turn-2", "text": "instead",
                                                         "expected_revision": session["revision"]})
        assert follow.status_code == 202

    with SessionJournal(path) as journal:
        assert journal.turn("alpha", session["session_id"], "turn-1")["status"] == "cancelled"


async def test_cancelling_twice_says_the_same_thing_and_a_retry_never_restarts_it(engine, tmp_path):
    """Terminal means terminal: the provider may already have been asked,
    so resending the same request id must not ask it again."""
    app, runtimes = configured(engine, tmp_path / "sessions.db", blocked=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "turn-1", "text": "regretted", "expected_revision": session["revision"]}
        await client.post(url + "/turns", json=body)
        await asyncio.wait_for(runtimes[0].started.wait(), 1)

        first = (await client.post(url + "/turns/turn-1/cancel")).json()
        runtimes[0].release.set()
        again = (await client.post(url + "/turns/turn-1/cancel")).json()
        assert first["status"] == again["status"] == "cancelled"

        retried = await client.post(url + "/turns", json=body)
        assert retried.json()["status"] == "cancelled"
        assert runtimes[0].calls == 1, "the provider was asked a second time"


async def test_a_settled_turn_cannot_be_cancelled_after_the_fact(engine, tmp_path):
    """A completed turn stays completed. Cancelling it would rewrite what
    happened, and its reply is already in memory."""
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "Juniper",
                                                "expected_revision": session["revision"]})
        for _ in range(200):
            if (await client.get(url + "/turns/turn-1")).json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        assert (await client.post(url + "/turns/turn-1/cancel")).status_code == 409
        assert (await client.get(url + "/turns/turn-1")).json()["status"] == "completed"


async def test_cancel_is_scoped_and_names_only_turns_that_exist(engine, tmp_path):
    app, runtimes = configured(engine, tmp_path / "sessions.db", blocked=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "x",
                                                "expected_revision": session["revision"]})
        await asyncio.wait_for(runtimes[0].started.wait(), 1)

        assert (await client.post(url + "/turns/never-sent/cancel")).status_code == 404
        elsewhere = await client.post(url + "/turns/turn-1/cancel",
                                      headers={"Authorization": "Bearer beta-key"})
        assert elsewhere.status_code == 404
        assert (await client.get(url + "/turns/turn-1")).json()["status"] == "pending"
        runtimes[0].release.set()
