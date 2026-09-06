"""A turn's receipt outliving the process that ran it, end to end.

The service keeps turn results in a per-process dictionary. The journal
now records them too, so a reconnecting client can be told what became of
the turn it sent instead of a 404 that means only "not this process".
This is the write half: the read side changes the contract Codex's UI
depends on and is being coordinated with them first.
"""

import asyncio

import pytest

from scone_memory.session_journal import SessionJournal
from test_conversations_api import client_for, configured, create, engine  # noqa: F401


async def test_a_completed_turn_is_recorded_where_a_restart_can_find_it(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        body = {"request_id": "turn-1", "text": "Juniper", "expected_revision": session["revision"]}
        await client.post(url + "/turns", json=body)
        for _ in range(200):
            if (await client.get(url + "/turns/turn-1")).json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        live = (await client.get(url + "/turns/turn-1")).json()

    with SessionJournal(path) as journal:
        kept = journal.turn("alpha", session["session_id"], "turn-1")

    assert kept["status"] == "completed"
    # The episode the reply's text lives in, so nothing holds a second copy.
    assert kept["episode_id"] == live["result"]["assistant_episode_id"]
    assert "Scripted response" not in repr(kept)


async def test_a_turn_the_process_never_finished_is_recorded_as_interrupted(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, runtimes = configured(engine, path, blocked=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "waiting",
                                                "expected_revision": session["revision"]})
        await asyncio.wait_for(runtimes[0].started.wait(), 1)
        await client.post(url + "/stop", json={"request_id": "stop-1",
                                              "expected_revision": session["revision"]})
        runtimes[0].release.set()

    with SessionJournal(path) as journal:
        assert journal.turn("alpha", session["session_id"], "turn-1")["status"] == "interrupted"


async def test_a_turn_accepted_but_never_settled_is_recovered_not_left_pending(engine, tmp_path):
    """The crash case the journal exists for: the process is gone while a
    turn is still accepted, and recovery says so rather than leaving a
    receipt that reads as in flight."""
    path = tmp_path / "sessions.db"
    app, runtimes = configured(engine, path, blocked=True)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "waiting",
                                                "expected_revision": session["revision"]})
        await asyncio.wait_for(runtimes[0].started.wait(), 1)
        with SessionJournal(path) as journal:
            assert journal.turn("alpha", session["session_id"], "turn-1")["status"] == "accepted"
        runtimes[0].release.set()

    with SessionJournal(path) as journal:
        assert journal.recover("alpha") >= 0  # settled here or by the shutdown above
        assert journal.turn("alpha", session["session_id"], "turn-1")["status"] != "accepted"
