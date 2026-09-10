"""A turn's receipt outliving the process that ran it, end to end.

The service keeps turn results in a per-process dictionary. The journal
now records them too, so a reconnecting client can be told what became of
the turn it sent instead of a 404 that means only "not this process".
This is the write half: the read side changes the contract Codex's UI
depends on and is being coordinated with them first.
"""

import asyncio

import pytest

from scone_memory.realtime.session_journal import SessionJournal
from ..api.test_conversations_api import client_for, configured, create, engine  # noqa: F401


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


async def test_a_reconnecting_client_is_answered_from_the_journal(engine, tmp_path):
    """404 used to mean two different things: nobody sent that turn, and
    the process that ran it is gone. The second one is the reconnect case,
    and the answer is the receipt, with the reply's text read from the
    episode that holds it."""
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "Juniper",
                                                "expected_revision": session["revision"]})
        for _ in range(200):
            if (await client.get(url + "/turns/turn-1")).json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

    restarted, _ = configured(engine, path)
    async with client_for(restarted) as client:
        again = (await client.get(url + "/turns/turn-1")).json()
        assert again["status"] == "completed"
        assert again["result"]["text"] == "Scripted response to Juniper"
        assert again["result_state"] == "available"
        # A request id nobody ever sent is still simply absent.
        assert (await client.get(url + "/turns/never-sent")).status_code == 404


async def test_forgetting_the_episode_takes_the_text_out_of_the_receipt(engine, tmp_path):
    """Terminal, not failed: the turn still completed. The text is gone
    because the memory holding it was deleted, and that must never read as
    a failure or invite a resend."""
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "Juniper",
                                                "expected_revision": session["revision"]})
        for _ in range(200):
            live = (await client.get(url + "/turns/turn-1")).json()
            if live["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        assert live["result_state"] == "available"

        await engine.forget("alpha", live["result"]["assistant_episode_id"])

        # The live process still holds its own copy of the reply; it must
        # not serve it, or "no stale transcript copies" holds only across
        # a restart, which is the easy half.
        after = (await client.get(url + "/turns/turn-1")).json()
        assert after["status"] == "completed"
        assert after["result"] is None and after["result_state"] == "forgotten"


async def test_the_turns_of_a_session_can_be_listed_and_do_not_cross_spaces(engine, tmp_path):
    """A reloaded page knows the session id and nothing else, so it needs
    a way to find the turns without guessing their request ids."""
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "Juniper",
                                                "expected_revision": session["revision"]})
        for _ in range(200):
            if (await client.get(url + "/turns/turn-1")).json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        listed = (await client.get(url + "/turns")).json()
        assert [t["request_id"] for t in listed["turns"]] == ["turn-1"]
        assert listed["turns"][0]["result_state"] == "available"
        assert listed["has_more"] is False

        elsewhere = await client.get(url + "/turns", headers={"Authorization": "Bearer beta-key"})
        assert elsewhere.status_code == 404


async def test_a_store_that_cannot_be_read_is_not_reported_as_a_deletion(engine, tmp_path):
    """Calling a storage failure "forgotten" would report a deletion
    nobody performed, and the UI would show the text as gone for good. It
    says unreadable instead, which asserts nothing and can be retried."""
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/turns", json={"request_id": "turn-1", "text": "Juniper",
                                                "expected_revision": session["revision"]})
        for _ in range(200):
            if (await client.get(url + "/turns/turn-1")).json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        working = engine.episode

        async def failing(*_args, **_kwargs):
            raise RuntimeError("the document store is unreachable")

        engine.episode = failing
        try:
            broken = (await client.get(url + "/turns/turn-1")).json()
        finally:
            engine.episode = working

        assert broken["status"] == "completed"
        assert broken["result"] is None and broken["result_state"] == "unreadable"
        assert (await client.get(url + "/turns/turn-1")).json()["result_state"] == "available"


async def test_starting_the_service_settles_turns_a_dead_process_left_behind(engine, tmp_path):
    """The lifespan already interrupts sessions that were running when
    their process died. Their turns were left reading as in flight, which
    is a receipt that can never become true: the process that would have
    settled it is gone. The service takes an exclusive lock on the journal
    at startup, so nothing else can own those turns while it does."""
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        created = journal.create("alpha", "create-1")
        sid = created["session_id"]
        journal.transition("alpha", sid, "start-1", "start", created["revision"])
        journal.start_turn("alpha", sid, "turn-1", {"text": "asked but never answered"})
        assert journal.turn("alpha", sid, "turn-1")["status"] == "accepted"

    app, _ = configured(engine, path)
    async with client_for(app) as client:
        answered = (await client.get(f"/v1/conversations/{sid}/turns/turn-1")).json()

    assert answered["status"] == "interrupted"
    assert answered["result"] is None
    with SessionJournal(path) as journal:
        assert "did not finish" in journal.turn("alpha", sid, "turn-1")["error"]
