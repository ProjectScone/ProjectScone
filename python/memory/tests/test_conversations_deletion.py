"""Deleting a conversation, and everything it left in memory.

A transcript is native episodes plus journal rows. Removing the session
has to remove both, in that order, and reach nothing outside the session
it names.
"""

import asyncio

import pytest

from scone_memory.core.errors import NotFound
from scone_memory.realtime.session_journal import SessionJournal
from test_conversations_api import client_for, configured, create, engine  # noqa: F401


async def spoke(client, url, revision, request_id="turn-1", text="Juniper"):
    await client.post(url + "/turns", json={"request_id": request_id, "text": text,
                                            "expected_revision": revision})
    for _ in range(200):
        if (await client.get(url + f"/turns/{request_id}")).json()["status"] == "completed":
            return (await client.get(url + f"/turns/{request_id}")).json()
        await asyncio.sleep(0.01)
    raise AssertionError("the scripted turn never completed")


async def test_deleting_a_session_removes_its_transcript_and_its_receipts(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        session = await create(client)
        sid = session["session_id"]
        url = "/v1/conversations/" + sid
        turn = await spoke(client, url, session["revision"])
        await client.post(url + "/stop", json={"request_id": "stop-1",
                                              "expected_revision": session["revision"]})

        assert (await client.delete(url)).status_code == 204
        assert (await client.get(url)).status_code == 404
        assert (await client.get(url + "/turns/turn-1")).status_code == 404

    with pytest.raises(NotFound):
        await engine.episode("alpha", turn["result"]["assistant_episode_id"])
    with SessionJournal(path) as journal:
        with pytest.raises(NotFound):
            journal.get("alpha", sid)


async def test_deleting_one_session_leaves_the_others_alone(engine, tmp_path):
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        first = await create(client, "new-1")
        second = await create(client, "new-2")
        kept = await spoke(client, "/v1/conversations/" + second["session_id"], second["revision"],
                           text="Ana")
        going = await spoke(client, "/v1/conversations/" + first["session_id"], first["revision"],
                            text="Juniper")
        for session in (first, second):
            await client.post(f"/v1/conversations/{session['session_id']}/stop",
                              json={"request_id": "stop", "expected_revision": session["revision"]})

        assert (await client.delete("/v1/conversations/" + first["session_id"])).status_code == 204
        surviving = await client.get(f"/v1/conversations/{second['session_id']}/turns/turn-1")
        assert surviving.json()["result_state"] == "available"

    assert (await engine.episode("alpha", kept["result"]["assistant_episode_id"])).episode_id
    with pytest.raises(NotFound):
        await engine.episode("alpha", going["result"]["assistant_episode_id"])


async def test_retrying_a_deleted_create_does_not_leave_an_orphan_session(engine, tmp_path):
    app, runtimes = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        stopped = await client.post(url + "/stop", json={"request_id": "stop",
                                                        "expected_revision": session["revision"]})
        assert stopped.status_code == 200
        assert (await client.delete(url)).status_code == 204
        for _ in range(2):
            replay = await client.post("/v1/conversations", json={"request_id": "new", "capture": True})
            assert replay.status_code == 404
            assert (await client.get("/v1/conversations")).json()["items"] == []
        assert len(runtimes) == 1


async def test_a_running_conversation_is_not_deleted_out_from_under_itself(engine, tmp_path):
    """Stop it first. Deleting a session a process is still serving would
    leave that process answering from a transcript that no longer exists."""
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        assert (await client.delete(url)).status_code == 409
        assert (await client.get(url)).json()["state"] == "running"

        await client.post(url + "/stop", json={"request_id": "stop-1",
                                              "expected_revision": session["revision"]})
        assert (await client.delete(url)).status_code == 204


async def test_a_key_cannot_delete_another_spaces_conversation(engine, tmp_path):
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        url = "/v1/conversations/" + session["session_id"]
        await client.post(url + "/stop", json={"request_id": "stop-1",
                                              "expected_revision": session["revision"]})

        assert (await client.delete(url, headers={"Authorization": "Bearer beta-key"})).status_code == 404
        assert (await client.get(url)).json()["state"] == "ended"
        # And deleting what is already gone says so rather than pretending.
        assert (await client.delete(url)).status_code == 204
        assert (await client.delete(url)).status_code == 404


async def test_a_reply_two_conversations_share_is_not_deleted_with_one_of_them(engine, tmp_path):
    """Identical text deduplicates: two conversations that say the same
    thing hold ONE episode between them. Deleting one of them must not
    take the other's transcript with it, so deletion removes only what no
    other conversation's receipt still names."""
    path = tmp_path / "sessions.db"
    app, _ = configured(engine, path)
    async with client_for(app) as client:
        first = await create(client, "new-1")
        second = await create(client, "new-2")
        mine = await spoke(client, "/v1/conversations/" + first["session_id"], first["revision"])
        theirs = await spoke(client, "/v1/conversations/" + second["session_id"], second["revision"])
        assert mine["result"]["assistant_episode_id"] == theirs["result"]["assistant_episode_id"], (
            "the fixture no longer deduplicates; this test needs identical replies"
        )
        for session in (first, second):
            await client.post(f"/v1/conversations/{session['session_id']}/stop",
                              json={"request_id": "stop", "expected_revision": session["revision"]})

        assert (await client.delete("/v1/conversations/" + first["session_id"])).status_code == 204
        survivor = await client.get(f"/v1/conversations/{second['session_id']}/turns/turn-1")
        assert survivor.json()["result_state"] == "available"
        assert (await engine.episode("alpha", theirs["result"]["assistant_episode_id"])).episode_id

        # And now the second one, which is the case metadata alone cannot
        # see: the shared episode carries the FIRST session's session_id,
        # so only this session's receipt still names it.
        assert (await client.delete("/v1/conversations/" + second["session_id"])).status_code == 204

    with pytest.raises(NotFound):
        await engine.episode("alpha", theirs["result"]["assistant_episode_id"])
