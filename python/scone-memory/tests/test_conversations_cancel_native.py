"""Cancel one turn, then complete the next, against the real runtime.

test_conversations_cancel.py proves the route with a scripted runtime
that never closes. Codex pointed out that the production runtime CAN
close on cancel, so a 202 for the next turn proved nothing about whether
it would complete. This runs the same sequence through TextConversation:
one cancel that leaves the runtime reusable and a completed turn after
it, and one cancel that closes the runtime and must interrupt the session
instead of promising a next turn it cannot serve.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation


class Model:
    """Streams one chunk, then waits to be released before completing.
    ``broken_cleanup`` makes aclose fail, which is the runtime's own
    signal that it cannot vouch for its state after a cancel."""

    def __init__(self, *, broken_cleanup=False):
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.broken_cleanup = broken_cleanup

    async def respond(self, messages):
        yield TextDelta("thinking")
        self.started.set()
        await self.release.wait()
        yield TextDelta(" done")
        yield ReplyCompleted()

    async def aclose(self):
        if self.broken_cleanup:
            raise RuntimeError("provider left resources behind")


async def app_with(memory, models, path):
    def factory():
        return models.pop(0)

    return create_conversation_app(
        memory, {"alpha-key": "alpha"}, path,
        lambda space, sid: TextConversation(memory, space, sid, factory),
    )


async def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test",
                             headers={"Authorization": "Bearer alpha-key"})


async def until(client, url, request_id, status):
    for _ in range(300):
        receipt = (await client.get(f"{url}/turns/{request_id}")).json()
        if receipt["status"] == status:
            return receipt
        await asyncio.sleep(0.01)
    raise AssertionError(f"turn {request_id} never reached {status}: {receipt}")


@pytest.fixture
async def memory():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


async def test_a_clean_cancel_leaves_the_real_runtime_able_to_complete_the_next_turn(memory, tmp_path):
    first, second = Model(), Model()
    app = await app_with(memory, [first, second], tmp_path / "journal.db")
    async with app.router.lifespan_context(app), await client_for(app) as client:
        session = (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).json()
        url = "/v1/conversations/" + session["session_id"]

        await client.post(url + "/turns", json={"request_id": "one", "text": "first question",
                                                "expected_revision": session["revision"]})
        await asyncio.wait_for(first.started.wait(), 2)

        cancelled = await client.post(url + "/turns/one/cancel")
        assert cancelled.json()["status"] == "cancelled", cancelled.text
        assert (await client.get(url)).json()["state"] == "running", "a clean cancel keeps the conversation"

        second.release.set()
        sent = await client.post(url + "/turns", json={"request_id": "two", "text": "second question",
                                                       "expected_revision": session["revision"]})
        assert sent.status_code == 202, sent.text
        done = await until(client, url, "two", "completed")
        assert done["result"]["text"] == "thinking done"

        # The cancelled turn's user text was captured before the cancel;
        # its reply never was, and the second turn's reply is there.
        transcript = (await client.get(url + "/transcript")).json()["episodes"]
        assert [e["content"] for e in transcript] == ["first question", "second question", "thinking done"]


async def test_a_cancel_the_runtime_cannot_vouch_for_interrupts_the_session_instead(memory, tmp_path):
    """When cleanup fails after a cancel the runtime reports itself
    closed. Answering the next turn with 202 would promise work that can
    never complete, so the session is interrupted and says so."""
    first = Model(broken_cleanup=True)
    app = await app_with(memory, [first, Model()], tmp_path / "journal.db")
    async with app.router.lifespan_context(app), await client_for(app) as client:
        session = (await client.post("/v1/conversations", json={"request_id": "new", "capture": True})).json()
        url = "/v1/conversations/" + session["session_id"]

        await client.post(url + "/turns", json={"request_id": "one", "text": "first question",
                                                "expected_revision": session["revision"]})
        await asyncio.wait_for(first.started.wait(), 2)

        cancelled = await client.post(url + "/turns/one/cancel")
        assert cancelled.json()["status"] == "cancelled"
        for _ in range(300):
            if (await client.get(url)).json()["state"] != "running":
                break
            await asyncio.sleep(0.01)
        assert (await client.get(url)).json()["state"] == "interrupted"
        # What this proves is the route's response to a cancel it cannot vouch
        # for. The runtime's own closure on failed cleanup is proven where it
        # lives, in test_text_conversation.py; asserting runtime.closed here
        # would observe the route's cleanup, which closes it either way.

        refused = await client.post(url + "/turns", json={"request_id": "two", "text": "second question",
                                                          "expected_revision": session["revision"]})
        assert refused.status_code == 409, "no next turn is promised on a runtime that cannot serve it"
