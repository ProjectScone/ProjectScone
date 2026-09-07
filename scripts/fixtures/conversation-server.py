"""Isolated browser fixture: real API/Scone, scripted model, temporary state."""

import asyncio
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory

# Optional installed extras can be supplied explicitly in a local test environment.
if os.environ.get("SCONE_TEST_EXTRA_SITEPACKAGES"):
    sys.path.append(os.environ["SCONE_TEST_EXTRA_SITEPACKAGES"])
# Import the package installed in the selected test interpreter. Forcing the old
# checkout onto sys.path would conceal broken src-layout or wheel installations.

import httpx
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import conversations
from scone_memory.api.conversation_server import create_server
from scone_memory.realtime.text import TextConversation
from scone_memory.realtime.events import TextDelta, ReplyCompleted

release_reply = asyncio.Event()

class ScriptedModel:
    async def aclose(self):
        pass

    async def respond(self, messages):
        assert any("Juniper is calibrated with Polaris." in m["content"] for m in messages)
        if messages[-1]["content"] == "Stream Juniper":
            yield TextDelta("Juniper 🌿 ")
            print("CONVERSATIONS_MODEL_WAITING", flush=True)
            await release_reply.wait()
            yield TextDelta("uses Polaris.")
            yield ReplyCompleted()
            return
        if messages[-1]["content"] == "Wait for cancellation":
            print("CONVERSATIONS_MODEL_WAITING", flush=True)
            await asyncio.Event().wait()
        if messages[-1]["content"] == "What was my question?":
            assert {"role": "user", "content": "How is Juniper calibrated?"} in messages
            assert {"role": "assistant", "content": "Scripted answer: use Polaris."} in messages
            reply = "Scripted follow-up: you asked about Juniper calibration."
        else:
            reply = "Scripted answer: use Polaris."
        yield TextDelta(reply)
        yield ReplyCompleted()


class ScopedModel:
    async def aclose(self):
        pass

    async def respond(self, messages):
        context = repr(messages)
        no_matches = "No matching" in messages[-1]["content"]
        assert ("scope-eligible-guide" in context) is not no_matches
        assert "scope-excluded-guide" not in context
        yield TextDelta("Scoped answer: no retrieved context." if no_matches else "Scoped answer: the selected guide.")
        yield ReplyCompleted()


async def run():
    conversations.PLAYGROUND = Path(sys.argv[1])
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Juniper is calibrated with Polaris.", metadata={"collection": "manuals"})
    if "--scoped" in sys.argv[2:]:
        await memory.remember("alpha", "Juniper scope-eligible-guide.", kind="file", source="docs/guide.md",
                              created_at="2026-09-02", metadata={"collection": "manuals"})
        await memory.remember("alpha", "Juniper scope-excluded-guide.", kind="file", source="private/guide.md",
                              created_at="2026-09-02", metadata={"collection": "manuals"})
    with TemporaryDirectory(prefix="scone-browser-conversations-") as temporary:
        app = conversations.create_conversation_app(
            memory, {"conversation-fixture-alpha": "alpha", "conversation-fixture-beta": "beta"},
            Path(temporary) / "sessions.db",
            lambda space, sid: TextConversation(memory, space, sid, ScriptedModel,
                                                      where={"collection": "manuals"}),
            console=True,
            public_text_streaming="--streaming" in sys.argv[2:],
            **({"scoped_runtime_factory": lambda space, sid, scope: TextConversation(
                memory, space, sid, ScopedModel, **scope.kwargs())} if "--scoped" in sys.argv[2:] else {}),
        )
        if "--reopened" in sys.argv[2:]:
            # Exercise real service teardown/recreation with a retained
            # journal and memory engine. No provider or fixture-only API.
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture",
                                             headers={"Authorization": "Bearer conversation-fixture-alpha"}) as client:
                    created = await client.post("/v1/conversations", json={"request_id": "create-recovery", "capture": True})
                    created.raise_for_status()
                    session = created.json()
                    url = "/v1/conversations/" + session["session_id"]
                    for request_id, text in [("z-earlier", "How is Juniper calibrated?"),
                                             ("a-newer", "What was my question?")]:
                        accepted = await client.post(url + "/turns", json={"request_id": request_id, "text": text,
                                                                          "expected_revision": session["revision"]})
                        accepted.raise_for_status()
                        async with asyncio.timeout(10):
                            while True:
                                receipt = (await client.get(url + "/turns/" + request_id)).json()
                                if receipt["status"] != "pending":
                                    break
                                await asyncio.sleep(0.01)
                        assert receipt["status"] == "completed"
                    await memory.forget("alpha", receipt["result"]["assistant_episode_id"])
            app = conversations.create_conversation_app(
                memory, {"conversation-fixture-alpha": "alpha", "conversation-fixture-beta": "beta"},
                Path(temporary) / "sessions.db", None, console=True,
            )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            print("CONVERSATIONS_READY " + str(sock.getsockname()[1]), flush=True)
            server = create_server(app, host="127.0.0.1", port=0)
            loop = asyncio.get_running_loop()
            def stop_from_parent():
                command = os.read(sys.stdin.fileno(), 64)
                if command == b"release\n":
                    release_reply.set()
                    return
                loop.remove_reader(sys.stdin.fileno())
                server.should_exit = True
            loop.add_reader(sys.stdin.fileno(), stop_from_parent)
            try:
                await server.serve(sockets=[sock])
            finally:
                loop.remove_reader(sys.stdin.fileno())



asyncio.run(run())
