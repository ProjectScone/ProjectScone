"""Isolated browser fixture: real API/Pipecat, scripted model, temporary state."""

import asyncio
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory

# Optional installed extras can be supplied explicitly in a local test environment.
if os.environ.get("SCONE_TEST_EXTRA_SITEPACKAGES"):
    sys.path.append(os.environ["SCONE_TEST_EXTRA_SITEPACKAGES"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/scone-memory"))

import uvicorn
from pipecat.frames.frames import (
    LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
    LLMTextFrame, LLMThoughtTextFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import conversations
from scone_memory.integrations.pipecat_text import PipecatTextConversation


class ScriptedModel(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        messages = frame.context.get_messages()
        assert any("Juniper is calibrated with Polaris." in m["content"] for m in messages)
        if messages[-1]["content"] == "What was my question?":
            assert {"role": "user", "content": "How is Juniper calibrated?"} in messages
            assert {"role": "assistant", "content": "Scripted answer: use Polaris."} in messages
            reply = "Scripted follow-up: you asked about Juniper calibration."
        else:
            reply = "Scripted answer: use Polaris."
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMThoughtTextFrame("private-fixture-thought-must-not-be-captured"))
        await self.push_frame(LLMTextFrame(reply))
        await self.push_frame(LLMFullResponseEndFrame())


async def run():
    conversations.PLAYGROUND = Path(sys.argv[1])
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        await memory.remember("alpha", "Juniper is calibrated with Polaris.", metadata={"collection": "manuals"})
        with TemporaryDirectory(prefix="scone-browser-conversations-") as temporary:
            app = conversations.create_conversation_app(
                memory, {"conversation-fixture-alpha": "alpha", "conversation-fixture-beta": "beta"},
                Path(temporary) / "sessions.db",
                lambda space, sid: PipecatTextConversation(memory, space, sid, ScriptedModel,
                                                          where={"collection": "manuals"}),
                console=True,
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                print("CONVERSATIONS_READY " + str(sock.getsockname()[1]), flush=True)
                await uvicorn.Server(uvicorn.Config(app, log_level="error")).serve(sockets=[sock])
    finally:
        await memory.close()


asyncio.run(run())
