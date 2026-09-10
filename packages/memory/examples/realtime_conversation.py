"""Run with Python 3.11+: python examples/realtime_conversation.py.

Native Scone memory and scheduling; the external model alone is scripted.
No credentials, database files, microphone or network requests.
"""

import asyncio
import json

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from scone_memory.realtime.text import TextConversation


async def main():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("example", "Juniper is calibrated with Polaris.", metadata={"collection": "manuals"})
    supplied = False

    class ScriptedModel:
        async def respond(self, messages):
            nonlocal supplied
            supplied = any("Juniper is calibrated with Polaris." in m["content"] for m in messages)
            yield TextDelta("Use ")
            yield TextDelta("Polaris.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    conversation = TextConversation(memory, "example", "example-session", ScriptedModel,
                                    where={"collection": "manuals"})
    try:
        result = await conversation.reply("How is Juniper calibrated?")
        transcript = await memory.episodes("example", {"session_id": "example-session"})
        print(json.dumps(dict(mode="scripted provider; no inference or live media",
                              memory_context=result["memory_context"], source_supplied=supplied,
                              transcript=[e.content for e in transcript])))
    finally:
        await conversation.close()


if __name__ == "__main__":
    asyncio.run(main())
