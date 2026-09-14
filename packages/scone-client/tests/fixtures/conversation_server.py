"""Real local conversation HTTP fixture: a journal on disk and a scripted text runtime that streams."""
import asyncio
from pathlib import Path
import sys

import uvicorn
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app


class Runtime:
    """Replies in two pieces with a pause between, so a reader on the socket watches the reply arrive."""

    def __init__(self, engine, space, sid):
        self.engine, self.space, self.sid = engine, space, sid

    async def reply(self, text, *, on_text=None):
        answer = 'Scripted reply to ' + text
        if on_text is not None:
            head, tail = answer[:len(answer) // 2], answer[len(answer) // 2:]
            await on_text(head)
            await asyncio.sleep(0.3)
            await on_text(tail)
        item = await self.engine.remember(self.space, answer, metadata={'session_id': self.sid})
        return {'text': answer, 'assistant_episode_id': item.episode_id, 'provider_completion': 'unverified'}

    async def close(self):
        return None


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_conversation_app(memory, {'conversation-fixture': 'alpha'}, state / 'sessions.sqlite',
                                  lambda space, sid: Runtime(memory, space, sid), public_text_streaming=True)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))

    async def commands():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True

    commands_task = asyncio.create_task(commands())
    try:
        await server.serve()
    finally:
        commands_task.cancel()
        await asyncio.gather(commands_task, return_exceptions=True)
        await memory.close()


asyncio.run(run())
