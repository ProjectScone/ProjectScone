"""Native voice browser fixture: real Scone, temporary memory, scripted providers."""
import asyncio
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.api import conversations
from scone_memory.api.conversation_server import create_server
from scone_memory.realtime.audio import AudioChunk, ReplyCompleted, SpeechStarted, TextDelta, Transcript
from scone_memory.realtime.catalog import bind_catalog
from scone_memory.realtime.persona import Persona
from scone_memory.realtime.providers import ProviderRegistry
interrupt = "--interrupt" in sys.argv
ui = os.environ.get('SCONE_CONVERSATIONS_HTML')
if ui:
    conversations.PLAYGROUND = Path(ui)
first_spoken = asyncio.Event()


class Recognizer:
    async def transcribe(self, audio):
        said = False
        interrupted = False
        async for chunk in audio:
            assert chunk.channels == 1 and len(chunk.pcm) == round(chunk.sample_rate * .02) * 2
            if not said:
                said = True
                yield SpeechStarted()
                yield Transcript("How is Juniper calibrated?")
            elif interrupt and first_spoken.is_set() and not interrupted:
                interrupted = True
                yield SpeechStarted()
                yield Transcript("Make that shorter.")

    async def aclose(self):
        pass


class Model:
    async def respond(self, messages):
        assert any("Juniper is calibrated with Polaris." in m["content"] for m in messages)
        yield TextDelta("Use Polaris." if messages[-1]["content"] == "Make that shorter."
                        else "Use Polaris to calibrate Juniper.")
        yield ReplyCompleted()

    async def aclose(self):
        pass


class Speech:
    async def synthesize(self, text):
        assert "Polaris" in text
        if interrupt and "calibrate" in text:
            yield AudioChunk(b"\x00\x20" * 32000, 16000)
            first_spoken.set()
            # This first utterance remains unfinished until real native barge-in.
            await asyncio.Event().wait()
        yield AudioChunk(b"\x00\x20" * 1600, 16000)

    async def aclose(self):
        pass


async def run():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Juniper is calibrated with Polaris.")
    persona = Persona.model_validate({
        "schema_version": 1, "id": "guide", "name": "Voice guide", "instructions": "Answer briefly.",
        "reply": {"provider": "scripted", "model": "reply"},
        "transcription": {"provider": "scripted", "model": "ears"},
        "speech": {"provider": "scripted", "model": "speech", "voice": "warm"},
    })
    registry = ProviderRegistry(reply={("scripted", "reply"): Model},
                                transcription={("scripted", "ears"): Recognizer},
                                speech={("scripted", "speech", "warm"): Speech})
    with TemporaryDirectory(prefix="scone-browser-voice-") as temporary:
        app = create_conversation_app(memory, {"voice-alpha": "alpha", "voice-beta": "beta"},
                                      Path(temporary) / "journal.db", None,
                                      catalog=bind_catalog([persona], registry), console=bool(ui))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            print("VOICE_READY " + str(sock.getsockname()[1]), flush=True)
            server = create_server(app, host="127.0.0.1", port=0)
            loop = asyncio.get_running_loop()

            def stop():
                os.read(sys.stdin.fileno(), 64)
                loop.remove_reader(sys.stdin.fileno())
                server.should_exit = True

            loop.add_reader(sys.stdin.fileno(), stop)
            try:
                await server.serve(sockets=[sock])
            finally:
                loop.remove_reader(sys.stdin.fileno())


asyncio.run(run())
