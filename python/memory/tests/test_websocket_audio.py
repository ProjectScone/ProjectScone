"""A browser's audio over one WebSocket is a Scone AudioTransport: PCM in
binary frames both ways, control in JSON text frames, and every failure
visible to the session rather than swallowed on the wire."""

from __future__ import annotations

import asyncio
import json
import struct

import pytest
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.audio import AudioChunk, ReplyCompleted, SpeechStarted, TextDelta, Transcript
from scone_memory.realtime.voice import VoiceSession
from scone_memory.realtime.websocket import WebSocketAudioTransport, frame, parse_frame

PCM = b"\x01\x00" * 160


def app_running(script):
    """A socket route that hands the accepted socket to ``script`` as a transport."""
    app = FastAPI()
    outcome = {}

    @app.websocket("/audio")
    async def audio(websocket: WebSocket):
        await websocket.accept()
        transport = WebSocketAudioTransport(websocket, sample_rate=16000, channels=1)
        try:
            outcome["result"] = await script(transport)
        except Exception as error:  # the test reads what the session would have seen
            outcome["error"] = error
        finally:
            await transport.aclose()

    return app, outcome


def test_frames_carry_the_turn_and_format_and_round_trip():
    chunk = AudioChunk(PCM, 16000, 1)
    data = frame(chunk, "turn-1")
    assert data[0] == len(b"turn-1") and data[1:7] == b"turn-1"
    assert struct.unpack("<IB", data[7:12]) == (16000, 1) and data[12:] == PCM
    assert parse_frame(data) == ("turn-1", chunk)
    with pytest.raises(ValueError):
        frame(chunk, "")
    with pytest.raises(ValueError):
        parse_frame(data[:5])


def test_audio_flows_both_ways_and_the_client_ends_its_input():
    async def script(transport):
        heard = [chunk async for chunk in transport.receive()]
        await transport.send(AudioChunk(PCM * 2, 16000), "reply-turn")
        await transport.clear("reply-turn")
        return heard

    app, outcome = app_running(script)
    with TestClient(app).websocket_connect("/audio") as socket:
        socket.send_bytes(PCM)
        socket.send_bytes(PCM * 3)
        socket.send_text(json.dumps({"type": "end"}))
        turn, spoken = parse_frame(socket.receive_bytes())
        assert (turn, spoken.pcm, spoken.sample_rate, spoken.channels) == ("reply-turn", PCM * 2, 16000, 1)
        assert json.loads(socket.receive_text()) == {"type": "clear", "turn_id": "reply-turn"}
    assert [c.pcm for c in outcome["result"]] == [PCM, PCM * 3]
    assert all(c.sample_rate == 16000 and c.channels == 1 for c in outcome["result"])


@pytest.mark.parametrize("bad", [b"\x01\x00\x02", json.dumps({"type": "hello"}), "not json", b"\x00" * 70000])
def test_a_frame_the_format_does_not_admit_fails_the_session_instead_of_being_reinterpreted(bad):
    async def script(transport):
        return [chunk async for chunk in transport.receive()]

    app, outcome = app_running(script)
    with TestClient(app).websocket_connect("/audio") as socket:
        (socket.send_bytes if isinstance(bad, bytes) else socket.send_text)(bad)
        # If the bad frame were admitted the input would simply end here and
        # the socket close, so a regression fails instead of waiting forever.
        socket.send_text(json.dumps({"type": "end"}))
        assert json.loads(socket.receive_text())["type"] == "error"
    assert isinstance(outcome["error"], ValueError)


def test_a_client_that_leaves_ends_the_input_and_every_later_send_raises():
    seen = asyncio.Event() if False else None
    states = {}

    async def script(transport):
        heard = [chunk async for chunk in transport.receive()]
        states["heard"] = len(heard)
        for call in (lambda: transport.send(AudioChunk(PCM, 16000), "t"), lambda: transport.clear("t")):
            try:
                await call()
            except RuntimeError as error:
                states.setdefault("raised", []).append(str(error))
        return True

    app, outcome = app_running(script)
    with TestClient(app).websocket_connect("/audio") as socket:
        socket.send_bytes(PCM)
    assert outcome["result"] is True and states["heard"] == 1
    assert len(states["raised"]) == 2 and all("disconnected" in reason for reason in states["raised"])


class Recognizer:
    async def transcribe(self, audio):
        async for chunk in audio:
            yield SpeechStarted()
            yield Transcript("Where does Juniper point?")

    async def aclose(self):
        pass


class Model:
    async def respond(self, messages):
        yield TextDelta("Juniper points to Polaris.")
        yield ReplyCompleted()

    async def aclose(self):
        pass


class Synthesizer:
    async def synthesize(self, text):
        yield AudioChunk(b"\x07\x00" * 320, 16000)

    async def aclose(self):
        pass


def test_a_voice_session_runs_over_the_socket_and_captures_both_sides():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    sessions = []

    async def script(transport):
        session = VoiceSession(engine, "voice", "socket-session", transport_factory=lambda: transport,
                               stt_factory=Recognizer, model_factory=Model, tts_factory=Synthesizer,
                               capture=True, session_timeout=5, turn_timeout=3)
        sessions.append(session)
        await session.run()
        return session.state

    app, outcome = app_running(script)
    with TestClient(app).websocket_connect("/audio") as socket:
        socket.send_bytes(PCM)
        turn, spoken = parse_frame(socket.receive_bytes())
        assert spoken.pcm == b"\x07\x00" * 320 and len(turn) == 32
        socket.send_text(json.dumps({"type": "end"}))
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()  # the session closes its transport when the input ends
    assert "error" not in outcome, repr(outcome.get("error"))
    assert outcome["result"] == "ended" and sessions[0].stored_count == 2
    stored = asyncio.run(engine.episodes("voice", {"session_id": "socket-session"}))
    assert sorted(e.metadata["role"] for e in stored) == ["assistant", "user"]
