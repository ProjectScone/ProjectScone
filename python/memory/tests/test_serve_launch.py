"""``serve`` as a real process: composed with a catalog, spoken to over TCP,
and shut down cleanly with a voice socket still open."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from scone_memory.realtime.session_journal import SessionJournal
from scone_memory.realtime.websocket import parse_frame

REGISTRY = '''
from scone_memory.realtime.audio import AudioChunk, ReplyCompleted, SpeechStarted, TextDelta, Transcript
from scone_memory.realtime.providers import ProviderRegistry


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
        yield AudioChunk(b"\\x07\\x00" * 320, 16000)

    async def aclose(self):
        pass


def registry():
    return ProviderRegistry(reply={("stub", "echo"): Model}, transcription={("stub", "ears"): Recognizer},
                            speech={("stub", "mouth", "alto"): Synthesizer})
'''
HELPER = {"schema_version": 1, "id": "helper", "name": "Helper", "instructions": "Answer briefly.",
          "reply": {"provider": "stub", "model": "echo"}, "transcription": {"provider": "stub", "model": "ears"},
          "speech": {"provider": "stub", "model": "mouth", "voice": "alto"}}
KEY = "launcher-key"
PCM = b"\x01\x00" * 160


@pytest.fixture
def composed(tmp_path):
    (tmp_path / "launch_registry.py").write_text(REGISTRY)
    (tmp_path / "personas.json").write_text(json.dumps([HELPER]))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SCONE_")}
    env.update(SCONE_SQLITE_PATH=str(tmp_path / "memory.db"), SCONE_EMBEDDER="hash", SCONE_API_KEY=KEY,
               SCONE_HOST="127.0.0.1", SCONE_PORT=str(port), SCONE_CONVERSATIONS_JOURNAL=str(tmp_path / "sessions.db"),
               SCONE_CONVERSATIONS_PERSONAS=str(tmp_path / "personas.json"),
               SCONE_CONVERSATIONS_REGISTRY="launch_registry:registry", PYTHONPATH=str(tmp_path))
    process = subprocess.Popen([sys.executable, "-m", "scone_memory.runtime.cli", "serve"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {KEY}"}, timeout=2)
    deadline = time.monotonic() + 20
    while True:
        if process.poll() is not None:
            raise AssertionError("serve exited: " + process.communicate()[1])
        try:
            if client.get("/healthz").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            process.kill()
            raise AssertionError("serve did not become ready")
        time.sleep(0.025)
    yield client, port, process, tmp_path / "sessions.db"
    client.close()
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        _, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise AssertionError("serve required forced termination")
    assert process.returncode in (0, 130), stderr
    assert KEY not in stderr and "Traceback" not in stderr, stderr


async def speak(port, sid, *, end=True):
    """One browser-shaped exchange over TCP: hello, a PCM frame, the reply."""
    import websockets.asyncio.client as ws

    async with ws.connect(f"ws://127.0.0.1:{port}/v1/conversations/{sid}/audio") as sock:
        await sock.send(json.dumps({"type": "hello", "key": KEY, "sample_rate": 16000, "channels": 1}))
        assert json.loads(await sock.recv()) == {"type": "ready", "session_id": sid}
        await sock.send(PCM)
        turn, spoken = parse_frame(await sock.recv())
        assert spoken.pcm == b"\x07\x00" * 320 and len(turn) == 32
        if end:
            await sock.send(json.dumps({"type": "end"}))
            with pytest.raises(Exception):
                await asyncio.wait_for(sock.recv(), 5)  # the session closes the socket
        return turn


def test_a_launched_serve_composes_memory_catalog_and_voice_over_tcp(composed):
    client, port, _, _ = composed
    assert client.get("/v1/capabilities").json()["features"]["conversations"] is True
    ready = client.get("/v1/conversations/capabilities").json()
    assert ready["personas"] == 1 and ready["voice"] is True and ready["text_configured"] is True
    assert KEY not in client.get("/memory").text
    listing = client.get("/v1/conversations/personas").json()
    fingerprint = listing["personas"][0]["fingerprint"]

    text = client.post("/v1/conversations", json={"request_id": "t", "capture": True, "persona": "helper",
                                                  "persona_fingerprint": fingerprint}).json()
    body = {"request_id": "turn", "text": "Hello", "expected_revision": text["revision"]}
    assert client.post(f"/v1/conversations/{text['session_id']}/turns", json=body).status_code == 202
    deadline = time.monotonic() + 10
    while (receipt := client.get(f"/v1/conversations/{text['session_id']}/turns/turn").json())["status"] == "pending":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert receipt["status"] == "completed" and "Polaris" in receipt["result"]["text"]

    voice = client.post("/v1/conversations", json={"request_id": "v", "capture": True, "persona": "helper",
                                                   "mode": "voice", "persona_fingerprint": fingerprint}).json()
    assert voice["state"] == "created"
    asyncio.run(speak(port, voice["session_id"]))
    deadline = time.monotonic() + 10
    while (after := client.get(f"/v1/conversations/{voice['session_id']}").json())["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert after["state"] == "ended" and after["persona"]["current"] is True
    roles = sorted(e["metadata"]["role"] for e in client.get(f"/v1/conversations/{voice['session_id']}/transcript").json()["episodes"])
    assert roles == ["assistant", "user"]


def test_shutdown_with_a_voice_socket_open_completes_and_records_the_interruption(composed):
    client, port, process, journal = composed
    voice = client.post("/v1/conversations", json={"request_id": "v", "capture": True, "persona": "helper", "mode": "voice"}).json()
    sid = voice["session_id"]

    async def hold_open():
        import websockets.asyncio.client as ws

        async with ws.connect(f"ws://127.0.0.1:{port}/v1/conversations/{sid}/audio") as sock:
            await sock.send(json.dumps({"type": "hello", "key": KEY, "sample_rate": 16000, "channels": 1}))
            assert json.loads(await sock.recv())["type"] == "ready"
            process.send_signal(signal.SIGINT)
            with pytest.raises(Exception):
                await asyncio.wait_for(sock.recv(), 8)  # the service closes it on the way down

    asyncio.run(hold_open())
    _, stderr = process.communicate(timeout=8)
    assert process.returncode in (0, 130) and "Traceback" not in stderr, stderr
    with SessionJournal(journal) as read:
        assert read.get("default", sid)["state"] == "interrupted"
