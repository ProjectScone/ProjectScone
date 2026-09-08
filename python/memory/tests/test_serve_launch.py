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
#: A one-minute load average above this means the machine, not the server,
#: is what the clock is measuring. A quiet host and CI never reach it.
BUSY = 12.0
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
    # A loaded machine (another gate, a cargo build, a benchmark pinning
    # every core) has taken more than 20 s to import and bind, and at 90 s
    # this failed twice during a bench run while passing alone in three
    # seconds. The bound is generous so the test measures the server
    # rather than its neighbours, still fails rather than hanging, and
    # says what stderr held when it does.
    deadline = time.monotonic() + 240
    while True:
        if process.poll() is not None:
            raise AssertionError("serve exited: " + process.communicate()[1])
        try:
            if client.get("/healthz").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            alive = process.poll() is None
            load = os.getloadavg()[0]
            process.kill()
            _, stderr = process.communicate()
            # A server that never binds is a failure. A server the machine
            # never scheduled is a fact about the machine, and calling it a
            # failure would train everyone to ignore this test. The two are
            # told apart by whether the process is still alive and what the
            # machine was doing, and the skip says so out loud.
            if alive and load > BUSY:
                pytest.skip(f"the machine was too loaded to time a server start (load {load:.0f}); "
                            "run this again when it is quiet")
            raise AssertionError(f"serve did not become ready within 240 s (load {load:.0f}); "
                                 "stderr: " + stderr[-2000:])
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
        frame = await sock.recv()
        assert isinstance(frame, bytes), "voice replies must use binary audio frames"
        turn, spoken = parse_frame(frame)
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
    # A single-key loopback launch bootstraps the browser connection. This
    # does not grant unauthenticated API access or trust arbitrary Host values.
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as browser:
        page = browser.get("/memory")
        assert page.status_code == 200 and KEY in page.text
        assert "__SCONE_TOKEN__" not in page.text
        assert page.headers["cache-control"] == "no-store"
        assert browser.get("/v1/status").status_code == 401
        untrusted = browser.get("/memory", headers={"Host": "untrusted.example"})
        assert untrusted.status_code == 200 and KEY not in untrusted.text
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
