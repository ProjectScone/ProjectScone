"""A voice session over the service: created with a persona, spoken to over
the audio socket, and accounted for in the journal like any other session."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.audio import AudioChunk, ReplyCompleted, SpeechStarted, TextDelta, Transcript
from scone_memory.realtime.catalog import bind_catalog
from scone_memory.realtime.persona import Persona
from scone_memory.realtime.providers import ProviderRegistry
from scone_memory.realtime.websocket import parse_frame

HELPER = {"schema_version": 1, "id": "helper", "name": "Helper", "instructions": "Answer briefly.",
          "reply": {"provider": "stub", "model": "echo"}, "transcription": {"provider": "stub", "model": "ears"},
          "speech": {"provider": "stub", "model": "mouth", "voice": "alto"}}
KEYS = {"alpha-key": "alpha", "beta-key": "beta"}
AUTH = {"Authorization": "Bearer alpha-key"}
PCM = b"\x01\x00" * 160
HELLO = {"type": "hello", "key": "alpha-key", "sample_rate": 16000, "channels": 1}


class Recognizer:
    async def transcribe(self, audio):
        async for chunk in audio:
            yield SpeechStarted()
            yield Transcript("Where does Juniper point?")

    async def aclose(self):
        pass


class Model:
    seen: list = []

    async def respond(self, messages):
        Model.seen.append(messages)
        yield TextDelta("Juniper points to Polaris.")
        yield ReplyCompleted()

    async def aclose(self):
        pass


class Synthesizer:
    async def synthesize(self, text):
        yield AudioChunk(b"\x07\x00" * 320, 16000)

    async def aclose(self):
        pass


def catalog():
    registry = ProviderRegistry(reply={("stub", "echo"): Model}, transcription={("stub", "ears"): Recognizer},
                                speech={("stub", "mouth", "alto"): Synthesizer})
    return bind_catalog([Persona.model_validate(HELPER)], registry)


@pytest.fixture
def service(tmp_path):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    app = create_conversation_app(engine, KEYS, tmp_path / "j.db", None, catalog=catalog(), public_text_streaming=True)
    with TestClient(app) as client:
        client.headers.update(AUTH)
        yield client


def create_voice(client, request_id="v1", **extra):
    return client.post("/v1/conversations", json={"request_id": request_id, "capture": True, "persona": "helper",
                                                  "mode": "voice", **extra})

def test_voice_discovery_negotiates_the_browser_protocol(service):
    body = service.get('/v1/conversations/capabilities').json()
    assert body.get('voice_stream') == {
        'schema_version': 1, 'transport': 'websocket', 'protocol': 'scone-pcm-v1',
        'authentication': 'hello', 'reconnect': False, 'pcm': 's16le',
        'input_channels': [1, 2], 'min_sample_rate': 8000,
        'max_sample_rate': 192000, 'max_input_frame_bytes': 64000,
    }

def test_recreated_waiting_voice_cannot_attach_to_a_changed_persona(tmp_path):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    original = catalog()
    journal = tmp_path / 'waiting.db'
    app = create_conversation_app(engine, KEYS, journal, None, catalog=original)
    with TestClient(app) as client:
        client.headers.update(AUTH)
        receipt = create_voice(client).json()
    changed = Persona.model_validate({**HELPER, 'instructions': 'A different instruction set.'})
    registry = ProviderRegistry(reply={('stub', 'echo'): Model}, transcription={('stub', 'ears'): Recognizer},
                                speech={('stub', 'mouth', 'alto'): Synthesizer})
    newer = bind_catalog([changed], registry)
    Model.seen.clear()
    with TestClient(create_conversation_app(engine, KEYS, journal, None, catalog=newer)) as client:
        client.headers.update(AUTH)
        with client.websocket_connect(f"/v1/conversations/{receipt['session_id']}/audio") as socket:
            socket.send_text(json.dumps(HELLO))
            refusal = json.loads(socket.receive_text())
            assert refusal['type'] == 'error' and 'waiting' in refusal['reason']
        after = client.get(f"/v1/conversations/{receipt['session_id']}").json()
        assert after['state'] == 'interrupted' and after['revision'] > receipt['revision']
        assert Model.seen == []


def test_a_voice_session_needs_a_persona_and_waits_for_its_audio(service):
    ready = service.get("/v1/conversations/capabilities").json()
    assert ready["voice"] is True
    assert service.get("/v1/conversations/personas").json()["personas"][0]["voice_ready"] is True
    no_persona = service.post("/v1/conversations", json={"request_id": "x", "capture": True, "mode": "voice"})
    assert no_persona.status_code == 422 and "persona" in no_persona.text
    created = create_voice(service)
    assert created.status_code == 200
    receipt = created.json()
    assert receipt["mode"] == "voice" and receipt["state"] == "created" and receipt["persona"]["id"] == "helper"
    assert create_voice(service).json()["session_id"] == receipt["session_id"], "idempotent like any create"
    sid = receipt["session_id"]
    turn = service.post(f"/v1/conversations/{sid}/turns", json={"request_id": "t", "text": "hi", "expected_revision": receipt["revision"]})
    assert turn.status_code == 409 and "audio" in turn.text
    assert service.delete(f"/v1/conversations/{sid}").status_code == 204, "nothing serves it yet, so it can go"


@pytest.mark.parametrize("hello, reason", [
    ({**HELLO, "key": "wrong-key"}, "key"),
    ({**HELLO, "key": "beta-key"}, "session"),
    ({**HELLO, "sample_rate": 12}, "format"),
    ({"type": "end"}, "hello"),
])
def test_the_socket_refuses_a_bad_hello_before_any_audio(service, hello, reason):
    sid = create_voice(service).json()["session_id"]
    with service.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
        socket.send_text(json.dumps(hello))
        error = json.loads(socket.receive_text())
        assert error["type"] == "error" and reason in error["reason"] and "alpha-key" not in json.dumps(error)
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
        assert closed.value.code == 1008
    assert service.get(f"/v1/conversations/{sid}").json()["state"] == "created", "a refused socket changes nothing"


def test_a_text_session_has_no_audio_socket(service):
    sid = service.post("/v1/conversations", json={"request_id": "t1", "capture": True, "persona": "helper"}).json()["session_id"]
    with service.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
        socket.send_text(json.dumps(HELLO))
        assert "voice" in json.loads(socket.receive_text())["reason"]


def test_audio_runs_the_persona_and_the_journal_follows_the_session_to_its_end(service):
    Model.seen.clear()
    sid = create_voice(service).json()["session_id"]
    with service.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
        socket.send_text(json.dumps(HELLO))
        assert json.loads(socket.receive_text()) == {"type": "ready", "session_id": sid}
        assert service.get(f"/v1/conversations/{sid}").json()["state"] == "running"
        socket.send_bytes(PCM)
        turn, spoken = parse_frame(socket.receive_bytes())
        assert spoken.pcm == b"\x07\x00" * 320 and spoken.sample_rate == 16000
        socket.send_text(json.dumps({"type": "end"}))
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()
    after = service.get(f"/v1/conversations/{sid}").json()
    assert after["state"] == "ended" and after["persona"]["id"] == "helper"
    assert Model.seen[0][0] == {"role": "system", "content": "Answer briefly."}
    roles = [e["metadata"]["role"] for e in service.get(f"/v1/conversations/{sid}/transcript").json()["episodes"]]
    assert sorted(roles) == ["assistant", "user"]
    second = service.websocket_connect(f"/v1/conversations/{sid}/audio")
    with second as socket:
        socket.send_text(json.dumps(HELLO))
        assert "waiting" in json.loads(socket.receive_text())["reason"], "an ended session takes no more audio"


def test_stop_closes_a_running_voice_session_and_the_socket(service):
    sid = create_voice(service).json()["session_id"]
    with service.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
        socket.send_text(json.dumps(HELLO))
        assert json.loads(socket.receive_text())["type"] == "ready"
        running = service.get(f"/v1/conversations/{sid}").json()
        stopped = service.post(f"/v1/conversations/{sid}/stop", json={"request_id": "s", "expected_revision": running["revision"]})
        assert stopped.status_code == 200 and stopped.json()["state"] == "ended"
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()
    assert service.get(f"/v1/conversations/{sid}").json()["state"] == "ended"


def test_a_client_that_leaves_ends_its_session_and_a_failing_provider_fails_it(service, tmp_path):
    sid = create_voice(service).json()["session_id"]
    with service.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
        socket.send_text(json.dumps(HELLO))
        assert json.loads(socket.receive_text())["type"] == "ready"
        socket.close()  # the browser tab goes away; the input simply ends
        for _ in range(100):
            if service.get(f"/v1/conversations/{sid}").json()["state"] != "running":
                break
            time.sleep(0.02)
    assert service.get(f"/v1/conversations/{sid}").json()["state"] == "ended"

    sid = create_voice(service, request_id="v2").json()["session_id"]
    with service.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
        socket.send_text(json.dumps(HELLO))
        assert json.loads(socket.receive_text())["type"] == "ready"
        socket.send_bytes(b"\x01\x00\x02")  # an incomplete sample frame the format does not admit
        assert json.loads(socket.receive_text())["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            socket.receive_text()
    assert service.get(f"/v1/conversations/{sid}").json()["state"] == "failed"


def test_a_voice_session_needs_a_persona_even_where_text_does_not(tmp_path):
    """A bare text runtime serves unnamed text sessions; it has no ears or
    voice, so a voice session must still name a persona there."""
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())

    class Bare:
        async def reply(self, text):
            return {"text": "bare", "assistant_episode_id": None, "provider_completion": "unverified"}

        async def close(self):
            pass

    app = create_conversation_app(engine, KEYS, tmp_path / "bare.db", lambda space, sid: Bare(), catalog=catalog())
    with TestClient(app) as client:
        client.headers.update(AUTH)
        text = client.post("/v1/conversations", json={"request_id": "t", "capture": True})
        assert text.status_code == 200 and text.json()["persona"] is None
        voice = client.post("/v1/conversations", json={"request_id": "v", "capture": True, "mode": "voice"})
        assert voice.status_code == 422 and "persona" in voice.text
        assert [s["mode"] for s in client.get("/v1/conversations").json()["items"]] == ["text"]


class PausingRecognizer(Recognizer):
    """Hears one utterance as two final transcripts, as an energy gate
    does when the speaker pauses mid-clause for longer than its stop."""

    async def transcribe(self, audio):
        async for chunk in audio:
            yield SpeechStarted()
            yield Transcript("Tell me which star Juniper points to")
            yield SpeechStarted()
            yield Transcript("at night.")


@pytest.mark.parametrize("semantic", [False, True])
def test_semantic_turns_are_the_hosts_choice_and_hold_an_open_clause(tmp_path, semantic):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    registry = ProviderRegistry(reply={("stub", "echo"): Model}, transcription={("stub", "ears"): PausingRecognizer},
                                speech={("stub", "mouth", "alto"): Synthesizer})
    app = create_conversation_app(engine, KEYS, tmp_path / "turns.db", None, public_text_streaming=True,
                                  catalog=bind_catalog([Persona.model_validate(HELPER)], registry),
                                  semantic_turn=semantic)
    with TestClient(app) as client:
        client.headers.update(AUTH)
        assert client.get("/v1/conversations/capabilities").json()["voice_turn_end"] == ("semantic" if semantic else "silence")
        sid = create_voice(client).json()["session_id"]
        with client.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
            socket.send_text(json.dumps(HELLO))
            assert json.loads(socket.receive_text())["type"] == "ready"
            socket.send_bytes(PCM)
            socket.send_text(json.dumps({"type": "end"}))
            while socket.receive()["type"] != "websocket.close":
                pass  # spoken frames, and with silence alone a clear for the superseded reply
        episodes = client.get(f"/v1/conversations/{sid}/transcript").json()["episodes"]
    said = [(e["content"], e["metadata"]["turn_end"]) for e in episodes if e["metadata"]["role"] == "user"]
    if semantic:
        assert said == [("Tell me which star Juniper points to at night.", "semantic_complete")]
    else:
        assert sorted(said) == [("Tell me which star Juniper points to", "silence"), ("at night.", "silence")]


@pytest.mark.parametrize("keypad", ["off", "collect"])
def test_keys_from_the_client_are_the_hosts_choice_and_become_one_turn(tmp_path, keypad):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    app = create_conversation_app(engine, KEYS, tmp_path / "keys.db", None, public_text_streaming=True,
                                  catalog=catalog(), voice_keypad=keypad)
    with TestClient(app) as client:
        client.headers.update(AUTH)
        assert client.get("/v1/conversations/capabilities").json()["voice_keypad"] == keypad
        sid = create_voice(client).json()["session_id"]
        with client.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
            socket.send_text(json.dumps(HELLO))
            assert json.loads(socket.receive_text())["type"] == "ready"
            for key in "42#":
                socket.send_text(json.dumps({"type": "keypad", "key": key}))
            # Admitted keys and then the end would simply finish the input, so a
            # regression that took keys while off fails here instead of waiting.
            socket.send_text(json.dumps({"type": "end"}))
            if keypad == "off":
                assert json.loads(socket.receive_text()) == {"type": "error", "reason": "unsupported control message"}
            while socket.receive()["type"] != "websocket.close":
                pass
        state = client.get(f"/v1/conversations/{sid}").json()["state"]
        episodes = client.get(f"/v1/conversations/{sid}/transcript").json()["episodes"]
    users = [e for e in episodes if e["metadata"]["role"] == "user"]
    if keypad == "off":
        assert state == "failed" and users == []
    else:
        [user] = users
        assert (user["content"], user["metadata"]["turn_end"], user["metadata"]["keypad_ended"]) == \
            ("[keypad] 42#", "keypad", "terminator")
        assert user["metadata"]["keypad_sources"] == "eee"


def test_a_keypad_mode_the_service_does_not_know_is_refused(tmp_path):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with pytest.raises(ValueError, match="voice_keypad"):
        create_conversation_app(engine, KEYS, tmp_path / "keys.db", None, catalog=catalog(), voice_keypad="loud")


class QuietRecognizer:
    """Takes the audio and says nothing: the caller has gone quiet."""

    async def transcribe(self, audio):
        async for chunk in audio:
            if False:
                yield chunk

    async def aclose(self):
        pass


def test_idle_and_turn_strategy_are_the_hosts_choice_and_reported(tmp_path):
    from scone_memory.realtime.idle import IdlePolicy
    from scone_memory.realtime.turn_strategy import KeypadSubmit

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    off = create_conversation_app(engine, KEYS, tmp_path / "off.db", None, catalog=catalog())
    with TestClient(off) as client:
        client.headers.update(AUTH)
        capabilities = client.get("/v1/conversations/capabilities").json()
    assert (capabilities["voice_idle"], capabilities["voice_turn_strategy"]) == (None, "end_of_turn")
    on = create_conversation_app(engine, KEYS, tmp_path / "on.db", None, catalog=catalog(), voice_keypad="collect",
                                 voice_idle=IdlePolicy(8, prompt=None, end_after=2), voice_turn_strategy=KeypadSubmit())
    with TestClient(on) as client:
        client.headers.update(AUTH)
        capabilities = client.get("/v1/conversations/capabilities").json()
    assert capabilities["voice_idle"] == {"timeout_s": 8, "prompt": False, "end_after": 2}
    assert capabilities["voice_turn_strategy"] == "keypad_submit"


def test_a_served_session_whose_caller_goes_quiet_is_prompted_and_ended(tmp_path):
    from scone_memory.realtime.idle import IdlePolicy

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    registry = ProviderRegistry(reply={("stub", "echo"): Model}, transcription={("stub", "ears"): QuietRecognizer},
                                speech={("stub", "mouth", "alto"): Synthesizer})
    app = create_conversation_app(engine, KEYS, tmp_path / "idle.db", None, public_text_streaming=True,
                                  catalog=bind_catalog([Persona.model_validate(HELPER)], registry),
                                  voice_idle=IdlePolicy(.1, prompt="Still with me?", end_after=2))
    with TestClient(app) as client:
        client.headers.update(AUTH)
        sid = create_voice(client).json()["session_id"]
        with client.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
            socket.send_text(json.dumps(HELLO))
            assert json.loads(socket.receive_text())["type"] == "ready"
            socket.send_bytes(PCM)
            frames = []
            while (message := socket.receive())["type"] != "websocket.close":
                frames.append(message)
        state = client.get(f"/v1/conversations/{sid}").json()["state"]
        episodes = client.get(f"/v1/conversations/{sid}/transcript").json()["episodes"]
    assert state == "ended", "the idle ended the session; the caller did not fail it"
    assert any(frame.get("bytes") for frame in frames), "the prompt was spoken to the caller"
    [said] = episodes
    assert (said["content"], said["metadata"]["idle_action"]) == ("Still with me?", "prompt")


@pytest.mark.parametrize("strategy", [None, "min_speech"])
def test_a_served_session_takes_its_turn_when_the_host_s_strategy_says(tmp_path, strategy):
    from scone_memory.realtime.turn_strategy import MinSpeech

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    app = create_conversation_app(engine, KEYS, tmp_path / "strategy.db", None, public_text_streaming=True,
                                  catalog=catalog(), voice_turn_strategy=MinSpeech(5) if strategy else None)
    with TestClient(app) as client:
        client.headers.update(AUTH)
        sid = create_voice(client).json()["session_id"]
        with client.websocket_connect(f"/v1/conversations/{sid}/audio") as socket:
            socket.send_text(json.dumps(HELLO))
            assert json.loads(socket.receive_text())["type"] == "ready"
            socket.send_bytes(PCM)
            socket.send_text(json.dumps({"type": "end"}))
            while socket.receive()["type"] != "websocket.close":
                pass
        episodes = client.get(f"/v1/conversations/{sid}/transcript").json()["episodes"]
    [user] = [e for e in episodes if e["metadata"]["role"] == "user"]
    # Too short for five seconds of speech, the turn waited for more and was answered when input ended.
    assert user["metadata"]["turn_end"] == ("input_ended" if strategy else "silence")


@pytest.mark.parametrize("options, match", [
    ({"voice_idle": 8}, "voice_idle"),
    ({"voice_turn_strategy": "min_speech"}, "voice_turn_strategy"),
    ({"voice_turn_strategy": object()}, "voice_turn_strategy"),
    ({"voice_turn_strategy": type("Undecided", (), {"name": "undecided", "submit": None})()}, "voice_turn_strategy"),
    ({"voice_turn_strategy": type("Nameless", (), {"decide": lambda *a: None})()}, "voice_turn_strategy"),
])
def test_an_idle_policy_or_turn_strategy_the_service_cannot_use_is_refused(tmp_path, options, match):
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with pytest.raises(ValueError, match=match):
        create_conversation_app(engine, KEYS, tmp_path / "bad.db", None, catalog=catalog(), **options)


def test_a_submit_key_strategy_needs_the_keypad_on(tmp_path):
    from scone_memory.realtime.turn_strategy import KeypadSubmit

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with pytest.raises(ValueError, match="voice_keypad"):
        create_conversation_app(engine, KEYS, tmp_path / "bad.db", None, catalog=catalog(),
                                voice_turn_strategy=KeypadSubmit())
