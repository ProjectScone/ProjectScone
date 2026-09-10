"""Persona choices drive native sessions, never dynamic imports or credentials."""

import importlib.util
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
from ..media.test_voice import Transport, Recognizer, Model, Synthesizer, Resource


def api():
    assert importlib.util.find_spec("scone_memory.realtime.persona") is not None, "Native personas are not implemented"
    from scone_memory.realtime.persona import Persona
    from scone_memory.realtime.providers import ProviderRegistry
    return Persona, ProviderRegistry


def document():
    return {
        "schema_version": 1, "id": "juniper", "name": "Juniper",
        "instructions": "Be concise. Explain the source behind each answer.",
        "reply": {"provider": "local", "model": "reply-v1"},
        "transcription": {"provider": "transcriber", "model": "speech-v1"},
        "speech": {"provider": "voice-a", "model": "tts-v1", "voice": "alto"},
        "activity": {"provider": "detector", "model": "vad-v1"},
    }


def test_persona_json_roundtrip_preserves_choices_and_literal_instructions():
    Persona, _ = api()
    data = document()
    data["instructions"] = "Use **clear** answers.\nKeep `code` exact."
    persona = Persona.model_validate_json(json.dumps(data))
    assert json.loads(persona.model_dump_json()) == data
    with pytest.raises(ValueError):
        persona.speech.voice = "changed"


@pytest.mark.parametrize("field,value", [
    ("api_key", "secret-canary"), ("endpoint", "https://unexpected.invalid"),
    ("factory", "os:system"), ("where", {"space": "other"}),
    ("schema_version", True), ("schema_version", 2),
    ("instructions", "   "), ("instructions", "x" * 16001),
    ("id", "../private"),
])
def test_persona_rejects_authority_fields_and_invalid_identity(field, value):
    Persona, _ = api()
    data = document(); data[field] = value
    with pytest.raises(ValueError) as exc:
        Persona.model_validate(data)
    assert "secret-canary" not in str(exc.value)


@pytest.mark.parametrize("stage", ["reply", "transcription", "speech", "activity"])
def test_nested_provider_credentials_and_endpoints_are_not_persona_fields(stage):
    Persona, _ = api()
    data = document(); data[stage]["api_key"] = "secret-canary"
    with pytest.raises(ValueError) as exc:
        Persona.model_validate(data)
    assert "secret-canary" not in str(exc.value)


def test_every_selection_resolves_before_any_resource_is_created():
    Persona, Registry = api()
    calls = []
    def create():
        calls.append("created")
        return Resource()
    registry = Registry(reply={("local", "reply-v1"): create},
        transcription={("transcriber", "speech-v1"): create},
        speech={("voice-a", "tts-v1", "alto"): create}, activity={})
    with pytest.raises(ValueError, match="activity"):
        registry.resolve(Persona.model_validate(document()))
    assert calls == []


@pytest.mark.parametrize("stage,field,value", [
    ("reply", "model", "unapproved"), ("transcription", "provider", "unknown"),
    ("speech", "voice", "unapproved"), ("activity", "model", "unknown"),
])
def test_unknown_choices_never_fall_back_to_a_registered_provider(stage, field, value):
    Persona, Registry = api()
    calls = []
    def create(): calls.append("created")
    registry = Registry(reply={("local", "reply-v1"): create},
        transcription={("transcriber", "speech-v1"): create},
        speech={("voice-a", "tts-v1", "alto"): create},
        activity={("detector", "vad-v1"): create})
    data = document(); data[stage][field] = value
    with pytest.raises(ValueError, match=stage):
        registry.resolve(Persona.model_validate(data))
    assert calls == []


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    return engine


async def test_switching_speech_provider_keeps_persona_and_authorized_memory_scope(memory):
    Persona, Registry = api()
    await memory.remember_many("personas", [
        Record("Juniper points to Polaris.", metadata={"collection":"manual"}),
        Record("Juniper private points to Mars.", metadata={"collection":"private"}),
    ])
    observed = []
    class Detector(Resource):
        async def detect(self, audio): return False
    def model():
        instance = Model(); observed.append(instance); return instance
    spoken = []
    class Voice(Synthesizer):
        def __init__(self, name): super().__init__(); self.name = name
        async def synthesize(self, text):
            spoken.append((self.name, text))
            async for packet in super().synthesize(text): yield packet
    registry = Registry(reply={("local", "reply-v1"): model},
        transcription={("transcriber", "speech-v1"): Recognizer},
        speech={("voice-a", "tts-v1", "alto"): lambda: Voice("alto"),
                ("voice-b", "tts-v2", "tenor"): lambda: Voice("tenor")},
        activity={("detector", "vad-v1"): Detector})
    for index, choice in enumerate([document()["speech"], {"provider":"voice-b", "model":"tts-v2", "voice":"tenor"}]):
        data = document(); data["speech"] = choice
        bound = registry.resolve(Persona.model_validate(data))
        transport = Transport()
        session = bound.voice(memory, "personas", f"session-{index}",
            transport_factory=lambda: transport, capture=True, where={"collection": "manual"},
            session_timeout=3)
        from scone_memory.realtime.audio import AudioChunk
        await transport.input.put(AudioChunk(b"\0\0" * 320, 16000))
        await transport.input.put(None)
        await session.run()
        context = str(observed[-1].contexts[0])
        assert "Juniper points to Polaris." in context
        assert "Juniper private points to Mars." not in context
        saved = await memory.episodes("personas", {"session_id":f"session-{index}"})
        assert [episode.metadata["role"] for episode in saved] == ["user", "assistant"]
    assert [name for name, text in spoken] == ["alto", "tenor"]
    assert all(model.contexts[0][0] == {"role":"system", "content":document()["instructions"]} for model in observed)


async def test_bound_text_uses_fresh_selected_models_and_never_starts_audio(memory):
    Persona, Registry = api()
    models = []
    def model():
        instance = Model(); models.append(instance); return instance
    def audio(): raise AssertionError("text must not open audio providers")
    reply = {("local", "reply-v1"): model}
    registry = Registry(reply=reply, transcription={("transcriber", "speech-v1"):audio},
        speech={("voice-a", "tts-v1", "alto"):audio}, activity={("detector", "vad-v1"):audio})
    bound = registry.resolve(Persona.model_validate(document()))
    reply.clear()  # caller mutation cannot rewrite an admitted binding
    session = bound.text(memory, "personas", "text-persona")
    await session.reply("First question")
    await session.reply("Second question")
    await session.close()
    assert len(models) == 2 and models[0] is not models[1]
    assert all(model.closes == 1 for model in models)
    assert models[1].contexts[0][0]["content"] == document()["instructions"]


def test_persona_cannot_override_host_factories_via_session_options():
    Persona, Registry = api()
    data = document(); data["activity"] = None
    registry = Registry(reply={("local", "reply-v1"):Model},
        transcription={("transcriber", "speech-v1"):Recognizer},
        speech={("voice-a", "tts-v1", "alto"):Synthesizer})
    bound = registry.resolve(Persona.model_validate(data))
    with pytest.raises(TypeError):
        bound.text(None, "personas", "test", system_prompt="silently changed")


def test_resolution_revalidates_a_caller_constructed_persona():
    Persona, Registry = api()
    persona = Persona.model_validate(document()).model_copy(update={"instructions":""})
    registry = Registry(reply={}, transcription={}, speech={})
    with pytest.raises(ValueError, match="instructions"):
        registry.resolve(persona)


@pytest.mark.parametrize("key,factory", [("local", Model), (("local",), Model), (("local","reply-v1"), None)])
def test_registry_rejects_invalid_registration_before_binding(key, factory):
    _, Registry = api()
    with pytest.raises(ValueError, match="registration"):
        Registry(reply={key:factory}, transcription={}, speech={})
