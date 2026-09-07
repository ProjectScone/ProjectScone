"""Operator-owned admission of exact provider/model/voice choices.

Factories close over credentials and provider configuration outside persona JSON.
They must be synchronous and return fresh resources, as required by native sessions.
Registering an ID does not install a provider or prove its remote availability.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .audio import AudioTransport, SpeechActivityDetector, SpeechRecognizer, SpeechSynthesizer
from .events import TextModel
from .persona import Persona

ModelKey = tuple[str, str]
VoiceKey = tuple[str, str, str]


@dataclass(frozen=True)
class BoundPersona:
    persona: Persona
    model_factory: Callable[[], TextModel]
    stt_factory: Callable[[], SpeechRecognizer]
    tts_factory: Callable[[], SpeechSynthesizer]
    activity_factory: Callable[[], SpeechActivityDetector] | None

    def text(self, memory, space: str, session_id: str, **options):
        """Create a native text session; audio choices create no resources."""
        from .text import TextConversation
        return TextConversation(memory, space, session_id, self.model_factory,
                                system_prompt=self.persona.instructions, **options)

    def voice(self, memory, space: str, session_id: str, *,
              transport_factory: Callable[[], AudioTransport], capture: bool, **options):
        """Create a native voice session; host still owns consent and recall scope."""
        from .voice import VoiceSession
        return VoiceSession(memory, space, session_id, transport_factory=transport_factory,
                            stt_factory=self.stt_factory, model_factory=self.model_factory,
                            tts_factory=self.tts_factory, activity_factory=self.activity_factory,
                            capture=capture, system_prompt=self.persona.instructions, **options)


class ProviderRegistry:
    """Immutable snapshots of host-authorized choices; no implicit fallback.

    Models are keyed by (provider, model); synthesis adds a voice ID. A registry
    should be scoped by the host to the requesting user's allowed providers.
    Resolve checks all choices before constructing any resource, even transport.
    Availability, credentials and PCM compatibility are the adapters' contract.
    """

    def __init__(self, *, reply: Mapping[ModelKey, Callable[[], TextModel]],
                 transcription: Mapping[ModelKey, Callable[[], SpeechRecognizer]],
                 speech: Mapping[VoiceKey, Callable[[], SpeechSynthesizer]],
                 activity: Mapping[ModelKey, Callable[[], SpeechActivityDetector]] | None = None):
        self._choices = {}
        for stage, choices, size in (("reply", reply, 2), ("transcription", transcription, 2),
                                      ("speech", speech, 3), ("activity", activity or {}, 2)):
            copied = dict(choices)
            if any(type(key) is not tuple or len(key) != size
                   or any(not isinstance(part, str) or not part for part in key)
                   or not callable(factory) for key, factory in copied.items()):
                raise ValueError(f"invalid {stage} provider registration")
            self._choices[stage] = MappingProxyType(copied)

    def resolve(self, persona: Persona) -> BoundPersona:
        # Revalidate even a caller-created model_copy/model_construct instance.
        persona = Persona.model_validate(persona)
        factories = []
        for stage in ("reply", "transcription", "speech", "activity"):
            choice = getattr(persona, stage)
            if choice is None:
                factories.append(None)
                continue
            key = (choice.provider, choice.model)
            if stage == "speech":
                key += (choice.voice,)
            try:
                factory = self._choices[stage][key]
            except KeyError:
                raise ValueError(f"{stage} provider/model/voice choice is not registered") from None
            factories.append(factory)
        return BoundPersona(persona, *factories)
