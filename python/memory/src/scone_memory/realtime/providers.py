"""Operator-owned admission of exact provider/model/voice choices.

Factories close over credentials and provider configuration outside persona JSON.
They must be synchronous and return fresh resources, as required by native sessions.
Registering an ID does not install a provider or prove its remote availability.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

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
    The historical local-openai ID and self-hosted-openai are aliases of the
    same exact registered model/voice; explicit registrations take precedence.
    """

    def __init__(self, *, reply: Mapping[ModelKey, Callable[[], TextModel]],
                 transcription: Mapping[ModelKey, Callable[[], SpeechRecognizer]],
                 speech: Mapping[VoiceKey, Callable[[], SpeechSynthesizer]],
                 activity: Mapping[ModelKey, Callable[[], SpeechActivityDetector]] | None = None):
        self._choices: dict[str, Mapping[tuple[str, ...], Callable[[], object]]] = {}
        for stage, choices, size in (("reply", reply, 2), ("transcription", transcription, 2),
                                      ("speech", speech, 3), ("activity", activity or {}, 2)):
            copied: dict[tuple[str, ...], Callable[[], object]] = {key: factory for key, factory in choices.items()}
            if any(type(key) is not tuple or len(key) != size
                   or any(not isinstance(part, str) or not part for part in key)
                   or not callable(factory) for key, factory in copied.items()):
                raise ValueError(f"invalid {stage} provider registration")
            self._choices[stage] = MappingProxyType(copied)

    def resolve(self, persona: Persona) -> BoundPersona:
        # Revalidate even a caller-created model_copy/model_construct instance.
        persona = Persona.model_validate(persona)
        factories: list[Callable[[], object] | None] = []
        for stage in ("reply", "transcription", "speech", "activity"):
            choice = getattr(persona, stage)
            if choice is None:
                factories.append(None)
                continue
            key: tuple[str, ...] = (choice.provider, choice.model)
            if stage == "speech":
                key += (choice.voice,)
            # The deployment rename aliases the same exact registered model
            # and voice. An explicit registration always takes precedence.
            if key not in self._choices[stage]:
                alias = {'local-openai': 'self-hosted-openai', 'self-hosted-openai': 'local-openai'}.get(key[0])
                if alias is not None:
                    key = (alias, *key[1:])
            try:
                factory = self._choices[stage][key]
            except KeyError:
                raise ValueError(f"{stage} provider/model/voice choice is not registered") from None
            factories.append(factory)
        return BoundPersona(persona,
            cast(Callable[[], TextModel], factories[0]),
            cast(Callable[[], SpeechRecognizer], factories[1]),
            cast(Callable[[], SpeechSynthesizer], factories[2]),
            cast(Callable[[], SpeechActivityDetector] | None, factories[3]))
