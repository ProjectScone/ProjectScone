"""A saved persona catalog: the operator's file, bound once through the registry.

The file is intent only (see persona.py); the registry is authority. Binding
every persona at startup means a host either serves its whole catalog or
refuses to start, so a client never learns about a choice that cannot run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from pydantic import ValidationError

from .persona import Persona
from .providers import BoundPersona, ProviderRegistry


def load_personas(path) -> list[Persona]:
    """A JSON array of Persona documents, in file order. Errors name a
    document by its id or position and the offending fields, never its text."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"persona catalog is not readable JSON ({type(error).__name__})") from None
    if not isinstance(raw, list):
        raise ValueError("persona catalog must be a JSON array of persona documents")
    personas, seen = [], set()
    for position, document in enumerate(raw):
        label = document["id"] if isinstance(document, dict) and isinstance(document.get("id"), str) else f"#{position}"
        try:
            persona = Persona.model_validate(document)
        except ValidationError as error:
            fields = sorted({".".join(str(part) for part in item["loc"]) or "document" for item in error.errors()})
            raise ValueError(f"persona {label} is invalid: {', '.join(fields)}") from None
        if persona.id in seen:
            raise ValueError(f"persona id is not unique: {persona.id}")
        seen.add(persona.id)
        personas.append(persona)
    return personas


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def fingerprint_of(persona: Persona) -> str:
    """Identity of a persona's configuration: its id, name and the four
    choices. Readiness and the registry behind the choices are not part of
    it, so a client can tell "the same choice" from "changed underneath"."""
    return _digest({"id": persona.id, "name": persona.name, "reply": persona.reply.model_dump(),
                    "transcription": persona.transcription.model_dump(), "speech": persona.speech.model_dump(),
                    "activity": persona.activity.model_dump() if persona.activity is not None else None})


@dataclass(frozen=True)
class PersonaCatalog:
    personas: tuple[Persona, ...]
    bound: Mapping[str, BoundPersona]

    @property
    def revision(self) -> str:
        """Identity of the whole catalog, in order; changes when any entry does."""
        return _digest([fingerprint_of(persona) for persona in self.personas])

    def get(self, persona_id: str) -> BoundPersona | None:
        return self.bound.get(persona_id)

    def fingerprint(self, persona_id: str) -> str | None:
        found = self.bound.get(persona_id)
        return fingerprint_of(found.persona) if found is not None else None

    def name(self, persona_id: str) -> str | None:
        found = self.bound.get(persona_id)
        return found.persona.name if found is not None else None

    def public(self, *, voice: bool = False) -> list[dict]:
        """What a client may see: the choices and their readiness. Never the
        instructions, and nothing the registry closed over. Every persona
        here is text-ready because binding admitted it; the host says
        whether it can carry voice (it has an audio transport route)."""
        return [{
            "id": persona.id, "name": persona.name,
            "reply": persona.reply.model_dump(), "transcription": persona.transcription.model_dump(),
            "speech": persona.speech.model_dump(),
            "activity": persona.activity.model_dump() if persona.activity is not None else None,
            "fingerprint": fingerprint_of(persona),
            "text_ready": True, "voice_ready": voice,
        } for persona in self.personas]


def bind_catalog(personas: Iterable[Persona], registry: ProviderRegistry) -> PersonaCatalog:
    """Every persona admitted by the registry, or a ValueError naming the
    first that is not and the stage that refused it."""
    personas = tuple(personas)
    bound = {}
    for persona in personas:
        try:
            bound[persona.id] = registry.resolve(persona)
        except ValueError as error:
            raise ValueError(f"persona {persona.id}: {error}") from None
    return PersonaCatalog(personas, MappingProxyType(bound))
