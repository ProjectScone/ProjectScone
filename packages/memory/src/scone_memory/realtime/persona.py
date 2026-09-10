"""Serializable persona intent, separate from provider credentials and authority."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

ProviderId = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")]
ModelId = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True,
                              hide_input_in_errors=True, revalidate_instances="always")


class ModelChoice(_Config):
    provider: ProviderId
    model: ModelId


class VoiceChoice(ModelChoice):
    voice: ModelId


class Persona(_Config):
    """A named instruction set with independent reply/STT/TTS/VAD selections.

    Save with model_dump_json; load with model_validate_json. This document does
    not grant memory access, select an endpoint, import code, carry credentials,
    or start a provider. A host must admit each exact choice through its registry.
    None activity uses the recognizer's speech events without a local detector.
    """

    schema_version: Literal[1] = 1
    id: Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    name: str = Field(min_length=1, max_length=120)
    instructions: str = Field(min_length=1, max_length=16000)
    reply: ModelChoice
    transcription: ModelChoice
    speech: VoiceChoice
    activity: ModelChoice | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("name", "instructions")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank")
        return value
