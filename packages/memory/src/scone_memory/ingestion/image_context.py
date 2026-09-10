"""Source assertions associated with an image, distinct from visual recognition."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ImageAttribute(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    kind: Literal['alt', 'title', 'caption', 'description', 'surrounding_text', 'filename', 'metadata']
    value: str = Field(min_length=1, max_length=16_000)
    origin: Literal['supplied', 'html', 'sidecar', 'embedded', 'model_generated'] = 'supplied'
    name: str | None = Field(default=None, min_length=1, max_length=128)
    locator: str | None = Field(default=None, min_length=1, max_length=1024)

    @field_validator('value')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or '\x00' in value:
            raise ValueError('image attribute must contain nonempty text without NUL')
        return value


class ImageEntity(BaseModel):
    """Caller-supplied association, supported by the occurrence's attributes."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    entity_id: str = Field(min_length=1, max_length=256, pattern=r'^[^\x00-\x20\x7f]+$')
    name: str = Field(min_length=1, max_length=256)
    aliases: tuple[str, ...] = Field(default=(), max_length=16)
    relationship: Literal['depicts', 'mentions', 'associated_with']
    attribute_indexes: tuple[int, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode='after')
    def valid_labels(self) -> ImageEntity:
        if any(not item.strip() or '\x00' in item or len(item) > 256 for item in (self.name, *self.aliases)):
            raise ValueError('entity names and aliases must contain bounded nonempty text')
        return self


class ImageContext(BaseModel):
    """One occurrence. Source and locator are identifiers, never fetch targets."""
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    source: str = Field(min_length=1, max_length=4096)
    locator: str | None = Field(default=None, min_length=1, max_length=1024)
    source_sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    attributes: tuple[ImageAttribute, ...] = Field(min_length=1, max_length=64)
    entities: tuple[ImageEntity, ...] = Field(default=(), max_length=32)

    @model_validator(mode='after')
    def valid_context(self) -> ImageContext:
        if not self.source.strip() or '\x00' in self.source:
            raise ValueError('image source must contain nonempty text without NUL')
        if len({entity.entity_id for entity in self.entities}) != len(self.entities):
            raise ValueError('entity IDs must be unique within an image occurrence')
        for entity in self.entities:
            if any(index < 0 or index >= len(self.attributes) for index in entity.attribute_indexes):
                raise ValueError('entity evidence must refer to existing image attributes')
        if len(self.model_dump_json().encode()) > 128_000:
            raise ValueError('image context exceeds its byte limit')
        return self


def context_text(context: ImageContext) -> str:
    lines = ['Image context (source assertions)']
    for attribute in context.attributes:
        label = attribute.kind if attribute.name is None else f'{attribute.kind} ({attribute.name})'
        lines.append(f'{label}: {attribute.value}')
    for entity in context.entities:
        labels = ' / '.join((entity.name, *entity.aliases))
        lines.append(f'Entity [{entity.relationship}]: {labels}')
    return '\n'.join(lines)
