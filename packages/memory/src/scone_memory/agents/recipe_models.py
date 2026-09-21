"""Bounded declarative tools authored as data, never executable source."""
from __future__ import annotations

import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .custom_tools import _encoded, _RESERVED

RecipeName = Annotated[str, Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,63}$')]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]


class RecipeModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)


class RecipeInput(RecipeModel):
    name: RecipeName
    kind: Literal['string', 'integer', 'number', 'boolean']


class RecipeLiteral(RecipeModel):
    kind: Literal['literal']
    value_json: str = Field(max_length=4096)

    @model_validator(mode='after')
    def bounded(self) -> Self:
        _encoded(json.loads(self.value_json), 4096)
        return self


class RecipeReference(RecipeModel):
    kind: Literal['input', 'step']
    name: RecipeName
    path: tuple[Annotated[str, Field(max_length=128)] | Annotated[int, Field(ge=0, le=4095)], ...] = Field(default=(), max_length=8)

    @model_validator(mode='after')
    def scalar_input(self) -> Self:
        if self.kind == 'input' and self.path:
            raise ValueError('recipe inputs are scalars')
        return self


RecipeValue = Annotated[RecipeLiteral | RecipeReference, Field(discriminator='kind')]


class RecipeStep(RecipeModel):
    name: RecipeName
    tool: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    digest: Digest
    arguments: dict[RecipeName, RecipeValue] = Field(max_length=32)


class ToolRecipe(RecipeModel):
    name: RecipeName
    description: str = Field(min_length=1, max_length=1000)
    requirements: str = Field(min_length=1, max_length=4000)
    inputs: tuple[RecipeInput, ...] = Field(max_length=32)
    steps: tuple[RecipeStep, ...] = Field(min_length=1, max_length=8)
    result: RecipeValue

    @model_validator(mode='after')
    def valid(self) -> Self:
        if self.name in _RESERVED | {'recipe_capabilities', 'propose_tool_recipe',
                                    'inspect_tool_recipe', 'invoke_tool_recipe'}:
            raise ValueError('reserved recipe name')
        if not self.description.strip() or not self.requirements.strip():
            raise ValueError('recipe purpose required')
        inputs = {item.name for item in self.inputs}
        if len(inputs) != len(self.inputs):
            raise ValueError('duplicate recipe inputs')
        prior: set[str] = set()

        def reference(value: RecipeValue) -> None:
            if isinstance(value, RecipeReference):
                names = inputs if value.kind == 'input' else prior
                if value.name not in names:
                    raise ValueError('unknown or forward recipe reference')

        for step in self.steps:
            if step.name in prior or step.tool == self.name:
                raise ValueError('duplicate or recursive recipe step')
            for value in step.arguments.values():
                reference(value)
            prior.add(step.name)
        reference(self.result)
        if len(self.model_dump_json().encode()) > 16000:
            raise ValueError('recipe byte limit')
        return self

    def parameters(self) -> dict[str, object]:
        return {'type': 'object', 'properties': {item.name: {'type': item.kind} for item in self.inputs},
                'required': [item.name for item in self.inputs], 'additionalProperties': False}
