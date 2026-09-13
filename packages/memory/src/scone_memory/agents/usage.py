"""Provider-reported token counts; missing observations are never estimates."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TokenField = Literal['prompt_tokens', 'completion_tokens', 'total_tokens']


def _tokens(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10**9 else None


def _consistent(prompt: int | None, completion: int | None, total: int | None) -> bool:
    known = [count for count in (prompt, completion) if count is not None]
    return total is None or (total >= sum(known) and (len(known) < 2 or total == sum(known)))


class ModelTokenUsage(BaseModel):
    """One model response's optional counts, excluding arbitrary provider data."""

    model_config = ConfigDict(strict=True, frozen=True, extra='forbid', hide_input_in_errors=True)
    prompt_tokens: int | None = Field(default=None, ge=0, le=10**9)
    completion_tokens: int | None = Field(default=None, ge=0, le=10**9)
    total_tokens: int | None = Field(default=None, ge=0, le=10**9)

    @model_validator(mode='after')
    def consistent(self) -> Self:
        if not _consistent(self.prompt_tokens, self.completion_tokens, self.total_tokens):
            raise ValueError('inconsistent token total')
        return self

    @classmethod
    def from_provider(cls, value: object) -> Self:
        """Ignore invalid optional telemetry without discarding a valid answer."""
        if not isinstance(value, dict):
            return cls()
        prompt = _tokens(value.get('prompt_tokens'))
        completion = _tokens(value.get('completion_tokens'))
        total = _tokens(value.get('total_tokens'))
        if not _consistent(prompt, completion, total):
            total = None
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


class ToolTokenUsage(BaseModel):
    """Ordered accepted model-call reports and totals with complete coverage.

    Empty reports describe legacy results without accounting, not zero usage.
    The existing loop permits at most sixteen tool rounds and a final answer.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra='forbid', hide_input_in_errors=True)
    calls: tuple[ModelTokenUsage, ...] = Field(default=(), max_length=17)

    def _total(self, field: TokenField) -> int | None:
        if field == 'prompt_tokens':
            values = [call.prompt_tokens for call in self.calls]
        elif field == 'completion_tokens':
            values = [call.completion_tokens for call in self.calls]
        else:
            values = [call.total_tokens for call in self.calls]
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    @property
    def prompt_tokens(self) -> int | None:
        return self._total('prompt_tokens')

    @property
    def completion_tokens(self) -> int | None:
        return self._total('completion_tokens')

    @property
    def total_tokens(self) -> int | None:
        return self._total('total_tokens')
