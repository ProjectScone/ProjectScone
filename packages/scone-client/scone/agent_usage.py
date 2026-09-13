"""Immutable provider-reported counts; absent observations are never estimates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ._wire import integer, invalid, items, record


def _count(value: object) -> Optional[int]:
    return None if value is None else integer(value, 0, 10**9)


@dataclass(frozen=True)
class ModelTokenUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        prompt, completion, total = (_count(self.prompt_tokens), _count(self.completion_tokens),
                                     _count(self.total_tokens))
        known = [value for value in (prompt, completion) if value is not None]
        if total is not None and (total < sum(known) or (len(known) == 2 and total != sum(known))):
            raise invalid('token usage consistency')

    @classmethod
    def from_json(cls, value: object) -> ModelTokenUsage:
        row = record(value)
        if set(row) != {'prompt_tokens', 'completion_tokens', 'total_tokens'}:
            raise invalid('token usage fields')
        return cls(_count(row['prompt_tokens']), _count(row['completion_tokens']), _count(row['total_tokens']))


@dataclass(frozen=True)
class ToolTokenUsage:
    calls: tuple[ModelTokenUsage, ...] = ()

    def __post_init__(self) -> None:
        if type(self.calls) is not tuple or len(self.calls) > 17 or any(
                not isinstance(call, ModelTokenUsage) for call in self.calls):
            raise invalid('token usage calls')

    @classmethod
    def from_json(cls, value: object, *, model_calls: int) -> ToolTokenUsage:
        integer(model_calls, 1, 17)
        row = record(value)
        if set(row) != {'calls'}:
            raise invalid('token usage fields')
        calls = tuple(ModelTokenUsage.from_json(raw) for raw in items(row['calls'], 17))
        if len(calls) != model_calls:
            raise invalid('token usage call count')
        return cls(calls)

    def _total(self, category: Literal['prompt', 'completion', 'total']) -> Optional[int]:
        if category == 'prompt':
            values = [call.prompt_tokens for call in self.calls]
        elif category == 'completion':
            values = [call.completion_tokens for call in self.calls]
        else:
            values = [call.total_tokens for call in self.calls]
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    @property
    def prompt_tokens(self) -> Optional[int]:
        return self._total('prompt')

    @property
    def completion_tokens(self) -> Optional[int]:
        return self._total('completion')

    @property
    def total_tokens(self) -> Optional[int]:
        return self._total('total')
