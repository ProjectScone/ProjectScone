"""Host-owned answer instructions and deterministic publication constraints."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LINE_BREAK = re.compile(r'\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]')


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError('nonstandard JSON constant')


class AnswerRequirements(BaseModel):
    """Instructions guide models; accepts() checks only format, bytes and lines.

    JSON mode requires one object, without fences, duplicate keys or nonstandard
    constants. It does not validate an application schema or factual accuracy.
    Nothing is stripped, truncated or repaired. A trailing line break counts as
    another line. Consumers must withhold rejected output from their observers.
    """

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    instructions: str = Field(default='', max_length=8000)
    max_bytes: int = Field(default=64000, ge=1, le=128000)
    max_lines: int | None = Field(default=None, ge=1, le=1000)
    format: Literal['text', 'json_object'] = 'text'

    @field_validator('instructions')
    @classmethod
    def bounded_instructions(cls, value: str) -> str:
        if len(value.encode('utf-8')) > 8000:
            raise ValueError('answer instructions exceed UTF-8 byte limit')
        return value

    def accepts(self, answer: str) -> bool:
        if type(answer) is not str or not answer.strip():
            return False
        try:
            if len(answer.encode('utf-8')) > self.max_bytes:
                return False
            if self.max_lines is not None and len(_LINE_BREAK.findall(answer)) + 1 > self.max_lines:
                return False
            if self.format == 'json_object':
                # Preserve JSON number syntax without imposing Python integer or
                # float range limits: this gate checks syntax, not numeric values.
                value = json.loads(answer, object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant, parse_int=str, parse_float=str)
                return isinstance(value, dict)
        except (ValueError, RecursionError):
            return False
        return True

    def prompt(self) -> str:
        """The same explicit host constraints supplied to generation and review."""
        return 'Answer requirements: ' + json.dumps(self.model_dump(mode='json'),
            ensure_ascii=False, separators=(',', ':'))


def validated_requirements(value: AnswerRequirements | None) -> AnswerRequirements | None:
    """Snapshot even Pydantic instances that bypassed construction validation."""
    if value is None:
        return None
    if not isinstance(value, AnswerRequirements):
        raise ValueError('requirements must be AnswerRequirements')
    return AnswerRequirements.model_validate(dict(vars(value)), strict=True)
