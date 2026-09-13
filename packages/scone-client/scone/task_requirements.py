"""Bounded authored task contracts; execution schema validation belongs to the host."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Literal, Mapping, Optional

from ._wire import integer, invalid, record


def _schema_snapshot(value: object) -> Mapping[str, object]:
    frozen_input = isinstance(value, MappingProxyType)
    if type(value) is not dict and not frozen_input:
        raise invalid('output schema object')
    nodes = 0
    size = 0

    def charge(amount: int) -> None:
        nonlocal size
        size += amount
        if size > 32768:
            raise invalid('output schema byte limit')

    def scalar(item: object) -> None:
        if isinstance(item, str) and len(item) > 32768:
            raise invalid('output schema byte limit')
        try:
            encoded = json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError, OverflowError):
            raise invalid('output schema JSON') from None
        charge(len(encoded))

    def visit(item: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > 4096:
            raise invalid('output schema structure limit')
        if isinstance(item, Mapping) and (type(item) is dict or frozen_input and isinstance(item, MappingProxyType)):
            if len(item) > 4096 - nodes:
                raise invalid('output schema structure limit')
            charge(2 + max(0, len(item) - 1))
            result: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise invalid('output schema key')
                scalar(key)
                charge(1)
                result[key] = visit(child, depth + 1)
            return MappingProxyType(result)
        if type(item) is list or frozen_input and type(item) is tuple:
            assert isinstance(item, (list, tuple))
            if len(item) > 4096 - nodes:
                raise invalid('output schema structure limit')
            charge(2 + max(0, len(item) - 1))
            return tuple(visit(child, depth + 1) for child in item)
        if type(item) not in (str, bool, int, float, type(None)):
            raise invalid('output schema JSON')
        scalar(item)
        return item

    snapshot = visit(value, 0)
    assert isinstance(snapshot, Mapping)
    return snapshot


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class TaskAnswerRequirements:
    instructions: str = ''
    max_bytes: int = 64000
    max_lines: Optional[int] = None
    format: Literal['text', 'json_object'] = 'text'
    output_schema: Optional[Mapping[str, object]] = field(default=None, compare=False)
    _schema_identity: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, str) or len(self.instructions) > 8000:
            raise invalid('answer instructions')
        try:
            if len(self.instructions.encode('utf-8')) > 8000:
                raise invalid('answer instructions byte limit')
        except UnicodeError:
            raise invalid('answer instructions UTF-8') from None
        integer(self.max_bytes, 1, 128000)
        if self.max_lines is not None:
            integer(self.max_lines, 1, 1000)
        if self.format not in ('text', 'json_object'):
            raise invalid('answer format')
        if self.output_schema is not None:
            if self.format != 'json_object':
                raise invalid('output schema requires json_object')
            schema = _schema_snapshot(self.output_schema)
            object.__setattr__(self, 'output_schema', schema)
            object.__setattr__(self, '_schema_identity', json.dumps(_thaw(schema), ensure_ascii=False,
                allow_nan=False, sort_keys=True, separators=(',', ':')))

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {'instructions': self.instructions, 'max_bytes': self.max_bytes,
                                    'max_lines': self.max_lines, 'format': self.format}
        if self.output_schema is not None:
            result['output_schema'] = _thaw(self.output_schema)
        return result

    @classmethod
    def from_json(cls, value: object) -> TaskAnswerRequirements:
        row = record(value)
        if set(row) - {'instructions', 'max_bytes', 'max_lines', 'format', 'output_schema'}:
            raise invalid('answer requirement fields')
        instructions = row.get('instructions', '')
        if not isinstance(instructions, str):
            raise invalid('answer instructions')
        format_value = row.get('format', 'text')
        if format_value not in ('text', 'json_object'):
            raise invalid('answer format')
        schema = row.get('output_schema')
        if schema is not None and not isinstance(schema, dict):
            raise invalid('output schema object')
        return cls(instructions, integer(row.get('max_bytes', 64000), 1, 128000),
            integer(row['max_lines'], 1, 1000) if row.get('max_lines') is not None else None,
            'json_object' if format_value == 'json_object' else 'text', schema)
