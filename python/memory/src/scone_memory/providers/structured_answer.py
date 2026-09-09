"""Render a generated JSON object without changing its values or number tokens."""
from __future__ import annotations

import json

from ..realtime.answer_requirements import _reject_constant, _unique_object

_WHITESPACE = ' \t\r\n'


def _after_whitespace(text: str, offset: int) -> int:
    while offset < len(text) and text[offset] in _WHITESPACE:
        offset += 1
    return offset


def _compact(text: str) -> str:
    output: list[str] = []
    quoted = escaped = False
    for character in text:
        if quoted:
            output.append(character)
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
            output.append(character)
        elif character not in _WHITESPACE:
            output.append(character)
    return ''.join(output)


def object_answer(text: str, *, action_envelope: bool = False) -> str | None:
    """Return compact JSON; None means this envelope selected another action.

    Syntax and duplicate keys are validated before extracting the original
    object span. Only structural whitespace is removed. Decimal precision,
    exponent spelling, escapes, key order and string whitespace are preserved.
    Fences, trailing prose and wrong answer types are rejected, never repaired.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object,
        parse_constant=_reject_constant, parse_int=str, parse_float=str)
    value = decoder.decode(text)
    if not isinstance(value, dict):
        raise ValueError('JSON answer must be an object')
    if action_envelope:
        if value.get('action') != 'answer':
            return None
        if set(value) != {'action', 'answer'} or not isinstance(value['answer'], dict):
            raise ValueError('invalid JSON answer action')
        # The complete object has already been validated. Walk its two members
        # with the same decoder to locate the answer without re-encoding values.
        cursor = _after_whitespace(text, 0) + 1
        for _ in range(2):
            key, cursor = decoder.raw_decode(text, _after_whitespace(text, cursor))
            start = _after_whitespace(text, _after_whitespace(text, cursor) + 1)
            _, end = decoder.raw_decode(text, start)
            if key == 'answer':
                text = text[start:end]
                break
            cursor = _after_whitespace(text, end) + 1
    result = _compact(text)
    if len(result.encode('utf-8')) > 64000:
        raise ValueError('JSON answer byte limit')
    return result
