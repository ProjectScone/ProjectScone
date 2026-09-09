"""Bounded, self-contained Draft 2020-12 contracts for application output.

Schemas are application configuration. Validation is synchronous; byte/depth
limits are not a CPU deadline for complex regular expressions or combinations.
Format is an annotation, as in Draft 2020-12 without a FormatChecker.
"""
from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, Protocol, cast
from urllib.parse import unquote

_SCHEMA_BYTES = 32768
_MAPS = {'properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'}
_SINGLE = {'items', 'additionalProperties', 'not', 'contains', 'if', 'then', 'else',
           'propertyNames', 'unevaluatedItems', 'unevaluatedProperties', 'contentSchema'}
_ARRAYS = {'allOf', 'anyOf', 'oneOf', 'prefixItems'}
_ANNOTATIONS = {'title', 'description', 'default', 'examples', 'deprecated', 'readOnly',
                'writeOnly', '$comment', 'contentEncoding', 'contentMediaType', '$schema'}
_CONTAINS_BOUNDS = {'minContains', 'maxContains'}
_DIALECT = 'https://json-schema.org/draft/2020-12/schema'


class _Validator(Protocol):
    def is_valid(self, instance: object) -> bool: ...
    def is_type(self, instance: object, type: str) -> bool: ...


def _document(value: object) -> str:
    """Reject non-JSON input, cycles and excessive nesting before serializing."""
    pending: list[tuple[object, int, frozenset[int]]] = [(value, 0, frozenset())]
    count = 0
    while pending:
        item, depth, ancestors = pending.pop()
        count += 1
        if depth > 32 or count > 4096:
            raise ValueError('output schema structure limit')
        if type(item) in (dict, list):
            if id(item) in ancestors:
                raise ValueError('cyclic output schema')
            lineage = ancestors | {id(item)}
            children: Iterable[object]
            if isinstance(item, dict):
                if any(type(key) is not str for key in item):
                    raise ValueError('output schema keys must be strings')
                children = item.values()
            else:
                assert isinstance(item, list)
                children = item
            pending.extend((child, depth + 1, lineage) for child in children)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError('output schema must contain JSON values')
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(',', ':'))
    if len(encoded.encode('utf-8')) > _SCHEMA_BYTES:
        raise ValueError('output schema byte limit')
    return encoded


def _pointer(reference: object, root: dict[str, object]) -> tuple[object, tuple[str, ...]]:
    if not isinstance(reference, str) or not reference.startswith('#'):
        raise ValueError('output schema references must be local JSON pointers')
    fragment = unquote(reference[1:], errors='strict')
    if fragment and not fragment.startswith('/'):
        raise ValueError('output schema references must be local JSON pointers')
    path = tuple(part.replace('~1', '/').replace('~0', '~') for part in fragment[1:].split('/')) if fragment else ()
    current: object = root
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isascii() and part.isdigit() and str(int(part)) == part and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError('unresolved output schema reference')
    return current, path


def _inline(root: dict[str, object], keywords: set[str]) -> dict[str, object]:
    count = 0

    def walk(value: object, path: tuple[str, ...], active: frozenset[tuple[str, ...]], depth: int) -> object:
        nonlocal count
        count += 1
        if count > 256 or depth > 16:
            raise ValueError('expanded output schema structure limit')
        if path in active:
            raise ValueError('recursive output schema reference')
        active = active | {path}
        if isinstance(value, bool):
            return value
        if not isinstance(value, dict) or set(value) - keywords:
            raise ValueError('unsupported output schema keyword or schema node')
        if '$schema' in value and value['$schema'] not in (_DIALECT, _DIALECT + '#'):
            raise ValueError('output schema must use Draft 2020-12')
        result: dict[str, object] = {}
        for key, child in value.items():
            if key in ('$ref', '$schema'):
                continue
            if key in _MAPS:
                if not isinstance(child, dict):
                    raise ValueError('invalid output schema map')
                mapped = {name:walk(node, (*path, key, name), active, depth + 1) for name, node in child.items()}
                if key not in ('$defs', 'definitions'):
                    result[key] = mapped
            elif key in _SINGLE:
                result[key] = walk(child, (*path, key), active, depth + 1)
            elif key in _ARRAYS:
                if not isinstance(child, list):
                    raise ValueError('invalid output schema list')
                result[key] = [walk(node, (*path, key, str(index)), active, depth + 1)
                               for index, node in enumerate(child)]
            else:
                result[key] = child
        if '$ref' in value:
            target, target_path = _pointer(value['$ref'], root)
            resolved = walk(target, target_path, active, depth + 1)
            existing = result.get('allOf', [])
            assert isinstance(existing, list)
            # Sibling unevaluatedProperties must see annotations from the ref.
            result['allOf'] = [resolved, *existing]
        return result

    compiled = walk(root, (), frozenset(), 0)
    assert isinstance(compiled, dict)
    return compiled


def _number(token: str) -> Decimal:
    value = Decimal(token)
    exponent = value.as_tuple().exponent
    if not value.is_finite() or not isinstance(exponent, int) or len(value.as_tuple().digits) > 4096 or abs(exponent) > 4096:
        raise ValueError('output schema numeric limit')
    return value


def _integer(checker: object, value: object) -> bool:
    return type(value) is int or isinstance(value, Decimal) and value.is_finite() and value == value.to_integral_value()


def _multiple_of(validator: _Validator, divisor: object, value: object, schema: object):
    from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

    if not validator.is_type(value, 'number'):
        return
    assert isinstance(value, (int, Decimal)) and isinstance(divisor, (int, Decimal))
    if (Fraction(value) / Fraction(divisor)).denominator != 1:
        yield ValidationError('output does not satisfy multipleOf')


@lru_cache(maxsize=128)
def _compiled(encoded: str) -> tuple[str, _Validator]:
    try:
        from jsonschema import Draft202012Validator, validators  # type: ignore[import-untyped]
        from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
        from referencing import Registry
    except ImportError as error:
        raise ImportError('output_schema requires scone-memory[structured-output]') from error

    root = json.loads(encoded)
    if not isinstance(root, dict) or root.get('type', 'object') != 'object':
        raise ValueError('output schema root must be an object schema')
    try:
        Draft202012Validator.check_schema(root)
        keywords = ((set(Draft202012Validator.VALIDATORS) - {'$dynamicRef'})
                    | _MAPS | _SINGLE | _ARRAYS | _ANNOTATIONS | _CONTAINS_BOUNDS)
        compiled = _inline(root, keywords)
        compiled.setdefault('type', 'object')
        Draft202012Validator.check_schema(compiled)
        canonical = _document(compiled)
        exact = json.loads(canonical, parse_int=_number, parse_float=_number)
        validator_type = validators.extend(Draft202012Validator, {'multipleOf':_multiple_of},
            type_checker=Draft202012Validator.TYPE_CHECKER.redefine('integer', _integer))
        validator = validator_type(exact, registry=Registry())
    except SchemaError:
        raise ValueError('invalid Draft 2020-12 output schema') from None
    return canonical, cast(_Validator, validator)


def compile_schema(value: object) -> dict[str, object]:
    """Copy and resolve application schemas without fetching any reference."""
    if type(value) is not dict:
        raise ValueError('output schema must be an object')
    canonical, _ = _compiled(_document(value))
    return cast(dict[str, object], json.loads(canonical))


def accepts_schema(answer: str, schema: dict[str, object]) -> bool:
    from .answer_requirements import _reject_constant, _unique_object

    try:
        _, validator = _compiled(_document(schema))
        value = json.loads(answer, parse_int=_number, parse_float=_number,
            object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        return isinstance(value, dict) and validator.is_valid(value)
    except (ValueError, RecursionError, ArithmeticError):
        return False
