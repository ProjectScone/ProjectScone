"""Bounded annotation-to-JSON contracts without evaluating annotation code.

Only explicit JSON-shaped annotations are supported. Union conversion prefers
an exact recursive type match, then a unique result; it never picks an arbitrary
branch. Public schemas are detached copies of the execution contract.
"""
from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import types
import typing


@dataclass(frozen=True)
class _Form:
    kind: str
    arguments: tuple[object, ...] = ()
    applied: bool = False


_FORMS = {'List': 'list', 'Dict': 'dict', 'Tuple': 'tuple', 'Union': 'union',
          'Optional': 'optional', 'Literal': 'literal', 'Annotated': 'annotated'}
_BUILTINS: dict[str, object] = {'str': str, 'int': int, 'float': float, 'bool': bool,
                              'list': list, 'dict': dict, 'tuple': tuple, 'NoneType': type(None)}


def _kind(value: object) -> str | None:
    if isinstance(value, _Form) and not value.applied:
        return value.kind
    for candidate, kind in ((list, 'list'), (dict, 'dict'), (tuple, 'tuple'),
                            (typing.List, 'list'), (typing.Dict, 'dict'), (typing.Tuple, 'tuple'),
                            (typing.Union, 'union'), (typing.Optional, 'optional'),
                            (typing.Literal, 'literal'), (typing.Annotated, 'annotated')):
        if value is candidate:
            return kind
    return None


def _literal_ast(node: ast.expr, namespace: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)):
        number = node.operand.value
        assert isinstance(number, (int, float))
        return -number if isinstance(node.op, ast.USub) else number
    if isinstance(node, ast.Name) and isinstance(namespace.get(node.id), Enum):
        return namespace[node.id]
    raise ValueError('unsupported annotation literal')


def resolve_annotation(annotation: object, namespace: Mapping[str, object]) -> object:
    """Resolve bounded known syntax and direct host names, never eval or getattr."""
    return _resolve(annotation, namespace, 0, _Budget(256, 16))


def _resolve(annotation: object, namespace: Mapping[str, object], depth: int, budget: _Budget) -> object:
    if depth > 16:
        raise ValueError('annotation depth limit')
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if not isinstance(annotation, str):
        return annotation
    try:
        if len(annotation) > 32768 or len(annotation.encode('utf-8')) > 32768:
            raise ValueError('annotation byte limit')
        tree = ast.parse(annotation, mode='eval')
    except (SyntaxError, UnicodeError, RecursionError):
        raise ValueError('invalid annotation expression') from None
    pending: list[tuple[ast.AST, int]] = [(tree.body, depth)]
    while pending:
        node, level = pending.pop()
        budget.visit(level)
        pending.extend((child, level + 1) for child in ast.iter_child_nodes(node))

    def read(node: ast.expr) -> object:
        if isinstance(node, ast.Name):
            if node.id in namespace:
                return _resolve(namespace[node.id], namespace, depth + 1, budget)
            if node.id in _BUILTINS:
                return _BUILTINS[node.id]
            if node.id in _FORMS:
                return _Form(_FORMS[node.id])
            raise ValueError('unknown annotation name')
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'typing'
                and namespace.get('typing') is typing and node.attr in _FORMS):
            return _Form(_FORMS[node.attr])
        if isinstance(node, ast.Constant):
            if node.value is None:
                return type(None)
            if isinstance(node.value, str):
                return _resolve(node.value, namespace, depth + 1, budget)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return _Form('union', (read(node.left), read(node.right)), True)
        if isinstance(node, ast.Subscript):
            kind = _kind(read(node.value))
            if kind is None:
                raise ValueError('unsupported annotation generic')
            arguments = tuple(node.slice.elts) if isinstance(node.slice, ast.Tuple) else (node.slice,)
            if kind == 'literal':
                return _Form(kind, tuple(_literal_ast(item, namespace) for item in arguments), True)
            if kind == 'annotated':
                if len(arguments) < 2:
                    raise ValueError('annotation description is missing')
                return _Form(kind, (read(arguments[0]), *(_literal_ast(item, namespace) for item in arguments[1:])), True)
            return _Form(kind, tuple(Ellipsis if isinstance(item, ast.Constant) and item.value is Ellipsis else read(item)
                                     for item in arguments), True)
        raise ValueError('unsupported annotation expression')

    return read(tree.body)


class _LimitError(ValueError):
    pass


@dataclass
class _Budget:
    maximum: int
    depth: int
    count: int = 0
    bytes: int = 0

    def visit(self, depth: int) -> None:
        self.count += 1
        if self.count > self.maximum or depth > self.depth:
            raise _LimitError('function contract structure limit')

    def scalar(self, value: object) -> object:
        result = _scalar(value)
        self.bytes += len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8'))
        if self.bytes > 128000:
            raise _LimitError('function value byte limit')
        return result


@dataclass(frozen=True)
class _Node:
    kind: str
    children: tuple[_Node, ...] = ()
    literals: tuple[tuple[object, object], ...] = ()
    description: str | None = None

    def schema(self) -> dict[str, object]:
        scalar = {'str': 'string', 'int': 'integer', 'float': 'number', 'bool': 'boolean', 'null': 'null'}
        if self.kind in scalar:
            result: dict[str, object] = {'type': scalar[self.kind]}
        elif self.kind in ('list', 'tuple_many'):
            result = {'type': 'array', 'items': self.children[0].schema()}
        elif self.kind == 'tuple':
            result = {'type': 'array', 'prefixItems': [child.schema() for child in self.children],
                      'items': False, 'minItems': len(self.children), 'maxItems': len(self.children)}
            if not self.children:
                result.pop('prefixItems')
        elif self.kind == 'dict':
            result = {'type': 'object', 'additionalProperties': self.children[0].schema()}
        elif self.kind == 'union':
            result = {'anyOf': [child.schema() for child in self.children]}
        elif self.kind in ('literal', 'enum'):
            result = {'enum': [value for value, _ in self.literals]}
        else:
            raise ValueError('unsupported function contract')
        if self.description is not None:
            result['description'] = self.description
        return result


def _scalar(value: object) -> object:
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('function contract requires a JSON scalar')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('function contract requires finite numbers')
    try:
        if isinstance(value, str) and len(value) > 128000:
            raise ValueError('function contract value byte limit')
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 128000:
            raise ValueError('function contract value byte limit')
    except (UnicodeError, OverflowError):
        raise ValueError('invalid function contract scalar') from None
    return value


def _compile(annotation: object, namespace: Mapping[str, object], budget: _Budget, depth: int) -> _Node:
    budget.visit(depth)
    annotation = resolve_annotation(annotation, namespace)
    kind: str | None
    for candidate, kind in ((str, 'str'), (int, 'int'), (float, 'float'), (bool, 'bool'), (type(None), 'null'), (None, 'null')):
        if annotation is candidate:
            assert kind is not None
            return _Node(kind)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = typing.cast(dict[str, Enum], type.__getattribute__(annotation, '_member_map_'))
        entries: list[tuple[object, object]] = []
        seen: set[int] = set()
        for member in members.values():
            if id(member) in seen:
                continue
            seen.add(id(member))
            budget.visit(depth + 1)
            value = _scalar(object.__getattribute__(member, '_value_'))
            entries.append((value, member))
        if not entries:
            raise ValueError('empty function contract enum')
        return _Node('enum', literals=tuple(entries))
    if isinstance(annotation, _Form):
        if not annotation.applied:
            raise ValueError('generic annotation arguments required')
        kind, arguments = annotation.kind, annotation.arguments
    else:
        if _kind(annotation) is not None:
            raise ValueError('generic annotation arguments required')
        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        kind = 'union' if origin in (typing.Union, types.UnionType) else _kind(origin)
    if kind is None:
        raise ValueError('unsupported function parameter annotation')
    if kind == 'optional':
        if len(arguments) != 1:
            raise ValueError('Optional requires one annotation')
        kind, arguments = 'union', (*arguments, type(None))
    if kind == 'annotated':
        if len(arguments) != 2 or type(arguments[1]) is not str:
            raise ValueError('Annotated requires one string description')
        description = arguments[1]
        assert isinstance(description, str)
        _scalar(description)
        if not description.strip():
            raise ValueError('annotation description must not be blank')
        base = _compile(arguments[0], namespace, budget, depth + 1)
        return _Node(base.kind, base.children, base.literals, description)
    if kind == 'literal':
        if not arguments:
            raise ValueError('Literal requires values')
        pairs: list[tuple[object, object]] = []
        for argument in arguments:
            budget.visit(depth + 1)
            value = object.__getattribute__(argument, '_value_') if isinstance(argument, Enum) else argument
            pairs.append((_scalar(value), argument))
        return _Node('literal', literals=tuple(pairs))
    if kind == 'list' or kind == 'dict':
        expected = 1 if kind == 'list' else 2
        if len(arguments) != expected or (kind == 'dict' and resolve_annotation(arguments[0], namespace) is not str):
            raise ValueError('typed list or string-keyed dict required')
        return _Node(kind, (_compile(arguments[-1], namespace, budget, depth + 1),))
    if kind == 'tuple':
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return _Node('tuple_many', (_compile(arguments[0], namespace, budget, depth + 1),))
        return _Node(kind, tuple(_compile(item, namespace, budget, depth + 1) for item in arguments))
    if kind == 'union' and len(arguments) >= 2:
        return _Node(kind, tuple(_compile(item, namespace, budget, depth + 1) for item in arguments))
    raise ValueError('unsupported function parameter annotation')


def _same(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Enum):
        return left is right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    return bool(left == right)


def _convert(node: _Node, value: object, encode: bool, budget: _Budget, depth: int = 0) -> object:
    budget.visit(depth)
    if node.kind == 'union':
        results: list[object] = []
        exact: list[object] = []
        for child in node.children:
            try:
                result = _convert(child, value, encode, budget, depth + 1)
            except _LimitError:
                raise
            except ValueError:
                continue
            if not any(_same(result, prior) for prior in results):
                results.append(result)
            # For encoding, the chosen branch must round-trip to the same
            # Python shape (tuples and enums deliberately serialize differently).
            restored = _convert(child, result, False, budget, depth + 1) if encode else result
            if _same(restored, value) and not any(_same(result, prior) for prior in exact):
                exact.append(result)
        preferred = exact or results
        if len(preferred) != 1:
            raise ValueError('ambiguous or unmatched function union value')
        return preferred[0]
    if node.kind in ('literal', 'enum'):
        for scalar, member in node.literals:
            if _same(value, member if encode else scalar):
                budget.scalar(scalar)
                return scalar if encode else member
        raise ValueError('function literal or enum value mismatch')
    if node.kind in ('list', 'tuple', 'tuple_many'):
        expected = tuple if encode and node.kind.startswith('tuple') else list
        if type(value) is not expected:
            raise ValueError('function array type mismatch')
        assert isinstance(value, (list, tuple))
        if len(value) > 4096 or (node.kind == 'tuple' and len(value) != len(node.children)):
            raise ValueError('function array length limit')
        result_list = [_convert(node.children[index] if node.kind == 'tuple' else node.children[0], child, encode, budget, depth + 1)
                       for index, child in enumerate(value)]
        return tuple(result_list) if not encode and node.kind.startswith('tuple') else result_list
    if node.kind == 'dict':
        if type(value) is not dict:
            raise ValueError('function object type mismatch')
        assert isinstance(value, dict)
        if len(value) > 4096 or any(type(key) is not str for key in value):
            raise ValueError('function object keys or size limit')
        result_dict: dict[str, object] = {}
        for key, child in value.items():
            budget.scalar(key)
            result_dict[key] = _convert(node.children[0], child, encode, budget, depth + 1)
        return result_dict
    expected_scalar: dict[str, type] = {'str': str, 'int': int, 'bool': bool, 'null': type(None)}
    if node.kind == 'float':
        if type(value) not in (int, float):
            raise ValueError('function number type mismatch')
        assert isinstance(value, (int, float))
        try:
            number = float(value)
            if type(value) is int and number != value:
                raise ValueError('function number conversion would lose precision')
            return budget.scalar(number)
        except OverflowError:
            raise ValueError('function number out of range') from None
    if type(value) is not expected_scalar[node.kind]:
        raise ValueError('function scalar type mismatch')
    return budget.scalar(value)


@dataclass(frozen=True)
class ParameterType:
    _node: _Node = field(repr=False)
    _schema_json: str = field(repr=False)

    @property
    def schema(self) -> dict[str, object]:
        return typing.cast(dict[str, object], json.loads(self._schema_json))

    def encode(self, value: object) -> object:
        result = _convert(self._node, value, True, _Budget(4096, 32))
        _value_size(result)
        return result

    def decode(self, value: object) -> object:
        # Bound the JSON input before any conversion creates Python objects.
        _value_size(value)
        return _convert(self._node, value, False, _Budget(4096, 32))

    def encode_default(self, value: object) -> object:
        """Refuse defaults whose JSON round-trip changes Python type or value.

        For example float defaults must be floats, and an Enum default in an
        Enum|str union cannot become a string merely because it was omitted.
        """
        encoded = self.encode(value)
        if not _same(self.decode(encoded), value):
            raise ValueError('function default does not preserve its declared type and value')
        return encoded


def _value_size(value: object) -> None:
    pending = [(value, 0)]
    budget = _Budget(4096, 32)
    while pending:
        item, depth = pending.pop()
        budget.visit(depth)
        if type(item) in (dict, list):
            assert isinstance(item, (dict, list))
            if len(item) > 4096:
                raise ValueError('function value structure limit')
            if isinstance(item, dict):
                for key in item:
                    if type(key) is not str:
                        raise ValueError('function value keys must be strings')
                    budget.scalar(key)
                children: Iterable[object] = item.values()
            else:
                children = item
            pending.extend((child, depth + 1) for child in children)
        else:
            budget.scalar(item)
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')) > 128000:
        raise ValueError('function value byte limit')


def parameter_type(annotation: object, namespace: Mapping[str, object]) -> ParameterType:
    node = _compile(annotation, namespace, _Budget(256, 16), 0)
    encoded = json.dumps(node.schema(), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    try:
        if len(encoded.encode('utf-8')) > 32768:
            raise ValueError('function schema byte limit')
    except UnicodeError:
        raise ValueError('invalid function schema text') from None
    return ParameterType(node, encoded)


__all__ = ['ParameterType', 'parameter_type', 'resolve_annotation']
