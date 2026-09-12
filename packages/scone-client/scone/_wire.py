"""Strict JSON boundary helpers for independently installable workflow clients."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Protocol
from urllib.parse import quote

from .errors import SconeError


def invalid(label: str) -> SconeError:
    return SconeError('invalid workflow ' + label)


def record(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise invalid('object')
    return {str(key): item for key, item in value.items()}


def text(value: object, maximum: int, label: str = 'text') -> str:
    if not isinstance(value, str) or not value.strip():
        raise invalid(label)
    try:
        size = len(value.encode('utf-8'))
    except UnicodeError:
        raise invalid(label) from None
    if size > maximum:
        raise invalid(label)
    return value


def identifier(value: object) -> str:
    name = text(value, 128, 'identifier')
    if name in ('.', '..') or re.fullmatch(r'[A-Za-z0-9._:-]+', name) is None:
        raise invalid('identifier')
    return name


def address(value: object) -> str:
    return quote(identifier(value), safe='')


def integer(value: object, minimum: int, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise invalid('integer')
    return value


def boolean(value: object) -> bool:
    if type(value) is not bool:
        raise invalid('boolean')
    return value


def items(value: object, maximum: int) -> List[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise invalid('list')
    return list(value)


def names(value: object, maximum: int = 32) -> tuple[str, ...]:
    result = tuple(identifier(item) for item in items(value, maximum))
    if len(set(result)) != len(result):
        raise invalid('duplicate identifiers')
    return result


def digest(value: object) -> str:
    result = text(value, 64, 'digest')
    if re.fullmatch(r'[a-f0-9]{64}', result) is None:
        raise invalid('digest')
    return result


def timestamp(value: object) -> str:
    result = text(value, 64, 'timestamp')
    try:
        parsed = datetime.fromisoformat(result.replace('Z', '+00:00'))
    except ValueError:
        raise invalid('timestamp') from None
    if parsed.tzinfo is None:
        raise invalid('timestamp')
    return result


@dataclass(frozen=True)
class Capabilities:
    implementation: str
    features: Mapping[str, bool]

    @classmethod
    def from_json(cls, value: object) -> Capabilities:
        row = record(value)
        if integer(row.get('schema_version'), 1, 1) != 1:
            raise invalid('capability schema')
        features = {key: boolean(value) for key, value in record(row.get('features')).items()}
        return cls(text(row.get('implementation'), 64), MappingProxyType(features))

    def supports(self, name: str) -> bool:
        return self.features.get(name) is True

    def require(self, name: str) -> None:
        if not self.supports(name):
            raise SconeError('server does not advertise ' + name)


class WorkflowTransport(Protocol):
    def _request(self, method: str, path: str, *, params: Optional[Mapping[str, str]] = None,
                 json: Optional[Dict[str, object]] = None) -> object: ...
    def capabilities(self) -> Capabilities: ...


class ResourceClient:
    def __init__(self, client: WorkflowTransport, *, expected_space: str) -> None:
        self._client = client
        self.expected_space = identifier(expected_space)

    def _check(self, capability: str, *, mutation: bool = False) -> Capabilities:
        capabilities = self._client.capabilities()
        capabilities.require(capability)
        if mutation:
            self._space(record(self._client._request('GET', '/v1/status')))
        return capabilities

    def _space(self, row: Mapping[str, object]) -> None:
        if row.get('space') != self.expected_space:
            raise SconeError('workflow response does not match the expected space')


def cursor(value: object) -> Optional[str]:
    if value is None:
        return None
    result = text(value, 129, 'cursor')
    if re.fullmatch(r'[a-f0-9]{64}:[a-f0-9]{64}', result) is None:
        raise invalid('cursor')
    return result


def bounded_body(value: Dict[str, object], maximum: int = 8192) -> Dict[str, object]:
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))
    except (TypeError, ValueError):
        raise invalid('request body') from None
    if size > maximum:
        raise invalid('request byte limit')
    return value
