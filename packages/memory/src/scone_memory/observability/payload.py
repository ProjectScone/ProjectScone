"""Narrow recorded JSON fields before arithmetic or nested access."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import SupportsFloat, SupportsIndex, SupportsInt


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("recorded event field must be a JSON object")
    return value


def sequence(value: object) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("recorded event field must be a sequence")
    return value


def items(value: object) -> list[Mapping[str, object]]:
    return [mapping(item) for item in sequence(value)]


def number(value: object) -> float:
    if not isinstance(value, (str, bytes, bytearray, SupportsFloat, SupportsIndex)):
        raise TypeError("recorded event field must be numeric")
    return float(value)


def integer(value: object) -> int:
    if not isinstance(value, (str, bytes, bytearray, SupportsInt, SupportsIndex)):
        raise TypeError("recorded event field must be an integer")
    return int(value)
