"""Bounded immutable evidence packets; current source validity belongs to the server."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Mapping, Tuple, Union

from ._wire import integer, invalid, items, record, text, timestamp

FrozenJSON = Union[None, bool, int, float, str, Tuple['FrozenJSON', ...], Mapping[str, 'FrozenJSON']]


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise invalid('duplicate evidence key')
        result[key] = value
    return result


def _freeze(value: object, depth: int = 0) -> FrozenJSON:
    if depth > 64:
        raise invalid('evidence nesting')
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        text('x' + value, 64001, 'evidence text')
        return value
    if isinstance(value, (int, float)):
        _number(value)
        return value
    if isinstance(value, list):
        return tuple(_freeze(item, depth + 1) for item in value)
    values = record(value)
    for key in values:
        text('x' + key, 64001, 'evidence key')
    return MappingProxyType({key: _freeze(item, depth + 1) for key, item in values.items()})


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid('evidence number')
    try:
        result = float(value)
    except OverflowError:
        raise invalid('evidence number') from None
    if not math.isfinite(result):
        raise invalid('evidence number')
    return result


def _rows(value: object, maximum: int) -> tuple[dict[str, object], ...]:
    return tuple(record(item) for item in items(value, maximum))


def _chunk(row: dict[str, object]) -> None:
    if set(row) != {'chunk_id', 'episode_id', 'text', 'source', 'created_at', 'score'}:
        raise invalid('evidence chunk fields')
    integer(row.get('episode_id'), 1)
    text(row.get('text'), 64000)
    timestamp(row.get('created_at'))
    _number(row.get('score'))
    if row.get('source') is not None:
        if not isinstance(row['source'], str):
            raise invalid('evidence source')
        text('x' + row['source'], 64001)


def _claim(row: dict[str, object]) -> None:
    if (set(row) != {'fact_id', 'subject', 'predicate', 'object', 'origin', 'status', 'valid_from',
                     'valid_until', 'confidence', 'source_episode_id', 'quote'}
            or row.get('status') != 'active' or row.get('origin') not in ('stated', 'extracted', 'inferred')):
        raise invalid('evidence claim fields')
    for field in ('subject', 'predicate', 'object', 'quote'):
        text(row.get(field), 64000)
    integer(row.get('source_episode_id'), 1)
    timestamp(row.get('valid_from'))
    if row.get('valid_until') is not None:
        timestamp(row['valid_until'])
    if not 0 <= _number(row.get('confidence')) <= 1:
        raise invalid('evidence confidence')


def _paths(raw: object, claims: dict[int, dict[str, object]], links: dict[int, dict[str, object]]) -> None:
    for path in _rows(raw, 8):
        if set(path) - {'fact_ids', 'steps', 'ordered_evidence'}:
            raise invalid('evidence path fields')
        ids = tuple(integer(value, 1) for value in items(path.get('fact_ids'), 7))
        steps = _rows(path.get('steps'), 6)
        if len(ids) < 2 or len(set(ids)) != len(ids) or set(ids) - set(claims) or len(steps) != len(ids) - 1:
            raise invalid('evidence path')
        for index, step in enumerate(steps):
            direction = step.get('direction')
            left, right = ids[index:index + 2]
            if direction not in ('forward', 'reverse'):
                raise invalid('evidence path direction')
            endpoints = (left, right) if direction == 'forward' else (right, left)
            if (integer(step.get('from_fact'), 1), integer(step.get('to_fact'), 1)) != endpoints:
                raise invalid('evidence path endpoints')
            fields = {'from_fact', 'to_fact', 'kind', 'direction'}
            if step.get('kind') == 'subject_object':
                if set(step) != fields or direction != 'forward':
                    raise invalid('evidence join')
                source = text(claims[left]['object'], 64000)
                target = text(claims[right]['subject'], 64000)
                if ' '.join(source.casefold().split()) != ' '.join(target.casefold().split()):
                    raise invalid('evidence join identity')
            else:
                link = links.get(integer(step.get('link_id'), 1))
                if (set(step) != fields | {'link_id'} or link is None or step.get('kind') == 'contradicts'
                        or any(step.get(key) != link[key] for key in ('from_fact', 'to_fact', 'kind'))):
                    raise invalid('evidence path link')
        if 'ordered_evidence' in path:
            expected = [{'fact_id': key, 'source_episode_id': claims[key]['source_episode_id'], 'quote': claims[key]['quote']} for key in ids]
            ordered = _rows(path['ordered_evidence'], 7)
            for entry in ordered:
                integer(entry.get('fact_id'), 1)
                integer(entry.get('source_episode_id'), 1)
            if list(ordered) != expected:
                raise invalid('ordered evidence')


@dataclass(frozen=True)
class EvidencePacket:
    raw_json: str
    evidence_ids: tuple[str, ...]
    payload: Mapping[str, FrozenJSON]

    @classmethod
    def from_json(cls, value: object) -> EvidencePacket:
        raw = text(value, 64000, 'evidence packet')
        try:
            row = record(json.loads(raw, object_pairs_hook=_unique))
        except (ValueError, RecursionError):
            raise invalid('evidence JSON') from None
        chunks = _rows(row.get('items', []), 20)
        claims = _rows(row.get('facts', []), 20) + _rows(row.get('claims', []), 16)
        links = _rows(row.get('relations', []), 32)
        ids = tuple(kind + ':' + str(integer(item.get(key), 1)) for records, kind, key in (
            (chunks, 'chunk', 'chunk_id'), (claims, 'fact', 'fact_id'), (links, 'link', 'link_id')) for item in records)
        if not ids or len(ids) != len(set(ids)) or row.get('status') != 'prepared':
            raise invalid('retained evidence identifiers')
        for chunk in chunks:
            _chunk(chunk)
        for claim in claims:
            _claim(claim)
        facts = {integer(claim['fact_id'], 1): claim for claim in claims}
        for link in links:
            if (set(link) != {'link_id', 'from_fact', 'to_fact', 'kind', 'source_episode_id', 'quote'}
                    or integer(link.get('from_fact'), 1) not in facts or integer(link.get('to_fact'), 1) not in facts
                    or link.get('kind') not in ('extends', 'derived_from', 'contradicts', 'supports')):
                raise invalid('evidence relation')
            integer(link.get('source_episode_id'), 1)
            text(link.get('quote'), 64000)
        _paths(row.get('paths', []), facts, {integer(link['link_id'], 1): link for link in links})
        payload = MappingProxyType({key: _freeze(item) for key, item in row.items()})
        return cls(raw, ids, payload)
