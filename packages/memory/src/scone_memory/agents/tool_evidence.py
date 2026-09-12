"""Immutable snapshots of scoped tool evidence, rechecked before publication."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
import math
import sqlite3
import time

from ..core.models import Episode, Fact
from ..core.ports import TextFilter
from ..core.timeutil import parse_rfc3339
from ..memory.engine import MemoryEngine
from ..retrieval.multihop import PointFactLinks, _source_matches
from ..retrieval.recall_scope import RecallScope
from ..realtime.review_evidence import _claim, _id, _json, _paths, _records, _relation


@dataclass(frozen=True)
class PreparedToolEvidence:
    payload: str
    evidence_ids: tuple[str, ...]
    _validator: Callable[[], Awaitable[bool]] = field(repr=False, compare=False)

    async def validate(self) -> bool:
        return await self._validator()


async def prepare_tool_evidence(memory: MemoryEngine, space: str, scope: RecallScope,
                                excluded_session: str | None, result: dict[str, object],
                                timeout_s: float, *, raise_unavailable: bool = False) -> PreparedToolEvidence:
    # The serialized result is detached from mutable store/provider models.
    payload = _json(result)
    if len(payload.encode()) > 64000:
        raise ValueError('tool evidence unavailable')
    packet = json.loads(payload)
    items = _records(packet.get('items', []), 20)
    claims = _records(packet.get('facts', []), 20) + _records(packet.get('claims', []), 16)
    relations = _records(packet.get('relations', []), 32)
    paths = _records(packet.get('paths', []), 8)
    ids = tuple(f'{kind}:{_id(record.get(key))}' for records, kind, key in (
        (items, 'chunk', 'chunk_id'), (claims, 'fact', 'fact_id'), (relations, 'link', 'link_id')) for record in records)
    if len(ids) != len(set(ids)) or (packet.get('status') != 'prepared' and ids):
        raise ValueError('tool evidence unavailable')
    _paths(paths, {_id(row.get('fact_id')): row for row in claims},
           {_id(row.get('link_id')): row for row in relations})
    scope = RecallScope.validated(**scope.kwargs())
    documents = memory.documents
    revision: int | None = None

    async def capture() -> tuple[str, ...]:
        nonlocal revision
        if not ids:
            return ()
        current_revision = await documents.revision(space)
        if revision is None:
            revision = current_revision
        if current_revision != revision:
            raise ValueError('tool evidence changed')
        boundary = memory.clock()
        now = parse_rfc3339(boundary)
        sources: dict[int, Episode] = {}
        snapshots: list[str] = []

        async def source(episode_id: int) -> Episode:
            if episode_id in sources:
                return sources[episode_id]
            if len(sources) >= 64:
                raise ValueError('tool evidence unavailable')
            episode = await documents.get_episode(space, episode_id)
            if episode is None:
                raise ValueError('tool evidence changed')
            episode = Episode.model_validate(episode.model_dump(warnings=False), strict=True).model_copy(deep=True)
            if (episode.episode_id != episode_id or episode.space != space
                    or not _source_matches(episode, TextFilter(**scope.kwargs()), excluded_session)
                    or parse_rfc3339(episode.created_at) > now):
                raise ValueError('tool evidence changed')
            snapshots.append(_json(episode.model_dump(mode='json')))
            sources[episode_id] = episode.model_copy(deep=True)
            return sources[episode_id]

        for row in items:
            chunk_id, episode_id = _id(row.get('chunk_id')), _id(row.get('episode_id'))
            chunks = await documents.get_chunks(space, [chunk_id])
            episode = await source(episode_id)
            if len(chunks) != 1:
                raise ValueError('tool evidence changed')
            chunk = chunks[0]
            score = row.get('score')
            if (chunk.chunk_id != chunk_id or chunk.episode_id != episode_id or chunk.space != space
                    or chunk.created_at != episode.created_at or chunk.start < 0 or chunk.end < chunk.start
                    or chunk.end > len(episode.content.encode())
                    or episode.content.encode()[chunk.start:chunk.end] != chunk.text.encode()
                    or set(row) != {'chunk_id', 'episode_id', 'text', 'source', 'created_at', 'score'}
                    or row['text'] != chunk.text or row['source'] != episode.source
                    or row['created_at'] != chunk.created_at
                    or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
                raise ValueError('tool evidence changed')
            snapshots.append(_json(chunk.model_dump(mode='json')))
        facts: dict[int, Fact] = {}
        for row in claims:
            fact_id = _id(row.get('fact_id'))
            fact = await documents.get_fact(space, fact_id)
            if (fact is None or fact.fact_id != fact_id or fact.space != space or fact.excluded
                    or fact.status != 'active' or not fact.holds_at(boundary) or _json(row) != _json(_claim(fact))):
                raise ValueError('tool evidence changed')
            episode = await source(_id(fact.source_episode_id))
            if not fact.quote or fact.quote not in episode.content:
                raise ValueError('tool evidence changed')
            snapshots.append(_json(fact.model_dump(mode='json')))
            facts[fact_id] = fact.model_copy(deep=True)
        for row in relations:
            if not isinstance(documents, PointFactLinks):
                raise ValueError('tool evidence unavailable')
            link_id = _id(row.get('link_id'))
            link = await documents.get_fact_link(space, link_id)
            if (link is None or link.link_id != link_id or link.space != space or link.from_fact not in facts
                    or link.to_fact not in facts or parse_rfc3339(link.created_at) > now
                    or _json(row) != _json(_relation(link))):
                raise ValueError('tool evidence changed')
            episode = await source(_id(link.source_episode_id))
            if not link.quote or link.quote not in episode.content:
                raise ValueError('tool evidence changed')
            snapshots.append(_json(link.model_dump(mode='json')))
        if await documents.revision(space) != revision:
            raise ValueError('tool evidence changed')
        return tuple(snapshots)

    async def bounded_capture() -> tuple[str, ...]:
        deadline = time.monotonic() + timeout_s
        async with asyncio.timeout(timeout_s):
            snapshot = await asyncio.create_task(capture())
            active = asyncio.current_task()
            if active is not None and active.cancelling():
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError('tool evidence unavailable')
            return snapshot

    original = await bounded_capture()

    async def validate() -> bool:
        try:
            return await bounded_capture() == original
        except asyncio.CancelledError:
            raise
        except (OSError, sqlite3.OperationalError):
            if raise_unavailable:
                raise
            return False
        except Exception:
            return False

    return PreparedToolEvidence(payload, ids, validate)
