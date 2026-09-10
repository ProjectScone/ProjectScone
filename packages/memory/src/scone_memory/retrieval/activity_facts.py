"""Bound graph fact selection without falling back to a whole-ledger scan."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal, cast

from ..core.graph_read import GraphFactReader, MAX_GRAPH_FACTS
from ..core.models import Fact
from ..core.ports import DocumentStore


@dataclass(frozen=True)
class ActivityFacts:
    facts: list[Fact]
    truncated: bool = False
    status: Literal['bounded', 'unavailable', 'failed'] = 'bounded'


def _checked(value: list[Fact], space: str, source: int | None, limit: int) -> list[Fact]:
    if type(value) is not list or len(value) > limit:
        raise ValueError('unbounded fact response')
    previous = 0
    for fact in value:
        if (not isinstance(fact, Fact) or fact.space != space or type(fact.fact_id) is not int
                or fact.fact_id <= previous or (source is not None and fact.source_episode_id != source)):
            raise ValueError('invalid scoped fact response')
        previous = fact.fact_id
    return value


async def read_activity_facts(documents: DocumentStore, space: str,
                              source_ids: set[int] | None, limit: int) -> ActivityFacts:
    """Source groups in episode-ID order, then sort selected facts by fact ID.

    A focused graph never spends its budget reading unrelated source groups.
    The budget is global; unvisited groups conservatively mean a partial result.
    The host bounds the source set through its event window. Errors discard the
    fact snapshot, never trigger an unbounded fallback or expose provider text.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_GRAPH_FACTS:
        raise ValueError('graph fact limit must be in 1..2000')
    if source_ids is not None and len(source_ids) > 2001:
        raise ValueError('too many focused source groups')
    if source_ids == set():
        return ActivityFacts([])
    if not callable(getattr(documents, 'facts_for_graph', None)):
        return ActivityFacts([], truncated=True, status='unavailable')
    reader = cast(GraphFactReader, documents)
    sources: list[int | None] = [None] if source_ids is None else list(sorted(source_ids))
    found: list[Fact] = []
    seen: set[int] = set()
    truncated = False
    deadline = time.monotonic() + 2.0
    try:
        async with asyncio.timeout_at(deadline):
            for index, source in enumerate(sources):
                remaining = limit - len(found)
                rows = _checked(await reader.facts_for_graph(space, source, remaining + 1),
                                space, source, remaining + 1)
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise asyncio.CancelledError()
                if time.monotonic() >= deadline:
                    raise TimeoutError()
                if any(row.fact_id in seen for row in rows):
                    raise ValueError('duplicate fact across source groups')
                seen.update(row.fact_id for row in rows)
                found.extend(rows[:remaining])
                if len(rows) > remaining or (len(found) == limit and index + 1 < len(sources)):
                    truncated = True
                    break
    except asyncio.CancelledError:
        raise
    except Exception:
        return ActivityFacts([], truncated=True, status='failed')
    return ActivityFacts(sorted(found, key=lambda fact:fact.fact_id), truncated)
