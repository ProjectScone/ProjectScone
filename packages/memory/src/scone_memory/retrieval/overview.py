"""Bounded, recent source evidence for questions without search terms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.errors import InvalidInput
from ..core.models import Episode, RecallItem
from ..core.ports import DocumentStore, EpisodeInventory, TextFilter


@dataclass(frozen=True)
class OverviewResult:
    """Recent inserted evidence, never a claim to cover the entire history.

    ``considered`` counts examined inventory rows, including rejected rows.
    ``has_more`` means unvisited rows remain or the scan budget was reached;
    a continuation may find no eligible evidence. The cursor is the last
    examined source ID, so a continuation also progresses past rejected rows.
    """

    items: list[RecallItem]
    considered: int
    has_more: bool
    next_before: int | None


def _integer(value: int, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise InvalidInput(f"{name} must be an integer from 1 through {maximum}")


def _filters(
    space: str, where: Mapping[str, str] | None, kind: str | None,
    source_prefix: str | None, since: str | None, until: str | None,
    exclude_session_id: str | None,
) -> TextFilter:
    # Reuse the engine's public boundary rules without an import cycle.
    from ..memory.engine import KINDS, MAX_SOURCE, check_space, normalise_metadata, normalise_time

    if not isinstance(space, str):
        raise InvalidInput("space must be a string")
    check_space(space)
    if where is not None and (not isinstance(where, Mapping) or any(not isinstance(k, str) for k in where)):
        raise InvalidInput("where must map metadata keys to string values")
    clean_where = normalise_metadata(where if where is not None else {})
    if kind is not None and (not isinstance(kind, str) or kind not in KINDS):
        raise InvalidInput(f"kind must be one of {KINDS}")
    if source_prefix is not None and (not isinstance(source_prefix, str) or len(source_prefix) > MAX_SOURCE):
        raise InvalidInput(f"source_prefix must be a string of at most {MAX_SOURCE} chars")
    for name, value in (("since", since), ("until", until)):
        if value is not None and (not isinstance(value, str) or not value):
            raise InvalidInput(f"{name} must be a timestamp string")
    since_at = normalise_time(since) if since is not None else None
    until_at = normalise_time(until) if until is not None else None
    if since_at is not None and until_at is not None and since_at > until_at:
        raise InvalidInput("since must be at or before until")
    if exclude_session_id is not None:
        normalise_metadata({"session_id": exclude_session_id})
    return TextFilter(where=clean_where, kind=kind, source_prefix=source_prefix, since=since_at, until=until_at)


def _matches(episode: Episode, space: str, filters: TextFilter, exclude_session_id: str | None) -> bool:
    if episode.space != space or (filters.kind is not None and episode.kind != filters.kind):
        return False
    if any(episode.metadata.get(key) != value for key, value in filters.where.items()):
        return False
    if exclude_session_id is not None and episode.metadata.get("session_id") == exclude_session_id:
        return False
    if filters.source_prefix is not None and (episode.source is None or not episode.source.startswith(filters.source_prefix)):
        return False
    if filters.since is not None and episode.created_at < filters.since:
        return False
    return filters.until is None or episode.created_at <= filters.until


async def overview(
    documents: DocumentStore, space: str, *, limit: int = 20, before: int | None = None,
    where: Mapping[str, str] | None = None, kind: str | None = None,
    source_prefix: str | None = None, since: str | None = None, until: str | None = None,
    exclude_session_id: str | None = None, max_records: int = 200,
) -> OverviewResult:
    """Walk at most ``max_records`` recent sources within the supplied space.

    One full retained chunk represents each eligible episode, in descending
    insertion ID order. This reads no embeddings, facts, model, or event log.
    Callers must authorize ``space`` before invoking this engine operation.
    Inventory is a live keyset walk, not a frozen snapshot; new inserts are
    found by restarting it. Stores without bounded inventory are unsupported.
    """
    _integer(limit, "limit", 50)
    _integer(max_records, "max_records", 1000)
    if before is not None:
        _integer(before, "before", 2**63 - 1)
    filters = _filters(space, where, kind, source_prefix, since, until, exclude_session_id)
    if not isinstance(documents, EpisodeInventory) or not callable(documents.page_episodes):
        raise InvalidInput("this document store does not implement source inventory")
    items: list[RecallItem] = []
    cursor, considered = before, 0
    seen: set[int] = set()
    while considered < max_records:
        requested = min(100, max_records - considered)
        batch = await documents.page_episodes(space, cursor, requested, kind)
        if not batch:
            return OverviewResult(items, considered, False, None)
        for index, episode in enumerate(batch[:requested]):
            considered += 1
            if episode.episode_id in seen:
                continue
            if cursor is not None and episode.episode_id >= cursor:
                raise InvalidInput("source inventory must return descending episode IDs")
            cursor = episode.episode_id
            seen.add(cursor)
            if _matches(episode, space, filters, exclude_session_id):
                chunks = await documents.chunks_of(space, episode.episode_id)
                chunk = next((chunk for chunk in chunks if chunk.space == space
                              and chunk.episode_id == episode.episode_id and chunk.text.strip()), None)
                if chunk is not None:
                    items.append(RecallItem(
                        chunk_id=chunk.chunk_id, episode_id=episode.episode_id, text=chunk.text,
                        score=1.0, lanes={"overview": len(items) + 1}, created_at=episode.created_at,
                        source=episode.source, tags=episode.tags, metadata=dict(episode.metadata),
                    ))
            if len(items) == limit:
                more = index + 1 < len(batch) or len(batch) >= requested
                return OverviewResult(items, considered, more, cursor if more else None)
        if len(batch) < requested:
            return OverviewResult(items, considered, False, None)
    return OverviewResult(items, considered, True, cursor)
