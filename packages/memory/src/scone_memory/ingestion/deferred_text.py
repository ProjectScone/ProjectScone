"""Retain conversation text before indexing, using the native recovery journal.

The owner serializes captures/indexing for a source. Deletion may interrupt an
embedding or vector write; every publication checks the retained source again.
This is not a transaction across independent store clients.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from ..core import forget_after
from ..core.errors import InvalidInput
from ..core.models import Chunk, Episode
from ..core.ports import NewChunk, NewEpisode, VectorPoint
from ..core.retirement import supports_retirement
from ..core.space_deletion import supports_space_deletion
from ..core.timeutil import parse_rfc3339
from .batch import IngestionRuntime, _embed_chunks, chunk_record, embedding_inputs, validated_record
from .records import Record, key_hash


@dataclass(frozen=True)
class CapturedText:
    episode_id: int
    indexing: Literal['pending', 'ready']


def is_deferred(episode: Episode) -> bool:
    """Recognize the full native contract, never a lone user metadata flag."""
    metadata = episode.metadata
    if (episode.kind != 'conversation' or metadata.get('capture_indexing') != 'deferred'
            or metadata.get('capture_schema') != '1' or metadata.get('chunking') != 'length'
            or 'chunking_profile' in metadata or 'semantic_merge_threshold' in metadata
            or not episode.content.strip() or len(episode.content.encode()) > 128000):
        return False
    try:
        return episode.content_hash == key_hash(episode.space, metadata.get('capture_key', ''))
    except InvalidInput:
        return False


def _same_record(episode: Episode, new: NewEpisode, *, explicit_time: bool = False) -> bool:
    return (episode.space == new.space and episode.content_hash == new.content_hash
            and episode.content == new.content and episode.source == new.source
            and episode.kind == new.kind and episode.metadata == new.metadata
            and episode.tags == new.tags and (not explicit_time or episode.created_at == new.created_at))


def _new_episode(episode: Episode) -> NewEpisode:
    return NewEpisode(space=episode.space, kind=episode.kind, content=episode.content,
        content_hash=episode.content_hash, created_at=episode.created_at, ingested_at=episode.ingested_at,
        source=episode.source, tags=episode.tags, metadata=dict(episode.metadata))


def _validate_chunks(episode: Episode, chunks: Sequence[Chunk]) -> None:
    """Check exact byte spans, ownership, order and complete nonblank coverage."""
    raw = episode.content.encode()
    previous_start, covered = -1, 0
    ids: set[int] = set()
    for ordinal, chunk in enumerate(chunks):
        if (chunk.space != episode.space or chunk.episode_id != episode.episode_id
                or type(chunk.chunk_id) is not int or chunk.chunk_id < 1 or chunk.chunk_id in ids
                or chunk.ordinal != ordinal or chunk.created_at != episode.created_at
                or not previous_start < chunk.start < chunk.end <= len(raw) or chunk.end <= covered
                or raw[chunk.start:chunk.end] != chunk.text.encode()):
            raise InvalidInput('captured chunk identity or source span is invalid')
        if chunk.start > covered and raw[covered:chunk.start].decode().strip():
            raise InvalidInput('captured chunks omit source text')
        previous_start, covered = chunk.start, chunk.end
        ids.add(chunk.chunk_id)
    if not chunks or raw[covered:].decode().strip():
        raise InvalidInput('captured chunks are incomplete; run native recovery')


async def _space_alive(runtime: IngestionRuntime, space: str) -> bool:
    if supports_space_deletion(runtime.documents) and await runtime.documents.space_deletion(space) is not None:
        return False
    return await runtime.documents.space_deleted(space) is None


async def _retained(runtime: IngestionRuntime, episode: Episode) -> bool:
    current = await runtime.documents.get_episode(episode.space, episode.episode_id)
    if (current is None or current.model_dump_json() != episode.model_dump_json()
            or forget_after.is_due(current.metadata, parse_rfc3339(runtime.clock()))):
        return False
    if not await _space_alive(runtime, episode.space):
        return False
    if supports_retirement(runtime.documents) and await runtime.documents.retirement(episode.space, episode.episode_id) is not None:
        return False
    return await runtime.documents.tombstone(episode.space, episode.episode_id) is None


async def _clear_discarded(runtime: IngestionRuntime, episode: Episode) -> None:
    # A new explicit capture under the old key owns its own recovery mark.
    async with runtime.deferred_write_lock:
        current = await runtime.documents.episode_by_hash(episode.space, episode.content_hash)
        if current is None or (current.model_dump_json() == episode.model_dump_json()
                               and forget_after.is_due(current.metadata, parse_rfc3339(runtime.clock()))):
            await runtime.documents.clear_inflight(episode.space, episode.content_hash)


async def _chunks(runtime: IngestionRuntime, episode: Episode, *, repair: bool) -> list[Chunk]:
    chunks = await runtime.documents.chunks_of(episode.space, episode.episode_id)
    if not chunks and repair:
        new = _new_episode(episode)
        cut = await chunk_record(runtime, new)
        if not await _retained(runtime, episode):
            return []
        chunks = await runtime.documents.insert_chunks([
            NewChunk(episode_id=episode.episode_id, space=episode.space, ordinal=ordinal,
                     start=start, end=end, text=text, created_at=episode.created_at)
            for ordinal, ((start, end), text) in enumerate(zip(cut.spans, cut.texts))
        ])
        if not await _retained(runtime, episode):
            # Chunk identities are never reused. Remove only this source's rows
            # if it disappeared during the interrupted insert.
            if await runtime.documents.get_episode(episode.space, episode.episode_id) is None:
                await runtime.documents.delete_episode(episode.space, episode.episode_id)
            return []
        if runtime.context_lane:
            from .context_terms import index_episode_context
            await index_episode_context(runtime.documents, episode.space, new, chunks)
    _validate_chunks(episode, chunks)
    return chunks


async def retain(runtime: IngestionRuntime, space: str, record: Record) -> CapturedText:
    # Capture never awaits an embedder. Keep its intent publication separate
    # from old indexing/retirement acknowledgments for the same source key.
    async with runtime.deferred_write_lock:
        return await _retain(runtime, space, record)


async def _retain(runtime: IngestionRuntime, space: str, record: Record) -> CapturedText:
    if (record.kind != 'conversation' or not record.dedup_key or record.content_hash is not None
            or record.chunking not in (None, 'length') or record.chunking_profile is not None
            or record.semantic_merge_threshold is not None):
        raise InvalidInput('deferred capture requires keyed, length-chunked conversation text')
    if not isinstance(record.content, str) or len(record.content.encode()) > 128000:
        raise InvalidInput('deferred conversation text exceeds 128000 bytes')
    markers = {'capture_indexing': 'deferred', 'capture_schema': '1', 'capture_key': record.dedup_key}
    if any(name in record.metadata and record.metadata[name] != value for name, value in markers.items()):
        raise InvalidInput('deferred capture metadata conflicts with its native identity')
    new = validated_record(space, replace(record, chunking='length', metadata={**record.metadata, **markers}),
                           runtime.clock(), _deferred_capture=True)
    existing = await runtime.documents.episode_by_hash(space, new.content_hash)
    if existing is not None:
        if not is_deferred(existing) or not _same_record(existing, new, explicit_time=record.created_at is not None):
            raise InvalidInput('capture key already belongs to a different record')
        if not await _retained(runtime, existing):
            raise InvalidInput('captured source is no longer retained')
        if (space, new.content_hash) not in await runtime.documents.inflight():
            _validate_chunks(existing, await runtime.documents.chunks_of(space, existing.episode_id))
            return CapturedText(existing.episode_id, 'ready')
    if not await _space_alive(runtime, space):
        raise InvalidInput('capture space was deleted')
    await runtime.documents.mark_inflight(space, new.content_hash)
    episode = existing if existing is not None else await runtime.documents.insert_episode(new)
    if not _same_record(episode, new, explicit_time=record.created_at is not None):
        raise InvalidInput('capture source does not match its retained record')
    chunks = await _chunks(runtime, episode, repair=True)
    if not chunks or not await _retained(runtime, episode):
        raise InvalidInput('captured source changed before retention completed')
    await runtime.documents.bump_revision(space)
    # Intentionally leave the journal mark until vectors have been written.
    return CapturedText(episode.episode_id, 'pending')


async def _delete_vectors(runtime: IngestionRuntime, ids: Sequence[int]) -> None:
    cleanup = asyncio.create_task(runtime.vectors.delete(ids))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    cleanup.result()


async def index(runtime: IngestionRuntime, space: str, episode_id: int, *, repair: bool = False) -> bool:
    episode = await runtime.documents.get_episode(space, episode_id)
    if episode is None:
        return False
    if episode.space != space or episode.episode_id != episode_id or not is_deferred(episode):
        raise InvalidInput('source was not captured for native deferred indexing')
    # Freeze mutable metadata before any awaited provider/store operation.
    episode = episode.model_copy(deep=True)
    if not await _retained(runtime, episode):
        await _clear_discarded(runtime, episode)
        return False
    if (space, episode.content_hash) not in await runtime.documents.inflight():
        return True
    async with runtime.deferred_write_lock:
        chunks = await _chunks(runtime, episode, repair=repair)
    if not chunks:
        await _clear_discarded(runtime, episode)
        return False
    chunk_snapshot = [chunk.model_dump_json() for chunk in chunks]

    async def retained() -> bool:
        return (await _retained(runtime, episode)
                and [chunk.model_dump_json() for chunk in await runtime.documents.chunks_of(space, episode_id)] == chunk_snapshot)

    texts = await embedding_inputs(runtime, _new_episode(episode), [(c.start, c.end) for c in chunks], [c.text for c in chunks])
    if not await retained():
        await _clear_discarded(runtime, episode)
        return False
    vectors = await _embed_chunks(runtime.embedder, texts, cache=runtime.embedding_cache)
    if not await retained():
        await _clear_discarded(runtime, episode)
        return False
    ids = [chunk.chunk_id for chunk in chunks]
    try:
        await runtime.vectors.upsert([VectorPoint(chunk_id=chunk.chunk_id, space=space, episode_id=episode_id,
            created_at=episode.created_at, vector=vector, tags=episode.tags, metadata=episode.metadata)
            for chunk, vector in zip(chunks, vectors)])
        if not await retained():
            await _delete_vectors(runtime, ids)
            await _clear_discarded(runtime, episode)
            return False
    except BaseException:
        # A failed/partial vector write keeps its durable recovery intent.
        await _delete_vectors(runtime, ids)
        raise
    # Once vectors are verified, an uncertain journal acknowledgement must not
    # erase them: the clear may already have committed. Either outcome is safe
    # to retry, whereas deleting here could lose both vectors and their intent.
    async with runtime.deferred_write_lock:
        if await retained():
            await runtime.documents.clear_inflight(space, episode.content_hash)
            if await retained():
                return True
    await _delete_vectors(runtime, ids)
    await _clear_discarded(runtime, episode)
    return False
