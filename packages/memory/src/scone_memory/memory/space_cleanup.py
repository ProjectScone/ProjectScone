"""Restartable cleanup of an accepted whole-space deletion."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.errors import InvalidInput
from ..core.models import SpaceReceipt
from ..core.retirement import Retirement, supports_retirement
from ..core.space_deletion import SpaceDeletion, SpaceDeletionStore, supports_space_deletion, validate_deletion
from ..core.validation import check_space

if TYPE_CHECKING:
    from .retention import RetentionRuntime


def cleanup_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise InvalidInput('space_deletion_limit must be 1..1000')


def _store(runtime: RetentionRuntime) -> SpaceDeletionStore:
    if not supports_space_deletion(runtime.documents):
        raise InvalidInput('this document store does not implement durable space deletion')
    if not callable(getattr(runtime.documents, 'delete_space', None)):
        raise InvalidInput('this document store does not implement delete-space')
    return runtime.documents


async def delete(runtime: RetentionRuntime, space: str) -> SpaceReceipt:
    check_space(space)
    store = _store(runtime)
    pending = await store.space_deletion(space)
    if pending is None:
        await runtime.living(space)
        chunk_index = getattr(runtime.documents, 'chunk_index', None)
        if not callable(chunk_index):
            raise InvalidInput('space deletion requires a document chunk inventory')
        preview = await runtime.space_receipt(space)
        chunk_ids = tuple(sorted(chunk_id for chunk_id, _ in await chunk_index(space)))
        pending = await store.record_space_deletion(SpaceDeletion(
            space=space, requested_at=runtime.clock(), chunk_ids=chunk_ids, receipt=preview,
            retired_chunk_ids=await _retired_chunks(runtime, space),
        ))
    return await _finish(runtime, store, validate_deletion(pending, space))


async def _retired_chunks(runtime: RetentionRuntime, space: str) -> tuple[int, ...]:
    if not supports_retirement(runtime.documents):
        return ()
    ids: set[int] = set()
    after = None
    while True:
        page = await runtime.documents.page_retirements(after, 100)
        if not page:
            return tuple(sorted(ids))
        for raw in page:
            record = Retirement.model_validate(raw)
            if record.space > space:
                return tuple(sorted(ids))
            if record.space == space:
                ids.update(record.chunk_ids)
        after = (page[-1].space, page[-1].episode_id)


async def _finish(runtime: RetentionRuntime, store: SpaceDeletionStore, pending: SpaceDeletion) -> SpaceReceipt:
    # Repeating each step also handles a write that succeeded but lost its reply.
    # The original counts survive both that ambiguity and already removed rows.
    pending = validate_deletion(pending)
    space = pending.space
    await runtime.blobs.release_space(space)
    await runtime.documents.delete_space(space, pending.requested_at)
    for index in (runtime.vectors, runtime.image_vectors):
        if index is None:
            continue
        sweep = getattr(index, 'delete_space', None)
        if callable(sweep):
            await sweep(space)
        else:
            await index.delete(sorted(set(pending.chunk_ids) | set(pending.retired_chunk_ids)))
    if runtime.events is not None:
        await runtime.events.purge(space)
    await store.clear_space_deletion(space)
    return pending.receipt.model_copy(update={'deleted_at': pending.requested_at})


async def recover(runtime: RetentionRuntime, limit: int) -> tuple[int, bool]:
    cleanup_limit(limit)
    if not supports_space_deletion(runtime.documents):
        return 0, False
    store = _store(runtime)
    completed = 0
    while completed < limit:
        pending = await store.page_space_deletions(None, min(100, limit - completed))
        if not pending:
            return completed, False
        for record in pending:
            await _finish(runtime, store, record)
            completed += 1
    return completed, bool(await store.page_space_deletions(None, 1))
