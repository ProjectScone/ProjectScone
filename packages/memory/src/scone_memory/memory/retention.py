"""Source retention, integrity inspection and whole-space deletion.

Deletion spans independent stores in an explicit order, not one distributed
transaction. Failures propagate; completed store writes are not rolled back.
Source deletion preserves claims as ledger history and invalidates evidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Optional

from ..backends.blobs import BlobStore
from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import DoctorReport, Episode, ExpiryReport, ForgetReceipt, SpaceReceipt, Tombstone
from ..core.ports import DocumentStore, EventLog, NewTombstone, VectorIndex
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space, retention_policy


@dataclass(frozen=True)
class RetentionRuntime:
    documents: DocumentStore
    vectors: VectorIndex
    blobs: BlobStore
    events: Optional[EventLog]
    clock: Callable[[], str]
    emit: Callable[[str, str, dict[str, object]], Awaitable[object]]
    episode_or_gone: Callable[[str, int], Awaitable[Episode]]
    impact: Callable[[str, int], Awaitable[ForgetReceipt]]
    forget: Callable[[str, int], Awaitable[ForgetReceipt]]
    living: Callable[[str], Awaitable[None]]
    space_deleted: Callable[[str], Awaitable[Optional[str]]]
    space_receipt: Callable[[str], Awaitable[SpaceReceipt]]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


async def episode_or_gone(runtime: RetentionRuntime, space: str, episode_id: int) -> Episode:
    """The episode, or Gone when a tombstone says it was forgotten, or
    NotFound when the id never meant anything here."""
    found = await runtime.documents.get_episode(space, episode_id)
    if found is not None:
        return found
    stone = await runtime.documents.tombstone(space, episode_id)
    if stone is not None:
        raise Gone(f"episode {episode_id} was forgotten on {stone.forgotten_at}", stone.forgotten_at)
    raise NotFound(f"episode {episode_id} not found in {space!r}")


async def tombstone(runtime: RetentionRuntime, space: str, episode_id: int) -> Optional[Tombstone]:
    """The record that an episode was forgotten, or None."""
    check_space(space)
    return await runtime.documents.tombstone(space, episode_id)


async def doctor(runtime: RetentionRuntime, space: str) -> DoctorReport:
    """What references what across the space's stores, read only: chunks
    whose episode is gone, vectors whose chunk is gone, facts citing a
    forgotten or an unknown episode, links with a missing end, held
    attachments no episode carries. A store that cannot be walked is
    named in not_inspected rather than reported clean."""
    check_space(space)
    counts = await runtime.documents.counts(space)
    facts = await runtime.documents.list_facts(space, include_closed=True)
    report = DoctorReport(space=space, episodes=counts.episodes, chunks=counts.chunks, facts=len(facts))
    episode_ids = {e.episode_id for e in await runtime.documents.recent_episodes(space, max(counts.episodes, 1))}
    stones = {t.episode_id for t in await runtime.documents.list_tombstones(space)}
    report.tombstones = len(stones)
    for fact in facts:
        source = fact.source_episode_id
        if source is None or source in episode_ids:
            continue
        (report.facts_citing_forgotten if source in stones else report.facts_citing_unknown).append(fact.fact_id)
    fact_ids = {f.fact_id for f in facts}
    seen: set[int] = set()
    for fact in facts:
        for link in await runtime.documents.fact_links(space, fact.fact_id):
            if link.link_id in seen:
                continue
            seen.add(link.link_id)
            if link.from_fact not in fact_ids or link.to_fact not in fact_ids:
                report.links_with_missing_ends.append(link.link_id)
    report.links = len(seen)
    chunk_index = getattr(runtime.documents, "chunk_index", None)
    chunk_ids: Optional[set[int]] = None
    if callable(chunk_index):
        pairs = await chunk_index(space)
        chunk_ids = {chunk_id for chunk_id, _ in pairs}
        report.chunks_without_episode = sorted(chunk_id for chunk_id, episode_id in pairs if episode_id not in episode_ids)
    else:
        report.not_inspected.append("chunks")
    vector_ids = getattr(runtime.vectors, "ids", None)
    if callable(vector_ids) and chunk_ids is not None:
        report.vectors_without_chunk = sorted(v for v in await vector_ids(space) if v not in chunk_ids)
    else:
        report.not_inspected.append("vectors")
    held = getattr(runtime.blobs, "held", None)
    if callable(held):
        linked = await runtime.blobs.linked(space)
        report.attachments_unlinked = [a for a in await held(space) if a not in linked]
    else:
        report.not_inspected.append("attachments")
    report.healthy = not any([
        report.chunks_without_episode, report.vectors_without_chunk, report.facts_citing_forgotten,
        report.facts_citing_unknown, report.links_with_missing_ends, report.attachments_unlinked,
    ])
    return report


async def expire(runtime: RetentionRuntime, space: str, policy: Mapping[str, float], *, limit: int = 100,
                 dry_run: bool = False) -> ExpiryReport:
    """Forget the episodes a retention policy no longer keeps: for each
    kind in ``policy``, those whose own time is more than that many
    days before this engine's clock, oldest first, at most ``limit``
    in one pass. Facts never expire; the claims that cited a forgotten
    episode stand. ``dry_run`` reports and forgets nothing."""
    check_space(space)
    clean = retention_policy(policy)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
        raise InvalidInput("limit must be an integer in 1..10000")
    now = parse_rfc3339(runtime.clock())
    counts = await runtime.documents.counts(space)
    due = []
    for episode in await runtime.documents.recent_episodes(space, max(counts.episodes, 1)):
        days = clean.get(episode.kind)
        if days is not None and (now - parse_rfc3339(episode.created_at)).total_seconds() > days * 86400:
            due.append(episode)
    due.sort(key=lambda e: (parse_rfc3339(e.created_at), e.episode_id))
    report = ExpiryReport(space=space, policy=dict(clean), remaining=len(due), dry_run=dry_run)
    if dry_run:
        return report
    for episode in due[:limit]:
        report.receipts.append(await runtime.forget(space, episode.episode_id))
        report.forgotten.append(episode.episode_id)
    report.remaining = len(due) - len(report.forgotten)
    await runtime.emit(space, "expire", {"policy": dict(clean), "forgotten": len(report.forgotten),
                                       "remaining": report.remaining, "limit": limit})
    return report


async def impact(runtime: RetentionRuntime, space: str, episode_id: int) -> ForgetReceipt:
    """What forgetting the episode would take with it and leave, with
    nothing removed. The claims and links that cite it are reported,
    not closed: a source being gone is a fact about the evidence."""
    check_space(space)
    await runtime.episode_or_gone(space, episode_id)
    carried = [a.attachment_id for a in await runtime.blobs.for_episode(space, episode_id)]
    released = await runtime.blobs.released_by(space, episode_id)
    facts = await runtime.documents.list_facts(space, include_closed=True)
    citing_links: dict[int, int] = {}
    for fact in facts:
        for link in await runtime.documents.fact_links(space, fact.fact_id):
            if link.source_episode_id == episode_id:
                citing_links[link.link_id] = link.link_id
    return ForgetReceipt(
        episode_id=episode_id,
        chunks=len(await runtime.documents.chunks_of(space, episode_id)),
        attachments_released=released,
        attachments_kept=[a for a in carried if a not in released],
        facts_citing=sorted(f.fact_id for f in facts if f.source_episode_id == episode_id),
        links_citing=sorted(citing_links),
    )


async def forget(runtime: RetentionRuntime, space: str, episode_id: int) -> ForgetReceipt:
    """Remove the episode, its chunks and vectors, and release the
    attachments nothing else carries; return the receipt that
    ``impact`` would have shown. Claims and links stand."""
    check_space(space)
    started = time.perf_counter()
    try:
        receipt = await runtime.impact(space, episode_id)
    except NotFound as error:
        await runtime.emit(space, "forget", {"episode_id": episode_id, "error": type(error).__name__, "latency_ms": _ms(started)})
        raise
    episode = await runtime.episode_or_gone(space, episode_id)
    removed = await runtime.documents.delete_episode(space, episode_id)
    await runtime.vectors.delete(removed)
    await runtime.blobs.unlink(space, episode_id)
    # The tombstone outlives the episode: the id keeps meaning something
    # and the decision is not undone by a later import.
    stone = await runtime.documents.record_tombstone(NewTombstone(
        space=space, episode_id=episode_id, content_hash=episode.content_hash, forgotten_at=runtime.clock(),
    ))
    receipt = receipt.model_copy(update={"forgotten_at": stone.forgotten_at})
    await runtime.documents.bump_revision(space)
    await runtime.emit(space, "forget", {
        "episode_id": episode_id, "chunks_removed": len(removed),
        "attachments_released": len(receipt.attachments_released),
        "facts_citing": len(receipt.facts_citing), "links_citing": len(receipt.links_citing),
        "latency_ms": _ms(started),
    })
    return receipt


async def living(runtime: RetentionRuntime, space: str) -> None:
    """Refuse a space that was deleted: a key left in config must not
    re-create what was erased."""
    when = await runtime.space_deleted(space)
    if when is not None:
        raise NotFound(f"space {space!r} was deleted at {when}")


async def space_deleted(runtime: RetentionRuntime, space: str) -> Optional[str]:
    """When the space was deleted, or None while it lives. A store
    that cannot record a deletion has never deleted one."""
    check_space(space)
    deleted = getattr(runtime.documents, "space_deleted", None)
    return await deleted(space) if callable(deleted) else None


async def space_receipt(runtime: RetentionRuntime, space: str) -> SpaceReceipt:
    counts = await runtime.documents.counts(space)
    facts = await runtime.documents.list_facts(space, include_closed=True)
    links: set[int] = set()
    for fact in facts:
        for link in await runtime.documents.fact_links(space, fact.fact_id):
            links.add(link.link_id)
    released, kept = await runtime.blobs.release_space(space, preview=True)
    return SpaceReceipt(
        space=space, episodes=counts.episodes, chunks=counts.chunks, facts=len(facts), links=len(links),
        tombstones=len(await runtime.documents.list_tombstones(space)),
        events=(await runtime.events.purge(space, preview=True)) if runtime.events is not None else 0,
        attachments_released=released, attachments_kept=kept,
    )


async def space_impact(runtime: RetentionRuntime, space: str) -> SpaceReceipt:
    """What deleting the space would take with it, with nothing removed."""
    check_space(space)
    await runtime.living(space)
    return await runtime.space_receipt(space)


async def delete_space(runtime: RetentionRuntime, space: str) -> SpaceReceipt:
    """Remove everything the space holds, in order: attachment holds
    (bytes only when no other space holds them), the records of the
    space with their vectors, then the event trail; mark the space
    deleted so no write re-creates it. Returns the receipt
    ``space_impact`` would have shown, with the counts of the deed."""
    check_space(space)
    await runtime.living(space)
    if not callable(getattr(runtime.documents, "delete_space", None)):
        raise InvalidInput("this document store does not implement delete-space")
    preview = await runtime.space_receipt(space)
    deleted_at = runtime.clock()
    released, kept = await runtime.blobs.release_space(space)
    gone = await runtime.documents.delete_space(space, deleted_at)
    sweep = getattr(runtime.vectors, "delete_space", None)
    if sweep is not None:
        await sweep(space)
    else:
        await runtime.vectors.delete(list(gone.chunk_ids))
    events = (await runtime.events.purge(space)) if runtime.events is not None else 0
    return preview.model_copy(update={
        "episodes": gone.episodes, "chunks": len(gone.chunk_ids), "facts": gone.facts, "links": gone.links,
        "tombstones": gone.tombstones, "events": events, "attachments_released": released,
        "attachments_kept": kept, "deleted_at": deleted_at,
    })
