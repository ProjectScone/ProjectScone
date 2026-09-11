"""Read the current source under a stable key without mutating its state."""
from __future__ import annotations

from ..backends.blobs import BlobStore
from ..core.errors import Conflict, Gone, NotFound
from ..core.models import Episode, Tombstone
from ..core.ports import DocumentStore
from ..core.validation import check_space
from ..ingestion.records import key_hash


def _check_identity(record: Episode | Tombstone | None, space: str, digest: str) -> None:
    if record is not None and (record.space != space or record.content_hash != digest):
        raise RuntimeError("document store returned a mismatched source-key identity")


async def episode_by_key(documents: DocumentStore, blobs: BlobStore, space: str, dedup_key: str) -> Episode:
    check_space(space)
    digest = key_hash(space, dedup_key)
    for _ in range(3):
        found = await documents.episode_by_hash(space, digest)
        _check_identity(found, space, digest)
        tombstone = await documents.tombstone_by_hash(space, digest) if found is None else None
        _check_identity(tombstone, space, digest)
        carried = await blobs.for_episode(space, found.episode_id) if found is not None else ()
        # A source may change while its separate attachment store is read.
        # Recheck identity, including an absent key's last deletion, before
        # presenting the evidence. This is a read, not a lock for a next write.
        latest_tombstone = await documents.tombstone_by_hash(space, digest) if found is None else None
        _check_identity(latest_tombstone, space, digest)
        current = await documents.episode_by_hash(space, digest)
        _check_identity(current, space, digest)
        if ((current.episode_id if current is not None else None)
                != (found.episode_id if found is not None else None) or tombstone != latest_tombstone):
            continue
        if found is not None:
            return found.model_copy(update={"attachments": tuple(carried)})
        if tombstone is not None:
            raise Gone(f"source key's episode {tombstone.episode_id} was forgotten", tombstone.forgotten_at)
        raise NotFound("source key not found in this space")
    raise Conflict("source key changed during lookup; retry the read", await documents.revision(space))
