"""Verified video source snapshots shared by pixel reads and explicit inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import Attachment, Episode
from .files import DocumentProvenance, digest, document_provenance

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine


@dataclass(frozen=True)
class VideoSource:
    evidence: DocumentProvenance
    episode: Episode
    original: Attachment
    manifest: Attachment
    raw: bytes
    encoded: bytes


async def read_video_source(engine: MemoryEngine, space: str, episode_id: int, ordinal: int) -> VideoSource:
    evidence = await document_provenance(engine, space, episode_id)
    retained = evidence.video
    if retained is None or evidence.parser != 'video-frame-ocr':
        raise InvalidInput('document has no retained video frame evidence')
    if ordinal not in {item.ordinal for item in retained.frames}:
        raise InvalidInput('requested frame is not in the retained video evidence')
    original, raw = await engine.attachment(space, evidence.original.attachment_id)
    manifest, encoded = await engine.attachment(space, evidence.manifest.attachment_id)
    source = await engine.episode(space, episode_id)
    required_links = {original.attachment_id, manifest.attachment_id}
    if (source.kind != 'file' or original != evidence.original or manifest != evidence.manifest
            or digest(raw) != original.attachment_id or digest(encoded) != manifest.attachment_id
            or source.metadata.get('document_original') != original.attachment_id
            or source.metadata.get('document_manifest') != manifest.attachment_id
            or source.metadata.get('document_format') != evidence.format
            or source.content != '\n\n'.join(segment.text for segment in evidence.segments)
            or not required_links <= {item.attachment_id for item in source.attachments}):
        raise InvalidInput('video document changed before frame decoding')
    return VideoSource(evidence, source.model_copy(deep=True), original, manifest, raw, encoded)


async def verify_video_source(engine: MemoryEngine, space: str, episode_id: int, saved: VideoSource) -> None:
    # Hydration awaits storage; observe the raw source row last to catch a
    # forget during hydration. These observations are not a global transaction.
    await engine.episode(space, episode_id)
    original, raw = await engine.attachment(space, saved.original.attachment_id)
    manifest, encoded = await engine.attachment(space, saved.manifest.attachment_id)
    await engine.episode(space, episode_id)
    links = await engine.blobs.for_episode(space, episode_id)
    current = await engine.documents.get_episode(space, episode_id)
    if current is None:
        tombstone = await engine.tombstone(space, episode_id)
        if tombstone is not None:
            raise Gone('This source was forgotten', tombstone.forgotten_at)
        raise NotFound('Video source unavailable in this space')
    source = saved.episode
    if (not {original.attachment_id, manifest.attachment_id} <= {item.attachment_id for item in links}
            or (current.kind, current.source, current.content, current.metadata)
                != (source.kind, source.source, source.content, source.metadata)
            or original != saved.original or manifest != saved.manifest
            or raw != saved.raw or encoded != saved.encoded):
        raise InvalidInput('video source changed during frame verification')
