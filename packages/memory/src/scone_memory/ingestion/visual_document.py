"""Retain textless video sources only against complete, hash-bound evidence."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..core.errors import InvalidInput
from .files import DocumentManifest, digest
from .formats.types import DocumentLimits, validate_document, visual_only
from .records import Record, RetainedVideoRecord, content_hash

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine


async def verify_visual_record(memory: MemoryEngine, space: str, record: Record,
                               episode_id: int | None = None) -> None:
    """Verify retained bytes before insertion; repair links before completing a write."""
    if not isinstance(record, RetainedVideoRecord) or record.content != '' or record.kind != 'file':
        raise InvalidInput('visual-only retention requires an explicit empty document record')
    original_id = record.metadata.get('document_original')
    manifest_id = record.metadata.get('document_manifest')
    if (not original_id or not manifest_id or original_id == manifest_id
            or record.source != f'attachment:{original_id}'
            or record.metadata.get('evidence_origin') != 'sampled_video_frames'
            or (record.dedup_key is None and record.content_hash is None)):
        raise InvalidInput('visual-only document identity is invalid')
    original, raw = await memory.attachment(space, original_id)
    retained, encoded = await memory.attachment(space, manifest_id)
    if (original.attachment_id != original_id or retained.attachment_id != manifest_id
            or digest(raw) != original_id or digest(encoded) != manifest_id
            or retained.media_type != 'application/json'):
        raise InvalidInput('visual-only document attachments do not match their retained identities')
    try:
        manifest = DocumentManifest.model_validate_json(encoded)
    except ValidationError:
        raise InvalidInput('visual-only document manifest is invalid') from None
    validate_document(manifest.parsed, DocumentLimits())
    if (manifest.schema_version != 7 or not visual_only(manifest.parsed)
            or manifest.original_sha256 != original_id
            or record.metadata.get('document_format') != manifest.parsed.format):
        raise InvalidInput('visual-only document does not match its retained evidence')
    if episode_id is None:
        return
    episode = await memory.episode(space, episode_id)
    expected = record.content_hash or content_hash(space, '', record.dedup_key)
    if (episode.content != '' or episode.kind != 'file' or episode.source != record.source
            or episode.content_hash != expected or episode.metadata != dict(record.metadata)):
        raise InvalidInput('visual-only document episode does not match its retained evidence')
    for attachment_id in (original_id, manifest_id):
        await memory.blobs.link(space, attachment_id, episode_id)
