"""Index attributed image occurrences and resolve search hits to retained images."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import Added, Attachment, RecallResult
from ..core.validation import check_space
from ..providers.vision import SUPPORTED_IMAGE_TYPES
from ..ocr.process import run_bounded
from .image_context import ImageAttribute, ImageContext, ImageEntity, context_text

__all__ = ['ImageAttribute', 'ImageContext', 'ImageEntity', 'ImageIngested', 'ImageProvenance',
    'ImageMatch', 'ImageRecall', 'ingest_image', 'image_provenance', 'recall_images']


if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine


class ImageManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1] = 1
    original_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    text_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    media_type: Literal['image/png', 'image/jpeg', 'image/webp']
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    context: ImageContext


@dataclass(frozen=True)
class ImageIngested:
    added: Added
    image: Attachment
    manifest: Attachment


@dataclass(frozen=True)
class ImageProvenance:
    image: Attachment
    manifest: Attachment
    context: ImageContext
    width: int
    height: int


@dataclass(frozen=True)
class ImageMatch:
    episode_id: int
    chunk_id: int
    score: float
    matched_text: str
    image: Attachment
    context: ImageContext
    width: int
    height: int


@dataclass(frozen=True)
class ImageRecall:
    matches: tuple[ImageMatch, ...]
    recall: RecallResult
    unresolved_episode_ids: tuple[int, ...]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entity_tag(entity_id: str) -> str:
    if not isinstance(entity_id, str) or not entity_id or len(entity_id) > 256:
        raise InvalidInput('image entity ID must contain 1..256 characters')
    # Preserve case-sensitive external identity through casefolded engine tags.
    return _sha(('image-entity:' + entity_id).encode())


async def ingest_image(memory: MemoryEngine, space: str, data: bytes, *, media_type: Literal['image/png', 'image/jpeg', 'image/webp'],
                       context: ImageContext, filename: str | None = None) -> ImageIngested:
    """Retain original + context manifest, then index attribution without inference.

    Writes use existing primitives, not a transaction. Exact retries repair links.
    Same bytes in different source occurrences share a blob, not their descriptions.
    """
    check_space(space)
    if not isinstance(data, bytes) or not 0 < len(data) <= min(10_000_000, memory.max_attachment_bytes):
        raise InvalidInput('image input is empty or exceeds its byte limit')
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise InvalidInput('image context supports still PNG, JPEG and WebP images')
    context = ImageContext.model_validate_json(context.model_dump_json())
    output = await run_bounded([sys.executable, '-m', 'scone_memory.ingestion._image_worker', media_type],
        data, timeout=15., max_output=4096)
    dimensions = json.loads(output)
    if 'error' in dimensions:
        raise InvalidInput(str(dimensions['error']))
    width, height = dimensions['width'], dimensions['height']
    text = context_text(context)
    manifest = ImageManifest(original_sha256=_sha(data), text_sha256=_sha(text.encode()),
        media_type=media_type, width=width, height=height, context=context)
    encoded = manifest.model_dump_json().encode()
    if len(encoded) > memory.max_attachment_bytes:
        raise InvalidInput('image context exceeds the attachment byte limit')
    image = await memory.attach(space, data, media_type=media_type, filename=filename)
    if image.media_type != media_type:
        raise InvalidInput('retained image has an incompatible media type')
    retained = await memory.attach(space, encoded, media_type='application/json', filename='image-context.json')
    if retained.media_type != 'application/json':
        raise InvalidInput('retained image context has an incompatible media type')
    added = await memory.remember(space, text, kind='file', source=context.source,
        tags=tuple(_entity_tag(entity.entity_id) for entity in context.entities),
        attachment_ids=(image.attachment_id, retained.attachment_id),
        dedup_key=f'image-v1:{image.attachment_id}:{retained.attachment_id}',
        metadata={'document_format': 'image', 'evidence_origin': 'image_context',
            'image_original': image.attachment_id, 'image_manifest': retained.attachment_id})
    return ImageIngested(added, image, retained)


async def image_provenance(memory: MemoryEngine, space: str, episode_id: int) -> ImageProvenance:
    episode = await memory.episode(space, episode_id)
    linked = {attachment.attachment_id for attachment in episode.attachments}
    original_id, manifest_id = episode.metadata.get('image_original'), episode.metadata.get('image_manifest')
    if not original_id or not manifest_id or not {original_id, manifest_id} <= linked:
        raise InvalidInput('image and context must be linked to the source episode')
    retained, encoded = await memory.attachment(space, manifest_id)
    try:
        manifest = ImageManifest.model_validate_json(encoded)
    except ValidationError as error:
        raise InvalidInput('image context manifest is invalid') from error
    if (_sha(encoded) != manifest_id or retained.media_type != 'application/json'
            or manifest.original_sha256 != original_id
            or manifest.text_sha256 != _sha(episode.content.encode())
            or context_text(manifest.context) != episode.content or episode.source != manifest.context.source):
        raise InvalidInput('image context does not match its retained source')
    image = next(attachment for attachment in episode.attachments if attachment.attachment_id == original_id)
    if image.media_type != manifest.media_type:
        raise InvalidInput('image media type does not match its context')
    # The immutable blob ID is retained in the result. Download through attachment()
    # only when needed; search does not fetch each potentially multi-MB image.
    return ImageProvenance(image, retained, manifest.context, manifest.width, manifest.height)


async def recall_images(memory: MemoryEngine, space: str, query: str, *, limit: int = 5,
                        entity_id: str | None = None) -> ImageRecall:
    """Use the configured hybrid index; return authorized references, not base64 blobs."""
    if type(limit) is not int or not 1 <= limit <= 25:
        raise InvalidInput('image result limit must be between 1 and 25')
    tags = () if entity_id is None else (_entity_tag(entity_id),)
    recalled = await memory.recall(space, query, limit=limit * 4,
        where={'document_format': 'image'}, tags=tags)
    matches: list[ImageMatch] = []
    unresolved: list[int] = []
    seen: set[int] = set()
    for item in recalled.items:
        if item.episode_id in seen:
            continue
        seen.add(item.episode_id)
        try:
            provenance = await image_provenance(memory, space, item.episode_id)
            if entity_id is not None and not any(entity.entity_id == entity_id for entity in provenance.context.entities):
                raise InvalidInput('image entity tag does not match its attributed entities')
        except (InvalidInput, NotFound, Gone):
            unresolved.append(item.episode_id)
            continue
        matches.append(ImageMatch(item.episode_id, item.chunk_id, item.score, item.text,
            provenance.image, provenance.context, provenance.width, provenance.height))
        if len(matches) == limit:
            break
    return ImageRecall(tuple(matches), recalled, tuple(unresolved))
