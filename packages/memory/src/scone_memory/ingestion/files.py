"""Retained originals and source locators for every document parser."""
from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import hashlib
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.errors import InvalidInput
from ..core.models import Added, Attachment
from ..core.validation import MAX_METADATA_VALUE, check_space
from .formats.registry import BuiltinDocumentParser, DocumentParser, extension
from .formats.types import DocumentLimits, DocumentSegment, ParsedDocument, validate_document
from .document_source import DocumentSource, source_revision_key

if TYPE_CHECKING:
    from ..core.ports import EmbeddingCheckpoint
    from ..memory.engine import MemoryEngine


FILE_MEDIA_TYPES = {
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.doc': 'application/msword', '.xls': 'application/vnd.ms-excel', '.ppt': 'application/vnd.ms-powerpoint',
    '.xlsb': 'application/vnd.ms-excel.sheet.binary.macroEnabled.12',
    '.odt': 'application/vnd.oasis.opendocument.text', '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
    '.odp': 'application/vnd.oasis.opendocument.presentation', '.epub': 'application/epub+zip',
    '.pdf': 'application/pdf', '.json': 'application/json', '.ipynb': 'application/json', '.jsonl': 'application/x-ndjson',
    '.ndjson': 'application/x-ndjson', '.xml': 'application/xml', '.html': 'text/html', '.htm': 'text/html',
    '.csv': 'text/csv', '.tsv': 'text/tab-separated-values', '.eml': 'message/rfc822',
    '.msg': 'application/vnd.ms-outlook', '.rtf': 'application/rtf',
}


class DocumentManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1, 2, 3, 4, 5] = 1
    offset_unit: Literal['extracted_text_utf8_bytes'] = 'extracted_text_utf8_bytes'
    original_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    filename: str = Field(min_length=1, max_length=1024)
    parsed: ParsedDocument

    @model_validator(mode='after')
    def validate_version(self) -> Self:
        _validate_manifest_version(self)
        return self


def _validate_manifest_version(manifest: DocumentManifest) -> None:
    if manifest.schema_version < 5 and any(c.context or c.merged_locators
            for s in manifest.parsed.segments for c in s.table_cells):
        raise ValueError('document merged table sources require manifest version five')
    if manifest.schema_version < 4 and any(s.table_cells for s in manifest.parsed.segments):
        raise ValueError('document table cells require manifest version four')
    if manifest.schema_version == 1 and any(segment.regions for segment in manifest.parsed.segments):
        raise ValueError('document regions require manifest version two')
    if manifest.schema_version < 3 and any('ocr_reading_order' in segment.metadata
            or any(r.provider_index is not None for r in segment.regions) for segment in manifest.parsed.segments):
        raise ValueError('document OCR reading order requires manifest version three')


def encode_manifest(manifest: DocumentManifest) -> bytes:
    """Preserve legacy attachment and dedup identities for documents without regions."""
    try:
        _validate_manifest_version(manifest)
    except ValueError as error:
        raise InvalidInput(str(error)) from error
    if manifest.schema_version == 1:
        if any(segment.regions for segment in manifest.parsed.segments):
            raise InvalidInput('document regions require manifest version two')
        return manifest.model_dump_json(exclude={'parsed': {'segments': {'__all__': {'regions'}}}}).encode()
    return manifest.model_dump_json().encode()


@dataclass(frozen=True)
class DocumentIngested:
    added: Added
    original: Attachment
    manifest: Attachment
    format: str
    segments: int
    filename: str


@dataclass(frozen=True)
class DocumentProvenance:
    original: Attachment
    manifest: Attachment
    filename: str
    format: str
    parser: str
    segments: tuple[DocumentSegment, ...]
    metadata: dict[str, str] = field(default_factory=dict)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extraction_filename(original: Attachment, filename: str | None = None) -> str:
    """Choose a parse label independently of the blob's first-upload metadata."""
    selected = original.filename if filename is None else filename
    if selected is None:
        raise InvalidInput('document requires an extraction filename with its format extension')
    extension(selected)
    return selected


async def prepare_document(data: bytes, filename: str, *, parser: DocumentParser,
                           limits: DocumentLimits) -> DocumentManifest:
    extension(filename)
    if not isinstance(data, bytes) or not data or len(data) > limits.max_input_bytes:
        raise InvalidInput('document exceeds its input byte limit or is empty')
    try:
        parsed = await asyncio.wait_for(parser.parse(data, filename, limits), limits.timeout_seconds)
    except asyncio.TimeoutError:
        raise InvalidInput('document parser exceeded its wall time limit') from None
    validate_document(parsed, limits)
    has_order = any('ocr_reading_order' in s.metadata for s in parsed.segments)
    has_tables = any(s.table_cells for s in parsed.segments)
    has_merges = any(c.context or c.merged_locators for s in parsed.segments for c in s.table_cells)
    return DocumentManifest(schema_version=5 if has_merges else 4 if has_tables else 3 if has_order else 2 if any(s.regions for s in parsed.segments) else 1,
                            original_sha256=digest(data), filename=filename, parsed=parsed)


async def store_document(memory: MemoryEngine, space: str, original: Attachment,
                         manifest: DocumentManifest, *,
                         source: DocumentSource | None = None,
                         embedding_checkpoint: EmbeddingCheckpoint | None = None) -> DocumentIngested:
    """Index prepared extraction. Replays repair links using content identities."""
    validate_document(manifest.parsed, DocumentLimits())
    if original.attachment_id != manifest.original_sha256:
        raise InvalidInput('document extraction does not match its original')
    encoded = encode_manifest(manifest)
    if len(encoded) > memory.max_attachment_bytes:
        raise InvalidInput('document manifest exceeds its attachment byte limit')
    identity = source_revision_key(source, original.attachment_id, digest(encoded)) if source is not None else None
    source_metadata = source.metadata() if source is not None else {}
    retained = await memory.attach(space, encoded, 'application/json', filename='document-provenance.json')
    if retained.media_type != 'application/json':
        raise InvalidInput('document manifest has an incompatible retained media type')
    content = '\n\n'.join(segment.text for segment in manifest.parsed.segments)
    added = await memory.remember(space, content, kind='file', source=f'attachment:{original.attachment_id}',
        dedup_key=identity or f'document-v1:{original.attachment_id}:{retained.attachment_id}',
        attachment_ids=(original.attachment_id, retained.attachment_id),
        embedding_checkpoint=embedding_checkpoint,
        metadata={'document_format': manifest.parsed.format,
                  **({'document_filename': manifest.filename} if len(manifest.filename) <= MAX_METADATA_VALUE else {}),
                  'document_original': original.attachment_id,
                  'document_manifest': retained.attachment_id, 'evidence_origin': 'extracted_text',
                  **source_metadata})
    return DocumentIngested(added, original, retained, manifest.parsed.format,
                            len(manifest.parsed.segments), manifest.filename)


async def ingest_document(memory: MemoryEngine, space: str, data: bytes, *, filename: str,
                           parser: DocumentParser | None = None,
                           limits: DocumentLimits = DocumentLimits()) -> DocumentIngested:
    """Parse, retain, index and link a file. Use the workflow for durable retries."""
    check_space(space)
    if len(data) > memory.max_attachment_bytes:
        raise InvalidInput('document exceeds the attachment byte limit')
    manifest = await prepare_document(data, filename, parser=parser or BuiltinDocumentParser(), limits=limits)
    if len(encode_manifest(manifest)) > memory.max_attachment_bytes:
        raise InvalidInput('document manifest exceeds its attachment byte limit')
    original = await memory.attach(space, data, FILE_MEDIA_TYPES.get(extension(filename), 'application/octet-stream'),
                                   filename=filename)
    return await store_document(memory, space, original, manifest)


async def document_provenance(memory: MemoryEngine, space: str, episode_id: int, *,
                              chunk_id: int | None = None) -> DocumentProvenance:
    """Resolve a retrieved chunk to file-local rows, cells, slides or other locators."""
    episode = await memory.episode(space, episode_id)
    linked = {attachment.attachment_id for attachment in episode.attachments}
    original_id = episode.metadata.get('document_original')
    manifest_id = episode.metadata.get('document_manifest')
    if not original_id or not manifest_id or not {original_id, manifest_id} <= linked:
        raise InvalidInput('document original and manifest must be linked to the episode')
    original, raw = await memory.attachment(space, original_id)
    retained, encoded = await memory.attachment(space, manifest_id)
    if digest(raw) != original_id or digest(encoded) != manifest_id or retained.media_type != 'application/json':
        raise InvalidInput('document evidence does not match its retained identity')
    try:
        manifest = DocumentManifest.model_validate_json(encoded)
    except ValidationError:
        raise InvalidInput('document provenance manifest is invalid') from None
    validate_document(manifest.parsed, DocumentLimits())
    content = '\n\n'.join(segment.text for segment in manifest.parsed.segments).encode()
    if (manifest.original_sha256 != original_id or content != episode.content.encode()
            or episode.metadata.get('document_format') != manifest.parsed.format):
        raise InvalidInput('document provenance does not match its original or extracted text')
    start, end = 0, len(content)
    if chunk_id is not None:
        if type(chunk_id) is not int or chunk_id <= 0:
            raise InvalidInput('document chunk id must be a positive integer')
        chunks = await memory.documents.get_chunks(space, [chunk_id])
        if len(chunks) != 1 or chunks[0].episode_id != episode_id:
            raise InvalidInput('document chunk does not belong to the episode')
        chunk = chunks[0]
        if content[chunk.start:chunk.end] != chunk.text.encode():
            raise InvalidInput('document chunk does not match its extracted text')
        start, end = chunk.start, chunk.end
    offset = 0
    selected: list[DocumentSegment] = []
    for segment in manifest.parsed.segments:
        stop = offset + len(segment.text.encode())
        if offset < end and stop > start:
            regions = tuple(region for region in segment.regions
                            if offset + region.start < end and offset + region.end > start)
            cells = segment.table_cells if chunk_id is None else tuple(cell for cell in segment.table_cells
                          if offset + cell.start < end and offset + cell.end > start)
            selected.append(segment.model_copy(update={'regions': regions, 'table_cells': cells}))
        offset = stop + 2
    return DocumentProvenance(original, retained, manifest.filename, manifest.parsed.format,
                              manifest.parsed.parser, tuple(selected), dict(manifest.parsed.metadata))
