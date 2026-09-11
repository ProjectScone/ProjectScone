"""Original-backed PDF ingestion and validated, scoped page provenance."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.errors import InvalidInput
from ..core.models import Added, Attachment
from ..core.validation import check_space
from .pdf import ParsedPdf, PdfLimits, PdfPage, PdfParser, PypdfParser, validate_pdf

if TYPE_CHECKING:
    from ..core.ports import EmbeddingCheckpoint
    from ..memory.engine import MemoryEngine


class PdfManifest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1, 2, 3] = 1
    offset_unit: Literal['extracted_text_utf8_bytes'] = 'extracted_text_utf8_bytes'
    geometry: Literal['unrotated_media_box_points'] = 'unrotated_media_box_points'
    original_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    text_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    parser: str = Field(min_length=1, max_length=128)
    pages: tuple[PdfPage, ...] = Field(min_length=1, max_length=1000)

    @model_validator(mode='after')
    def consistent_version(self) -> PdfManifest:
        if self.schema_version == 1 and any(page.extraction == 'ocr' for page in self.pages):
            raise ValueError('OCR provenance requires manifest schema version 2')
        if self.schema_version < 3 and any(page.reading_order is not None for page in self.pages):
            raise ValueError('OCR reading order requires manifest schema version 3')
        return self


@dataclass(frozen=True)
class PdfIngested:
    added: Added
    original: Attachment
    manifest: Attachment
    empty_pages: tuple[int, ...]


@dataclass(frozen=True)
class PdfProvenance:
    original: Attachment
    manifest: Attachment
    parser: str
    pages: tuple[PdfPage, ...]
    empty_pages: tuple[int, ...]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _coverage(pages: tuple[PdfPage, ...]) -> str:
    if any(page.empty for page in pages):
        return 'partial'
    methods = {page.extraction for page in pages}
    return 'mixed' if len(methods) > 1 else next(iter(methods))


async def ingest_pdf(memory: MemoryEngine, space: str, data: bytes, *, filename: str | None = None,
                     limits: PdfLimits = PdfLimits(), parser: PdfParser | None = None,
                     embedding_checkpoint: EmbeddingCheckpoint | None = None) -> PdfIngested:
    """Parse first, then store original, manifest and searchable derived episode.

    Uses existing attachment/write primitives; this is not an atomic ingest job.
    A failed write may leave attachments or a partially linked episode. Retrying
    the same document/parser output reuses its identity and repairs missing links.
    """
    check_space(space)
    if not isinstance(data, bytes) or not data or len(data) > min(limits.max_input_bytes, memory.max_attachment_bytes):
        raise InvalidInput('PDF input exceeds its byte limit or has no bytes')
    if not data.startswith(b'%PDF-'):
        raise InvalidInput('input does not have a PDF header')
    parsed = await (parser or PypdfParser()).parse(data, limits)
    validate_pdf(parsed, limits)
    has_ocr = any(page.extraction == 'ocr' for page in parsed.pages)
    has_order = any(page.reading_order is not None for page in parsed.pages)
    manifest = PdfManifest(schema_version=3 if has_order else 2 if has_ocr else 1, original_sha256=_sha(data), text_sha256=_sha(parsed.text.encode()),
        parser=parsed.parser, pages=parsed.pages)
    encoded = manifest.model_dump_json(exclude=None if has_ocr else {
        'pages': {'__all__': {'extraction', 'region_geometry', 'regions', 'ocr_engine'}}}).encode()
    if len(encoded) > memory.max_attachment_bytes:
        raise InvalidInput('PDF manifest exceeds its attachment byte limit')
    # Include actual output and parser version: new source bytes or new parser
    # output must never inherit another source's deduplicated episode metadata.
    identity = f'pdf-v1:{_sha(data)}:{_sha(encoded)}'
    original = await memory.attach(space, data, media_type='application/pdf', filename=filename)
    if original.media_type != 'application/pdf':
        raise InvalidInput('PDF original is already retained with an incompatible media type')
    retained = await memory.attach(space, encoded, media_type='application/json', filename='pdf-provenance.json')
    if retained.media_type != 'application/json':
        raise InvalidInput('PDF manifest is already retained with an incompatible media type')
    empty = tuple(page.number for page in parsed.pages if page.empty)
    added = await memory.remember(space, parsed.text, kind='file', source=f'attachment:{original.attachment_id}',
        dedup_key=identity, attachment_ids=(original.attachment_id, retained.attachment_id),
        embedding_checkpoint=embedding_checkpoint,
        metadata={'document_format': 'pdf', 'evidence_origin': 'extracted_text',
            'pdf_original': original.attachment_id, 'pdf_manifest': retained.attachment_id,
            'pdf_coverage': _coverage(parsed.pages)})
    return PdfIngested(added, original, retained, empty)


async def pdf_provenance(memory: MemoryEngine, space: str, episode_id: int, *,
                         start: int = 0, end: int | None = None,
                         chunk_id: int | None = None) -> PdfProvenance:
    """Resolve an extracted UTF-8 span to retained PDF pages, within its space.

    Validates hashes, links, schema and spans. Records are not a signature or an
    independent verification that a supplied parser interpreted the PDF correctly.
    """
    episode = await memory.episode(space, episode_id)
    linked = {item.attachment_id: item for item in episode.attachments}
    original_id, manifest_id = episode.metadata.get('pdf_original'), episode.metadata.get('pdf_manifest')
    if not original_id or not manifest_id or original_id not in linked or manifest_id not in linked:
        raise InvalidInput('PDF original and manifest must be linked to this episode')
    original, raw = await memory.attachment(space, original_id)
    retained, encoded = await memory.attachment(space, manifest_id)
    if (original.media_type != 'application/pdf' or retained.media_type != 'application/json'
            or _sha(raw) != original_id or _sha(encoded) != manifest_id):
        raise InvalidInput('PDF retained attachment content or type does not match its identity')
    try:
        manifest = PdfManifest.model_validate_json(encoded)
    except ValidationError as error:
        raise InvalidInput('PDF provenance manifest is invalid') from error
    content = episode.content.encode('utf-8')
    if manifest.original_sha256 != original_id or manifest.text_sha256 != _sha(content):
        raise InvalidInput('PDF provenance does not match its original or extracted text')
    validate_pdf(ParsedPdf(text=episode.content, parser=manifest.parser, pages=manifest.pages), PdfLimits(max_pages=1000))
    if chunk_id is not None:
        if type(chunk_id) is not int or chunk_id <= 0 or start != 0 or end is not None:
            raise InvalidInput('provide a positive chunk id or an explicit span, not both')
        chunks = await memory.documents.get_chunks(space, [chunk_id])
        if len(chunks) != 1 or chunks[0].episode_id != episode_id:
            raise InvalidInput('PDF chunk does not belong to this episode')
        chunk = chunks[0]
        if content[chunk.start:chunk.end] != chunk.text.encode('utf-8'):
            raise InvalidInput('PDF chunk does not match the extracted text')
        start, end = chunk.start, chunk.end
    stop = len(content) if end is None else end
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(content):
        raise InvalidInput('PDF span must be a nonempty range in extracted text bytes')
    try:
        content[:start].decode('utf-8')
        content[:stop].decode('utf-8')
    except UnicodeDecodeError as error:
        raise InvalidInput('PDF span splits a UTF-8 character') from error
    selected = tuple(page for page in manifest.pages if page.start < stop and page.end > start and not page.empty)
    return PdfProvenance(original, retained, manifest.parser, selected,
        tuple(page.number for page in manifest.pages if page.empty))
