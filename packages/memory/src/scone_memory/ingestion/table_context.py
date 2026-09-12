"""Embedding context backed by retained, source-declared table relationships."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
import hashlib
import json
import re

from pydantic import ValidationError

from ..backends.blobs import BlobStore
from ..core.errors import InvalidInput
from ..core.ports import NewEpisode
from .files import DocumentManifest
from .formats.types import DocumentLimits, ParsedDocument, validate_document
from .formats.table_types import DocumentTableCell

TABLE_CONTEXT_VERSION = 'source-table-headers-v1'
MAX_CONTEXT_BYTES = 8192
MAX_EPISODE_CONTEXT_BYTES = 8 * 1024 * 1024
_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate manifest key')
        result[key] = value
    return result


def context_inputs(parsed: ParsedDocument, spans: Sequence[tuple[int, int]]) -> list[str]:
    """Add exact cell/header quotes to embedding inputs; never change the source."""
    validate_document(parsed, DocumentLimits())
    content = '\n\n'.join(segment.text for segment in parsed.segments).encode()
    cells: list[tuple[int, int, DocumentTableCell]] = []
    offset = 0
    for segment in parsed.segments:
        for cell in segment.table_cells:
            if cell.is_header or not cell.text or not (cell.headers or cell.context):
                continue
            cells.append((offset + cell.start, offset + cell.end, cell))
        offset += len(segment.text.encode()) + 2
    ends = [end for _, end, _ in cells]
    result: list[str] = []
    total = previous_end = 0
    for start, end in spans:
        if (type(start) is not int or type(end) is not int
                or not previous_end <= start < end <= len(content)):
            raise InvalidInput('document embedding chunk spans are invalid')
        previous_end = end
        try:
            excerpt = content[start:end].decode('utf-8')
        except UnicodeError:
            raise InvalidInput('document embedding chunk span splits UTF-8') from None
        selected: list[str] = []
        size = 2
        for index in range(bisect_right(ends, start), len(cells)):
            cell_start, cell_end, cell = cells[index]
            if cell_start >= end:
                break
            headers = [header for header in cell.headers if header.text not in excerpt]
            context = [reference for reference in cell.context if reference.text not in excerpt]
            if not headers and not context:
                continue
            values = [header.text for header in headers] + [reference.text for reference in context]
            raw = ' / '.join(dict.fromkeys(values))
            if raw in selected:
                continue
            size += len(raw.encode()) + (1 if selected else 0)
            if size > MAX_CONTEXT_BYTES:
                raise InvalidInput('document table embedding context exceeds its chunk byte limit')
            selected.append(raw)
        if not selected:
            result.append(excerpt)
            continue
        prefix = 'Table context: ' + ' | '.join(selected) + '\n\n'
        total += len(prefix.encode())
        if len(prefix.encode()) > MAX_CONTEXT_BYTES or total > MAX_EPISODE_CONTEXT_BYTES:
            raise InvalidInput('document table embedding context exceeds its byte limit')
        result.append(prefix + excerpt)
    return result


async def embedding_inputs(episode: NewEpisode, spans: Sequence[tuple[int, int]], *,
                           blobs: BlobStore) -> list[str]:
    """Load within the record's space, including before initial attachment linking."""
    original_id = episode.metadata.get('document_original')
    manifest_id = episode.metadata.get('document_manifest')
    if episode.kind != 'file' or (not original_id and not manifest_id):
        content = episode.content.encode()
        return [content[start:end].decode() for start, end in spans]
    if (not original_id or not manifest_id
            or re.fullmatch(r'[a-f0-9]{64}', original_id) is None
            or re.fullmatch(r'[a-f0-9]{64}', manifest_id) is None
            or episode.source != f'attachment:{original_id}'):
        raise InvalidInput('document table embedding context has invalid source identities')
    original, raw = await blobs.get(episode.space, original_id)
    retained, encoded = await blobs.get(episode.space, manifest_id)
    if (not 0 < len(raw) <= _MAX_ATTACHMENT_BYTES or not 0 < len(encoded) <= _MAX_ATTACHMENT_BYTES
            or original.attachment_id != original_id or retained.attachment_id != manifest_id
            or hashlib.sha256(raw).hexdigest() != original_id
            or hashlib.sha256(encoded).hexdigest() != manifest_id or retained.media_type != 'application/json'):
        raise InvalidInput('document table embedding context does not match its retained sources')
    try:
        json.loads(encoded, object_pairs_hook=_unique)
        manifest = DocumentManifest.model_validate_json(encoded)
    except (ValueError, RecursionError, ValidationError):
        raise InvalidInput('document table embedding context requires a valid manifest') from None
    if (manifest.original_sha256 != original_id
            or manifest.parsed.format != episode.metadata.get('document_format')
            or '\n\n'.join(segment.text for segment in manifest.parsed.segments) != episode.content
            or ('document_filename' in episode.metadata
                and episode.metadata['document_filename'] != manifest.filename)):
        raise InvalidInput('document table embedding context does not match the episode')
    return context_inputs(manifest.parsed, spans)
