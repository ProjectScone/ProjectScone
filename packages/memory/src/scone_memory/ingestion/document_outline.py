"""What an imported file said about its own shape, placed in the text it was stored as.

The stored content of an imported file is its segments' text joined by blank lines, which
says nothing of the headings or units the reader found. The readers keep a heading's level
beside its segment (``heading_level``) and name each segment by where it came from (its
locator). The cutter and the heading path read the content alone; this gives them the
headings and units back, from the manifest kept with the episode, so ingestion and recovery
read the same ones.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import re
from typing import TYPE_CHECKING

from ..backends.blobs import BlobStore
from ..core.errors import InvalidInput
from ..core.ports import NewEpisode
from .formats.types import DocumentSegment
from .pdf import PdfPage
from .structure import Heading, SourceUnit
from .table_context import retained_manifest

if TYPE_CHECKING:
    from .documents import PdfManifest

__all__ = ["DocumentOutline", "Heading", "SourceUnit", "document_outline", "outline", "pdf_units", "source_units",
           "unit_of"]

_SEPARATOR = len(b"\n\n")
_MAX_PDF_MANIFEST_BYTES = 25 * 1024 * 1024
#: Locator stems that name a unit, first match wins: a page holds its tables' rows.
_UNITS = (re.compile(r"^((?:page|slide|frame|video:stream:\d+/frame):\d+)(?:/|$)"),
          re.compile(r"^(audio:\d+/segment:\d+)(?:/|$)"),
          re.compile(r"^(line:\d+)#"),
          re.compile(r"^((?:[^/]+/)*?row:\d+)(?:/|$)"),
          # The legacy Excel reader joins sheet and row with a colon; the sheet name is escaped.
          re.compile(r"^(sheet:[^/:]*:row:\d+)$"))
_CELL = re.compile(r"^(sheet:[^/]+)/cell:[A-Za-z]{1,3}(\d+)$")


@dataclass(frozen=True)
class DocumentOutline:
    headings: tuple[Heading, ...] = ()
    units: tuple[SourceUnit, ...] = ()


def outline(segments: Sequence[DocumentSegment]) -> tuple[Heading, ...]:
    """Each heading segment's byte offset in the joined text, its level and its title, with
    runs of whitespace in the title made one space."""
    found: list[Heading] = []
    offset = 0
    for segment in segments:
        written = segment.metadata.get("heading_level", "")
        title = " ".join(segment.text.split())
        if written.isdecimal() and int(written) > 0 and title:
            found.append(Heading(offset, int(written), title))
        offset += len(segment.text.encode()) + _SEPARATOR
    return tuple(found)


def unit_of(locator: str) -> str | None:
    """The unit a locator names: a page, slide, image frame or video frame; an audio segment;
    a JSON Lines record; a table or sheet row, a spreadsheet cell's row. None for anything else."""
    cell = _CELL.match(locator)
    if cell:
        return f"{cell[1]}/row:{cell[2]}"
    for pattern in _UNITS:
        found = pattern.match(locator)
        if found:
            return found[1]
    return None


def source_units(segments: Sequence[DocumentSegment]) -> tuple[SourceUnit, ...]:
    """Consecutive segments of one unit as that unit, and a run of segments in no unit as one
    ``text`` unit, each spanning its first segment's first byte to its last segment's last."""
    units: list[SourceUnit] = []
    offset = 0
    for segment in segments:
        size = len(segment.text.encode())
        label = unit_of(segment.locator) or "text"
        if units and units[-1].label == label and units[-1].end + _SEPARATOR == offset:
            units[-1] = SourceUnit(units[-1].start, offset + size, label)
        else:
            units.append(SourceUnit(offset, offset + size, label))
        offset += size + _SEPARATOR
    return tuple(units)


def pdf_units(pages: Sequence[PdfPage]) -> tuple[SourceUnit, ...]:
    """Each page with text, by the byte span the manifest gives it."""
    return tuple(SourceUnit(page.start, page.end, f"page:{page.number}") for page in pages
                 if not page.empty and page.end > page.start)


async def _retained_pdf(episode: NewEpisode, blobs: BlobStore) -> "PdfManifest | None":
    from .documents import PdfManifest
    from .pdf import ParsedPdf, PdfLimits, validate_pdf

    original_id = episode.metadata.get("pdf_original")
    manifest_id = episode.metadata.get("pdf_manifest")
    if episode.kind != "file" or (not original_id and not manifest_id):
        return None
    if (not original_id or not manifest_id or re.fullmatch(r"[a-f0-9]{64}", original_id) is None
            or re.fullmatch(r"[a-f0-9]{64}", manifest_id) is None or episode.source != f"attachment:{original_id}"):
        raise InvalidInput("PDF outline has invalid source identities")
    original, raw = await blobs.get(episode.space, original_id)
    retained, encoded = await blobs.get(episode.space, manifest_id)
    if (not 0 < len(encoded) <= _MAX_PDF_MANIFEST_BYTES or original.media_type != "application/pdf"
            or hashlib.sha256(raw).hexdigest() != original_id
            or hashlib.sha256(encoded).hexdigest() != manifest_id or retained.media_type != "application/json"):
        raise InvalidInput("PDF outline does not match its retained sources")
    try:
        manifest = PdfManifest.model_validate_json(encoded)
    except (ValueError, RecursionError):
        raise InvalidInput("PDF outline requires a valid manifest") from None
    if (manifest.original_sha256 != original_id
            or manifest.text_sha256 != hashlib.sha256(episode.content.encode()).hexdigest()):
        raise InvalidInput("PDF outline does not match the episode")
    # The pages must tile the text as the provenance route requires, or their spans cannot be cut.
    validate_pdf(ParsedPdf(text=episode.content, parser=manifest.parser, pages=manifest.pages), PdfLimits(max_pages=1000))
    return manifest


async def document_outline(episode: NewEpisode, *, blobs: BlobStore) -> DocumentOutline | None:
    """The headings and units of a file episode's retained manifest; None for an episode that
    is not an imported file. A manifest that does not match the episode is refused, as for
    table context."""
    manifest = await retained_manifest(episode, blobs=blobs, purpose="document heading outline")
    if manifest is not None:
        segments = manifest.parsed.segments
        return DocumentOutline(outline(segments), source_units(segments))
    pdf = await _retained_pdf(episode, blobs)
    return None if pdf is None else DocumentOutline(units=pdf_units(pdf.pages))
