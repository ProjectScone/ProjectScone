"""The headings an imported file said it had, placed in the text it was stored as.

The readers keep a heading's level beside its segment (``heading_level``), because the
stored content is the segments' text joined by blank lines and says nothing of it. The
cutter and the heading path read that content alone; this gives them the headings back,
from the manifest kept with the episode, so ingestion and recovery read the same ones.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..backends.blobs import BlobStore
from ..core.ports import NewEpisode
from .formats.types import DocumentSegment
from .structure import Heading
from .table_context import retained_manifest

__all__ = ["Heading", "document_headings", "outline"]


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
        offset += len(segment.text.encode()) + len(b"\n\n")
    return tuple(found)


async def document_headings(episode: NewEpisode, *, blobs: BlobStore) -> tuple[Heading, ...] | None:
    """The outline of a file episode's retained manifest; None for an episode that is not an
    imported file. A manifest that does not match the episode is refused, as for table
    context."""
    manifest = await retained_manifest(episode, blobs=blobs, purpose="document heading outline")
    return None if manifest is None else outline(manifest.parsed.segments)
