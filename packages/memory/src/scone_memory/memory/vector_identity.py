"""Which embedder wrote a store's vectors, and what happens when it is not this one.

A cosine between two vectors means something only when one embedder, under
one set of settings, made both. Reopening a store under another embedder of
the same width, or the same embedder with different token rules or
contextual prefixes, used to compare new queries with old vectors and rank
noise. Indexes that can record their writer now do, and the engine settles
the question when it opens:

- ``verified``: the index records this engine's writer, or held no vectors.
- ``rebuilt``: this engine re-embedded every stored chunk and recorded itself.
- ``declared``: an operator vouched that unrecorded vectors are this writer's.
- ``mismatch``: another writer is recorded; the vector lane stays off.
- ``unknown``: vectors exist with no recorded writer; the vector lane stays off.
- ``unverifiable``: the index cannot record a writer; used as before and reported.

An embedder that is local, free and deterministic (the hash embedder) is
rebuilt on open, as a derived index would be. Any other embedder may be slow
or paid, so its vectors are rebuilt only when asked, through
``MemoryEngine.reembed_vectors``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Literal, cast

from ..core.errors import InvalidInput
from ..core.ports import NewEpisode, RecordsVectorWriter, VectorPoint

if TYPE_CHECKING:
    from .engine import MemoryEngine

logger = logging.getLogger(__name__)
State = Literal["verified", "rebuilt", "declared", "mismatch", "unknown", "unverifiable"]
_PAGE = 100
_EMBED_BATCH = 64


@dataclass(frozen=True)
class VectorIdentity:
    state: State
    #: What this engine writes: the embedder id and whether chunks are
    #: embedded with their contextual prefix.
    writer: str
    #: What the index says wrote its vectors, if it records anything.
    recorded: str | None = None

    @property
    def blocked(self) -> str | None:
        """Why the vector lane is off, or None when its vectors can be compared."""
        if self.state == "mismatch":
            return (f"embedder mismatch: stored vectors were written by {self.recorded}, this engine "
                    f"writes {self.writer}; rebuild them with reembed_vectors()")
        if self.state == "unknown":
            return ("embedder unknown: stored vectors predate writer records; rebuild them with "
                    "reembed_vectors() or declare them with adopt_vector_identity()")
        return None


@dataclass(frozen=True)
class ReembedReport:
    spaces: tuple[str, ...]
    chunks: int
    orphans_removed: int


def writer_of(engine: "MemoryEngine") -> str:
    return f"{engine.embedder.id};contextual={int(engine.contextual_embeddings)}"


def _recording(engine: "MemoryEngine") -> RecordsVectorWriter | None:
    vectors = engine.vectors
    needed = ("written_by", "record_writer", "spaces_with_vectors", "ids")
    return cast(RecordsVectorWriter, vectors) if all(callable(getattr(vectors, name, None)) for name in needed) else None


async def settle(engine: "MemoryEngine") -> VectorIdentity:
    """Decide whether this engine may compare its vectors with the stored ones."""
    writer = writer_of(engine)
    index = _recording(engine)
    if index is None:
        return VectorIdentity("unverifiable", writer)
    recorded = await index.written_by()
    if recorded is not None and recorded[0] == writer:
        return VectorIdentity("declared" if recorded[1] == "declared" else "verified", writer, writer)
    if recorded is None and not await index.spaces_with_vectors():
        await index.record_writer(writer, "written")
        return VectorIdentity("verified", writer, writer)
    if getattr(engine.embedder, "cheap_to_rebuild", False):
        report = await rebuild(engine)
        logger.info("vectors.rebuilt", extra={"event": "vectors.rebuilt", "writer": writer,
                    "previous": None if recorded is None else recorded[0], "chunks": report.chunks,
                    "spaces": len(report.spaces), "orphans_removed": report.orphans_removed})
        return VectorIdentity("rebuilt", writer, writer)
    return VectorIdentity("mismatch" if recorded is not None else "unknown", writer,
                          None if recorded is None else recorded[0])


async def rebuild(engine: "MemoryEngine") -> ReembedReport:
    """Re-embed every stored chunk, drop vectors whose chunk is gone, then record the writer.

    The writer is recorded last, so an interrupted rebuild leaves the index
    unaccepted and the next attempt starts over; upserts make that safe.
    """
    index = _recording(engine)
    if index is None:
        raise InvalidInput(f"the {engine.vectors.name} vector index cannot record which embedder wrote "
                           "its vectors, so a rebuild could not be verified")
    page_episodes = getattr(engine.documents, "page_episodes", None)
    if not callable(page_episodes):
        raise InvalidInput(f"the {engine.documents.name} document store cannot list episodes, "
                           "so its chunks cannot be re-embedded")
    spaces = tuple(await index.spaces_with_vectors())
    chunks = orphans = 0
    for space in spaces:
        rebuilt: set[int] = set()
        before: int | None = None
        while episodes := await page_episodes(space, before, _PAGE, None):
            for episode in episodes:
                stored = await engine.documents.chunks_of(space, episode.episode_id)
                new = NewEpisode(space=space, kind=episode.kind, content=episode.content,
                                 content_hash=episode.content_hash, created_at=episode.created_at,
                                 ingested_at=episode.ingested_at, source=episode.source,
                                 tags=episode.tags, metadata=episode.metadata)
                for start in range(0, len(stored), _EMBED_BATCH):
                    batch = stored[start:start + _EMBED_BATCH]
                    embedded = await engine.embedder.embed([engine._embed_text(new, chunk.text) for chunk in batch])
                    if len(embedded) != len(batch):
                        # zip would pair the survivors with the wrong chunks.
                        raise ValueError(f"embedder returned {len(embedded)} vectors for {len(batch)} texts")
                    await engine.vectors.upsert([
                        VectorPoint(chunk_id=chunk.chunk_id, space=space, episode_id=episode.episode_id,
                                    created_at=episode.created_at, vector=vector, tags=episode.tags,
                                    metadata=episode.metadata)
                        for chunk, vector in zip(batch, embedded)])
                rebuilt.update(chunk.chunk_id for chunk in stored)
                chunks += len(stored)
            before = episodes[-1].episode_id
        stale = [chunk_id for chunk_id in await index.ids(space) if chunk_id not in rebuilt]
        if stale:
            await engine.vectors.delete(stale)
            orphans += len(stale)
    await index.record_writer(writer_of(engine), "written")
    return ReembedReport(spaces, chunks, orphans)


async def declare(engine: "MemoryEngine", current: VectorIdentity) -> VectorIdentity:
    """Record that unrecorded vectors were written by this engine's writer.

    Only vectors with no recorded writer can be vouched for. A recorded
    different writer is a fact about the vectors, and only a rebuild
    changes it.
    """
    index = _recording(engine)
    if index is None:
        raise InvalidInput(f"the {engine.vectors.name} vector index cannot record which embedder wrote its vectors")
    if current.state == "mismatch":
        raise InvalidInput(f"stored vectors were written by {current.recorded}; rebuild them with reembed_vectors()")
    if current.state != "unknown":
        return current
    await index.record_writer(current.writer, "declared")
    return VectorIdentity("declared", current.writer, current.writer)
