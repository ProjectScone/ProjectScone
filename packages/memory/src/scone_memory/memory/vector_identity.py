"""Which embedder wrote a store's vectors, and what happens when it is not this one.

A cosine between two vectors means something only when one embedder, under
one set of settings, made both. Reopening a store under another embedder of
the same width, or the same embedder with other token rules or contextual
prefixes, used to compare new queries with old vectors and rank noise.

Indexes that can record their writer keep the invariant in
``core.vector_writers``: a recorded writer really wrote every vector, and a
write that would make that untrue changes the record in the same
transaction. The engine reads the record whenever it is about to compare
vectors, not only when it opens, so an engine that another process rebuilt
underneath stops trusting its vectors at once. States:

- ``verified``: the index records this engine's writer.
- ``rebuilt``: this engine re-embedded every stored chunk and recorded itself.
- ``declared``: an operator vouched that unrecorded vectors are this writer's.
- ``mismatch``: another writer is recorded.
- ``unknown``: some vectors have no recorded writer.
- ``mixed``: vectors from more than one writer.
- ``interrupted``: a rebuild began and never finished.
- ``unverifiable``: the index cannot record a writer; used as before and reported.

Only the first three let recall compare vectors (and ``unverifiable``, which
is unchanged behaviour). The hash embedder is local, free and deterministic,
so a store it cannot trust is rebuilt when it opens, as a derived index
would be. Any other embedder may be slow or paid and is rebuilt only when
asked, through ``MemoryEngine.reembed_vectors``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Sequence, cast
from uuid import uuid4

from ..core.errors import InvalidInput, SconeError
from ..core.ports import NewEpisode, RecordsVectorWriter, VectorIndex, VectorPoint
from ..core.vector_writers import Writer, rebuilding_token

if TYPE_CHECKING:
    from .engine import MemoryEngine

logger = logging.getLogger(__name__)
State = Literal["verified", "rebuilt", "declared", "mismatch", "unknown", "mixed", "interrupted", "unverifiable"]
_TRUSTED = frozenset({"verified", "rebuilt", "declared", "unverifiable"})
_BY_BASIS: dict[str, State] = {"written": "verified", "declared": "declared"}


class VectorWriterChanged(SconeError):
    """Another writer changed the vectors while this one was settling them."""
_PAGE = 100
_EMBED_BATCH = 64


@dataclass(frozen=True)
class VectorIdentity:
    state: State
    #: What this engine writes: the embedder id and whether chunks are
    #: embedded with their contextual prefix.
    writer: str
    #: What the index records as having written its vectors, if anything.
    recorded: str | None = None

    @property
    def blocked(self) -> str | None:
        """Why the vector lane is off, or None when its vectors can be compared."""
        if self.state in _TRUSTED:
            return None
        why = {
            "mismatch": f"embedder mismatch: stored vectors were written by {self.recorded}, "
                        f"this engine writes {self.writer}",
            "unknown": "embedder unknown: some stored vectors predate writer records",
            "mixed": "embedder mixed: stored vectors come from more than one writer",
            "interrupted": "embedder rebuild interrupted: stored vectors are part old, part new",
        }[self.state]
        fix = "; rebuild them with reembed_vectors()"
        if self.state == "unknown":
            fix += " or declare them with adopt_vector_identity()"
        return why + fix


@dataclass(frozen=True)
class ReembedReport:
    spaces: tuple[str, ...]
    chunks: int
    orphans_removed: int


def writer_of(engine: "MemoryEngine") -> str:
    return f"{engine.embedder.id};contextual={int(engine.contextual_embeddings)}"


def recording(vectors: object) -> RecordsVectorWriter | None:
    needed = ("written_by", "holds_vectors", "swap_writer", "upsert_as", "spaces_with_vectors", "ids")
    return cast(RecordsVectorWriter, vectors) if all(callable(getattr(vectors, name, None)) for name in needed) else None


class GuardedVectors:
    """A vector index whose every write goes through its writer check.

    Everything but ``upsert`` passes straight through. Ingestion, recovery
    and rebuilds all write through here, so none of them can put one
    embedder's vectors under another's name.
    """

    def __init__(self, inner: VectorIndex, writer: Callable[[], str]) -> None:
        self._inner = inner
        self._writer = writer
        self.name = inner.name

    @property
    def inner(self) -> VectorIndex:
        return self._inner

    async def ensure(self, dim: int) -> None:
        await self._inner.ensure(dim)

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        await cast(RecordsVectorWriter, self._inner).upsert_as(points, self._writer())

    async def search(self, space: str, vector: Sequence[float], limit: int, as_of: Optional[str] = None,
                     tags: tuple[str, ...] = (), where: Mapping[str, str] | None = None) -> list[tuple[int, float]]:
        return await self._inner.search(space, vector, limit, as_of, tags, where)

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        await self._inner.delete(chunk_ids)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def guard(vectors: VectorIndex, writer: Callable[[], str]) -> VectorIndex:
    """Wrap an index that records writers; leave any other index as it is."""
    return cast(VectorIndex, GuardedVectors(vectors, writer)) if recording(vectors) is not None else vectors


def _index(engine: "MemoryEngine") -> RecordsVectorWriter | None:
    vectors = engine.vectors
    return recording(vectors.inner if isinstance(vectors, GuardedVectors) else vectors)


def _state(record: Writer | None, holds: bool, writer: str) -> State:
    if record is None:
        return "unknown" if holds else "verified"
    name, basis = record
    if basis == "rebuilding":
        return "interrupted"
    if basis == "invalidated":
        return "mixed"
    if name != writer:
        return "mismatch"
    return _BY_BASIS.get(basis, "unknown")


async def observe(engine: "MemoryEngine") -> VectorIdentity:
    """Read the index's record now and say whether its vectors can be compared."""
    writer = writer_of(engine)
    index = _index(engine)
    if index is None:
        return VectorIdentity("unverifiable", writer)
    record = await index.written_by()
    holds = record is None and await index.holds_vectors()
    return VectorIdentity(_state(record, holds, writer), writer, None if record is None else record[0])


async def settle(engine: "MemoryEngine") -> VectorIdentity:
    """Decide, on open, whether this engine may compare its vectors with the stored ones."""
    index = _index(engine)
    seen = await observe(engine)
    if index is None or seen.state != "verified" or seen.recorded is not None:
        if seen.blocked is not None and getattr(engine.embedder, "cheap_to_rebuild", False):
            report = await rebuild(engine)
            logger.info("vectors.rebuilt", extra={"event": "vectors.rebuilt", "writer": seen.writer,
                        "previous": seen.recorded, "previous_state": seen.state, "chunks": report.chunks,
                        "spaces": len(report.spaces), "orphans_removed": report.orphans_removed})
            return VectorIdentity("rebuilt", seen.writer, seen.writer)
        return seen
    # An empty index with no record: claim it, unless someone wrote first.
    if await index.swap_writer(None, (seen.writer, "written"), require_empty=True):
        return VectorIdentity("verified", seen.writer, seen.writer)
    return await settle(engine)


async def rebuild(engine: "MemoryEngine") -> ReembedReport:
    """Re-embed every stored chunk, drop vectors whose chunk is gone, then record the writer.

    Before the first vector changes, the record becomes a rebuild marker that
    matches no writer, so an interruption leaves nothing trusted. The writer
    is recorded only if the marker is still in place at the end: a write by
    another embedder during the rebuild turns the record to ``mixed`` and the
    rebuild reports it instead of vouching for vectors it did not write.
    """
    index = _index(engine)
    if index is None:
        raise InvalidInput(f"the {engine.vectors.name} vector index cannot record which embedder wrote "
                           "its vectors, so a rebuild could not be verified")
    page_episodes = getattr(engine.documents, "page_episodes", None)
    if not callable(page_episodes):
        raise InvalidInput(f"the {engine.documents.name} document store cannot list episodes, "
                           "so its chunks cannot be re-embedded")
    writer = writer_of(engine)
    marker = rebuilding_token(writer, uuid4().hex)
    while not await index.swap_writer(await index.written_by(), marker):
        pass
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
                    await index.upsert_as([
                        VectorPoint(chunk_id=chunk.chunk_id, space=space, episode_id=episode.episode_id,
                                    created_at=episode.created_at, vector=vector, tags=episode.tags,
                                    metadata=episode.metadata)
                        for chunk, vector in zip(batch, embedded)], writer)
                rebuilt.update(chunk.chunk_id for chunk in stored)
                chunks += len(stored)
            before = episodes[-1].episode_id
        stale = [chunk_id for chunk_id in await index.ids(space) if chunk_id not in rebuilt]
        if stale:
            await engine.vectors.delete(stale)
            orphans += len(stale)
    if not await index.swap_writer(marker, (writer, "written")):
        raise VectorWriterChanged("another embedder wrote vectors during the rebuild; run the rebuild again")
    return ReembedReport(spaces, chunks, orphans)


async def declare(engine: "MemoryEngine") -> VectorIdentity:
    """Record that vectors with no recorded writer were written by this engine's writer.

    Only vectors no one recorded can be vouched for. A recorded different
    writer, a mix of writers or an unfinished rebuild are facts about the
    vectors, and only a rebuild changes them.
    """
    index = _index(engine)
    if index is None:
        raise InvalidInput(f"the {engine.vectors.name} vector index cannot record which embedder wrote its vectors")
    current = await observe(engine)
    if current.state in _TRUSTED:
        return current
    if current.state != "unknown":
        raise InvalidInput(f"{current.blocked}; a declaration cannot settle this")
    record = await index.written_by()
    if record is not None and record[0] != current.writer:
        raise InvalidInput(f"stored vectors include ones written by {record[0]}; rebuild them with reembed_vectors()")
    if not await index.swap_writer(record, (current.writer, "declared")):
        raise VectorWriterChanged("the vectors' writer changed while declaring; check again")
    return VectorIdentity("declared", current.writer, current.writer)
