"""The image lane: stored images found by a text query in an image embedder's space.

An image ingested with its context (``ingestion.images``) is found by the
text lanes only through the words said about it. With an image embedder and
an index of its own configured on the engine, each stored image also gets
one vector made from its bytes, keyed by the first chunk of the episode that
carries it. A recall that asks for the lane embeds the query with the same
embedder, searches that index under the recall's own filters, and fuses the
chunks it finds with the other lanes by rank at ``IMAGE_WEIGHT``. The chunk
is the image's caption, so a result from the lane carries the episode's
``image_original`` (the image's attachment id) and ``image_manifest`` (the
retained context the caption was made from) like any other result for it.

The index is separate because a cosine between an image and a query means
something only when one model made both: vectors from the text embedder and
the image embedder never meet. Forgetting an episode through an engine with
the lane removes its image vector with its text vectors, since both are keyed
by its chunk ids; a vector written while that forget ran is taken out again
(``index_image``). A forget through an engine without the lane cannot reach
this index, and its receipt says so; an engine with the lane removes what such
forgets left when it opens (``remove_forgotten``) and whenever a recall's
search meets one (``search_images``).

Every image vector carries the id and width of the image embedder that made
it (``IMAGE_WRITER``, ``IMAGE_WIDTH``). An index that records its writer, like
the text vectors' (the in-memory and SQLite ones), refuses an engine whose
image embedder is not that one: it is told so when it opens (``writer_block``)
and writes nothing there, since one write would record the index as mixed and
leave it unusable by either embedder. Any other index cannot record a writer,
so the lane there searches only the vectors tagged with its own embedder's id
and ignores the rest (``records_writer``).

LlamaIndex's multi-modal index keeps image nodes in an ``image`` vector
store namespace and appends the image results to the text results; this lane
instead ranks them with the other lanes, so one ordering answers the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, cast

from ..core.errors import Gone, NotFound
from ..core.models import Episode
from ..core.ports import DocumentStore, ImageEmbedder, VectorIndex, VectorPoint
from ..core.vector_writers import VectorsNotComparable, vouches
from ..memory.vector_identity import GuardedVectors

#: The lane's weight in reciprocal-rank fusion beside the text lane's 1.
#: Not measured on any benchmark: there is no image retrieval set here and
#: no model to run one with. The context and entity lanes chose 2.0 on
#: their benchmarks; this one keeps the text lane's voice until it has one.
IMAGE_WEIGHT = 1.0

#: How many times one recall searches the image index. A search whose hits
#: include vectors of forgotten images (a forget through an engine without the
#: lane leaves them) removes those and searches again, so they cannot keep
#: live images out of the lane's window. A recall still finding them on its
#: last search ranks fewer images than it looks for, and says so.
IMAGE_SEARCHES = 2

#: Metadata on every image vector: the id of the image embedder that made it,
#: and its width. Where the index cannot record its writer, the lane searches
#: only vectors whose ``IMAGE_WRITER`` is its own embedder's id. The width is
#: recorded, not searched on: an index holds vectors of one width.
IMAGE_WRITER = "image_embedder"
IMAGE_WIDTH = "image_embedder_dim"

_PAGE = 100


@dataclass(frozen=True)
class ImageSearch:
    """What one recall's image lane found, and what it removed on the way."""

    #: Ranked (chunk_id, cosine) of live images only.
    hits: list[tuple[int, float]]
    #: Vectors of forgotten images the searches met and removed.
    removed: int
    #: The last search still met some: the lane ranked fewer images than it looks for.
    short: bool
    #: Rows the last search returned, removed ones included. A window they
    #: filled hid deeper images whether or not those rows were live.
    returned: int


@dataclass(frozen=True)
class ImageLane:
    """What a recall needs to run the lane: the embedder and the index only it writes."""

    embedder: ImageEmbedder
    vectors: VectorIndex


class _ConditionNarrowing(Protocol):
    async def search(self, space: str, vector: Sequence[float], limit: int, as_of: Optional[str] = None,
                     tags: tuple[str, ...] = (), where: Mapping[str, str] | None = None,
                     conditions: object = None) -> list[tuple[int, float]]: ...


async def index_image(documents: DocumentStore, embedder: ImageEmbedder, vectors: VectorIndex, space: str,
                      episode: Episode, data: bytes) -> int:
    """Write one vector for the image ``data`` that ``episode`` carries; return the chunk id it is keyed by.

    Written again for the same episode it replaces the one before, so an
    exact retry repairs a vector a failed write left out. When the episode is
    forgotten before the vector lands, no vector is left and it raises ``Gone``,
    or ``NotFound`` while that forget has not yet recorded its tombstone. When
    the index records another writer it writes nothing and raises
    ``VectorsNotComparable``."""
    # A context episode always has text, so a live one always has a first chunk.
    chunks = await documents.chunks_of(space, episode.episode_id)
    if not chunks:
        raise await _forgotten(documents, space, episode)
    chunk_id = chunks[0].chunk_id
    [vector] = await embedder.embed_images([data])
    # Read the record as late as possible: another process may have written since this engine opened.
    blocked = await writer_block(vectors, embedder.id)
    if blocked is not None:
        raise VectorsNotComparable(blocked)
    await vectors.upsert([VectorPoint(chunk_id=chunk_id, space=space, episode_id=episode.episode_id,
                                      created_at=episode.created_at, vector=vector, tags=episode.tags,
                                      metadata={**episode.metadata, IMAGE_WRITER: embedder.id,
                                                IMAGE_WIDTH: str(embedder.dim)})])
    # Forgetting deletes the chunks before it deletes the image vectors. A
    # forget that ran while the image was embedded has already deleted this
    # key, so the write above put a vector back for a chunk that is gone.
    if not await documents.get_chunks(space, [chunk_id]):
        await vectors.delete([chunk_id])
        raise await _forgotten(documents, space, episode)
    return chunk_id


async def _forgotten(documents: DocumentStore, space: str, episode: Episode) -> NotFound:
    stone = await documents.tombstone(space, episode.episode_id)
    if stone is None:
        return NotFound(f"episode {episode.episode_id} is being forgotten while its image was indexed; "
                        "no image vector is kept")
    return Gone(f"episode {episode.episode_id} was forgotten while its image was indexed; no image vector is kept",
                stone.forgotten_at)


def records_writer(vectors: VectorIndex) -> bool:
    """Whether the index records which image embedder wrote it, and so refuses
    another; the engine checks every write and search of such an index."""
    return isinstance(vectors, GuardedVectors)


async def writer_block(vectors: VectorIndex, embedder_id: str) -> str | None:
    """Why the image embedder ``embedder_id`` must not write to or compare with
    the index's vectors, as the index records them now; None when it may, or
    when the index records no writers at all."""
    written_by = getattr(vectors, "written_by", None)
    holds_vectors = getattr(vectors, "holds_vectors", None)
    if not callable(written_by) or not callable(holds_vectors):
        return None
    record = await written_by()
    if vouches(record, embedder_id, record is None and await holds_vectors()):
        return None
    name, basis = record if record is not None else ("no writer", "unrecorded")
    return (f"image embedder mismatch: stored image vectors are recorded as {name} ({basis}), "
            f"this engine's image embedder is {embedder_id}; give it an image index of its own "
            "and run ingest_image again for each image")


async def search_images(lane: ImageLane, documents: DocumentStore, space: str, query: str, depth: int,
                        as_of: Optional[str], tags: tuple[str, ...], where: Mapping[str, str],
                        conditions: object = None) -> ImageSearch:
    """The stored images nearest the query, under the recall's filters.

    A condition the index cannot evaluate is left to the recall's post-filter,
    as it is for the vector lane. A hit whose chunk is gone is a forgotten
    image's vector: it is removed and the index searched again, up to
    ``IMAGE_SEARCHES`` times."""
    [vector] = await lane.embedder.embed_texts([query])
    if not records_writer(lane.vectors):
        # Nothing refuses another model's vectors here: its own tag leaves them out.
        where = {**where, IMAGE_WRITER: lane.embedder.id}
    removed = 0
    for _ in range(IMAGE_SEARCHES):
        if conditions is not None and getattr(lane.vectors, "narrows_conditions", False):
            found = await cast(_ConditionNarrowing, lane.vectors).search(space, vector, depth, as_of, tags, where,
                                                                         conditions=conditions)
        else:
            found = await lane.vectors.search(space, vector, depth, as_of, tags, where)
        live = await _live(documents, space, [chunk_id for chunk_id, _ in found])
        stale = [chunk_id for chunk_id, _ in found if chunk_id not in live]
        if not stale:
            return ImageSearch(found, removed, False, len(found))
        await lane.vectors.delete(stale)
        removed += len(stale)
    return ImageSearch([hit for hit in found if hit[0] in live], removed, True, len(found))


async def remove_forgotten(documents: DocumentStore, vectors: VectorIndex) -> int | None:
    """Remove every image vector whose chunk is gone, in every space; return how many.

    Those are vectors a forget or a space deletion through an engine without
    the lane could not reach. None when the index cannot list what it holds."""
    spaces = getattr(vectors, "spaces_with_vectors", None)
    ids = getattr(vectors, "ids", None)
    if not callable(spaces) or not callable(ids):
        return None
    removed = 0
    for space in await spaces():
        held = await ids(space)
        live = await _live(documents, space, held)
        stale = [chunk_id for chunk_id in held if chunk_id not in live]
        if stale:
            await vectors.delete(stale)
            removed += len(stale)
    return removed


async def _live(documents: DocumentStore, space: str, chunk_ids: Sequence[int]) -> set[int]:
    live: set[int] = set()
    for start in range(0, len(chunk_ids), _PAGE):
        live.update(chunk.chunk_id for chunk in await documents.get_chunks(space, chunk_ids[start:start + _PAGE]))
    return live
