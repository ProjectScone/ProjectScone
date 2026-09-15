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
the image embedder never meet. Forgetting an episode removes its image
vector with its text vectors, since both are keyed by its chunk ids; a vector
written while that forget ran is taken out again (``index_image``).

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

#: The lane's weight in reciprocal-rank fusion beside the text lane's 1.
#: Not measured on any benchmark: there is no image retrieval set here and
#: no model to run one with. The context and entity lanes chose 2.0 on
#: their benchmarks; this one keeps the text lane's voice until it has one.
IMAGE_WEIGHT = 1.0


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
    or ``NotFound`` while that forget has not yet recorded its tombstone."""
    # A context episode always has text, so a live one always has a first chunk.
    chunks = await documents.chunks_of(space, episode.episode_id)
    if not chunks:
        raise await _forgotten(documents, space, episode)
    chunk_id = chunks[0].chunk_id
    [vector] = await embedder.embed_images([data])
    await vectors.upsert([VectorPoint(chunk_id=chunk_id, space=space, episode_id=episode.episode_id,
                                      created_at=episode.created_at, vector=vector, tags=episode.tags,
                                      metadata=episode.metadata)])
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


async def search_images(lane: ImageLane, space: str, query: str, depth: int, as_of: Optional[str],
                        tags: tuple[str, ...], where: Mapping[str, str],
                        conditions: object = None) -> list[tuple[int, float]]:
    """Ranked (chunk_id, cosine) of the stored images nearest the query, under the recall's filters.

    A condition the index cannot evaluate is left to the recall's post-filter,
    as it is for the vector lane."""
    [vector] = await lane.embedder.embed_texts([query])
    if conditions is not None and getattr(lane.vectors, "narrows_conditions", False):
        return await cast(_ConditionNarrowing, lane.vectors).search(space, vector, depth, as_of, tags, where,
                                                                    conditions=conditions)
    return await lane.vectors.search(space, vector, depth, as_of, tags, where)
