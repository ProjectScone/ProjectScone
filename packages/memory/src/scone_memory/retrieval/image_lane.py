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
and ignores the rest (``records_writer``), saying how many live images it
ignored when that leaves its window short (``search_images``). Such an index
serves one image model: an image has one vector, keyed by its chunk, so
another model's write replaces it.

LlamaIndex's multi-modal index keeps image nodes in an ``image`` vector
store namespace and appends the image results to the text results; this lane
instead ranks them with the other lanes, so one ordering answers the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, cast
from uuid import uuid4

from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import Episode, ImageRebuildReport
from ..core.ports import DocumentStore, ImageEmbedder, RecordsVectorWriter, VectorIndex, VectorPoint
from ..core.validation import check_space
from ..core.vector_writers import VectorsNotComparable, Writer, rebuilding_token, vouches
from ..memory.vector_identity import GuardedVectors, VectorWriterChanged

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

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

#: File episodes one ``reembed_images`` pass may read, at most.
MAX_REBUILD_PASS = 1_000

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
    #: On an index that cannot record its writer: live images under the
    #: recall's filters whose vector this embedder did not tag, so the lane
    #: left them out. Looked for only when the lane's own window was not full.
    ignored: int = 0
    #: The search for them filled its window too: ``ignored`` is a lower bound.
    ignored_at_least: bool = False


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
    blocked = await writer_block(vectors, embedder.id, writing=True)
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


async def writer_block(vectors: VectorIndex, embedder_id: str, *, writing: bool = False) -> str | None:
    """Why the image embedder ``embedder_id`` must not compare with the index's
    vectors (or, ``writing``, write to it), as the index records them now; None
    when it may, or when the index records no writers at all.

    While its own rebuild holds the record, the embedder may write, since a
    write under its own marker keeps the marker, but compares nothing."""
    written_by = getattr(vectors, "written_by", None)
    holds_vectors = getattr(vectors, "holds_vectors", None)
    if not callable(written_by) or not callable(holds_vectors):
        return None
    record = await written_by()
    if vouches(record, embedder_id, record is None and await holds_vectors()):
        return None
    if _rebuilding_by(record, embedder_id):
        return None if writing else (f"image lane rebuild in progress: {embedder_id} is re-embedding the stored "
                                     "images; run reembed_images() until a pass completes with no space pending")
    name, basis = record if record is not None else ("no writer", "unrecorded")
    return (f"image embedder mismatch: stored image vectors are recorded as {name} ({basis}), "
            f"this engine's image embedder is {embedder_id}; rebuild them with reembed_images(), "
            "or give it an image index of its own")


def _rebuilding_by(record: Writer | None, embedder_id: str) -> bool:
    return record is not None and record[0].startswith(f"rebuilding:{embedder_id}:")


async def reembed_images(engine: "MemoryEngine", space: str, *, limit: int = 100,
                         before: Optional[int] = None) -> ImageRebuildReport:
    """Embed a space's stored images again with the engine's image embedder, one bounded pass.

    Reads at most ``limit`` file episodes, newest id first, from ``before``
    (a previous pass's ``resume_before``), and writes each stored image's
    vector from its original bytes, tagged with this embedder (``index_image``).

    On an index that records its writer and does not vouch for this embedder,
    the pass first takes this embedder's rebuild marker, which refuses the lane
    and lets only this embedder write. The pass that completes a space's walk
    records the embedder as the writer only when no space holds a vector it
    did not tag, so a rebuild that covers one space of several, or leaves an
    image that failed, records nothing yet and names the spaces pending. On any
    other index nothing is recorded: the lane already ignores other tags."""
    check_space(space)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_REBUILD_PASS:
        raise InvalidInput(f"limit must be an integer in 1..{MAX_REBUILD_PASS}")
    if before is not None and (isinstance(before, bool) or not isinstance(before, int) or not 1 <= before <= 2**63 - 1):
        raise InvalidInput("before must be a positive signed 64-bit episode ID")
    vectors = engine.image_vectors
    if vectors is None:
        raise InvalidInput("image lane: this engine has no image embedder and image index")
    page = getattr(engine.documents, "page_episodes", None)
    if not callable(page):
        raise InvalidInput(f"the {engine.documents.name} document store cannot list episodes, "
                           "so its images cannot be re-embedded")
    embedder = cast(ImageEmbedder, engine.image_embedder)
    recorder = cast(RecordsVectorWriter, vectors.inner) if isinstance(vectors, GuardedVectors) else None
    if recorder is not None:
        await _take_rebuild(recorder, embedder.id)
    report = ImageRebuildReport(space=space, embedder=embedder.id, limit=limit)
    batch = await page(space, before, limit, "file")
    for episode in batch:
        report.scanned += 1
        original = episode.metadata.get("image_original")
        if original is None:
            continue
        try:
            _, data = await engine.attachment(space, original)
        except Exception as error:  # noqa: BLE001 - a live image whose bytes cannot be read is reported
            _fail(report, episode.episode_id, error)
            continue
        try:
            await index_image(engine.documents, embedder, vectors, space, episode, data)
        except NotFound:
            # Forgotten while it was embedded (Gone is a NotFound): index_image kept no vector.
            report.forgotten.append(episode.episode_id)
            continue
        except Exception as error:  # noqa: BLE001 - one image's failure does not stop the pass
            _fail(report, episode.episode_id, error)
            continue
        report.reembedded += 1
    if len(batch) == limit:
        # A full page: the end of the space and the bound are told apart by looking for one more.
        report.scan_complete = not await page(space, batch[-1].episode_id, 1, "file")
    if not report.scan_complete:
        report.resume_before = batch[-1].episode_id
    else:
        report.orphans_removed = await _remove_forgotten_in(engine.documents, vectors, space)
    if recorder is None:
        return report
    record = await recorder.written_by()
    if report.scan_complete and _rebuilding_by(record, embedder.id):
        report.spaces_pending = await _spaces_pending(recorder, embedder)
        if not report.spaces_pending:
            if not await recorder.swap_writer(record, (embedder.id, "written")):
                raise VectorWriterChanged("another image embedder wrote image vectors during the rebuild; "
                                          "run reembed_images() again")
            record = (embedder.id, "written")
    report.writer = ("recorded" if vouches(record, embedder.id, record is None and await recorder.holds_vectors())
                     else "rebuilding" if _rebuilding_by(record, embedder.id) else "refused")
    return report


def _fail(report: ImageRebuildReport, episode_id: int, error: Exception) -> None:
    report.failed.append(episode_id)
    if report.error is None:
        report.error = f"{type(error).__name__}: {error}"


async def _take_rebuild(index: RecordsVectorWriter, embedder_id: str) -> None:
    """Hold a rebuild marker for this embedder, unless the record already vouches for it.

    A marker this embedder already holds is taken again under a new nonce:
    what completes a rebuild is the tags, read at the end, not the nonce."""
    while True:
        record = await index.written_by()
        if vouches(record, embedder_id, record is None and await index.holds_vectors()):
            return
        if await index.swap_writer(record, rebuilding_token(embedder_id, uuid4().hex)):
            return


async def _spaces_pending(index: RecordsVectorWriter, embedder: ImageEmbedder) -> list[str]:
    """The spaces holding a vector not tagged with ``embedder``'s id.

    Read below the writer check, which refuses every search under a marker:
    one search per space, as deep as the space holds vectors."""
    probe = [1.0] + [0.0] * (embedder.dim - 1)
    pending: list[str] = []
    for space in await index.spaces_with_vectors():
        held = await index.ids(space)
        own = await cast(VectorIndex, index).search(space, probe, len(held), where={IMAGE_WRITER: embedder.id})
        if len(own) < len(held):
            pending.append(space)
    return pending


async def search_images(lane: ImageLane, documents: DocumentStore, space: str, query: str, depth: int,
                        as_of: Optional[str], tags: tuple[str, ...], where: Mapping[str, str],
                        conditions: object = None) -> ImageSearch:
    """The stored images nearest the query, under the recall's filters.

    A condition the index cannot evaluate is left to the recall's post-filter,
    as it is for the vector lane. A hit whose chunk is gone is a forgotten
    image's vector: it is removed and the index searched again, up to
    ``IMAGE_SEARCHES`` times.

    Where the index cannot record its writer, the search is narrowed to this
    embedder's tag. When that leaves the window short, the index is searched
    once more without the tag, and the live images it finds beyond the lane's
    own are counted as ``ignored``: another model's vectors, or ones written
    before vectors were tagged, which the lane cannot compare with."""
    [vector] = await lane.embedder.embed_texts([query])
    tagged = not records_writer(lane.vectors)
    # Nothing refuses another model's vectors here: its own tag leaves them out.
    own_where = {**where, IMAGE_WRITER: lane.embedder.id} if tagged else where
    removed, short = 0, True
    for _ in range(IMAGE_SEARCHES):
        found = await _search(lane.vectors, space, vector, depth, as_of, tags, own_where, conditions)
        live = await _live(documents, space, [chunk_id for chunk_id, _ in found])
        stale = [chunk_id for chunk_id, _ in found if chunk_id not in live]
        if not stale:
            short = False
            break
        await lane.vectors.delete(stale)
        removed += len(stale)
    hits = [hit for hit in found if hit[0] in live]
    ignored, at_least = 0, False
    if tagged and len(found) < depth:
        untagged = await _search(lane.vectors, space, vector, depth, as_of, tags, where, conditions)
        others = [chunk_id for chunk_id, _ in untagged if chunk_id not in live]
        ignored, at_least = len(await _live(documents, space, others)), len(untagged) >= depth
    return ImageSearch(hits, removed, short, len(found), ignored, at_least)


async def _search(vectors: VectorIndex, space: str, vector: Sequence[float], depth: int, as_of: Optional[str],
                  tags: tuple[str, ...], where: Mapping[str, str], conditions: object) -> list[tuple[int, float]]:
    if conditions is not None and getattr(vectors, "narrows_conditions", False):
        return await cast(_ConditionNarrowing, vectors).search(space, vector, depth, as_of, tags, where,
                                                               conditions=conditions)
    return await vectors.search(space, vector, depth, as_of, tags, where)


async def remove_forgotten(documents: DocumentStore, vectors: VectorIndex) -> int | None:
    """Remove every image vector whose chunk is gone, in every space; return how many.

    Those are vectors a forget or a space deletion through an engine without
    the lane could not reach. None when the index cannot list what it holds."""
    spaces = getattr(vectors, "spaces_with_vectors", None)
    if not callable(spaces) or not callable(getattr(vectors, "ids", None)):
        return None
    removed = 0
    for space in await spaces():
        removed += cast(int, await _remove_forgotten_in(documents, vectors, space))
    return removed


async def _remove_forgotten_in(documents: DocumentStore, vectors: VectorIndex, space: str) -> int | None:
    """Remove the space's image vectors whose chunk is gone; None when the index cannot list them."""
    ids = getattr(vectors, "ids", None)
    if not callable(ids):
        return None
    held = await ids(space)
    live = await _live(documents, space, held)
    stale = [chunk_id for chunk_id in held if chunk_id not in live]
    if stale:
        await vectors.delete(stale)
    return len(stale)


async def _live(documents: DocumentStore, space: str, chunk_ids: Sequence[int]) -> set[int]:
    live: set[int] = set()
    for start in range(0, len(chunk_ids), _PAGE):
        live.update(chunk.chunk_id for chunk in await documents.get_chunks(space, chunk_ids[start:start + _PAGE]))
    return live
