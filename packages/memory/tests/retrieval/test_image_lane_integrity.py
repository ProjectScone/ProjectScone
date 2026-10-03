"""The image lane's index kept true to the episodes and the embedder it serves.

Only an engine built with the image lane can reach its index, and the lane
is configured from Python alone: the HTTP server, the CLI and directory sync
run engines without it. A forget through one of those cannot delete the
image's vector, so it says so, and an engine with the lane removes what such
forgets left: when it opens, and when a recall meets one. An image embedder
the index does not record as its writer is named when the engine opens and
writes nothing there, so one ingest cannot leave the index unusable by
every embedder.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ImageAttribute, ImageContext, ingest_image
from scone_memory.retrieval.recall import LANE_DEPTH

CAPTIONS = ["Holiday album, first page", "Inventory card, shed shelf", "Survey sheet, north border",
            "Logbook cover, winter volume", "Catalogue insert, autumn edition", "Ticket stub, drawer",
            "Postcard, harbour view", "Recipe card, apple cake"]


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new("RGB", (16, 16), color).save(out, format="PNG")
    return out.getvalue()


def context(n: int) -> ImageContext:
    return ImageContext(source=f"album/{n}", attributes=(ImageAttribute(kind="alt", value=CAPTIONS[n], origin="html"),))


def embedder(salt: str = "") -> HashImageEmbedder:
    return HashImageEmbedder(dim=64, phrases={"red bicycle": picture("red")}, salt=salt)


@pytest.fixture(params=["memory", "sqlite"])
def handles(request, tmp_path):
    """A builder of store handles: new handles on the same files for SQLite, the same objects in memory."""
    if request.param == "memory":
        shared = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex(), None
        return lambda: shared
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    return lambda: (SqliteDocumentStore(tmp_path / "memory.db"), SqliteVectorIndex(tmp_path / "memory.db"),
                    SqliteVectorIndex(tmp_path / "image-vectors.db"), FileBlobStore(tmp_path / "blobs"))


async def with_lane(handles, image_embedder: HashImageEmbedder | None = None) -> MemoryEngine:
    documents, vectors, images, blobs = handles()
    return await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs,
                              image_embedder=image_embedder or embedder(), image_vectors=images).open()


async def without_lane(handles) -> MemoryEngine:
    documents, vectors, _, blobs = handles()
    return await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs).open()


async def test_a_forget_says_whether_it_removed_the_image_vector(handles):
    lane = await with_lane(handles)
    try:
        first = await ingest_image(lane, "s", picture("red"), media_type="image/png", context=context(0))
        second = await ingest_image(lane, "s", picture("blue"), media_type="image/png", context=context(1))
        note = await lane.remember("s", "A note with no image in it.")
        plain = await without_lane(handles)
        assert (await plain.impact("s", first.added.episode_id)).image_vector == "not_reached"
        receipt = await plain.forget("s", first.added.episode_id)
        assert receipt.forgotten_at is not None and receipt.image_vector == "not_reached"
        assert (await plain.impact("s", note.episode_id)).image_vector == "none"
        assert (await lane.impact("s", second.added.episode_id)).image_vector == "removed"
        assert (await lane.forget("s", second.added.episode_id)).image_vector == "removed"
        assert (await lane.forget("s", note.episode_id)).image_vector == "none"
    finally:
        await lane.close()


async def test_an_engine_with_the_lane_removes_vectors_left_by_forgets_without_it_when_it_opens(handles):
    lane = await with_lane(handles)
    kept = await ingest_image(lane, "s", picture("red"), media_type="image/png", context=context(0))
    gone = await ingest_image(lane, "s", picture("blue"), media_type="image/png", context=context(1))
    elsewhere = await ingest_image(lane, "t", picture("green"), media_type="image/png", context=context(2))
    await lane.close()
    plain = await without_lane(handles)
    await plain.forget("s", gone.added.episode_id)
    await plain.delete_space("t")
    await plain.close()
    _, _, images, _ = handles()
    assert len(await images.ids("s")) == 2 and len(await images.ids("t")) == 1, "a lane-less engine cannot reach them"
    reopened = await with_lane(handles)
    try:
        [chunk] = await reopened.documents.chunks_of("s", kept.added.episode_id)
        assert await reopened.image_vectors.ids("s") == [chunk.chunk_id]
        assert await reopened.image_vectors.ids("t") == []
        assert reopened.image_vectors_removed == 2
        assert elsewhere.image_lane == "indexed"
    finally:
        await reopened.close()


async def test_a_recall_removes_vectors_of_images_forgotten_while_it_was_open():
    documents, vectors, images = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex()
    lane = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(), image_vectors=images).open()
    plain = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    old = [await ingest_image(lane, "s", picture("red"), media_type="image/png", context=context(n))
           for n in range(LANE_DEPTH)]
    for saved in old:
        await plain.forget("s", saved.added.episode_id)
    live = await ingest_image(lane, "s", picture("red"), media_type="image/png", context=context(LANE_DEPTH))
    found = await lane.recall("s", "red bicycle", limit=1, image_lane=True)
    assert [(item.episode_id, item.lanes.get("image")) for item in found.items] == [(live.added.episode_id, 1)], \
        "the live image is not pushed out of the lane's window by forgotten ones"
    assert f"image lane: removed {LANE_DEPTH} vectors of images already forgotten" in " ".join(found.degraded)
    assert not any("still" in note for note in found.degraded)
    [chunk] = await documents.chunks_of("s", live.added.episode_id)
    assert await images.ids("s") == [chunk.chunk_id]
    again = await lane.recall("s", "red bicycle", limit=1, image_lane=True)
    assert not [note for note in again.degraded if note.startswith("image lane")]


async def test_a_recall_says_when_forgotten_images_still_fill_the_lane_after_its_searches():
    from scone_memory.retrieval.image_lane import IMAGE_SEARCHES

    from scone_memory import InMemoryEventLog

    documents, vectors, images, events = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex(), InMemoryEventLog()
    lane = await MemoryEngine(documents, vectors, HashEmbedder(), events=events,
                              image_embedder=embedder(), image_vectors=images).open()
    plain = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    window = 2
    old = [await ingest_image(lane, "s", picture("red"), media_type="image/png", context=context(n))
           for n in range(window * IMAGE_SEARCHES + 2)]
    for saved in old:
        await plain.forget("s", saved.added.episode_id)
    found = await lane.recall("s", "red bicycle", limit=1, candidate_limit=window, image_lane=True)
    notes = [note for note in found.degraded if note.startswith("image lane")]
    assert notes == [f"image lane: removed {window * IMAGE_SEARCHES} vectors of images already forgotten; "
                     f"forgotten images still filled its window after {IMAGE_SEARCHES} searches, "
                     "so it ranked fewer images than it looks for"]
    assert len(await images.ids("s")) == 2
    [recalled] = [event.payload for event in await events.query("s", kind="recall")]
    assert recalled["lane_candidates"]["image"] == 0, "a forgotten image's vector is never a candidate"


async def test_an_image_embedder_the_index_does_not_record_writes_nothing_there(handles):
    lane = await with_lane(handles)
    # Opened while the index is empty: nothing to refuse yet, so the writer is read again when it writes.
    early = await with_lane(handles, embedder("other"))
    try:
        assert early.image_block is None
        await ingest_image(lane, "s", picture("red"), media_type="image/png", context=context(0))
        _, _, images, _ = handles()
        record = await images.written_by()
        assert record == (embedder().id, "written")
        saved = await ingest_image(early, "s", picture("blue"), media_type="image/png", context=context(1))
        assert saved.image_lane == "blocked"
        assert saved.image_lane_blocked is not None and embedder().id in saved.image_lane_blocked
        assert await images.written_by() == record and len(await images.ids("s")) == 1
        late = await with_lane(handles, embedder("other"))
        assert late.image_block is not None and "image embedder mismatch" in late.image_block
        assert embedder().id in late.image_block and embedder("other").id in late.image_block
        assert lane.image_block is None
        found = await lane.recall("s", "red bicycle", limit=1, image_lane=True)
        assert found.items[0].lanes.get("image") == 1, "the index still answers the embedder that wrote it"
    finally:
        await lane.close()


async def test_image_vectors_no_writer_recorded_are_named_and_not_written_over():
    from scone_memory.core.ports import VectorPoint

    documents, vectors, images = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex()
    note = await (await MemoryEngine(documents, vectors, HashEmbedder()).open()).remember("s", "A note.")
    [chunk] = await documents.chunks_of("s", note.episode_id)
    await images.ensure(64)
    # Written around the record, as an index filled by some other tool would be, for a live chunk.
    await images.upsert([VectorPoint(chunk_id=chunk.chunk_id, space="s", episode_id=note.episode_id,
                                     created_at=chunk.created_at, vector=[1.0] + [0.0] * 63)])
    engine = await MemoryEngine(documents, vectors, HashEmbedder(),
                                image_embedder=embedder(), image_vectors=images).open()
    assert engine.image_block is not None and "recorded as no writer" in engine.image_block
    saved = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context(0))
    assert saved.image_lane == "blocked" and await images.written_by() is None


class _Unlisted:
    """An image index that records no writer and cannot list what it holds."""

    name = "unlisted"

    def __init__(self) -> None:
        self._inner = InMemoryVectorIndex()

    async def ensure(self, dim):
        await self._inner.ensure(dim)

    async def upsert(self, points):
        await self._inner.upsert(points)

    async def search(self, space, vector, limit, as_of=None, tags=(), where=None):
        return await self._inner.search(space, vector, limit, as_of, tags, where)

    async def delete(self, chunk_ids):
        await self._inner.delete(chunk_ids)


async def test_an_image_index_that_records_and_lists_nothing_is_used_as_it_is():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                image_embedder=embedder(), image_vectors=_Unlisted()).open()  # type: ignore[arg-type]
    assert engine.image_block is None and engine.image_vectors_removed is None
    saved = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context(0))
    assert saved.image_lane == "indexed"
    found = await engine.recall("s", "red bicycle", limit=1, image_lane=True)
    assert found.items[0].lanes.get("image") == 1


async def test_a_window_full_of_forgotten_images_still_says_the_bound_bit():
    """The images removed on a recall's last search filled its window, so images
    deeper than it looked went unseen: the bounds say so as for any full window."""
    from scone_memory.retrieval.image_lane import IMAGE_SEARCHES

    class PostFiltered(InMemoryVectorIndex):
        narrows_conditions = False

    documents, vectors, images = InMemoryDocumentStore(), PostFiltered(), PostFiltered()
    lane = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(), image_vectors=images).open()
    plain = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    window = 2
    # Two recalls, each removing a window per search, and a full window still left for the second's last search.
    old = [await ingest_image(lane, "s", picture("red"), media_type="image/png", context=ImageContext(
               source=f"album/{n}", attributes=(ImageAttribute(kind="alt", value=f"Scan {n}", origin="html"),)))
           for n in range(2 * window * IMAGE_SEARCHES + window)]
    for saved in old:
        await plain.forget("s", saved.added.episode_id)
    await lane.remember("s", "The red bicycle note is filed under receipts.")
    narrowed = await lane.recall("s", "red bicycle", limit=1, candidate_limit=window, image_lane=True,
                                 conditions={"field": "document_format", "is": "image"})
    report = narrowed.narrowing
    assert report is not None and report.image_lane == "postfiltered"
    assert report.image_returned == report.image_window == window
    assert report.postfiltered_out >= 1 and report.window_exhausted is True
    assert report.vector_returned < report.vector_window, "only the image lane's window can have bitten"
    phrased = await lane.recall("s", "red bicycle", limit=1, candidate_limit=window, lanes=["text"], image_lane=True,
                                require=["Ticket stub"])
    assert any("still filled its window" in note for note in phrased.degraded)
    assert phrased.phrases is not None and phrased.phrases.dropped_required >= 1
    assert phrased.phrases.short is True
