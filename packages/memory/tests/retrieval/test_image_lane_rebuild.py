"""Rebuilding the image lane: every stored image re-embedded with the engine's image embedder.

``reembed_images`` walks a space's stored images newest first, at most
``limit`` file episodes a pass, and embeds each one's original bytes again.
The report counts what it did and says where to walk on. On an index that
records its writer, a rebuild under another image model holds a rebuild
marker, so the lane is refused until it finishes; it records the new model
only once no space holds a vector the model did not tag. On any other index
the lane already ignores vectors tagged by another model.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.blobs import InMemoryBlobStore
from scone_memory.core.errors import InvalidInput
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ImageAttribute, ImageContext, ingest_image
from scone_memory.memory.vector_identity import VectorWriterChanged
from scone_memory.retrieval.image_lane import IMAGE_WRITER

COLORS = ["red", "blue", "green"]
CAPTIONS = ["Holiday album, first page", "Inventory card, shed shelf", "Survey sheet, north border"]


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new("RGB", (16, 16), color).save(out, format="PNG")
    return out.getvalue()


def context(n: int) -> ImageContext:
    return ImageContext(source=f"album/{n}", attributes=(ImageAttribute(kind="alt", value=CAPTIONS[n], origin="html"),))


PHRASES = {"red bicycle": picture("red"), "blue kettle": picture("blue"), "green parrot": picture("green")}


def embedder(salt: str = "") -> HashImageEmbedder:
    return HashImageEmbedder(dim=64, phrases=PHRASES, salt=salt)


class Unrecorded:
    """An image index that cannot record which embedder wrote it or list what it holds."""

    name = "unrecorded"

    def __init__(self) -> None:
        self.inner = InMemoryVectorIndex()

    async def ensure(self, dim):
        await self.inner.ensure(dim)

    async def upsert(self, points):
        await self.inner.upsert(points)

    async def search(self, space, vector, limit, as_of=None, tags=(), where=None):
        return await self.inner.search(space, vector, limit, as_of, tags, where)

    async def delete(self, chunk_ids):
        await self.inner.delete(chunk_ids)


@pytest.fixture(params=["memory", "sqlite"])
def handles(request, tmp_path):
    """New handles on the same files for SQLite; the same objects in memory."""
    if request.param == "memory":
        shared = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex(), InMemoryBlobStore()
        return lambda: shared
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    return lambda: (SqliteDocumentStore(tmp_path / "memory.db"), SqliteVectorIndex(tmp_path / "memory.db"),
                    SqliteVectorIndex(tmp_path / "image-vectors.db"), FileBlobStore(tmp_path / "blobs"))


async def open_lane(handles, image_embedder: HashImageEmbedder) -> MemoryEngine:
    documents, vectors, images, blobs = handles()
    return await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs,
                              image_embedder=image_embedder, image_vectors=images).open()


async def fill(engine: MemoryEngine, space: str = "s") -> list[int]:
    saved = [await ingest_image(engine, space, picture(color), media_type="image/png", context=context(n))
             for n, color in enumerate(COLORS)]
    assert all(one.image_lane == "indexed" for one in saved)
    return [one.added.episode_id for one in saved]


def image_ranks(result, episode_ids) -> dict[int, int]:
    return {item.episode_id: item.lanes["image"] for item in result.items
            if "image" in item.lanes and item.episode_id in episode_ids}


async def test_a_rebuild_under_another_image_model_is_walked_in_passes_and_turns_the_lane_on(handles):
    first = await open_lane(handles, embedder())
    # A note is not a file episode: the walk never reads it.
    await first.remember("s", "A plain note written before the album.")
    ids = await fill(first)
    # A file episode that carries no image: read by the walk, not embedded.
    await first.remember("s", "Invoice scan text, no picture attached.", kind="file")
    second = await open_lane(handles, embedder("other"))
    try:
        assert second.image_block is not None and "image embedder mismatch" in second.image_block
        assert "reembed_images()" in second.image_block

        one = await second.reembed_images("s", limit=3)
        assert (one.scanned, one.scan_complete, one.reembedded, one.forgotten, one.failed) == (3, False, 2, [], [])
        assert one.resume_before == ids[1], "newest first: the note and the two newest images"
        assert (one.writer, one.embedder, one.orphans_removed, one.spaces_pending) == ("rebuilding", embedder("other").id, None, [])
        assert second.image_block is not None and "rebuild in progress" in second.image_block
        refused = await second.recall("s", "red bicycle", limit=5, image_lane=True)
        assert image_ranks(refused, ids) == {}
        assert any(note.startswith("image lane: VectorsNotComparable") and note.endswith("rebuild them with reembed_images()")
                   for note in refused.degraded)
        # The model being rebuilt to may write while it runs; the model it replaces may not.
        assert (await ingest_image(first, "s", picture("red"), media_type="image/png", context=context(0))).image_lane == "blocked"
        assert (await ingest_image(second, "s", picture("blue"), media_type="image/png", context=context(1))).image_lane == "indexed"

        two = await second.reembed_images("s", limit=3, before=one.resume_before)
        assert (two.scanned, two.scan_complete, two.resume_before, two.reembedded) == (1, True, None, 1)
        assert (two.writer, two.orphans_removed, two.spaces_pending) == ("recorded", 0, [])
        assert second.image_block is None
        found = await second.recall("s", "red bicycle", limit=5, image_lane=True)
        assert image_ranks(found, ids)[ids[0]] == 1
        again = await open_lane(handles, embedder())
        assert again.image_block is not None and embedder("other").id in again.image_block
    finally:
        await first.close()


async def test_a_pass_whose_limit_ends_on_the_last_episode_says_the_walk_is_complete(handles):
    engine = await open_lane(handles, embedder())
    try:
        await engine.remember("s", "A plain note written before the album.")
        ids = await fill(engine)
        report = await engine.reembed_images("s", limit=3)
        assert (report.scanned, report.scan_complete, report.resume_before, report.reembedded) == (3, True, None, 3)
        # The record vouched for this model before the pass: no marker, the lane stayed on.
        assert (report.writer, report.orphans_removed) == ("recorded", 0)
        assert image_ranks(await engine.recall("s", "blue kettle", limit=5, image_lane=True), ids)[ids[1]] == 1
        empty = await engine.reembed_images("nothing-here", limit=3)
        assert (empty.scanned, empty.scan_complete, empty.resume_before, empty.orphans_removed) == (0, True, None, 0)
    finally:
        await engine.close()


async def test_the_new_model_is_recorded_only_once_no_space_holds_another_models_vectors(handles):
    first = await open_lane(handles, embedder())
    await fill(first, "s")
    await fill(first, "t")
    second = await open_lane(handles, embedder("other"))
    try:
        s = await second.reembed_images("s", limit=10)
        assert s.scan_complete and s.reembedded == 3
        assert (s.writer, s.spaces_pending) == ("rebuilding", ["t"])
        assert second.image_block is not None
        t = await second.reembed_images("t", limit=10)
        assert (t.writer, t.spaces_pending) == ("recorded", [])
        assert second.image_block is None
    finally:
        await first.close()


async def test_a_rebuild_counts_failures_and_forgets_and_removes_forgotten_images_vectors(handles):
    first = await open_lane(handles, embedder())
    ids = await fill(first)
    documents, vectors, _, blobs = handles()
    plain = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs).open()

    class Troubled(HashImageEmbedder):
        """Refuses the blue image; forgets the green one while embedding it."""

        refuse = True

        async def embed_images(self, images):
            if self.refuse and images == [picture("blue")]:
                raise ValueError("unsupported image")
            if images == [picture("green")] and (await documents.chunks_of("s", ids[2])):
                await plain.forget("s", ids[2])
            return await super().embed_images(images)

    troubled = Troubled(dim=64, phrases=PHRASES, salt="other")
    second = await open_lane(handles, troubled)
    try:
        # Forgotten through an engine without the lane, after this one opened: its vector stays behind.
        await plain.forget("s", ids[0])
        report = await second.reembed_images("s", limit=10)
        assert (report.scanned, report.reembedded, report.forgotten, report.failed) == (2, 0, [ids[2]], [ids[1]])
        assert report.error == "ValueError: unsupported image"
        assert report.scan_complete and report.orphans_removed == 1
        # The blue image still holds the old model's vector: nothing can be recorded yet.
        assert (report.writer, report.spaces_pending) == ("rebuilding", ["s"])
        troubled.refuse = False
        fixed = await second.reembed_images("s", limit=10)
        assert (fixed.reembedded, fixed.failed, fixed.error, fixed.writer) == (1, [], None, "recorded")
        _, _, images, _ = handles()
        assert await images.ids("s") == [(await documents.chunks_of("s", ids[1]))[0].chunk_id]
    finally:
        await first.close()


class IntrudedAtTheRecord(InMemoryVectorIndex):
    """Another image model writes the one vector in space ``s`` just as a rebuild records its own model."""

    armed = False

    async def swap_writer(self, expected, record, *, require_empty=False):
        if self.armed and record[1] == "written":
            self.armed = False
            [chunk_id] = await self.ids("s")
            [point] = [p for p in self._points.values() if p.chunk_id == chunk_id]
            await self.upsert_as([point], embedder("intruder").id)
        return await super().swap_writer(expected, record, require_empty=require_empty)


async def intruded_rebuild() -> tuple[MemoryEngine, IntrudedAtTheRecord]:
    """A second image model over one red image, its next rebuild armed to meet the intruder."""
    documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), IntrudedAtTheRecord(), InMemoryBlobStore()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                               image_vectors=images).open()
    await ingest_image(first, "s", picture("red"), media_type="image/png", context=context(0))
    second = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder("other"),
                                image_vectors=images).open()
    images.armed = True
    return second, images


async def test_another_model_writing_as_the_record_is_taken_fails_the_rebuild():
    second, images = await intruded_rebuild()
    with pytest.raises(VectorWriterChanged, match="another image embedder wrote"):
        await second.reembed_images("s", limit=10)
    assert (await images.written_by())[1] == "invalidated"


async def test_another_model_writing_during_a_pass_leaves_the_index_refused_and_the_next_pass_retakes_it():
    """Written after every image was re-embedded and tagged, so no vector is
    another model's, yet the record is mixed: only a marker held from the
    start of a pass lets the record be taken, so this pass records nothing."""

    class Intruded(InMemoryVectorIndex):
        armed = False

        async def ids(self, space):
            if self.armed:
                self.armed = False
                # Another process's write, around this engine's checks, of a vector already tagged.
                await self.upsert_as([next(iter(self._points.values()))], embedder().id)
            return await super().ids(space)

    documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), Intruded(), InMemoryBlobStore()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                               image_vectors=images).open()
    await fill(first)
    second = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder("other"),
                                image_vectors=images).open()
    images.armed = True
    report = await second.reembed_images("s", limit=10)
    assert (report.reembedded, report.failed, report.writer, report.spaces_pending) == (3, [], "refused", [])
    assert second.image_block is not None and "image embedder mismatch" in second.image_block
    again = await second.reembed_images("s", limit=10)
    assert (again.reembedded, again.writer) == (3, "recorded")


async def test_a_marker_lost_to_a_racing_write_is_taken_again_before_the_pass_writes():
    class Raced(InMemoryVectorIndex):
        races = 1

        async def swap_writer(self, expected, record, *, require_empty=False):
            if self.races and record[1] == "rebuilding":
                self.races -= 1
                return False
            return await super().swap_writer(expected, record, require_empty=require_empty)

    documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), Raced(), InMemoryBlobStore()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                               image_vectors=images).open()
    await fill(first)
    second = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder("other"),
                                image_vectors=images).open()
    report = await second.reembed_images("s", limit=10)
    assert (images.races, report.reembedded, report.failed, report.writer) == (0, 3, [], "recorded")


async def test_vectors_written_around_the_record_are_rebuilt_under_a_marker():
    from scone_memory.core.ports import VectorPoint

    documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex(), InMemoryBlobStore()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                               image_vectors=images).open()
    ids = await fill(first)
    [chunk] = await documents.chunks_of("s", ids[0])
    episode = await first.episode("s", ids[0])
    held = next(iter(images._points.values()))
    images._writer = None  # as an index filled by another tool, with no record at all
    await images.upsert([VectorPoint(chunk_id=chunk.chunk_id, space="s", episode_id=ids[0], created_at=episode.created_at,
                                     vector=held.vector, metadata=episode.metadata)])
    second = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                                image_vectors=images).open()
    assert second.image_block is not None and "recorded as no writer" in second.image_block
    report = await second.reembed_images("s", limit=10)
    assert (report.reembedded, report.writer) == (3, "recorded")


async def test_a_vector_written_around_the_record_during_an_empty_pass_is_not_reported_recorded():
    from scone_memory.core.ports import VectorPoint

    images = InMemoryVectorIndex()

    class Written(InMemoryDocumentStore):
        async def page_episodes(self, space, before, limit, kind):
            if not images._points:
                await images.upsert([VectorPoint(chunk_id=99, space="elsewhere", episode_id=99,
                                                 created_at="2026-01-01T00:00:00Z", vector=[1.0] + [0.0] * 63)])
            return await super().page_episodes(space, before, limit, kind)

    engine = await MemoryEngine(Written(), InMemoryVectorIndex(), HashEmbedder(), image_embedder=embedder(),
                                image_vectors=images).open()
    report = await engine.reembed_images("s", limit=10)
    assert (report.scanned, report.writer) == (0, "refused")


async def test_on_an_index_that_records_no_writer_a_rebuild_retags_every_image():
    from scone_memory.core.ports import VectorPoint

    documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), Unrecorded(), InMemoryBlobStore()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                               image_vectors=images).open()  # type: ignore[arg-type]
    ids = await fill(first)
    # One vector as an engine before tagging wrote it: no embedder id at all.
    episode = await first.episode("s", ids[2])
    [chunk] = await documents.chunks_of("s", ids[2])
    [vector] = await embedder().embed_images([picture("green")])
    await images.upsert([VectorPoint(chunk_id=chunk.chunk_id, space="s", episode_id=ids[2],
                                     created_at=episode.created_at, vector=vector, metadata=episode.metadata)])
    assert ids[2] not in image_ranks(await first.recall("s", "green parrot", limit=5, image_lane=True), ids)
    second = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder("other"),
                                image_vectors=images).open()  # type: ignore[arg-type]
    report = await second.reembed_images("s", limit=10)
    assert (report.reembedded, report.writer, report.orphans_removed, report.spaces_pending) == (3, "tagged", None, [])
    assert image_ranks(await second.recall("s", "green parrot", limit=5, image_lane=True), ids)[ids[2]] == 1
    assert image_ranks(await first.recall("s", "green parrot", limit=5, image_lane=True), ids) == {}
    probe = (await embedder("other").embed_images([picture("red")]))[0]
    assert len(await images.search("s", probe, 10, where={IMAGE_WRITER: embedder("other").id})) == 3


async def test_a_rebuild_is_refused_where_it_cannot_run():
    laneless = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with pytest.raises(InvalidInput, match="no image embedder and image index"):
        await laneless.reembed_images("s")

    class Unlisted(InMemoryDocumentStore):
        page_episodes = None  # type: ignore[assignment]

    unlisted = await MemoryEngine(Unlisted(), InMemoryVectorIndex(), HashEmbedder(), image_embedder=embedder(),
                                  image_vectors=InMemoryVectorIndex()).open()
    with pytest.raises(InvalidInput, match="cannot list episodes"):
        await unlisted.reembed_images("s")
    lane = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), image_embedder=embedder(),
                              image_vectors=InMemoryVectorIndex()).open()
    for limit in (0, 1001, True):
        with pytest.raises(InvalidInput, match="limit"):
            await lane.reembed_images("s", limit=limit)
    for before in (0, True, 2**63):
        with pytest.raises(InvalidInput, match="before"):
            await lane.reembed_images("s", before=before)


async def test_an_image_whose_bytes_cannot_be_read_is_counted_failed_not_forgotten():
    from scone_memory.core.errors import NotFound

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), image_embedder=embedder(),
                                image_vectors=InMemoryVectorIndex()).open()
    [red, _, _] = await fill(engine)

    async def lost(space, attachment_id):
        raise NotFound(f"attachment {attachment_id[:8]} not found")

    engine.attachment = lost  # type: ignore[method-assign]
    report = await engine.reembed_images("s", limit=1, before=red + 1)
    assert (report.scanned, report.reembedded, report.forgotten, report.failed) == (1, 0, [], [red])
    assert report.error is not None and report.error.startswith("NotFound: attachment")
