"""Every image vector names the image embedder that made it.

An index that records its writer (the in-memory and SQLite ones) refuses an
image embedder it does not record. Every other index cannot, so the lane
there compares only the vectors tagged with its own embedder's id and ignores
the rest; the engine says which of the two it relies on when it opens.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ImageAttribute, ImageContext, ingest_image
from scone_memory.retrieval.image_lane import IMAGE_WIDTH, IMAGE_WRITER


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new("RGB", (16, 16), color).save(out, format="PNG")
    return out.getvalue()


def context(caption: str, source: str) -> ImageContext:
    return ImageContext(source=source, attributes=(ImageAttribute(kind="alt", value=caption, origin="html"),))


def embedder(salt: str = "") -> HashImageEmbedder:
    return HashImageEmbedder(dim=64, phrases={"red bicycle": picture("red"), "blue kettle": picture("blue")},
                             salt=salt)


class Unrecorded:
    """An image index that cannot record which embedder wrote it, as Postgres,
    Qdrant, Elasticsearch and the other server indexes cannot."""

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
def recording(request, tmp_path):
    if request.param == "memory":
        return InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex()
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    return (SqliteDocumentStore(tmp_path / "memory.db"), SqliteVectorIndex(tmp_path / "memory.db"),
            SqliteVectorIndex(tmp_path / "image-vectors.db"))


async def test_every_image_vector_carries_its_embedders_id_and_width(recording):
    documents, vectors, images = recording
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(),
                                image_vectors=images).open()
    try:
        assert engine.image_writer_check == "recorded"
        saved = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                                   context=context("Holiday album, first page", "album/0"))
        [chunk] = await documents.chunks_of("s", saved.added.episode_id)
        probe = (await embedder().embed_images([picture("red")]))[0]
        # Searched below the writer check, as a tool reading the index would.
        own = {IMAGE_WRITER: embedder().id, IMAGE_WIDTH: "64"}
        assert [hit for hit, _ in await images.search("s", probe, 5, where=own)] == [chunk.chunk_id]
        assert await images.search("s", probe, 5, where={IMAGE_WRITER: embedder("other").id}) == []
        assert await images.search("s", probe, 5, where={IMAGE_WIDTH: "32"}) == []
    finally:
        await engine.close()


async def test_an_index_that_cannot_record_its_writer_ignores_another_embedders_vectors():
    documents, vectors, images = InMemoryDocumentStore(), InMemoryVectorIndex(), Unrecorded()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(),
                               image_vectors=images).open()  # type: ignore[arg-type]
    other = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder("other"),
                               image_vectors=images).open()  # type: ignore[arg-type]
    assert first.image_writer_check == other.image_writer_check == "tagged"
    assert (await MemoryEngine(documents, vectors, HashEmbedder()).open()).image_writer_check is None
    assert first.image_block is None and other.image_block is None, "nothing records a writer to refuse on"
    red = await ingest_image(first, "s", picture("red"), media_type="image/png",
                             context=context("Holiday album, first page", "album/0"))
    assert red.image_lane == "indexed"
    # The other model's lane ranks whatever vectors it is given, near or not:
    # the first model's red image is the only vector, so only its tag keeps it out.
    # The two images below are two keys; one image written by both models is one (below).
    ignored = await other.recall("s", "red bicycle", limit=5, image_lane=True)
    assert all("image" not in item.lanes for item in ignored.items)
    blue = await ingest_image(other, "s", picture("blue"), media_type="image/png",
                              context=context("Inventory card, shed shelf", "album/1"))
    assert blue.image_lane == "indexed"
    for engine, own in ((first, red), (other, blue)):
        found = await engine.recall("s", "red bicycle blue kettle", limit=5, image_lane=True)
        placed = {item.episode_id for item in found.items if "image" in item.lanes}
        assert placed == {own.added.episode_id}, f"{engine.image_embedder.id} ranked {placed}"


async def test_a_vector_tagged_by_no_embedder_is_ignored_where_the_index_records_no_writer():
    """Vectors written before image vectors were tagged carry no id. Where the
    index cannot record a writer they are ignored until the lane is rebuilt
    (``reembed_images``); where it records one, the record still vouches for them."""
    from scone_memory.core.ports import VectorPoint

    documents, vectors, images = InMemoryDocumentStore(), InMemoryVectorIndex(), Unrecorded()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(),
                                image_vectors=images).open()  # type: ignore[arg-type]
    saved = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                               context=context("Holiday album, first page", "album/0"))
    [chunk] = await documents.chunks_of("s", saved.added.episode_id)
    episode = await engine.episode("s", saved.added.episode_id)
    [vector] = await embedder().embed_images([picture("red")])
    await images.upsert([VectorPoint(chunk_id=chunk.chunk_id, space="s", episode_id=episode.episode_id,
                                     created_at=episode.created_at, vector=vector, tags=episode.tags,
                                     metadata=episode.metadata)])
    found = await engine.recall("s", "red bicycle", limit=5, image_lane=True)
    assert all("image" not in item.lanes for item in found.items)


async def test_a_vector_tagged_by_no_embedder_still_answers_where_the_index_records_its_writer(recording):
    from scone_memory.core.ports import VectorPoint

    documents, vectors, images = recording
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(),
                                image_vectors=images).open()
    try:
        saved = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                                   context=context("Holiday album, first page", "album/0"))
        [chunk] = await documents.chunks_of("s", saved.added.episode_id)
        episode = await engine.episode("s", saved.added.episode_id)
        [vector] = await embedder().embed_images([picture("red")])
        # As an engine before tagging wrote it: through the record, with the episode's metadata only.
        await images.upsert_as([VectorPoint(chunk_id=chunk.chunk_id, space="s", episode_id=episode.episode_id,
                                            created_at=episode.created_at, vector=vector, tags=episode.tags,
                                            metadata=episode.metadata)], embedder().id)
        assert await images.written_by() == (embedder().id, "written")
        found = await engine.recall("s", "red bicycle", limit=5, image_lane=True)
        assert [item.lanes.get("image") for item in found.items if item.episode_id == saved.added.episode_id] == [1]
    finally:
        await engine.close()


class CountedUnrecorded(Unrecorded):
    """Counts the searches a recall makes of it."""

    searches = 0

    async def search(self, space, vector, limit, as_of=None, tags=(), where=None):
        self.searches += 1
        return await super().search(space, vector, limit, as_of, tags, where)


def image_notes(result) -> list[str]:
    return [note for note in result.degraded if note.startswith("image lane")]


def ignored_note(count: str, embedder_id: str) -> str:
    return (f"image lane: ignored {count} image vectors under this recall's filters that {embedder_id} did not tag "
            "(another image model's, or written before image vectors were tagged); rebuild them with reembed_images()")


async def test_a_lane_that_ignores_vectors_it_did_not_tag_says_so_and_a_second_model_takes_the_image_over():
    """One image has one vector, whichever model wrote it last. Where the
    index cannot record its writer, the model that lost it is told so."""
    from scone_memory.backends.blobs import InMemoryBlobStore

    documents, vectors, images, blobs = InMemoryDocumentStore(), InMemoryVectorIndex(), CountedUnrecorded(), InMemoryBlobStore()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                               image_vectors=images).open()  # type: ignore[arg-type]
    saved = [await ingest_image(first, "s", picture(color), media_type="image/png",
                                context=context(caption, f"album/{n}"))
             for n, (color, caption) in enumerate((("red", "Holiday album, first page"),
                                                   ("blue", "Inventory card, shed shelf"),
                                                   ("green", "Survey sheet, north border")))]
    ids = {one.added.episode_id for one in saved}
    # Forgotten through an engine without the lane: its vector stays, and is not counted as ignored.
    plain = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs).open()
    await plain.forget("s", saved[2].added.episode_id)
    other = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder("other"),
                               image_vectors=images).open()  # type: ignore[arg-type]
    images.searches = 0
    ignored = await other.recall("s", "red bicycle", limit=5, image_lane=True)
    assert {item.episode_id for item in ignored.items if "image" in item.lanes} == set()
    assert image_notes(ignored) == [ignored_note("2", embedder("other").id)]
    assert images.searches == 2, "the own-tag search came back short, so the lane looked once more without its tag"

    rebuilt = await other.reembed_images("s", limit=10)
    assert (rebuilt.reembedded, rebuilt.writer) == (2, "tagged")
    taken = await other.recall("s", "red bicycle", limit=5, image_lane=True)
    assert {item.episode_id for item in taken.items if "image" in item.lanes} == ids - {saved[2].added.episode_id}
    assert image_notes(taken) == []
    # The first model's vectors were replaced, not joined: its lane has nothing left, and says so.
    lost = await first.recall("s", "red bicycle", limit=5, image_lane=True)
    assert {item.episode_id for item in lost.items if "image" in item.lanes} == set()
    # Its own tag still reaches the forgotten image's vector, which it removes.
    assert image_notes(lost) == ["image lane: removed 1 vectors of images already forgotten",
                                 ignored_note("2", embedder().id)]


async def test_a_full_own_window_costs_no_second_search_and_a_full_second_window_is_a_lower_bound():
    from scone_memory.retrieval.image_lane import ImageLane, search_images

    documents, vectors, images = InMemoryDocumentStore(), InMemoryVectorIndex(), CountedUnrecorded()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(),
                               image_vectors=images).open()  # type: ignore[arg-type]
    other = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder("other"),
                               image_vectors=images).open()  # type: ignore[arg-type]
    await ingest_image(first, "s", picture("red"), media_type="image/png", context=context("Holiday album, first page", "album/0"))
    await ingest_image(first, "s", picture("green"), media_type="image/png", context=context("Survey sheet, north border", "album/2"))
    await ingest_image(other, "s", picture("blue"), media_type="image/png", context=context("Inventory card, shed shelf", "album/1"))
    lane = ImageLane(embedder("other"), images)  # type: ignore[arg-type]

    images.searches = 0
    full = await search_images(lane, documents, "s", "blue kettle", 1, None, (), {})
    assert (len(full.hits), full.ignored, images.searches) == (1, 0, 1)

    images.searches = 0
    exact = await search_images(lane, documents, "s", "blue kettle", 5, None, (), {})
    assert (len(exact.hits), exact.ignored, exact.ignored_at_least, images.searches) == (1, 2, False, 2)

    alone = ImageLane(embedder("third"), images)  # type: ignore[arg-type]
    bound = await search_images(alone, documents, "s", "blue kettle", 2, None, (), {})
    assert (bound.hits, bound.ignored, bound.ignored_at_least) == ([], 2, True)
    third = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder("third"),
                               image_vectors=images).open()  # type: ignore[arg-type]
    found = await third.recall("s", "blue kettle", limit=5, image_lane=True, candidate_limit=2)
    assert image_notes(found) == [ignored_note("at least 2", embedder("third").id)]


class NarrowingUnrecorded(Unrecorded):
    """A tagged index that evaluates conditions itself."""

    narrows_conditions = True

    async def search(self, space, vector, limit, as_of=None, tags=(), where=None, conditions=None):
        return await self.inner.search(space, vector, limit, as_of, tags, where, conditions)


async def test_the_search_for_ignored_vectors_keeps_the_recalls_filters_and_a_recorded_index_is_searched_once():
    from scone_memory.retrieval.filters import parse_filter
    from scone_memory.retrieval.image_lane import ImageLane, search_images

    documents, vectors, images = InMemoryDocumentStore(), InMemoryVectorIndex(), NarrowingUnrecorded()
    first = await MemoryEngine(documents, vectors, HashEmbedder(), image_embedder=embedder(),
                               image_vectors=images).open()  # type: ignore[arg-type]
    await ingest_image(first, "s", picture("red"), media_type="image/png", context=context("Holiday album, first page", "album/0"))
    blue = await ingest_image(first, "s", picture("blue"), media_type="image/png",
                              context=context("Inventory card, shed shelf", "album/1"))
    lane = ImageLane(embedder("other"), images)  # type: ignore[arg-type]
    named = blue.image.attachment_id
    assert (await search_images(lane, documents, "s", "blue kettle", 10, None, (), {})).ignored == 2
    assert (await search_images(lane, documents, "s", "blue kettle", 10, None, (), {"image_original": named})).ignored == 1
    assert (await search_images(lane, documents, "s", "blue kettle", 10, None, (), {},
                                parse_filter({"field": "image_original", "is": named}))).ignored == 1
    assert (await search_images(lane, documents, "s", "blue kettle", 10, None, ("untagged",), {})).ignored == 0
    assert (await search_images(lane, documents, "s", "blue kettle", 10, "2000-01-01T00:00:00Z", (), {})).ignored == 0

    class Counted(InMemoryVectorIndex):
        searches = 0

        async def search(self, space, vector, limit, as_of=None, tags=(), where=None, conditions=None):
            self.searches += 1
            return await super().search(space, vector, limit, as_of, tags, where, conditions)

    recorded = Counted()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), image_embedder=embedder(),
                                image_vectors=recorded).open()
    await ingest_image(engine, "s", picture("red"), media_type="image/png", context=context("Holiday album, first page", "album/0"))
    recorded.searches = 0
    short = await engine.recall("s", "red bicycle", limit=5, image_lane=True)
    assert [item.lanes["image"] for item in short.items if "image" in item.lanes] == [1]
    assert recorded.searches == 1, "an index that records its writer holds no vector it would ignore"
