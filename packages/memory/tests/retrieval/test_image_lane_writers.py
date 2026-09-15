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
