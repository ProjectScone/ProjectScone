"""An image whose vector could not be made is stored, found by its caption, and says so.

``ingest_image`` stores the image, its context and the caption episode before
it asks the image embedder for a vector. When the embedder or the image index
then fails, the receipt says ``failed`` with the error instead of raising for
an image that is already stored; an exact retry writes the vector.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ImageAttribute, ImageContext, ingest_image


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new("RGB", (16, 16), color).save(out, format="PNG")
    return out.getvalue()


CAPTION = ImageContext(source="album/0", attributes=(
    ImageAttribute(kind="alt", value="Lighthouse keeper's logbook, winter volume", origin="html"),))


class Offline(HashImageEmbedder):
    """An image embedder whose model is unreachable until it is put back."""

    def __init__(self) -> None:
        super().__init__(dim=64, phrases={"red bicycle": picture("red")})
        self.down = True

    async def embed_images(self, images):
        if self.down:
            raise ConnectionError("image model unreachable")
        return await super().embed_images(images)


class Unwritable(InMemoryVectorIndex):
    """An image index that refuses writes until it is put back."""

    down = True

    async def upsert(self, points):
        if self.down:
            raise OSError("image index is read-only")
        await super().upsert(points)


@pytest.fixture(params=["memory", "sqlite"])
def stores(request, tmp_path):
    if request.param == "memory":
        return InMemoryDocumentStore(), InMemoryVectorIndex(), None
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    return SqliteDocumentStore(tmp_path / "memory.db"), SqliteVectorIndex(tmp_path / "memory.db"), FileBlobStore(tmp_path / "blobs")


async def test_an_image_the_embedder_could_not_embed_is_stored_found_by_caption_and_says_failed(stores):
    documents, vectors, blobs = stores
    model, images = Offline(), InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs,
                                image_embedder=model, image_vectors=images).open()
    try:
        saved = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=CAPTION)
        assert saved.image_lane == "failed"
        assert saved.image_lane_error == "ConnectionError: image model unreachable"
        assert saved.image_lane_blocked is None
        assert await images.ids("s") == []
        # Stored: the image, its context, and the caption the text lanes find it by.
        assert (await engine.attachment("s", saved.image.attachment_id))[1] == picture("red")
        found = await engine.recall("s", "lighthouse logbook", limit=3)
        assert [item.episode_id for item in found.items][:1] == [saved.added.episode_id]
        model.down = False
        again = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=CAPTION)
        assert (again.added.episode_id, again.image_lane, again.image_lane_error) == (saved.added.episode_id, "indexed", None)
        lane = await engine.recall("s", "red bicycle", limit=3, image_lane=True)
        assert [item.lanes.get("image") for item in lane.items if item.episode_id == saved.added.episode_id] == [1]
    finally:
        await engine.close()


async def test_an_image_the_index_would_not_take_is_stored_and_says_failed():
    documents, images = InMemoryDocumentStore(), Unwritable()
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(),
                                image_embedder=HashImageEmbedder(dim=64), image_vectors=images).open()
    saved = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=CAPTION)
    assert (saved.image_lane, saved.image_lane_error) == ("failed", "OSError: image index is read-only")
    assert (await engine.episode("s", saved.added.episode_id)).metadata["image_original"] == saved.image.attachment_id
    Unwritable.down = False
    try:
        again = await ingest_image(engine, "s", picture("red"), media_type="image/png", context=CAPTION)
    finally:
        Unwritable.down = True
    assert again.image_lane == "indexed" and len(await images.ids("s")) == 1


async def test_the_images_route_answers_a_stored_image_whose_vector_failed():
    from scone_memory.api import create_app

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                image_embedder=Offline(), image_vectors=InMemoryVectorIndex()).open()
    auth = {"authorization": "Bearer writer"}
    try:
        with TestClient(create_app(engine, {"writer": "alpha"})) as client:
            image = client.post("/v1/attachments", content=picture("red"),
                                headers={**auth, "content-type": "image/png"}).json()
            body = {"attachment_id": image["attachment_id"], "context": CAPTION.model_dump(mode="json")}
            saved = client.post("/v1/images", json=body, headers=auth)
            assert saved.status_code == 200, saved.text
            answer = saved.json()
            assert (answer["image_lane"], answer["image_lane_error"]) == ("failed", "ConnectionError: image model unreachable")
            found = client.get("/v1/images/search", params={"query": "lighthouse logbook"}, headers=auth).json()
            assert [match["episode_id"] for match in found["matches"]] == [answer["added"]["episode_id"]]
    finally:
        await engine.close()
