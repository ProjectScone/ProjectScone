"""`reembed-images` from the command line, and `POST /v1/images/reembed`."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.embedders import HashImageEmbedder
from scone_memory.ingestion.images import ingest_image
from scone_memory.runtime.cli import build_parser, run

from ..retrieval.test_image_lane_rebuild import COLORS, PHRASES, Unrecorded, context, picture


async def lane(salt: str = "", stores=None) -> MemoryEngine:
    documents, vectors, images, blobs = stores
    return await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_vectors=images,
                              image_embedder=HashImageEmbedder(dim=64, phrases=PHRASES, salt=salt)).open()


@pytest.fixture
def stores():
    from scone_memory.backends.blobs import InMemoryBlobStore
    return InMemoryDocumentStore(), InMemoryVectorIndex(), InMemoryVectorIndex(), InMemoryBlobStore()


async def filled(stores) -> tuple[MemoryEngine, list[int]]:
    first = await lane("", stores)
    ids = [(await ingest_image(first, "default", picture(color), media_type="image/png", context=context(n))).added.episode_id
           for n, color in enumerate(COLORS)]
    return await lane("other", stores), ids


async def cli(engine, *arguments: str) -> tuple[int, str]:
    out = io.StringIO()
    code = await run(build_parser().parse_args(list(arguments)), engine, io.StringIO(""), out)
    return code, out.getvalue()


async def test_reembed_images_walks_in_passes_and_says_where_to_walk_on(stores):
    engine, ids = await filled(stores)
    code, text = await cli(engine, "reembed-images", "--limit", "2")
    assert code == 0
    assert text.splitlines() == [
        "re-embedded 2 image(s) of 2 file episode(s) with hash-image-64-other; 0 forgotten meanwhile, 0 failed",
        f"  stopped at --limit 2: run again with --before {ids[1]}",
        "  image index: rebuild in progress; the image lane is off until a pass completes with no space pending",
    ], text
    code, text = await cli(engine, "--json", "reembed-images", "--limit", "2", "--before", str(ids[1]))
    report = json.loads(text)
    assert code == 0 and (report["scanned"], report["reembedded"], report["scan_complete"]) == (1, 1, True)
    assert (report["writer"], report["orphans_removed"], report["resume_before"]) == ("recorded", 0, None)
    code, text = await cli(engine, "reembed-images")
    assert text.splitlines()[1:] == ["  walk complete; removed 0 vector(s) of forgotten images",
                                     "  image index: records hash-image-64-other as its writer; the image lane is on"], text


async def test_reembed_images_names_failures_and_spaces_still_pending(stores):
    engine, ids = await filled(stores)

    async def lost(space, attachment_id):
        from scone_memory.core.errors import NotFound
        raise NotFound(f"attachment {attachment_id[:8]} not found")

    engine.attachment = lost  # type: ignore[method-assign]
    newest = (await engine.episode("default", ids[2])).metadata["image_original"]
    code, text = await cli(engine, "reembed-images")
    assert code == 0
    assert text.splitlines() == [
        "re-embedded 0 image(s) of 3 file episode(s) with hash-image-64-other; 0 forgotten meanwhile, 3 failed",
        f"  failed: {ids[2]}, {ids[1]}, {ids[0]} (first: NotFound: attachment {newest[:8]} not found)",
        "  walk complete; removed 0 vector(s) of forgotten images",
        "  image index: rebuild in progress; the image lane is off until a pass completes with no space pending",
        "  spaces still holding another image embedder's vectors: default",
    ], text


async def test_reembed_images_on_an_index_that_records_no_writer_says_what_it_could_not_do(stores):
    documents, vectors, _, blobs = stores
    engine, _ = await filled((documents, vectors, Unrecorded(), blobs))
    code, text = await cli(engine, "reembed-images")
    assert code == 0
    assert text.splitlines() == [
        "re-embedded 3 image(s) of 3 file episode(s) with hash-image-64-other; 0 forgotten meanwhile, 0 failed",
        "  walk complete; the image index cannot list its vectors, so vectors of forgotten images were not looked for",
        "  image index: cannot record a writer; the image lane ignores vectors another image embedder tagged",
    ], text


async def test_reembed_images_is_refused_by_an_engine_without_the_lane():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with pytest.raises(InvalidInput, match="no image embedder and image index"):
        await cli(engine, "reembed-images")


async def test_the_reembed_route_runs_one_pass_for_a_writer_and_says_when_it_cannot(stores):
    from scone_memory.api import create_app

    engine, ids = await filled(stores)
    laneless = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    keys = {"writer": "default", "reader": "default"}
    with TestClient(create_app(engine, keys, roles={"reader": "read"})) as client:
        writer = {"authorization": "Bearer writer"}
        assert client.get("/v1/capabilities", headers=writer).json()["features"]["images.reembed"] is True
        one = client.post("/v1/images/reembed", json={"limit": 2}, headers=writer)
        assert one.status_code == 200, one.text
        assert (one.json()["reembedded"], one.json()["resume_before"], one.json()["writer"]) == (2, ids[1], "rebuilding")
        two = client.post("/v1/images/reembed", json={"limit": 2, "before": ids[1]}, headers=writer).json()
        assert (two["reembedded"], two["scan_complete"], two["writer"]) == (1, True, "recorded")
        assert client.post("/v1/images/reembed", json={}, headers={"authorization": "Bearer reader"}).status_code == 403
        assert client.post("/v1/images/reembed", json={"limit": 2, "force": True}, headers=writer).status_code == 422
        assert client.post("/v1/images/reembed", json={"limit": 0}, headers=writer).status_code == 422
    with TestClient(create_app(laneless, keys)) as client:
        writer = {"authorization": "Bearer writer"}
        assert client.get("/v1/capabilities", headers=writer).json()["features"]["images.reembed"] is False
        refused = client.post("/v1/images/reembed", json={}, headers=writer)
        assert refused.status_code == 422 and "no image embedder and image index" in refused.text


async def test_the_reembed_route_is_unavailable_where_the_store_cannot_list_episodes():
    from scone_memory.api import create_app

    class Unlisted(InMemoryDocumentStore):
        page_episodes = None  # type: ignore[assignment]

    engine = await MemoryEngine(Unlisted(), InMemoryVectorIndex(), HashEmbedder(), image_vectors=InMemoryVectorIndex(),
                                image_embedder=HashImageEmbedder(dim=64)).open()
    with TestClient(create_app(engine, {"writer": "default"})) as client:
        writer = {"authorization": "Bearer writer"}
        assert client.get("/v1/capabilities", headers=writer).json()["features"]["images.reembed"] is False
        refused = client.post("/v1/images/reembed", json={}, headers=writer)
        assert refused.status_code == 422 and "cannot list episodes" in refused.text


async def test_the_reembed_route_waits_its_turn_behind_ingest():
    import asyncio

    import httpx

    from scone_memory.api import create_app
    from ..ingestion.test_ingest_backpressure import SlowEmbedder

    embedder = SlowEmbedder()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, image_vectors=InMemoryVectorIndex(),
                                image_embedder=HashImageEmbedder(dim=64)).open()
    app = create_app(engine, {"k": "default"}, ingest_concurrency=1)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test",
                                     headers={"Authorization": "Bearer k"}) as client:
            first = asyncio.create_task(client.post("/v1/episodes", json={"content": "the one being embedded"}))
            async with asyncio.timeout(5):
                while embedder.calls == 0:
                    await asyncio.sleep(0.01)
            busy = await asyncio.wait_for(client.post("/v1/images/reembed", json={}), 2)
            assert busy.status_code == 429 and busy.json()["code"] == "ingest_busy"
            embedder.release.set()
            assert (await first).status_code == 200
            assert (await client.post("/v1/images/reembed", json={})).status_code == 200
