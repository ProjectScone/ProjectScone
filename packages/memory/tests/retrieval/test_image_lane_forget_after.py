"""An image ingested with a schedule, with the image lane on: recall withholds it
from every lane once its time has come, and the sweep takes its image vector and
the image and context attachments nothing else carries, leaving the rest."""
from __future__ import annotations

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.ingestion.images import ingest_image
from scone_memory.testing import Clock

from .test_image_lane import QUESTIONS, context, embedder, picture, stores  # noqa: F401 - the stores fixture


async def test_the_sweep_takes_a_scheduled_image_its_vector_and_its_attachments(stores):
    documents, vectors, images, blobs = stores
    clock = Clock("2026-09-15T12:00:00.000Z")
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                                image_vectors=images, clock=clock).open()
    try:
        due = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                                 context=context("Holiday album, first page", "album/1"), forget_after="1h")
        kept = await ingest_image(engine, "s", picture("blue"), media_type="image/png",
                                  context=context("Inventory card, shed shelf", "album/2"))
        assert due.image_lane == kept.image_lane == "indexed"
        [kept_chunk] = await engine.documents.chunks_of("s", kept.added.episode_id)
        assert len(await images.ids("s")) == 2
        before = await engine.recall("s", QUESTIONS["red"], limit=5, image_lane=True)
        assert before.items and before.items[0].episode_id == due.added.episode_id, "only the lane finds it by its pixels"

        clock.now = "2026-09-15T13:00:00.000Z"
        withheld = await engine.recall("s", QUESTIONS["red"], limit=5, image_lane=True)
        assert all(item.episode_id != due.added.episode_id for item in withheld.items)
        assert withheld.past_forget_after is not None and withheld.past_forget_after["episode_ids"] == [due.added.episode_id]
        assert len(await images.ids("s")) == 2, "the read withholds; only the sweep removes"

        report = await engine.forget_due("s")
        assert report.forgotten == [due.added.episode_id]
        assert await images.ids("s") == [kept_chunk.chunk_id], "the image lane's vector goes with the episode"
        held = set(await engine.blobs.held("s"))
        assert due.image.attachment_id not in held and due.manifest.attachment_id not in held
        assert {kept.image.attachment_id, kept.manifest.attachment_id} <= held
    finally:
        await engine.close()


async def test_an_image_whose_schedule_passes_while_it_is_read_is_indexed_and_swept(stores, monkeypatch):
    from scone_memory.ingestion import images as module

    documents, vectors, images, blobs = stores
    clock = Clock("2026-09-15T12:00:00.000Z")
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs, image_embedder=embedder(),
                                image_vectors=images, clock=clock).open()
    real = module.run_bounded

    async def slow(*args, **kwargs):
        clock.now = "2026-09-15T12:05:00.000Z"
        return await real(*args, **kwargs)

    monkeypatch.setattr(module, "run_bounded", slow)
    try:
        due = await ingest_image(engine, "s", picture("red"), media_type="image/png",
                                 context=context("Holiday album, first page", "album/1"), forget_after="2m")
        assert due.image_lane == "indexed" and due.added.forget_after == "2026-09-15T12:02:00.000Z"
        assert (await engine.forget_due("s")).forgotten == [due.added.episode_id]
        assert await images.ids("s") == [] and await engine.blobs.held("s") == []
    finally:
        await engine.close()
