"""Scheduled forgetting on the ingestion paths: an image, a document and a page
fetched by URL carry ``forget_after`` exactly as ``remember`` does.

The schedule is resolved against the engine's clock before anything is stored,
so a refused one (past, or not a time) leaves no attachment behind and fetches
nothing. A re-import of unchanged content is a duplicate and keeps the schedule
the stored episode holds, as ``remember`` does; its receipt says which. A
re-import after the stored one's time has come forgets that one and stores the
new one afresh, with its image or file still linked.
"""
from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import Gone, InvalidInput
from scone_memory.ingestion.files import document_provenance, ingest_document
from scone_memory.ingestion.images import image_provenance, ingest_image
from scone_memory.ingestion.web import WebLimits, ingest_url
from scone_memory.runtime import cli
from scone_memory.testing import Clock

from .test_image_context import context, picture
from .test_url_import import LAB, server  # noqa: F401 - the page server fixture
from .test_video_ocr import video  # noqa: F401 - skips without ffmpeg

NOW = "2026-09-15T12:00:00.000Z"
NOTES = b"# Harbour\n\nThe harbour closes to sailing boats every November.\n"


@pytest.fixture
async def memory():
    clock = Clock(NOW)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock).open()
    engine.test_clock = clock
    yield engine
    await engine.close()


def at(engine, when: str) -> None:
    engine.test_clock.now = when


# -- images ------------------------------------------------------------------


async def test_an_image_takes_a_schedule_and_its_receipt_says_when(memory):
    saved = await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after="2d")
    assert saved.added.forget_after == "2026-09-17T12:00:00.000Z"
    episode = await memory.episode("s", saved.added.episode_id)
    assert episode.metadata["forget_after"] == "2026-09-17T12:00:00.000Z"
    unscheduled = await ingest_image(memory, "s", picture("blue"), media_type="image/png", context=context(source="catalog/blue"))
    assert unscheduled.added.forget_after is None
    assert "forget_after" not in (await memory.episode("s", unscheduled.added.episode_id)).metadata


@pytest.mark.parametrize("bad", ["2020-01-01", NOW, "tomorrow", "5y", "0d"])
async def test_a_refused_image_schedule_stores_nothing(memory, bad):
    with pytest.raises(InvalidInput, match="forget_after"):
        await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after=bad)
    assert await memory.blobs.held("s") == [], "a refused schedule leaves no image or manifest behind"
    assert (await memory.status("s")).episodes == 0


async def test_an_unchanged_image_keeps_the_schedule_it_holds(memory):
    first = await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after="3d")
    again = await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after="10d")
    assert again.added.outcome == "duplicate" and again.added.episode_id == first.added.episode_id
    assert again.added.forget_after == first.added.forget_after == "2026-09-18T12:00:00.000Z", \
        "the receipt says the schedule that holds, not the one asked for"
    assert (await memory.episode("s", first.added.episode_id)).metadata["forget_after"] == "2026-09-18T12:00:00.000Z"


async def test_an_image_imported_again_after_its_time_is_stored_afresh_with_its_image(memory):
    first = await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    again = await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after="1d")
    assert again.added.outcome == "accepted" and again.added.episode_id != first.added.episode_id
    assert again.added.forgot_overdue is not None and again.added.forgot_overdue.episode_id == first.added.episode_id
    assert again.added.forget_after == "2026-09-16T14:00:00.000Z"
    with pytest.raises(Gone):
        await memory.episode("s", first.added.episode_id)
    provenance = await image_provenance(memory, "s", again.added.episode_id)
    assert (await memory.attachment("s", provenance.image.attachment_id))[1] == picture()


# -- documents ---------------------------------------------------------------


async def test_a_document_takes_a_schedule_and_its_receipt_says_when(memory):
    saved = await ingest_document(memory, "s", NOTES, filename="harbour.md", forget_after="2026-10-01")
    assert saved.added.forget_after == "2026-10-01T00:00:00.000Z"
    episode = await memory.episode("s", saved.added.episode_id)
    assert episode.metadata["forget_after"] == "2026-10-01T00:00:00.000Z"
    assert episode.metadata["document_filename"] == "harbour.md"


@pytest.mark.parametrize("bad", ["2020-01-01", "1.5d", "x" * 65])
async def test_a_refused_document_schedule_stores_nothing(memory, bad):
    with pytest.raises(InvalidInput, match="forget_after"):
        await ingest_document(memory, "s", NOTES, filename="harbour.md", forget_after=bad)
    assert await memory.blobs.held("s") == [], "a refused schedule leaves no original or manifest behind"
    assert (await memory.status("s")).episodes == 0


async def test_storing_a_prepared_document_refuses_its_schedule_before_the_manifest(memory):
    from scone_memory.ingestion.files import prepare_document, store_document
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.formats.types import DocumentLimits

    manifest = await prepare_document(NOTES, "harbour.md", parser=BuiltinDocumentParser(), limits=DocumentLimits())
    original = await memory.attach("s", NOTES, "text/markdown", filename="harbour.md")
    with pytest.raises(InvalidInput, match="forget_after"):
        await store_document(memory, "s", original, manifest, forget_after="2020-01-01")
    assert await memory.blobs.held("s") == [original.attachment_id], "the manifest is not stored for a refused schedule"


async def test_an_unchanged_document_keeps_the_schedule_it_holds(memory):
    first = await ingest_document(memory, "s", NOTES, filename="harbour.md")
    again = await ingest_document(memory, "s", NOTES, filename="harbour.md", forget_after="1d")
    assert again.added.outcome == "duplicate" and again.added.episode_id == first.added.episode_id
    assert again.added.forget_after is None, "an unscheduled document stays unscheduled; the receipt says so"
    assert "forget_after" not in (await memory.episode("s", first.added.episode_id)).metadata


async def test_a_document_imported_again_after_its_time_is_stored_afresh_with_its_file(memory):
    first = await ingest_document(memory, "s", NOTES, filename="harbour.md", forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    again = await ingest_document(memory, "s", NOTES, filename="harbour.md")
    assert again.added.outcome == "accepted" and again.added.forgot_overdue is not None
    assert again.added.forgot_overdue.episode_id == first.added.episode_id and again.added.forget_after is None
    evidence = await document_provenance(memory, "s", again.added.episode_id)
    assert (await memory.attachment("s", evidence.original.attachment_id))[1] == NOTES


async def test_recall_withholds_a_scheduled_document_and_the_sweep_takes_its_file(memory):
    saved = await ingest_document(memory, "s", NOTES, filename="harbour.md", forget_after="1h")
    at(memory, "2026-09-15T13:00:00.000Z")
    found = await memory.recall("s", "when does the harbour close")
    assert not found.items and found.past_forget_after is not None
    report = await memory.forget_due("s")
    assert report.forgotten == [saved.added.episode_id]
    assert await memory.blobs.held("s") == [], "the original and its manifest go with the episode"


# -- a page by URL -----------------------------------------------------------


async def test_a_refused_url_schedule_fetches_nothing(memory):
    # Nothing listens on port 9: a fetch would be refused for that, not the schedule.
    with pytest.raises(InvalidInput, match="forget_after"):
        await ingest_url(memory, "s", "http://127.0.0.1:9/page.html", limits=WebLimits(allow_private=True, timeout_seconds=1),
                         forget_after="2020-01-01")


async def test_a_page_by_url_takes_a_schedule_and_its_record_says_when(memory, server):
    imported = await ingest_url(memory, "s", server + "/page.html", limits=LAB, forget_after="36h")
    assert imported.document.added.forget_after == "2026-09-17T00:00:00.000Z"
    assert imported.record()["forget_after"] == "2026-09-17T00:00:00.000Z"
    episode = await memory.episode("s", imported.document.added.episode_id)
    assert episode.metadata["forget_after"] == "2026-09-17T00:00:00.000Z"
    assert episode.metadata["document_url"] == server + "/page.html"
    plain = await ingest_url(memory, "s", server + "/notes.md", limits=LAB)
    assert plain.record()["forget_after"] is None


def test_the_import_url_command_takes_a_schedule(server, capsys):
    env = {"SCONE_DOCUMENTS": "memory", "SCONE_VECTORS": "memory", "SCONE_URL_IMPORT": "1", "SCONE_URL_IMPORT_PRIVATE": "1"}
    out = io.StringIO()
    code = cli.main(["--json", "import-url", server + "/page.html", "--forget-after", "2099-01-01"], env=env,
                    stdin=io.StringIO(""), out=out)
    assert code == 0, out.getvalue()
    assert json.loads(out.getvalue())["forget_after"] == "2099-01-01T00:00:00.000Z"
    out = io.StringIO()
    code = cli.main(["import-url", server + "/notes.md", "--forget-after", "2099-01-01"], env=env,
                    stdin=io.StringIO(""), out=out)
    assert code == 0 and "is to be forgotten after 2099-01-01T00:00:00.000Z" in out.getvalue(), out.getvalue()
    out = io.StringIO()
    code = cli.main(["import-url", server + "/notes.md"], env=env, stdin=io.StringIO(""), out=out)
    assert code == 0 and "imported" in out.getvalue() and "forgotten" not in out.getvalue(), out.getvalue()
    out = io.StringIO()
    code = cli.main(["import-url", server + "/page.html", "--forget-after", "2020-01-01"], env=env,
                    stdin=io.StringIO(""), out=out)
    assert code == 2 and "forget_after" in capsys.readouterr().err


# -- a video stored as sampled frames only ----------------------------------


async def test_a_frames_only_video_takes_a_schedule_and_keeps_the_one_it_holds(memory, video):
    from scone_memory.ingestion.files import prepare_document, store_document
    from scone_memory.ingestion.formats.types import DocumentLimits
    from scone_memory.ingestion.video_ocr import VideoDocumentParser

    from .test_video_ocr import ScriptedOcr

    data, decoder = video
    manifest = await prepare_document(data, "slides.mp4", parser=VideoDocumentParser(
        decoder, ScriptedOcr(empty=True), model_revision="fixture-v1"), limits=DocumentLimits())
    original = await memory.attach("s", data, media_type="video/mp4")
    stored = await store_document(memory, "s", original, manifest, forget_after="2d")
    assert stored.added.chunks == 0 and stored.added.forget_after == "2026-09-17T12:00:00.000Z"
    assert (await memory.episode("s", stored.added.episode_id)).metadata["forget_after"] == "2026-09-17T12:00:00.000Z"
    again = await store_document(memory, "s", original, manifest, forget_after="9d")
    assert again.added.deduplicated and again.added.forget_after == "2026-09-17T12:00:00.000Z"


async def frames_only(memory, video):
    from scone_memory.ingestion.files import prepare_document
    from scone_memory.ingestion.formats.types import DocumentLimits
    from scone_memory.ingestion.video_ocr import VideoDocumentParser

    from .test_video_ocr import ScriptedOcr

    data, decoder = video
    manifest = await prepare_document(data, "slides.mp4", parser=VideoDocumentParser(
        decoder, ScriptedOcr(empty=True), model_revision="fixture-v1"), limits=DocumentLimits())
    return await memory.attach("s", data, media_type="video/mp4"), manifest


async def test_a_frames_only_video_stored_again_after_its_time_is_stored_afresh(memory, video):
    from scone_memory.ingestion.files import store_document

    original, manifest = await frames_only(memory, video)
    first = await store_document(memory, "s", original, manifest, forget_after="1h")
    note = await memory.remember("s", "an ordinary note kept for an hour", forget_after="1h")
    at(memory, "2026-09-15T14:00:00.000Z")
    again = await store_document(memory, "s", original, manifest)
    assert again.added.outcome == "accepted" and again.added.episode_id != first.added.episode_id
    assert again.added.forgot_overdue is not None and again.added.forgot_overdue.episode_id == first.added.episode_id
    assert again.added.forget_after is None
    assert await memory.documents.inflight() == [], "a stored-again video leaves no unfinished mark"
    evidence = await document_provenance(memory, "s", again.added.episode_id)
    assert (await memory.attachment("s", evidence.original.attachment_id))[1] == video[0], "its original is still held"
    report = await memory.forget_due("s")
    assert report.forgotten == [note.episode_id], "the sweep still reaches the rest of the space"


async def test_an_interrupted_frames_only_video_past_its_time_still_lets_the_engine_open(tmp_path, video):
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    from scone_memory.ingestion.files import store_document

    clock = Clock(NOW)

    def opened():
        return MemoryEngine(SqliteDocumentStore(tmp_path / "m.db"), SqliteVectorIndex(tmp_path / "m.db"), HashEmbedder(),
                            blobs=FileBlobStore(tmp_path / "blobs"), clock=clock).open()

    engine = await opened()
    original, manifest = await frames_only(engine, video)
    stored = await store_document(engine, "s", original, manifest, forget_after="1h")
    held = await engine.documents.get_episode("s", stored.added.episode_id)
    # As a crash between the mark and its clearing leaves it.
    await engine.documents.mark_inflight("s", held.content_hash)
    await engine.close()
    clock.now = "2026-09-15T14:00:00.000Z"
    engine = await opened()
    try:
        assert await engine.documents.inflight() == [], "recovery finishes the video even though its time has come"
        assert (await engine.forget_due("s")).forgotten == [stored.added.episode_id]
    finally:
        await engine.close()


async def test_a_document_whose_schedule_passes_during_its_parse_is_stored_due_at_once(memory):
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser

    class SlowParser(BuiltinDocumentParser):
        async def parse(self, *args, **kwargs):
            at(memory, "2026-09-15T12:05:00.000Z")
            return await super().parse(*args, **kwargs)

    saved = await ingest_document(memory, "s", NOTES, filename="harbour.md", parser=SlowParser(), forget_after="2m")
    assert saved.added.forget_after == "2026-09-15T12:02:00.000Z", "the instant resolved at the door is the one stored"
    with pytest.raises(Gone):
        await memory.episode("s", saved.added.episode_id)
    assert (await memory.forget_due("s")).forgotten == [saved.added.episode_id]
    assert await memory.blobs.held("s") == [], "the sweep takes the original and manifest with it"


async def test_an_image_whose_schedule_passes_while_it_is_read_is_stored_due_at_once(memory, monkeypatch):
    from scone_memory.ingestion import images

    real = images.run_bounded

    async def slow(*args, **kwargs):
        at(memory, "2026-09-15T12:05:00.000Z")
        return await real(*args, **kwargs)

    monkeypatch.setattr(images, "run_bounded", slow)
    saved = await ingest_image(memory, "s", picture(), media_type="image/png", context=context(), forget_after="2m")
    assert saved.added.forget_after == "2026-09-15T12:02:00.000Z"
    assert (await memory.forget_due("s")).forgotten == [saved.added.episode_id]
    assert await memory.blobs.held("s") == []
