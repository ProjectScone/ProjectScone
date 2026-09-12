"""Crash-boundary source cleanup preserves targets and leaves claims standing."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.blobs import FileBlobStore, InMemoryBlobStore
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.errors import Gone, InvalidInput, NotFound
from scone_memory.observability.events import InMemoryEventLog, SqliteEventLog

NOW = "2026-09-11T00:00:00Z"


async def make_engine(tmp_path, backend):
    if backend == "sqlite":
        path = tmp_path / "catalog.db"
        return await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(),
                                  blobs=FileBlobStore(tmp_path / "blobs"), events=SqliteEventLog(path),
                                  clock=lambda: NOW).open()
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              blobs=InMemoryBlobStore(), events=InMemoryEventLog(), clock=lambda: NOW).open()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("stage", ["intent", "rows", "vectors", "blobs", "stone", "revision", "event", "ack"])
@pytest.mark.parametrize("after_write", [False, True])
async def test_cleanup_retries_each_boundary(tmp_path, monkeypatch, backend, stage, after_write):
    memory = await make_engine(tmp_path, backend)
    first_blob = await memory.attach("alpha", b"unique payload", "text/plain")
    shared = await memory.attach("alpha", b"shared payload", "text/plain")
    source = await memory.remember("alpha", "Alice works at Acme", attachment_ids=[first_blob.attachment_id, shared.attachment_id])
    other = await memory.remember("alpha", "other source", attachment_ids=[shared.attachment_id])
    fact = await memory.assert_fact("alpha", "Alice", "works_at", "Acme", source_episode_id=source.episode_id,
                                    quote="Alice works at Acme")
    old_ids = {c.chunk_id for c in await memory.documents.chunks_of("alpha", source.episode_id)}
    targets = {
        "intent": (memory.documents, "record_retirement"), "rows": (memory.documents, "delete_episode"),
        "vectors": (memory.vectors, "delete"), "blobs": (memory.blobs, "unlink"),
        "stone": (memory.documents, "record_tombstone"), "revision": (memory.documents, "bump_revision"),
        "event": (memory.events, "append"), "ack": (memory.documents, "clear_retirement"),
    }
    target, name = targets[stage]
    original = getattr(target, name)

    async def interrupt(*args, **kwargs):
        if after_write:
            await original(*args, **kwargs)
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(target, name, interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        await memory.forget("alpha", source.episode_id)
    monkeypatch.setattr(target, name, original)
    if stage == "intent" and not after_write:
        assert await memory.documents.get_episode("alpha", source.episode_id) is not None
        assert await memory.documents.retirement("alpha", source.episode_id) is None
        await memory.forget("alpha", source.episode_id)
    else:
        pending = await memory.documents.retirement("alpha", source.episode_id)
        assert pending is not None or (stage == "ack" and after_write)
        if backend == "sqlite":
            await memory.close()
            memory = await make_engine(tmp_path, backend)
        elif pending is not None:
            receipt = await memory.forget("alpha", source.episode_id)
            assert receipt.forgotten_at == NOW
    try:
        assert await memory.documents.retirement("alpha", source.episode_id) is None
        with pytest.raises(Gone):
            await memory.episode("alpha", source.episode_id)
        assert old_ids.isdisjoint(await memory.vectors.ids("alpha"))
        with pytest.raises(NotFound):
            await memory.attachment("alpha", first_blob.attachment_id)
        assert (await memory.attachment("alpha", shared.attachment_id))[1] == b"shared payload"
        assert (await memory.episode("alpha", other.episode_id)).content == "other source"
        assert (await memory.fact("alpha", fact.fact_id)).status == "active"
        assert len(await memory.events.query("alpha", kind="forget")) == 1
        assert (await memory.recover()).retired == 0
    finally:
        await memory.close()


async def test_unsupported_custom_catalog_fails_before_deleting(tmp_path):
    memory = await make_engine(tmp_path, "memory")
    source = await memory.remember("alpha", "preserve this source")
    memory.documents.record_retirement = None
    with pytest.raises(InvalidInput, match="retirement"):
        await memory.forget("alpha", source.episode_id)
    assert (await memory.episode("alpha", source.episode_id)).content == "preserve this source"


async def test_recovery_is_bounded(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, "memory")
    original = memory.documents.delete_episode

    async def interrupted(*args):
        raise RuntimeError("interrupted")

    for number in range(3):
        source = await memory.remember("alpha", f"source {number}")
        monkeypatch.setattr(memory.documents, "delete_episode", interrupted)
        with pytest.raises(RuntimeError):
            await memory.forget("alpha", source.episode_id)
        monkeypatch.setattr(memory.documents, "delete_episode", original)
    report = await memory.recover(retirement_limit=2)
    assert report.retired == 2 and report.retirements_pending
    report = await memory.recover(retirement_limit=2)
    assert report.retired == 1 and not report.retirements_pending
    assert (await memory.status("alpha")).episodes == 0


async def test_open_refuses_serving_a_remaining_cleanup_backlog(tmp_path):
    from scone_memory.core.models import ForgetReceipt
    from scone_memory.core.retirement import Retirement
    memory = await make_engine(tmp_path, "memory")
    for number in range(101):
        await memory.documents.record_retirement(Retirement(
            space="alpha", episode_id=number + 1, content_hash=f"source-{number}",
            requested_at=NOW, chunk_ids=(), receipt=ForgetReceipt(episode_id=number + 1, chunks=0),
        ))
    with pytest.raises(InvalidInput, match="source cleanup remains"):
        await memory.open()
    assert len(await memory.documents.page_retirements(None, 101)) == 1
    assert (await memory.recover()).retired == 1
    assert await memory.open() is memory


async def test_forget_refuses_unfinished_index_and_recovery_does_not_resurrect_retired_source(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, "memory")
    source = await memory.remember("alpha", "recover this evidence")
    episode = await memory.episode("alpha", source.episode_id)
    await memory.documents.mark_inflight("alpha", episode.content_hash)
    with pytest.raises(InvalidInput, match="indexing is unfinished"):
        await memory.forget("alpha", source.episode_id)
    assert await memory.documents.retirement("alpha", source.episode_id) is None
    await memory.recover()
    original = memory.documents.delete_episode

    async def interrupted(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(memory.documents, "delete_episode", interrupted)
    with pytest.raises(RuntimeError):
        await memory.forget("alpha", source.episode_id)
    monkeypatch.setattr(memory.documents, "delete_episode", original)
    await memory.documents.mark_inflight("alpha", episode.content_hash)
    report = await memory.recover()
    assert report.retired == 1 and report.completed == 0
    assert await memory.documents.inflight() == []
    with pytest.raises(Gone):
        await memory.episode("alpha", source.episode_id)


async def test_http_delete_retry_finishes_failed_cleanup_with_space_isolation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from scone_memory.api import create_app
    memory = await make_engine(tmp_path, "memory")
    source = await memory.remember("alpha", "HTTP source")
    original = memory.vectors.delete

    async def interrupted(*args):
        raise RuntimeError("interrupted")

    with TestClient(create_app(memory, {"owner": "alpha", "foreign": "bravo"}), raise_server_exceptions=False) as client:
        monkeypatch.setattr(memory.vectors, "delete", interrupted)
        assert client.delete(f"/v1/episodes/{source.episode_id}", headers={"Authorization": "Bearer owner"}).status_code == 500
        monkeypatch.setattr(memory.vectors, "delete", original)
        assert client.delete(f"/v1/episodes/{source.episode_id}", headers={"Authorization": "Bearer foreign"}).status_code == 404
        response = client.delete(f"/v1/episodes/{source.episode_id}", headers={"Authorization": "Bearer owner"})
        assert response.status_code == 200 and response.json()["forgotten_at"] == NOW
        assert client.get(f"/v1/episodes/{source.episode_id}", headers={"Authorization": "Bearer owner"}).status_code == 410


async def test_recovery_finishes_rows_after_failure_inside_document_cleanup(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, "memory")
    memory.chunk_target = 40
    source = await memory.remember("alpha", " ".join(f"Sentence {i} has many words for chunks." for i in range(15)))
    chunks = await memory.documents.chunks_of("alpha", source.episode_id)
    assert len(chunks) > 1
    index = memory.documents._bm25["alpha"]
    original = index.remove

    def interrupted(chunk_id):
        raise RuntimeError("lexical deletion interrupted")

    monkeypatch.setattr(index, "remove", interrupted)
    with pytest.raises(RuntimeError):
        await memory.forget("alpha", source.episode_id)
    monkeypatch.setattr(index, "remove", original)
    report = await memory.recover()
    assert report.retired == 1 and not report.retirements_pending
    assert await memory.documents.chunks_of("alpha", source.episode_id) == []
    assert (await memory.doctor("alpha")).chunks_without_episode == []
    assert index.search("Sentence", 100) == []


async def test_incomplete_document_acknowledgement_keeps_the_intent(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, "memory")
    source = await memory.remember("alpha", "source with incomplete cleanup")

    async def incomplete(*args):
        return []

    monkeypatch.setattr(memory.documents, "delete_episode", incomplete)
    with pytest.raises(InvalidInput, match="cleanup is incomplete"):
        await memory.forget("alpha", source.episode_id)
    assert await memory.documents.retirement("alpha", source.episode_id) is not None
    assert await memory.documents.tombstone("alpha", source.episode_id) is None
