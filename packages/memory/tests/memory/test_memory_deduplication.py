from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.deduplication import DeduplicationConfig, DocumentRevision
from scone_memory.deduplication.memory import MemoryDuplicateInspector


@pytest.fixture(params=["memory", "sqlite", "qdrant"])
async def memory(request, tmp_path):
    from scone_memory.backends import QdrantVectorIndex, SqliteDocumentStore, SqliteVectorIndex

    if request.param == "qdrant":
        pytest.importorskip("qdrant_client")
    documents = SqliteDocumentStore(tmp_path / "store.db") if request.param == "sqlite" else InMemoryDocumentStore()
    vectors = (QdrantVectorIndex(":memory:", "dedup_test") if request.param == "qdrant" else
               SqliteVectorIndex(tmp_path / "store.db") if request.param == "sqlite" else InMemoryVectorIndex())
    engine = MemoryEngine(documents, vectors, HashEmbedder())
    await engine.open()
    yield engine
    await engine.close()


async def test_inspection_reports_source_spans_without_changing_original(memory):
    original = "The Juniper calibration procedure uses Polaris as its reference point."
    saved = await memory.remember("alpha", original, source="manual.pdf")
    await memory.remember("private", original)
    inspector = MemoryDuplicateInspector(memory, DeduplicationConfig(semantic_enabled=False))
    incoming = DocumentRevision("alpha", "incoming.pdf", "1", original + " New instructions follow.")
    report = await inspector.inspect(incoming)
    assert report.requires_review and report.copied_fraction > .25
    assert report.matches
    assert {span.source_id for span in report.matches} == {f"episode:{saved.episode_id}"}
    assert not report.complete
    assert report.semantic_status == "disabled"
    stored = await memory.documents.get_episode("alpha", saved.episode_id)
    assert stored is not None and stored.content == original


async def test_hash_embedder_is_not_presented_as_paraphrase_understanding(memory):
    report = await MemoryDuplicateInspector(memory).inspect(DocumentRevision("alpha", "new", "1", "A new document."))
    assert report.semantic_status == "unavailable"


async def test_stored_document_does_not_report_itself(memory):
    saved = await memory.remember("alpha", "Juniper calibration uses the Polaris reference and a stable clock.")
    report = await MemoryDuplicateInspector(memory).inspect_episode("alpha", saved.episode_id)
    assert not report.matches


async def test_inspect_episode_enforces_space(memory):
    saved = await memory.remember("private", "The private document should never be visible in another space.")
    with pytest.raises(ValueError, match="not found"):
        await MemoryDuplicateInspector(memory).inspect_episode("alpha", saved.episode_id)
