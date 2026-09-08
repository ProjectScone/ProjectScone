"""Two framework APIs share Scone evidence without a model or network."""

import socket
import os

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_llamaindex_retrieval_in_langchain_workflow_preserves_scope_and_provenance(
    tmp_path, monkeypatch, backend,
):
    pytest.importorskip("langchain_core")
    pytest.importorskip("llama_index.core")
    from scone_memory.integrations.langchain import SconeRetriever as LangChainRetriever
    from langsmith import Client
    from langsmith.run_helpers import get_tracing_context

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        pytest.fail("framework composition attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(Client, "create_run", forbidden)
    monkeypatch.setattr(Client, "update_run", forbidden)
    from scone_memory.integrations.composition import build_retrieval_workflow, retrieve_without_tracing
    documents = InMemoryDocumentStore() if backend == "memory" else SqliteDocumentStore(str(tmp_path / "memory.db"))
    memory = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), chunk_target=200).open()
    try:
        content = "The telescope calibration runbook is in the observatory wiki. " * 20
        selected = await memory.remember("team", content, source="docs/calibration", created_at="2025-06-01",
                                          metadata={"project": "telescope", "user_id": "alice"})
        for space, project, owner in [("other-team", "telescope", "alice"), ("team", "other", "alice"),
                                      ("team", "telescope", "bob")]:
            await memory.remember(space, f"Private telescope calibration runbook: {space}/{project}/{owner}",
                                  metadata={"project": project, "user_id": owner})
        where = {"project": "telescope", "user_id": "alice"}
        fixed_where = dict(where)
        workflow = build_retrieval_workflow(memory, "team", where=where, limit=3)
        where["project"] = "other"  # Caller mutation must not broaden the built workflow.
        query = "Where is the telescope calibration runbook?"
        before = await memory.status("team")
        prior_tracing = get_tracing_context()
        composed = await retrieve_without_tracing(workflow, query)
        native = await retrieve_without_tracing(LangChainRetriever(memory=memory, space="team", where=fixed_where, limit=3), query)
        assert composed and composed == native
        chunks = {chunk.chunk_id: chunk for chunk in await memory.documents.chunks_of("team", selected.episode_id)}
        for document in composed:
            metadata = document.metadata
            assert metadata["episode_id"] == selected.episode_id
            assert document.page_content == chunks[metadata["chunk_id"]].text
            assert metadata["source"] == "docs/calibration"
            assert metadata["created_at"] == "2025-06-01T00:00:00.000Z"
            assert metadata["project"] == "telescope" and metadata["user_id"] == "alice"
        after = await memory.status("team")
        assert (after.episodes, after.chunks) == (before.episodes, before.chunks), "retrieval creates no memory"
        await memory.forget("team", selected.episode_id)
        assert await retrieve_without_tracing(workflow, query) == [], "neither stale nodes nor another scope may replace deleted evidence"
        assert attempts == []
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert get_tracing_context() == prior_tracing
    finally:
        await memory.close()
