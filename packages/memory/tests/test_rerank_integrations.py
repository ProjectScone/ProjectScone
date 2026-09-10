"""Framework adapters preserve original scores while exposing optional reranking."""

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, SyncMemoryEngine
from scone_memory.retrieval.reranking import RerankScore


class LocalReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, candidates):
        self.calls.append(candidates)
        return [RerankScore(chunk_id=item.chunk_id, score=100.0 if "target" in item.text else 10.0)
                for item in candidates]


@pytest.mark.parametrize("framework", ["langchain", "llamaindex"])
async def test_framework_candidate_override_and_rerank_opt_out_preserve_scoped_evidence(framework):
    if framework == "langchain":
        pytest.importorskip("langchain_core")
        from scone_memory.integrations.langchain import SconeRetriever
    else:
        pytest.importorskip("llama_index.core")
        from scone_memory.integrations.llamaindex import SconeRetriever
    ranker = LocalReranker()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                 candidate_limit=1, reranker=ranker).open()
    try:
        for number in range(2):
            await engine.remember("alpha", f"calibration notes {number}", metadata={"owner": "alice"})
        target = await engine.remember("alpha", "calibration target blueprint", source="docs/calibration",
                                       metadata={"owner": "alice", "rerank_score": "spoof"})
        await engine.remember("alpha", "calibration private distractor", metadata={"owner": "bob"})
        await engine.remember("beta", "calibration private tenant", metadata={"owner": "alice"})
        retriever = SconeRetriever(memory=engine, space="alpha", where={"owner": "alice"},
                                   limit=1, candidate_limit=3)
        if framework == "langchain":
            result = await retriever.ainvoke("calibration")
            text, metadata = result[0].page_content, result[0].metadata
        else:
            result = await retriever.aretrieve("calibration")
            text, metadata = result[0].node.text, result[0].node.metadata
        assert len(ranker.calls) == 1 and len(ranker.calls[0]) == 3
        assert text == "calibration target blueprint" and metadata["episode_id"] == target.episode_id
        assert metadata["owner"] == "alice" and metadata["source"] == "docs/calibration"
        assert metadata["rerank_score"] == 100.0 and 0 < metadata["score"] <= 1
        original = next(item for item in ranker.calls[0] if item.chunk_id == metadata["chunk_id"])
        assert metadata["score"] == pytest.approx(original.baseline_score, abs=1e-6)
        assert metadata["similarity"] == pytest.approx(original.similarity, abs=1e-6)
        assert metadata["lanes"] == dict(original.lanes)
        retriever.rerank = False
        if framework == "langchain":
            baseline = (await retriever.ainvoke("calibration"))[0].metadata
        else:
            baseline = (await retriever.aretrieve("calibration"))[0].node.metadata
        assert baseline["rerank_score"] is None and len(ranker.calls) == 1
    finally:
        await engine.close()


def test_sync_recall_forwards_candidate_override_and_rerank_opt_out():
    ranker = LocalReranker()
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                      candidate_limit=1, reranker=ranker)) as memory:
        for text in ["calibration notes", "calibration checklist", "calibration target blueprint"]:
            memory.remember("alpha", text)
        baseline = memory.recall("alpha", "calibration", limit=1, candidate_limit=3, rerank=False)
        assert len(baseline.items) == 1 and ranker.calls == []
        result = memory.recall("alpha", "calibration", limit=1, candidate_limit=3)
        assert result.items[0].text == "calibration target blueprint"
        assert result.items[0].rerank_score == 100.0 and len(ranker.calls[0]) == 3
