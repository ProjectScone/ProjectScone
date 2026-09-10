"""Recall controls exercise the real engine using isolated retained records."""
from __future__ import annotations

from dataclasses import asdict
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.retrieval.reranking import RerankCandidate, RerankScore

AUTH = {"authorization": "Bearer alpha-key"}


class RecordingReranker:
    def __init__(self, mode: str = "reverse") -> None:
        self.mode = mode
        self.calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    async def rerank(self, query: str, candidates: tuple[RerankCandidate, ...]) -> list[RerankScore]:
        self.calls.append((query, candidates))
        if self.mode == "error":
            raise RuntimeError("private adapter diagnostics must stay off the wire")
        if self.mode == "fabricated":
            return [RerankScore(987654321, 1.0) for _ in candidates]
        return [RerankScore(candidate.chunk_id, float(index)) for index, candidate in enumerate(candidates)]


@pytest.fixture
async def setup_engine():
    engines: list[MemoryEngine] = []

    async def make(reranker: RecordingReranker | None = None, **options: object) -> MemoryEngine:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                    reranker=reranker, **options).open()
        engines.append(engine)
        for name in ("Amber", "Birch", "Cedar", "Dahlia"):
            await engine.remember("alpha", f"NacreVault supports {name} workspace retrieval.")
        return engine

    yield make
    for engine in engines:
        await engine.close()


def client_for(engine: MemoryEngine) -> TestClient:
    return TestClient(create_app(engine, {"alpha-key": "alpha", "beta-key": "beta"}))


@pytest.mark.parametrize("value", ["0", "-1", "1001", "true", "false", "1.5", "nope", ""])
async def test_invalid_candidate_budget_is_rejected_before_retrieval(setup_engine, monkeypatch, value):
    ranker = RecordingReranker()
    engine = await setup_engine(ranker)
    spies = [AsyncMock(wraps=engine.embedder.embed), AsyncMock(wraps=engine.vectors.search),
             AsyncMock(wraps=engine.documents.search_text)]
    for owner, name, spy in zip((engine.embedder, engine.vectors, engine.documents),
                                 ("embed", "search", "search_text"), spies):
        monkeypatch.setattr(owner, name, spy)
    with client_for(engine) as client:
        response = client.get("/v1/recall", params={"q": "NacreVault", "candidate_limit": value}, headers=AUTH)
    assert response.status_code == 422
    assert "candidate_limit" in response.json()["error"]
    assert all(spy.await_count == 0 for spy in spies)
    assert ranker.calls == []


async def test_candidate_budget_controls_lanes_independently_of_page_and_rerank_can_be_bypassed(setup_engine, monkeypatch):
    ranker = RecordingReranker()
    engine = await setup_engine(ranker)
    vector = AsyncMock(wraps=engine.vectors.search)
    lexical = AsyncMock(wraps=engine.documents.search_text)
    embed = AsyncMock(wraps=engine.embedder.embed)
    monkeypatch.setattr(engine.vectors, "search", vector)
    monkeypatch.setattr(engine.documents, "search_text", lexical)
    monkeypatch.setattr(engine.embedder, "embed", embed)
    with client_for(engine) as client:
        response = client.get("/v1/recall", params={"q": "NacreVault", "limit": 1,
                              "candidate_limit": 3, "rerank": "false"}, headers=AUTH)
        capabilities = client.get("/v1/capabilities", headers=AUTH).json()["features"]
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1 and ranker.calls == []
    assert vector.await_count == lexical.await_count == embed.await_count == 1
    assert vector.call_args.args[2] == lexical.call_args.args[2] == 3
    assert lexical.call_args.args[1] == "NacreVault"
    assert embed.call_args.args[0] == ["NacreVault"]
    assert "rerank" not in body
    assert all("rerank_score" not in item for item in body["items"])
    assert capabilities["recall.candidate_budget"] is True and capabilities["recall.reranking"] is True


async def test_unconfigured_reranking_preserves_existing_wire_shape(setup_engine):
    engine = await setup_engine()
    with client_for(engine) as client:
        body = client.get("/v1/recall", params={"q": "NacreVault"}, headers=AUTH).json()
        capabilities = client.get("/v1/capabilities", headers=AUTH).json()["features"]
    assert body["items"] and "rerank" not in body
    assert all("rerank_score" not in item for item in body["items"])
    assert capabilities["recall.candidate_budget"] is True and capabilities["recall.reranking"] is False


@pytest.mark.parametrize("budget", [1, 1000])
async def test_candidate_budget_accepts_both_boundaries(setup_engine, monkeypatch, budget):
    engine = await setup_engine()
    lexical = AsyncMock(wraps=engine.documents.search_text)
    monkeypatch.setattr(engine.documents, "search_text", lexical)
    with client_for(engine) as client:
        response = client.get("/v1/recall", params={"q": "NacreVault", "candidate_limit": budget}, headers=AUTH)
    assert response.status_code == 200
    assert lexical.call_args.args[2] == budget


async def test_invalid_rerank_control_is_rejected_before_retrieval(setup_engine, monkeypatch):
    engine = await setup_engine(RecordingReranker())
    embed = AsyncMock(wraps=engine.embedder.embed)
    monkeypatch.setattr(engine.embedder, "embed", embed)
    with client_for(engine) as client:
        response = client.get("/v1/recall", params={"q": "NacreVault", "rerank": "sometimes"}, headers=AUTH)
    assert response.status_code == 422 and "rerank" in response.json()["error"]
    assert embed.await_count == 0


async def test_applied_trace_reports_exact_payload_and_original_chunk_identities(setup_engine, monkeypatch):
    ranker = RecordingReranker()
    engine = await setup_engine(ranker, rerank_limit=2)
    vector = AsyncMock(wraps=engine.vectors.search)
    lexical = AsyncMock(wraps=engine.documents.search_text)
    embed = AsyncMock(wraps=engine.embedder.embed)
    monkeypatch.setattr(engine.vectors, "search", vector)
    monkeypatch.setattr(engine.documents, "search_text", lexical)
    monkeypatch.setattr(engine.embedder, "embed", embed)
    with client_for(engine) as client:
        response = client.get("/v1/recall", params={"q": "NacreVault", "limit": 2,
                              "candidate_limit": 4}, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert len(ranker.calls) == 1 and vector.await_count == lexical.await_count == embed.await_count == 1
    query, candidates = ranker.calls[0]
    assert query == "NacreVault" and len(candidates) == 2
    assert [item["chunk_id"] for item in body["items"]] == [candidate.chunk_id for candidate in reversed(candidates)]
    assert [item["text"] for item in body["items"]] == [candidate.text for candidate in reversed(candidates)]
    assert [item["rerank_score"] for item in body["items"]] == [1.0, 0.0]
    trace = body["rerank"]
    assert (trace["status"], trace["ordering"]) == ("applied", "rerank")
    assert (trace["candidates_considered"], trace["candidates_sent"], trace["candidates_omitted"]) == (4, 2, 2)
    payload = json.dumps({"query": query, "candidates": [asdict(candidate) for candidate in candidates]},
                         ensure_ascii=False, separators=(",", ":")).encode()
    assert trace["payload_bytes"] == len(payload) and trace["duration_ms"] >= 0


@pytest.mark.parametrize("mode", ["error", "fabricated"])
async def test_failed_reranking_keeps_fusion_order_and_never_returns_invented_ids(setup_engine, mode):
    ranker = RecordingReranker(mode)
    engine = await setup_engine(ranker)
    with client_for(engine) as client:
        params = {"q": "NacreVault", "limit": 3, "candidate_limit": 4}
        baseline = client.get("/v1/recall", params={**params, "rerank": "false"}, headers=AUTH).json()
        response = client.get("/v1/recall", params=params, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == baseline["items"]
    assert body["rerank"]["status"] == "failed" and body["rerank"]["ordering"] == "fusion"
    assert body["rerank"]["candidates_considered"] == body["rerank"]["candidates_sent"] == 4
    assert body["rerank"]["candidates_omitted"] == 0 and body["rerank"]["payload_bytes"] > 0
    assert body["rerank"]["duration_ms"] >= 0
    assert len(ranker.calls) == 1
    assert "private adapter diagnostics" not in response.text
    assert "987654321" not in response.text


async def test_reranker_only_sees_retained_records_in_request_scope_even_with_stale_vector_hits(setup_engine, monkeypatch):
    ranker = RecordingReranker()
    engine = await setup_engine(ranker)
    common = {"source": "docs/approved", "kind": "note", "tags": ["public"],
              "metadata": {"status": "published"}, "created_at": "2025-03-05"}
    with client_for(engine) as client:
        def add(content: str, **overrides: object) -> int:
            response = client.post("/v1/episodes", json={"content": content, **common, **overrides}, headers=AUTH)
            assert response.status_code == 200
            return response.json()["episode_id"]

        allowed = add("NacreVault allowed retained text.")
        removed = add("NacreVault deleted secret text.")
        add("NacreVault wrong source text.", source="private/report")
        add("NacreVault wrong tag text.", tags=["private"])
        add("NacreVault wrong metadata text.", metadata={"status": "draft"})
        add("NacreVault future text.", created_at="2026-01-01")
        add("NacreVault older text.", created_at="2020-01-01")
        beta = client.post("/v1/episodes", json={"content": "NacreVault other tenant secret."},
                           headers={"authorization": "Bearer beta-key"})
        assert beta.status_code == 200
        stale = client.get("/v1/recall", params={"q": "NacreVault", "limit": 100,
                            "rerank": "false"}, headers=AUTH).json()["items"]
        assert client.delete(f"/v1/episodes/{removed}", headers=AUTH).status_code == 200
        # A stale vector index still advertises the forgotten ID. The document
        # store remains authoritative and must filter it before external ranking.
        monkeypatch.setattr(engine.vectors, "search", AsyncMock(return_value=[
            (item["chunk_id"], 0.9) for item in stale]))
        ranker.calls.clear()
        response = client.get("/v1/recall", params={"q": "NacreVault", "candidate_limit": 50,
            "source_prefix": "docs/", "kind": "note", "tags": "public", "where": "status:published",
            "since": "2025-01-01", "until": "2025-12-31"}, headers=AUTH)
    assert response.status_code == 200
    assert len(ranker.calls) == 1
    _, candidates = ranker.calls[0]
    assert [(candidate.episode_id, candidate.text) for candidate in candidates] == [
        (allowed, "NacreVault allowed retained text.")]
    assert [item["episode_id"] for item in response.json()["items"]] == [allowed]
