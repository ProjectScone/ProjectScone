from __future__ import annotations

import json
from pathlib import Path
import socket

import pytest

FIXTURE = Path(__file__).parent / "fixtures/edge_rag/qdrant_v1.json"


def test_varied_fixture_validates_ground_truth_and_temporal_scope(tmp_path):
    from scone_memory.testing.qdrant_comparison import load_comparison_fixture
    fixture = load_comparison_fixture(FIXTURE)
    assert len(fixture.cases) == 8
    assert {case.id for case in fixture.cases} >= {"table-limits", "unicode-owner", "unresolved-conflict", "multihop-journal"}
    data = json.loads(FIXTURE.read_text())
    data["cases"][3]["as_of"] = "2025-01-01T00:00:00Z"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="time scope"):
        load_comparison_fixture(path)


async def test_actual_backends_compare_offline_and_cleanup_only_owned_collection(monkeypatch):
    pytest.importorskip("qdrant_client")
    from scone_memory.testing.qdrant_comparison import run_comparison
    def blocked(*args, **kwargs):
        raise AssertionError("embedded comparison attempted network access")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    report = await run_comparison(FIXTURE, sizes=[0], repeats=1, k=3)
    assert report["qdrant_mode"] == "embedded-client; not server performance"
    assert report["collection_cleanup"] == "owned collections deleted"
    assert len(report["results"]) == 16
    assert len(report["parity"]) == 8
    assert all(row["scope_leaks"] == 0 for row in report["results"])
    assert all(0 <= row["evidence_recall_at_k"] <= 1 for row in report["results"])
    assert all(0 <= row["mrr"] <= 1 for row in report["results"])
    assert all(row["vector_latency_ms_p50"] >= 0 for row in report["results"])


def test_citation_validation_requires_prompt_source_and_literal_quote():
    from scone_memory.testing.qdrant_comparison import ContextEvidence, score_generation
    evidence = [ContextEvidence(document_id="probe", text="The probe uses 18 volts.")]
    good = score_generation('{"answer":"18 volts","citations":[{"document_id":"probe","quote":"18 volts"}]}', evidence, ("18 volts",), {"probe"})
    assert good["citation_precision"] == 1 and good["exact_answer_fact_coverage"] == 1
    bad = score_generation('{"answer":"99 volts","citations":[{"document_id":"foreign","quote":"99 volts"}]}', evidence, ("18 volts",), {"probe"})
    assert bad["citation_precision"] == 0 and bad["exact_answer_fact_coverage"] == 0
    numeric = score_generation('{"answer":"117 requests","citations":[]}', evidence, ("17",), {"probe"})
    assert numeric["exact_answer_fact_coverage"] == 0


async def test_generation_timeout_is_explicit_and_prompt_is_bounded():
    import asyncio
    from scone_memory.testing.qdrant_comparison import ContextEvidence, generate_answer
    class NeverAnswers:
        async def complete(self, system, user):
            await asyncio.sleep(10)
            return "unused"
    result = await generate_answer(NeverAnswers(), "query", [ContextEvidence(document_id="one", text="x" * 20000)],
                                   ("answer",), {"one"}, timeout=0.01, max_prompt_bytes=2000)
    assert result["status"] == "timeout"
    assert result["prompt_bytes"] <= 2000
    assert result["context_omitted"] == 1


def test_real_but_irrelevant_quotes_do_not_count_as_required_source_precision():
    from scone_memory.testing.qdrant_comparison import ContextEvidence, score_generation
    evidence = [ContextEvidence(document_id="probe", text="The probe uses 18 volts."),
                ContextEvidence(document_id="unrelated", text="The gate is open.")]
    answer = json.dumps({"answer": "18 volts", "citations": [
        {"document_id": "probe", "quote": "18 volts"},
        {"document_id": "unrelated", "quote": "The gate is open."}]})
    result = score_generation(answer, evidence, ("18 volts",), {"probe"})
    assert result["citation_validity"] == 1.0
    assert result["required_source_citation_precision"] == 0.5
    assert result["required_source_citation_recall"] == 1.0
    failed = score_generation("invalid JSON", evidence, ("18 volts",), {"probe"})
    assert failed["citation_validity"] == 0.0
    assert failed["required_source_citation_precision"] == 0.0


async def test_existing_qdrant_collection_is_untouched_and_checkpoints_persist(tmp_path, monkeypatch):
    from qdrant_client import AsyncQdrantClient, models
    from scone_memory.backends.qdrant import QdrantVectorIndex
    from scone_memory.testing import qdrant_comparison
    client = AsyncQdrantClient(":memory:")
    await client.create_collection("retained_control", vectors_config=models.VectorParams(size=256, distance=models.Distance.COSINE))
    original_close = client.close
    async def leave_open():
        return None
    monkeypatch.setattr(client, "close", leave_open)
    def isolated_index(url, collection):
        assert collection.startswith("scone_bench_")
        return QdrantVectorIndex(url, collection, client=client)
    monkeypatch.setattr(qdrant_comparison, "QdrantVectorIndex", isolated_index)
    output = tmp_path / "checkpoint.json"
    try:
        report = await qdrant_comparison.run_comparison(FIXTURE, sizes=[0], repeats=1, k=1, checkpoint_path=output)
        assert await client.collection_exists("retained_control")
        assert all([not await client.collection_exists(name) for name in report["owned_collections"]])
        saved = json.loads(output.read_text())
        assert len(saved["results"]) == 16
        assert saved["environment"]["qdrant_client_version"]
        assert saved["qdrant_collections"][0]["points_count"] >= 14
    finally:
        await original_close()
