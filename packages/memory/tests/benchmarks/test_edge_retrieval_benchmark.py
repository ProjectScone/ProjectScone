from __future__ import annotations

from ..paths import TESTS_ROOT

import json
from pathlib import Path
import socket

import pytest

from scone_memory.embedders.hash import HashEmbedder

FIXTURE = TESTS_ROOT / "fixtures/edge_rag/v1.json"


def test_fixture_loader_rejects_invalid_ground_truth_and_cross_scope(tmp_path):
    from scone_memory.testing.edge_retrieval_benchmark import load_fixture
    original = json.loads(FIXTURE.read_text())
    for change in ("quote", "scope", "duplicate", "reference", "unknown_field"):
        data = json.loads(json.dumps(original))
        if change == "quote":
            data["cases"][0]["required"][0]["quote"] = "invented evidence"
        elif change == "scope":
            data["cases"][1]["required"][0] = {"document_id": "foreign", "quote": "beacon routes requests to a public service."}
        elif change == "duplicate":
            data["documents"].append(data["documents"][0])
        elif change == "reference":
            data["facts"][0]["document_id"] = "missing"
        else:
            data["surprise"] = True
        path = tmp_path / f"{change}.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_fixture(path)


def test_fixture_fact_quotes_and_seed_scopes_are_validated(tmp_path):
    from scone_memory.testing.edge_retrieval_benchmark import load_fixture
    data = json.loads(FIXTURE.read_text())
    data["facts"][0]["quote"] = "plausible but not recorded"
    path = tmp_path / "bad-fact.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_fixture(path)
    data = json.loads(FIXTURE.read_text())
    data["cases"][1]["where"] = {"project": "other"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_fixture(path)


async def test_actual_sqlite_benchmark_runs_with_network_blocked(monkeypatch):
    from scone_memory.testing.edge_retrieval_benchmark import run
    def blocked(*args, **kwargs):
        raise AssertionError("benchmark attempted network access")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    report = await run(FIXTURE, [0, 3], 1)
    assert report["embedder"] == HashEmbedder(256).id
    assert len(report["results"]) == 4
    for row in report["results"]:
        assert row["repeats"] == 1
        assert row["query"] in {"Beacon checkpoint protocol", "beacon routes"}
        assert row["distractor_total"] == row["distractors_per_group"] * 3
        assert row["baseline"]["scope_leaks"] == 0
        assert row["enriched"]["scope_leaks"] == 0
        assert 0 <= row["baseline"]["quote_coverage"] <= 1
        assert row["enriched"]["quote_coverage"] >= row["baseline"]["quote_coverage"]
        assert row["baseline"]["latency_ms_median"] >= 0
        assert row["enriched"]["latency_ms_median"] >= 0
    structural = report["results"][0]
    assert structural["baseline"]["quote_coverage"] < 1
    assert structural["enriched"]["quote_coverage"] == 1
    assert structural["enriched"]["text_bytes"] > structural["baseline"]["text_bytes"]
    multihop = report["results"][1]
    assert multihop["seed_mode"] == "actual_recall_facts"
    assert multihop["explicit_seed_diagnostic"]["quote_coverage"] == 1
    assert multihop["work"]["store_calls"] <= multihop["caps"]["max_store_calls"]


async def test_missing_cached_model_fails_without_network(tmp_path, monkeypatch):
    from scone_memory.testing.edge_retrieval_benchmark import run
    def blocked(*args, **kwargs):
        raise AssertionError("cache miss attempted network access")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    with pytest.raises((ValueError, FileNotFoundError)):
        await run(FIXTURE, [0], 1, tmp_path / "missing")


async def test_invalid_run_sizes_are_rejected():
    from scone_memory.testing.edge_retrieval_benchmark import run
    with pytest.raises(ValueError):
        await run(FIXTURE, [-1], 1)


async def test_empty_existing_cache_refuses_downloads(tmp_path, monkeypatch):
    pytest.importorskip("fastembed")
    from scone_memory.testing.edge_retrieval_benchmark import run
    def blocked(*args, **kwargs):
        raise AssertionError("empty cache attempted public network access")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with pytest.raises(ValueError):
        await run(FIXTURE, [0], 1, tmp_path)


def test_valid_labels_do_not_authorize_a_foreign_seed(tmp_path):
    from scone_memory.testing.edge_retrieval_benchmark import load_fixture
    data = json.loads(FIXTURE.read_text())
    data["facts"][0]["document_id"] = "foreign"
    data["facts"][0]["quote"] = "beacon routes requests to a public service."
    path = tmp_path / "foreign-seed.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed fact is outside"):
        load_fixture(path)


def test_distinct_labels_cannot_alias_a_deduplicated_source(tmp_path):
    from scone_memory.testing.edge_retrieval_benchmark import load_fixture
    data = json.loads(FIXTURE.read_text())
    duplicate = dict(data["documents"][0], id="aliased-manual", source="different.md")
    data["documents"].append(duplicate)
    path = tmp_path / "duplicate-identity.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="deduplication identity"):
        load_fixture(path)
