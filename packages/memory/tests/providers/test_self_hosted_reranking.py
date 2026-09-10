"""Offline reranking diagnostics use real scoped recall and no chat provider."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import pytest

from scone_memory import HashEmbedder
from scone_memory.retrieval.reranking import RerankScore
from scone_memory.testing import self_hosted_reranking as diagnostic


def test_options_preserve_default_and_accept_explicit_cross_encoder(tmp_path):
    arguments = ["--model", "vendor/model", "--embedding-cache", str(tmp_path), "--output", str(tmp_path / "report.json")]
    default = diagnostic.parse_options(arguments)
    assert default.cross_encoder_dir is None
    assert diagnostic.Options(default.endpoint, default.model, default.embedding_cache, default.output) == default
    offline = diagnostic.parse_options([*arguments, "--cross-encoder-dir", str(tmp_path)])
    assert offline.cross_encoder_dir == tmp_path.resolve()
    assert offline.model == "vendor/model"


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_cross_encoder_directory_must_exist(tmp_path, path_kind):
    model_dir = tmp_path / "model"
    if path_kind == "file":
        model_dir.write_text("not a directory")
    with pytest.raises(SystemExit):
        diagnostic.parse_options(["--model", "vendor/model", "--embedding-cache", str(tmp_path),
            "--cross-encoder-dir", str(model_dir), "--output", str(tmp_path / "report.json")])


@pytest.mark.parametrize("failure", ["existing_output", "output_parent_file", "missing_cache", "missing_model"])
async def test_evaluate_validates_paths_before_initializing_models(tmp_path, monkeypatch, failure):
    def forbidden(*args, **kwargs):
        raise AssertionError("model initialization occurred before path validation")

    monkeypatch.setattr(diagnostic, "LocalEmbedder", forbidden)
    monkeypatch.setattr(diagnostic, "SelfHostedLLMReranker", forbidden)
    monkeypatch.setattr(diagnostic, "OfflineCrossEncoderReranker", forbidden)
    output = tmp_path / "report.json"
    cache = model_dir = tmp_path
    if failure == "existing_output":
        output.write_text("earlier diagnostic")
    elif failure == "output_parent_file":
        output.write_text("parent is a file")
        output = output / "report.json"
    elif failure == "missing_cache":
        cache = tmp_path / "missing"
    else:
        model_dir = tmp_path / "missing"
    with pytest.raises((ValueError, FileExistsError, NotADirectoryError)):
        await diagnostic.evaluate(diagnostic.Options("http://localhost:1234/v1", "vendor/model", cache, output, model_dir))
    if failure == "existing_output":
        assert output.read_text() == "earlier diagnostic"


async def test_offline_trial_uses_real_recall_without_chat_or_gold_scoring(tmp_path, monkeypatch, capsys):
    calls = []

    class Offline:
        def __init__(self, model_dir, *, model_name):
            assert model_dir == tmp_path
            assert model_name == "vendor/model"
            self.calls = 0
            self.model_identity = {"model_name": model_name, "sha256": {"model.onnx": "a" * 64}}

        async def rerank(self, query, candidates):
            self.calls += 1
            calls.append((query, candidates))
            # Deliberately neutral ranking: no case targets or source IDs used.
            return [RerankScore(candidate.chunk_id, -float(index)) for index, candidate in enumerate(reversed(candidates), 1)]

    def forbidden(*args, **kwargs):
        raise AssertionError("offline trial initialized chat or accessed the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(diagnostic, "SelfHostedLLMReranker", forbidden)
    monkeypatch.setattr(diagnostic, "OfflineCrossEncoderReranker", Offline)
    monkeypatch.setattr(diagnostic, "LocalEmbedder", lambda **kwargs: HashEmbedder())
    output = tmp_path / "offline.json"
    await diagnostic.evaluate(diagnostic.Options("http://localhost:1234/v1", "vendor/model", tmp_path, output, tmp_path))
    report = json.loads(output.read_text())
    assert report["state"] == "completed"
    assert report["reranker_backend"] == "offline_cross_encoder"
    assert report["endpoint"] is None
    assert report["model"] == "vendor/model"
    assert report["reranker_calls"] == report["model_calls"] == len(calls) == 3
    assert report["chat_model_calls"] == 0
    assert report["model_identity"]["sha256"] == {"model.onnx": "a" * 64}
    assert len(report["rows"]) == 9
    assert {row["variant"] for row in report["rows"]} == {
        "default_fusion", "expanded_fusion", "expanded_offline_cross_encoder"}
    ranked = [row for row in report["rows"] if row["variant"] == "expanded_offline_cross_encoder"]
    assert all(row["trace"]["status"] == "applied" for row in ranked)
    assert all(row["selected_rerank_scores"][0] < 0 for row in ranked)
    assert all(candidate.source.startswith(("design/", "policy/")) for _, candidates in calls for candidate in candidates)
    provenance = report["code_provenance"]
    package = Path(diagnostic.__file__).resolve().parent.parent
    assert "providers/offline_reranker.py" in provenance["files"]
    assert "retrieval/reranking.py" in provenance["files"]
    assert provenance["loaded_code_identity_verified"] is False
    for name, digest in provenance["files"].items():
        assert digest == hashlib.sha256((package / name).read_bytes()).hexdigest()
    assert any("confidence" in note for note in report["limitations"])
    assert len(capsys.readouterr().out.splitlines()) == 9
