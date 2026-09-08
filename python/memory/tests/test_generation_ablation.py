from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from scone_memory.realtime.events import ReplyCompleted, TextDelta


def fixture_file(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"schema_version": 1, "description": "offline evaluator fixture", "documents": [
        {"id": "manual", "content": "The cobalt probe uses 18 volts.", "space": "fixture", "source": "docs/manual.md",
         "tags": [], "where": {"project": "edge"}, "created_at": "2026-01-01"}], "facts": [], "cases": [
        {"id": "probe", "query": "What voltage does the cobalt probe use?", "mode": "structural", "space": "fixture",
         "where": {"project": "edge"}, "source_prefix": None, "seed_fact_ids": [],
         "required": [{"document_id": "manual", "quote": "The cobalt probe uses 18 volts."}],
         "split": "held_out", "answer_checks": [["18 volts", "EVALUATION_ONLY_CANARY"]]}]}))
    return path


def test_phrase_alternatives_preserve_public_quote_and_word_boundaries():
    from scone_memory.testing.generation_ablation import score_answer
    good = score_answer("It uses EIGHTEEN VOLTS.", (("18 volts", "eighteen volts"),))
    assert good["key_fact_coverage"] == 1
    assert good["checks"][0]["matched_quote"] == "EIGHTEEN VOLTS"
    assert score_answer("117 requests", (("17",),))["key_fact_coverage"] == 0


@pytest.mark.parametrize("mode,status", [("complete", "completed"), ("eof", "incomplete"), ("timeout", "timeout"), ("long", "output_limit")])
async def test_public_stream_status_bytes_and_provider_cleanup(mode, status):
    from scone_memory.testing.generation_ablation import capture_public_reply
    class Streaming:
        closed = False
        async def respond(self, messages):
            yield TextDelta("The probe uses 18 volts.")
            if mode == "timeout":
                await asyncio.sleep(1)
            if mode == "long":
                yield TextDelta("é" * 10000)
            if mode == "complete":
                yield ReplyCompleted()
        async def aclose(self):
            self.closed = True
    provider = Streaming()
    result = await capture_public_reply(provider, [{"role": "user", "content": "question"}], timeout=0.01)
    assert result["status"] == status
    assert len(result["answer_text"].encode()) <= 16000
    assert 0 <= result["first_token_ms"] <= result["total_ms"]
    assert provider.closed


async def test_actual_context_pair_does_not_put_labels_in_prompt_and_does_not_overwrite(tmp_path):
    from scone_memory.testing.generation_ablation import run_ablation
    calls = []
    class Streaming:
        async def respond(self, messages):
            calls.append(messages)
            yield TextDelta("The probe uses 18 volts.")
            yield ReplyCompleted()
        async def aclose(self):
            pass
    output = tmp_path / "report.json"
    report = await run_ablation(fixture_file(tmp_path), output=output, repeats=2, model_factory=Streaming)
    provenance = report["code_provenance"]
    assert provenance["kind"] == "disk_snapshot"
    assert provenance["capture_stage"] == "evaluator_import_before_local_imports"
    assert provenance["loaded_code_identity_verified"] is False
    assert provenance["captured_at_utc"].endswith("+00:00")
    package = Path(__file__).parent.parent / "src" / "scone_memory"
    expected = {"realtime/context.py", "retrieval/path_evidence.py", "retrieval/adaptive.py", "providers/llm.py",
                "providers/evidence_assessor.py", "retrieval/evidence_groups.py", "testing/generation_ablation.py"}
    assert set(provenance["files"]) == expected
    for relative, digest in provenance["files"].items():
        assert digest == hashlib.sha256((package / relative).read_bytes()).hexdigest()
    assert [row["structured_paths"] for row in report["results"]] == [False, True, True, False]
    assert [row["variant"] for row in report["results"]] == ["baseline", "candidate", "candidate", "baseline"]
    assert all(row["adaptive_retrieval"] is False for row in report["results"])
    assert all("EVALUATION_ONLY_CANARY" not in json.dumps(messages) for messages in calls)
    assert all(row["prompt_bytes"] <= 16000 for row in report["results"])
    assert all(row["evidence_coverage"] == 1 for row in report["results"])
    assert all(row["key_fact_coverage"] == 1 for row in report["results"])
    assert all(row["split"] == "held_out" for row in report["results"])
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        await run_ablation(fixture_file(tmp_path), output=output, model_factory=Streaming)
    assert output.read_bytes() == saved


def test_empty_answer_alternatives_are_rejected(tmp_path):
    from scone_memory.testing.generation_ablation import load_generation_fixture
    path = fixture_file(tmp_path)
    data = json.loads(path.read_text())
    data["cases"][0]["answer_checks"] = [[" "]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_generation_fixture(path)


@pytest.mark.parametrize("error,status", [("chat stream did not complete: 'length'", "truncated"),
                                         ("chat stream did not finish normally", "failed")])
async def test_provider_length_failure_retains_partial_text_without_completion(error, status):
    from scone_memory.providers.llm import ChatError
    from scone_memory.testing.generation_ablation import capture_public_reply
    class Truncated:
        closed = False
        async def respond(self, messages):
            yield TextDelta("A partial public answer")
            raise ChatError(error)
        async def aclose(self):
            self.closed = True
    provider = Truncated()
    result = await capture_public_reply(provider, [{"role": "user", "content": "query"}], timeout=1)
    assert result["status"] == status and not result["completed"]
    assert result["answer_text"] == "A partial public answer" and provider.closed


async def test_provider_is_closed_when_respond_raises_before_returning_an_iterator():
    from scone_memory.testing.generation_ablation import capture_public_reply
    class FailsEarly:
        closed = False
        def respond(self, messages):
            raise ValueError("adapter cannot start")
        async def aclose(self):
            self.closed = True
    provider = FailsEarly()
    result = await capture_public_reply(provider, [{"role": "user", "content": "query"}], timeout=1)
    assert result["status"] == "failed" and result["error_type"] == "ValueError"
    assert provider.closed


async def test_empty_completion_is_not_success():
    from scone_memory.testing.generation_ablation import capture_public_reply
    class Empty:
        async def respond(self, messages):
            yield TextDelta("  \n")
            yield ReplyCompleted()
        async def aclose(self):
            pass
    result = await capture_public_reply(Empty(), [], timeout=1)
    assert result["status"] == "empty" and result["completed"] is False


async def test_actual_three_fact_path_has_source_coverage_and_partial_answer_has_no_success_credit(tmp_path):
    from scone_memory.testing.generation_ablation import run_ablation
    data = json.loads((Path(__file__).parent / "fixtures/generation/v1.json").read_text())
    data["cases"] = [case for case in data["cases"] if case["id"] == "multihop-journal"]
    data["cases"][0]["answer_checks"][0].append("EVALUATION_ONLY_CANARY")
    fixture = tmp_path / "chain.json"
    fixture.write_text(json.dumps(data))
    calls = []
    class Partial:
        async def respond(self, messages):
            calls.append(messages)
            yield TextDelta("It uses an encrypted local journal.")
        async def aclose(self):
            pass
    report = await run_ablation(fixture, output=tmp_path / "chain-report.json", model_factory=Partial)
    candidate = next(row for row in report["results"] if row["structured_paths"])
    assert candidate["path_count"] > 0
    assert candidate["evidence_coverage"] == 1
    assert set(candidate["included_source_ids"]) >= {"route-a", "route-b", "route-c"}
    assert candidate["scope_leaks"] == candidate["invalid_provenance"] == 0
    assert candidate["key_fact_coverage"] == 1
    assert candidate["successful_key_fact_coverage"] == 0
    assert all("EVALUATION_ONLY_CANARY" not in json.dumps(messages) for messages in calls)
    assert report["ordered_quotes"] is False
    assert all(row["ordered_quotes"] is False for row in report["results"])
    quoted = await run_ablation(fixture, output=tmp_path / "quoted-report.json", model_factory=Partial, ordered_quotes=True)
    assert quoted["ordered_quotes"] is True
    assert [row["ordered_quotes"] for row in quoted["results"]] == [False, True]
    assert quoted["results"][0]["prompt_sha256"] == report["results"][0]["prompt_sha256"]
    for index, enabled in ((1, False), (3, True)):
        blocks = [json.loads(message["content"].partition("\n")[2]) for message in calls[index]
                  if message["content"].startswith("Scone retrieved source material:")]
        paths = [path for block in blocks for path in block.get("paths", [])]
        assert paths
        assert all(("ordered_evidence" in path) is enabled for path in paths)
    assert all("EVALUATION_ONLY_CANARY" not in json.dumps(messages) for messages in calls)


async def test_ordered_quotes_rejects_non_boolean(tmp_path):
    from scone_memory.testing.generation_ablation import run_ablation
    with pytest.raises(ValueError, match="ordered_quotes"):
        await run_ablation(fixture_file(tmp_path), output=tmp_path / "invalid.json", ordered_quotes=1)
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("group_relations", [False, True])
@pytest.mark.parametrize("baseline_paths", [False, True])
@pytest.mark.parametrize("assessment_fails", [False, True])
async def test_adaptive_pair_uses_real_context_and_keeps_failed_assessments_visible(tmp_path, monkeypatch,
                                                                                   baseline_paths, assessment_fails, group_relations):
    from scone_memory.retrieval.adaptive import EvidenceDecision
    from scone_memory.testing import generation_ablation
    expected_group_relations = group_relations
    assessment_inputs = []
    generation_inputs = []
    class Assessor:
        def __init__(self, endpoint, model, *, timeout, group_relations, max_evidence_bytes):
            assert group_relations is expected_group_relations
            assert max_evidence_bytes == 16000
            assert (endpoint, model, timeout) == ("http://127.0.0.1:1234/v1", "local-assessor", 20)
        async def assess(self, question, candidates):
            assessment_inputs.append((question, candidates))
            if assessment_fails:
                raise RuntimeError("private provider failure")
            return EvidenceDecision(status="sufficient", selected_ids=tuple(item.id for item in candidates))
    class Streaming:
        async def respond(self, messages):
            generation_inputs.append(messages)
            yield TextDelta("The probe uses 18 volts.")
            yield ReplyCompleted()
        async def aclose(self):
            pass
    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceAssessor", Assessor, raising=False)
    report = await generation_ablation.run_ablation(fixture_file(tmp_path), output=tmp_path / "adaptive.json",
        endpoint="http://127.0.0.1:1234/v1", model_factory=Streaming, repeats=2, ordered_quotes=True,
        adaptive_model="local-assessor", adaptive_timeout=20, adaptive_rounds=2, baseline_paths=baseline_paths, group_relations=group_relations)
    assert report["state"] == "completed"
    assert report["group_relations"] is group_relations
    assert report["assessment_max_evidence_bytes"] == 16000
    assert report["adaptive_model"] == "local-assessor"
    assert report["adaptive_timeout_seconds"] == 20
    assert report["assessment_timeout_seconds"] == 20
    assert report["adaptive_rounds"] == 2
    assert report["baseline_paths"] is baseline_paths
    rows = report["results"]
    assert [row["variant"] for row in rows] == ["baseline", "candidate", "candidate", "baseline"]
    assert [row["group_relations"] for row in rows] == [False, group_relations, group_relations, False]
    assert [row["adaptive_retrieval"] for row in rows] == [False, True, True, False]
    assert [row["structured_paths"] for row in rows] == [baseline_paths, True, True, baseline_paths]
    assert [row["ordered_quotes"] for row in rows] == [False, True, True, False]
    assert len(assessment_inputs) == 2
    assert all(row["status"] == "completed" for row in rows)
    for row in rows:
        if row["variant"] == "candidate":
            assert row["context_receipt"]["adaptive_status"] == ("uncertain" if assessment_fails else "sufficient")
            assert row["evidence_coverage"] == (0 if assessment_fails else 1)
        else:
            assert "adaptive_status" not in row["context_receipt"]
            assert row["evidence_coverage"] == 1
        assert row["scope_leaks"] == row["invalid_provenance"] == 0
    assert all("EVALUATION_ONLY_CANARY" not in json.dumps(messages) for messages in generation_inputs)
    assert all("EVALUATION_ONLY_CANARY" not in question + "".join(item.text for item in candidates)
               for question, candidates in assessment_inputs)
    assert "private provider failure" not in json.dumps(report)


@pytest.mark.parametrize("options,match", [
    ({"baseline_paths": 1}, "baseline_paths"),
    ({"baseline_paths": True}, "adaptive_model"),
    ({"group_relations": True}, "adaptive_model"),
    ({"group_relations": 1}, "group_relations"),
    ({"adaptive_model": "local-assessor"}, "endpoint"),
    ({"adaptive_model": " "}, "adaptive_model"),
    ({"adaptive_model": True}, "adaptive_model"),
    ({"adaptive_timeout": True}, "adaptive_timeout"),
    ({"adaptive_timeout": "30"}, "adaptive_timeout"),
    ({"adaptive_timeout": float("nan")}, "adaptive_timeout"),
    ({"adaptive_timeout": 0.5}, "adaptive_timeout"),
    ({"adaptive_timeout": 181}, "adaptive_timeout"),
    ({"adaptive_rounds": True}, "adaptive_rounds"),
    ({"adaptive_rounds": 1.5}, "adaptive_rounds"),
    ({"adaptive_rounds": 0}, "adaptive_rounds"),
    ({"adaptive_rounds": 5}, "adaptive_rounds"),
])
async def test_adaptive_options_validate_before_output_creation(tmp_path, options, match):
    from scone_memory.testing.generation_ablation import run_ablation
    output = tmp_path / "new-directory" / "invalid.json"
    with pytest.raises(ValueError, match=match):
        await run_ablation(fixture_file(tmp_path), output=output, **options)
    assert not output.parent.exists()
