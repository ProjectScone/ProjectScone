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
                "providers/evidence_assessor.py", "retrieval/evidence_groups.py", "retrieval/evidence_blend.py", "retrieval/adaptive_graph.py", "retrieval/multihop.py",
                "realtime/answer_review.py", "realtime/review_evidence.py", "realtime/evidence_answer.py", "realtime/text.py",
                "providers/answer_reviewer.py", "providers/evidence_selector.py", "testing/generation_ablation.py"}
    assert set(provenance["files"]) == expected
    for relative, digest in provenance["files"].items():
        assert digest == hashlib.sha256((package / relative).read_bytes()).hexdigest()
    assert [row["structured_paths"] for row in report["results"]] == [False, True, True, False]
    assert [row["variant"] for row in report["results"]] == ["baseline", "candidate", "candidate", "baseline"]
    assert all(row["adaptive_retrieval"] is False for row in report["results"])
    assert report["candidate_answer_mode"] == "model_generation"
    assert all(row["answer_mode"] == "model_generation" for row in report["results"])
    assert report["adaptive_empty_selection_policy"] is None
    assert report["adaptive_evidence_policy"] is None
    assert all(row["adaptive_evidence_policy"] is None for row in report["results"])
    assert all(row["adaptive_empty_selection_policy"] is None for row in report["results"])
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


@pytest.mark.parametrize("expand_relations", [False, True])
@pytest.mark.parametrize("failure_policy", ["empty", "retain_verified"])
@pytest.mark.parametrize("group_relations", [False, True])
@pytest.mark.parametrize("baseline_paths", [False, True])
@pytest.mark.parametrize("assessment_fails", [False, True])
async def test_adaptive_pair_uses_real_context_and_keeps_failed_assessments_visible(tmp_path, monkeypatch,
                                                                                   baseline_paths, assessment_fails, group_relations, failure_policy, expand_relations):
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
        adaptive_model="local-assessor", adaptive_timeout=20, adaptive_rounds=2, baseline_paths=baseline_paths, group_relations=group_relations, adaptive_failure_policy=failure_policy, expand_relations=expand_relations)
    assert report["state"] == "completed"
    assert report["adaptive_failure_policy"] == failure_policy
    assert report["adaptive_empty_selection_policy"] == "retain_verified"
    assert report["adaptive_evidence_policy"] == "model_selected"
    assert report["expand_relations"] is expand_relations
    assert (report["graph_limits"] is not None) is expand_relations
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
    assert [row["expand_relations"] for row in rows] == [False, expand_relations, expand_relations, False]
    assert [row["structured_paths"] for row in rows] == [baseline_paths, True, True, baseline_paths]
    assert [row["ordered_quotes"] for row in rows] == [False, True, True, False]
    assert len(assessment_inputs) == 2
    assert all(row["status"] == "completed" for row in rows)
    for row in rows:
        if row["variant"] == "candidate":
            assert row["context_receipt"]["adaptive_status"] == ("uncertain" if assessment_fails else "sufficient")
            retained_fallback = assessment_fails and failure_policy == "retain_verified"
            assert row["adaptive_failure_policy"] == failure_policy
            assert row["adaptive_empty_selection_policy"] == "retain_verified"
            assert row["adaptive_evidence_policy"] == "model_selected"
            assert row["evidence_coverage"] == (0 if assessment_fails and not retained_fallback else 1)
            assert row["context_receipt"]["adaptive_evidence_basis"] == (
                "verified_candidates" if retained_fallback else "none" if assessment_fails else "assessed_selection")
            assert row["context_receipt"]["adaptive_fallback_status"] == ("retained" if retained_fallback else "not_used")
            assert bool(row["context_receipt"]["adaptive_errors"]) is assessment_fails
        else:
            assert row["adaptive_failure_policy"] is None
            assert row["adaptive_empty_selection_policy"] is None
            assert row["adaptive_evidence_policy"] is None
            assert "adaptive_status" not in row["context_receipt"]
            assert row["evidence_coverage"] == 1
        assert row["scope_leaks"] == row["invalid_provenance"] == 0
    assert all("EVALUATION_ONLY_CANARY" not in json.dumps(messages) for messages in generation_inputs)
    assert all("EVALUATION_ONLY_CANARY" not in question + "".join(item.text for item in candidates)
               for question, candidates in assessment_inputs)
    assert "private provider failure" not in json.dumps(report)


@pytest.mark.parametrize("options,match", [
    ({"expand_relations": 1}, "expand_relations"),
    ({"expand_relations": True}, "adaptive_model"),
    ({"adaptive_failure_policy": "anything"}, "adaptive_failure_policy"),
    ({"adaptive_failure_policy": True}, "adaptive_failure_policy"),
    ({"adaptive_empty_selection_policy": "anything"}, "adaptive_empty_selection_policy"),
    ({"adaptive_empty_selection_policy": True}, "adaptive_empty_selection_policy"),
    ({"adaptive_empty_selection_policy": None}, "adaptive_empty_selection_policy"),
    ({"adaptive_evidence_policy": "anything"}, "adaptive_evidence_policy"),
    ({"adaptive_evidence_policy": True}, "adaptive_evidence_policy"),
    ({"adaptive_evidence_policy": None}, "adaptive_evidence_policy"),
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


@pytest.mark.parametrize("assessment_status", ["insufficient", "uncertain"])
async def test_empty_selection_policies_compare_actual_partial_route(tmp_path, monkeypatch, assessment_status):
    from scone_memory.retrieval.adaptive import EvidenceDecision
    from scone_memory.testing import generation_ablation
    fixture = json.loads((Path(__file__).parent / "fixtures/generation/v1.json").read_text())
    fixture["cases"] = [case for case in fixture["cases"] if case["id"] == "missing-destination"]
    case = fixture["cases"][0]
    source = tmp_path / "partial-route.json"
    source.write_text(json.dumps(fixture))
    assessment_inputs = []
    generation_inputs = []

    class EmptyAssessor:
        def __init__(self, *args, **kwargs):
            pass

        async def assess(self, question, candidates):
            assessment_inputs.append((question, candidates))
            return EvidenceDecision(status=assessment_status, selected_ids=())

    class Streaming:
        async def respond(self, messages):
            generation_inputs.append(messages)
            yield TextDelta("The final storage medium is not specified.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceAssessor", EmptyAssessor)
    reports = {}
    for policy in ("retain_verified", "empty"):
        reports[policy] = await generation_ablation.run_ablation(source, output=tmp_path / f"{policy}.json",
            endpoint="http://127.0.0.1:1234/v1", adaptive_model="empty-assessor", model_factory=Streaming,
            baseline_paths=True, group_relations=True, expand_relations=True,
            adaptive_empty_selection_policy=policy)
        baseline, candidate = reports[policy]["results"]
        receipt = candidate["context_receipt"]
        retained = policy == "retain_verified"
        assert reports[policy]["adaptive_empty_selection_policy"] == policy
        assert baseline["adaptive_empty_selection_policy"] is None
        assert candidate["adaptive_empty_selection_policy"] == policy
        assert receipt["adaptive_status"] == assessment_status
        assert receipt["adaptive_evidence_basis"] == ("unselected_candidates" if retained else "none")
        assert receipt["adaptive_fallback_status"] == "not_used"
        assert receipt["adaptive_errors"] == []
        assert candidate["evidence_coverage"] == (1 if retained else 0)
        assert candidate["scope_leaks"] == candidate["invalid_provenance"] == 0
        assert candidate["status"] == "completed"
    assert reports["retain_verified"]["results"][0]["prompt_sha256"] == reports["empty"]["results"][0]["prompt_sha256"]
    assert len(assessment_inputs) == 2
    for question, candidates in assessment_inputs:
        assert question == case["query"]
        assert any(item.text == "spruce forwards its records to unresolved-target." for item in candidates)
        assert all("public cloud archive" not in item.text for item in candidates)
    assert "spruce forwards its records to unresolved-target." in json.dumps(generation_inputs[1])
    assert "spruce forwards its records to unresolved-target." not in json.dumps(generation_inputs[3])
    for messages in generation_inputs:
        serialized = json.dumps(messages)
        assert "public cloud archive" not in serialized
        assert '"answer_checks"' not in serialized
        assert json.dumps(case["answer_checks"]) not in serialized
        assert '"required"' not in serialized


async def test_original_evidence_policy_preserves_partial_route_after_wrong_nonempty_selection(tmp_path, monkeypatch):
    from scone_memory.retrieval.adaptive import EvidenceDecision
    from scone_memory.testing import generation_ablation
    fixture = json.loads((Path(__file__).parent / "fixtures/generation/v1.json").read_text())
    fixture["cases"] = [case for case in fixture["cases"] if case["id"] == "missing-destination"]
    case = fixture["cases"][0]
    source = tmp_path / "wrong-followup.json"
    source.write_text(json.dumps(fixture))
    original_quote = "spruce forwards its records to unresolved-target."
    unrelated_quote = "talon uses a rotating disk archive."
    assessment_inputs = []
    generation_inputs = []

    class WrongAssessor:
        def __init__(self, *args, **kwargs):
            self.round = 0

        async def assess(self, question, candidates):
            assessment_inputs.append((question, candidates))
            self.round += 1
            if self.round == 1:
                assert any(item.text == original_quote for item in candidates)
                return EvidenceDecision(status="insufficient", selected_ids=(),
                    followup_queries=("What storage does talon use?",))
            selected = tuple(item.id for item in candidates if item.id.startswith("chunk:") and item.text == unrelated_quote)
            assert selected
            return EvidenceDecision(status="sufficient", selected_ids=selected)

    class Streaming:
        async def respond(self, messages):
            generation_inputs.append(messages)
            yield TextDelta("Scripted generation; answer accuracy is not measured by this regression.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceAssessor", WrongAssessor)
    reports = {}
    for policy in ("model_selected", "original_and_selected"):
        reports[policy] = await generation_ablation.run_ablation(source, output=tmp_path / f"{policy}.json",
            endpoint="http://127.0.0.1:1234/v1", adaptive_model="wrong-assessor", model_factory=Streaming,
            baseline_paths=True, group_relations=True, expand_relations=True, adaptive_evidence_policy=policy)
        baseline, candidate = reports[policy]["results"]
        receipt = candidate["context_receipt"]
        assert reports[policy]["adaptive_evidence_policy"] == policy
        assert baseline["adaptive_evidence_policy"] is None
        assert candidate["adaptive_evidence_policy"] == policy
        assert receipt["adaptive_status"] == "sufficient"
        assert receipt["adaptive_evidence_basis"] == ("original_and_selected" if policy == "original_and_selected" else "assessed_selection")
        assert receipt["adaptive_round_count"] == 2
        assert receipt["adaptive_fallback_status"] == "not_used"
        assert receipt["adaptive_errors"] == []
        assert candidate["evidence_coverage"] == (1 if policy == "original_and_selected" else 0)
        assert candidate["scope_leaks"] == candidate["invalid_provenance"] == 0
    assert reports["model_selected"]["results"][0]["prompt_sha256"] == reports["original_and_selected"]["results"][0]["prompt_sha256"]
    assert original_quote not in json.dumps(generation_inputs[1])
    assert original_quote in json.dumps(generation_inputs[3])
    assert all(unrelated_quote in json.dumps(generation_inputs[index]) for index in (1, 3))
    assert len(assessment_inputs) == 4
    assert all(question == case["query"] for question, _ in assessment_inputs)
    assert all("public cloud archive" not in item.text for _, candidates in assessment_inputs for item in candidates)
    for messages in generation_inputs:
        serialized = json.dumps(messages)
        assert "public cloud archive" not in serialized
        assert '"answer_checks"' not in serialized
        assert '"required"' not in serialized


@pytest.mark.parametrize("option,policy", [("empty_selection", "retain_verified"), ("empty_selection", "empty"),
                                         ("evidence", "model_selected"), ("evidence", "original_and_selected")])
def test_cli_forwards_adaptive_policy(tmp_path, monkeypatch, option, policy):
    from scone_memory.testing import generation_ablation
    captured = {}

    async def fake_run(fixture, **kwargs):
        captured.update(kwargs)
        return {"state": "completed"}

    monkeypatch.setattr(generation_ablation, "run_ablation", fake_run)
    monkeypatch.setattr("sys.argv", ["generation_ablation", "--fixture", str(tmp_path / "fixture.json"),
        "--output", str(tmp_path / "report.json"), "--model", "generator", "--endpoint", "http://localhost:1234/v1",
        "--adaptive-model", "assessor", f"--adaptive-{option.replace('_', '-')}-policy", policy])
    generation_ablation.main()
    assert captured[f"adaptive_{option}_policy"] == policy


@pytest.mark.parametrize("expand_relations", [False, True])
async def test_graph_option_delivers_missing_bridge_to_real_generation_context(tmp_path, monkeypatch, expand_relations):
    from scone_memory.retrieval.adaptive import EvidenceAssessmentError
    from scone_memory.testing import generation_ablation
    fixture = json.loads((Path(__file__).parent / "fixtures/generation/v1.json").read_text())
    fixture["cases"] = [case for case in fixture["cases"] if case["id"] == "multihop-journal"]
    source = tmp_path / "bridge.json"
    source.write_text(json.dumps(fixture))
    calls = []

    class FailingAssessor:
        def __init__(self, *args, **kwargs):
            pass

        async def assess(self, question, candidates):
            raise EvidenceAssessmentError("assessment_provider_failed")

    class Streaming:
        async def respond(self, messages):
            calls.append(messages)
            yield TextDelta("The journal is recorded in the supplied source.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceAssessor", FailingAssessor)
    report = await generation_ablation.run_ablation(source, output=tmp_path / "report.json",
        endpoint="http://127.0.0.1:1234/v1", adaptive_model="failing-assessor", model_factory=Streaming,
        baseline_paths=True, group_relations=True, expand_relations=expand_relations)
    candidate = report["results"][1]
    receipt = candidate["context_receipt"]
    assert receipt["adaptive_status"] == "uncertain"
    assert receipt["adaptive_fallback_status"] == "retained"
    assert receipt["adaptive_errors"] == ["assessment_provider_failed"]
    assert candidate["path_count"] == (1 if expand_relations else 0)
    data = next(json.loads(message["content"].split("\n", 1)[1]) for message in calls[1]
                if message["content"].startswith("Scone retrieved source material:"))
    if expand_relations:
        assert candidate["evidence_coverage"] == 1
        assert receipt["adaptive_graph_expansions"][0]["added_count"] >= 1
        assert any(len(path["fact_ids"]) == 3 for path in data["paths"])
    else:
        assert not data.get("paths")


@pytest.mark.parametrize("policy", ["report", "require_supported"])
async def test_answer_review_uses_delivered_evidence_and_preserves_public_draft(tmp_path, monkeypatch, policy):
    from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision
    from scone_memory.testing import generation_ablation
    reviews = []

    class Reviewer:
        def __init__(self, endpoint, model, **options):
            assert model == "reviewer"

        async def review(self, question, answer, evidence, evidence_ids):
            reviews.append((question, answer, evidence, evidence_ids))
            if answer == "The probe uses 12 volts.":
                return AnswerReviewDecision(status="needs_revision", issues=(AnswerIssue(code="contradiction",
                    answer_quote="12 volts", evidence_ids=(evidence_ids[0],)),), revised_answer="The probe uses 18 volts.")
            return AnswerReviewDecision(status="supported")

    class Streaming:
        async def respond(self, messages):
            yield TextDelta("The probe uses 12 volts.")
            yield ReplyCompleted()
        async def aclose(self):
            pass

    monkeypatch.setattr(generation_ablation, "SelfHostedAnswerReviewer", Reviewer)
    report = await generation_ablation.run_ablation(fixture_file(tmp_path), output=tmp_path / "review.json",
        endpoint="http://127.0.0.1:1234/v1", model_factory=Streaming, review_model="reviewer", review_policy=policy)
    baseline, candidate = report["results"]
    assert baseline["answer_text"] == "The probe uses 12 volts."
    assert "answer_review" not in baseline
    assert candidate["draft_answer_text"] == baseline["answer_text"]
    assert candidate["answer_text"] == "The probe uses 18 volts."
    assert candidate["answer_review"]["status"] == "supported"
    assert candidate["answer_review"]["revised"] is True
    assert candidate["answer_review"]["source_status"] == "retained"
    assert candidate["completed"] is True
    assert candidate["output_bytes"] == len(candidate["answer_text"].encode())
    assert candidate["review_ms"] >= 0
    assert len(reviews) == 2 and "18 volts" in reviews[0][2]
    assert all("EVALUATION_ONLY_CANARY" not in question + evidence for question, _, evidence, _ in reviews)
    assert report["review_model"] == "reviewer" and report["review_policy"] == policy


@pytest.mark.parametrize("options,match", [
    ({"review_model": " "}, "review_model"),
    ({"review_model": "reviewer"}, "endpoint"),
    ({"review_timeout": False}, "review_timeout"),
    ({"review_timeout": 0}, "review_timeout"),
    ({"review_timeout": float("inf")}, "review_timeout"),
    ({"review_policy": "anything"}, "review_policy"),
])
async def test_review_options_fail_before_output(tmp_path, options, match):
    from scone_memory.testing.generation_ablation import run_ablation
    with pytest.raises(ValueError, match=match):
        await run_ablation(fixture_file(tmp_path), output=tmp_path / "invalid-review.json", **options)
    assert not (tmp_path / "invalid-review.json").exists()


async def test_skipped_review_reports_buffered_delivery_time():
    from scone_memory.testing.generation_ablation import _review_reply
    row = {"answer_text": "Hello!", "completed": True, "status": "completed",
           "first_token_ms": 5.0, "total_ms": 500.0}
    await _review_reply(None, None, [], {"status": "empty"}, row, None, 20.0, "report")
    assert row["answer_review"]["status"] == "skipped"
    assert row["first_token_ms"] == 500.0
    assert row["draft_first_token_ms"] == 5.0


@pytest.mark.parametrize("options,match", [
    ({"evidence_selector_model": " "}, "evidence_selector_model"),
    ({"evidence_selector_model": True}, "evidence_selector_model"),
    ({"evidence_selector_model": "selector"}, "endpoint"),
    ({"evidence_selector_model": "selector", "review_model": "reviewer"}, "mutually exclusive"),
    ({"evidence_answer_timeout": True}, "evidence_answer_timeout"),
    ({"evidence_answer_timeout": "20"}, "evidence_answer_timeout"),
    ({"evidence_answer_timeout": float("nan")}, "evidence_answer_timeout"),
    ({"evidence_answer_timeout": 0}, "evidence_answer_timeout"),
    ({"evidence_answer_timeout": 181}, "evidence_answer_timeout"),
])
async def test_evidence_answer_options_validate_before_output(tmp_path, options, match):
    from scone_memory.testing.generation_ablation import run_ablation
    output = tmp_path / "new-directory" / "extractive.json"
    with pytest.raises(ValueError, match=match):
        await run_ablation(fixture_file(tmp_path), output=output, **options)
    assert not output.parent.exists()


@pytest.mark.parametrize("change_source,card_kind", [(False, "path"), (True, "path"), (False, "passage")])
async def test_extractive_candidate_selects_verified_complete_path_without_generator(tmp_path, monkeypatch, change_source, card_kind):
    from scone_memory.realtime.evidence_answer import EvidenceSelection
    from scone_memory.retrieval.adaptive import EvidenceDecision
    from scone_memory.testing import generation_ablation
    fixture = json.loads((Path(__file__).parent / "fixtures/generation/v1.json").read_text())
    fixture["cases"] = [case for case in fixture["cases"] if case["id"] == "multihop-journal"]
    source = tmp_path / "extractive-path.json"
    source.write_text(json.dumps(fixture))
    generation_inputs, selections = [], []
    captured = {}
    original_prepare = generation_ablation.prepare_review_evidence

    async def capture_material(memory, space, scope, session_id, request, receipt):
        captured.update(memory=memory, space=space)
        return await original_prepare(memory, space, scope, session_id, request, receipt)

    class Assessor:
        def __init__(self, *args, **kwargs):
            pass

        async def assess(self, question, candidates):
            return EvidenceDecision(status="sufficient", selected_ids=tuple(item.id for item in candidates))

    class Selector:
        def __init__(self, endpoint, model, *, timeout):
            assert (endpoint, model, timeout) == ("http://localhost:1234/v1", "selector", 20)

        async def select(self, question, cards):
            selections.append((question, cards))
            path = max((card for card in cards if card.kind == card_kind), key=lambda card: len(card.evidence_ids))
            captured["selected_text"] = path.text
            if change_source:
                fact_id = next(int(value.split(":")[1]) for value in path.evidence_ids if value.startswith("fact:"))
                fact = await captured["memory"].documents.get_fact(captured["space"], fact_id)
                await captured["memory"].forget(captured["space"], fact.source_episode_id)
            return EvidenceSelection(card_ids=(path.id,))

    class Streaming:
        async def respond(self, messages):
            generation_inputs.append(messages)
            yield TextDelta("GENERATOR_ONLY_INVENTED_TEXT")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    monkeypatch.setattr(generation_ablation, "prepare_review_evidence", capture_material)
    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceAssessor", Assessor)
    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceSelector", Selector)
    report = await generation_ablation.run_ablation(source, output=tmp_path / "extractive-report.json",
        endpoint="http://localhost:1234/v1", model_factory=Streaming, adaptive_model="assessor",
        expand_relations=True, group_relations=True, evidence_selector_model="selector")
    baseline, candidate = report["results"]
    assert len(generation_inputs) == len(selections) == 1
    assert baseline["answer_text"] == "GENERATOR_ONLY_INVENTED_TEXT"
    assert baseline["answer_mode"] == "model_generation"
    assert baseline["generation_provider_calls"] == 1
    assert "evidence_answer" not in baseline
    assert candidate["answer_mode"] == report["candidate_answer_mode"] == "extractive"
    assert candidate["generation_provider_calls"] == 0
    assert report["evidence_selector_model"] == "selector"
    assert report["evidence_answer_timeout_seconds"] == 20
    assert report["evidence_answer_max_cards"] == 24
    assert report["evidence_answer_max_evidence_bytes"] == report["evidence_answer_max_answer_bytes"] == 16000
    assert "GENERATOR_ONLY_INVENTED_TEXT" not in candidate["answer_text"]
    assert selections[0][0] == fixture["cases"][0]["query"]
    assert candidate["output_bytes"] == len(candidate["answer_text"].encode())
    assert candidate["evidence_answer"]["verified_accuracy"] is False
    if change_source:
        assert candidate["answer_text"] == ""
        assert candidate["completed"] is False
        assert candidate["first_token_ms"] is None
        assert candidate["evidence_answer"]["source_status"] in ("stale", "unavailable")
        assert candidate["selected_evidence_coverage"] == 0
    else:
        assert candidate["completed"] is True
        assert candidate["evidence_answer"]["status"] == "selected"
        assert candidate["evidence_answer"]["source_status"] == "retained"
        assert candidate["first_token_ms"] == candidate["total_ms"]
        selected_quotes = [required["quote"] for required in fixture["cases"][0]["required"]
                           if required["quote"] in captured["selected_text"]]
        assert candidate["evidence_coverage"] == 1
        assert candidate["selected_evidence_coverage"] == len(selected_quotes) / 3
        assert candidate["selected_evidence_scope_leaks"] == candidate["selected_evidence_invalid_provenance"] == 0
        assert all(quote in candidate["answer_text"] for quote in selected_quotes)
        assert len(selected_quotes) == (3 if card_kind == "path" else 0)
    assert any("inflate lexical coverage" in note for note in report["limitations"])


@pytest.mark.parametrize("failure", ["unprepared", "helper_timeout", "selector_error"])
async def test_extractive_skips_without_context_and_suppresses_prepared_failures(tmp_path, monkeypatch, failure):
    from scone_memory.testing import generation_ablation
    calls = []

    class Selector:
        def __init__(self, *args, **kwargs):
            pass

        async def select(self, question, cards):
            calls.append("selector")
            assert "EVALUATION_ONLY_CANARY" not in question + "".join(card.text for card in cards)
            raise RuntimeError("PRIVATE_SELECTOR_ERROR")

    class Streaming:
        async def respond(self, messages):
            calls.append("generator")
            yield TextDelta("Generated fallback.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    if failure == "unprepared":
        original = generation_ablation.MemoryContext.prepare

        async def unprepared(context, messages):
            _, receipt = await original(context, messages)
            return messages, {**receipt, "status": "empty", "context_sha256": None, "context_bytes": 0}

        monkeypatch.setattr(generation_ablation.MemoryContext, "prepare", unprepared)
    elif failure == "helper_timeout":
        async def stalled(*args, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError("absolute deadline must interrupt preparation")

        monkeypatch.setattr(generation_ablation, "prepare_review_evidence", stalled)
    monkeypatch.setattr(generation_ablation, "SelfHostedEvidenceSelector", Selector)
    report = await generation_ablation.run_ablation(fixture_file(tmp_path), output=tmp_path / "failure.json",
        model_factory=Streaming, endpoint="http://localhost:1234/v1", evidence_selector_model="selector", evidence_answer_timeout=1)
    baseline, candidate = report["results"]
    assert baseline["answer_text"] == "Generated fallback."
    if failure == "unprepared":
        assert calls == ["generator", "generator"]
        assert candidate["answer_mode"] == "model_generation"
        assert candidate["evidence_answer"]["status"] == "skipped"
        assert candidate["completed"] is True
    else:
        assert calls == (["generator", "selector"] if failure == "selector_error" else ["generator"])
        assert candidate["answer_text"] == ""
        assert candidate["completed"] is False
        assert candidate["first_token_ms"] is None
        assert candidate["generation_provider_calls"] == 0
        assert candidate["selected_evidence_coverage"] == 0
        assert candidate["evidence_answer"]["status"] == "unavailable"
        if failure == "helper_timeout":
            assert candidate["status"] == "timeout"
            assert candidate["total_ms"] < 2000
    assert "PRIVATE_SELECTOR_ERROR" not in json.dumps(report)


def test_cli_forwards_extractive_options(tmp_path, monkeypatch):
    from scone_memory.testing import generation_ablation
    captured = {}

    async def fake_run(fixture, **kwargs):
        captured.update(kwargs)
        return {"state": "completed"}

    monkeypatch.setattr(generation_ablation, "run_ablation", fake_run)
    monkeypatch.setattr("sys.argv", ["generation_ablation", "--fixture", str(tmp_path / "fixture.json"),
        "--output", str(tmp_path / "report.json"), "--model", "generator", "--endpoint", "http://localhost:1234/v1",
        "--evidence-selector-model", "selector", "--evidence-answer-timeout", "12"])
    generation_ablation.main()
    assert captured["evidence_selector_model"] == "selector"
    assert captured["evidence_answer_timeout"] == 12
