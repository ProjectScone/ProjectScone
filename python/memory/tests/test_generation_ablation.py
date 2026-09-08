from __future__ import annotations

import asyncio
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
    assert [row["structured_paths"] for row in report["results"]] == [False, True, True, False]
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
