"""Reviewer quality gates distinguish wrong approvals from abstention/failure."""
import asyncio
import json
from pathlib import Path

import pytest

from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision
from scone_memory.testing.answer_review_evaluation import ReviewFixture, evaluate_reviews, load_fixture


def fixture(expected):
    return ReviewFixture.model_validate_json(json.dumps({"schema_version": 1, "cases": [
        {"id": f"case-{index}", "category": "source_support", "split": "development",
         "question": "Which team?", "answer": "Platform", "expected_acceptable": acceptable,
         "sources": [{"id": "chunk:1", "text": "Mira works on Platform."}]}
        for index, acceptable in enumerate(expected)]}))


async def test_false_approvals_abstentions_and_failures_have_distinct_denominators():
    outcomes = iter(["supported", "supported", "needs_revision", "needs_revision", "uncertain", "failure"])
    calls = []

    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            calls.append((question, answer, evidence, evidence_ids))
            status = next(outcomes)
            if status == "failure":
                raise RuntimeError("PRIVATE provider detail")
            return AnswerReviewDecision(status=status, issues=(AnswerIssue(
                code="unsupported_claim", answer_quote=answer, evidence_ids=evidence_ids),)
                if status == "needs_revision" else ())

    report = await evaluate_reviews(Reviewer(), fixture([True, False, False, True, False, False]))
    metrics = report.metrics
    assert metrics.approved_acceptable == metrics.approved_unacceptable == 1
    assert metrics.rejected_acceptable == metrics.rejected_unacceptable == 1
    assert metrics.abstained == metrics.failed == 1
    assert metrics.approval_precision == 0.5
    assert metrics.false_approval_rate == 0.25
    assert metrics.acceptable_answer_recall == 0.5
    assert metrics.decision_coverage == pytest.approx(4 / 6)
    assert metrics.labeled_agreement == pytest.approx(2 / 6)
    assert "PRIVATE" not in report.model_dump_json()
    assert all(call == ("Which team?", "Platform", '[{"id":"chunk:1","text":"Mira works on Platform."}]', ("chunk:1",)) for call in calls)
    assert all(row.split == "development" for row in report.observations)


async def test_abstaining_on_everything_cannot_score_as_correct_rejection():
    class Reviewer:
        async def review(self, *args):
            return AnswerReviewDecision(status="uncertain")

    report = await evaluate_reviews(Reviewer(), fixture([False, False]), repeats=2)
    assert len(report.observations) == 4
    assert report.metrics.approval_precision is None
    assert report.metrics.acceptable_answer_recall is None
    assert report.metrics.labeled_agreement == report.metrics.decision_coverage == 0
    assert report.metrics.rejected_unacceptable == 0


async def test_forged_decision_with_unknown_evidence_is_failure():
    class Reviewer:
        async def review(self, *args):
            return AnswerReviewDecision(status="needs_revision", issues=(AnswerIssue(
                code="contradiction", answer_quote="Platform", evidence_ids=("chunk:999",)),))

    report = await evaluate_reviews(Reviewer(), fixture([False]))
    assert report.metrics.failed == 1
    assert report.observations[0].error == "invalid_review"
    assert report.metrics.labeled_agreement == 0


async def test_evaluation_checks_only_original_draft_without_adopting_revision():
    class Reviewer:
        async def review(self, *args):
            return AnswerReviewDecision(status="needs_revision", issues=(AnswerIssue(
                code="contradiction", answer_quote="Platform", evidence_ids=("chunk:1",)),),
                revised_answer="Replacement answer")

    report = await evaluate_reviews(Reviewer(), fixture([False]))
    assert report.metrics.rejected_unacceptable == 1
    assert report.observations[0].revision_proposed is True
    assert "Replacement answer" not in report.model_dump_json()


async def test_cancellation_is_not_counted_as_provider_failure():
    class Reviewer:
        async def review(self, *args):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await evaluate_reviews(Reviewer(), fixture([True]))


async def test_timeout_is_bounded_and_not_counted_as_a_correct_rejection():
    class Reviewer:
        async def review(self, *args):
            await asyncio.Event().wait()

    report = await evaluate_reviews(Reviewer(), fixture([False]), timeout_s=1)
    assert report.observations[0].error == "review_timeout"
    assert report.metrics.failed == 1
    assert report.metrics.labeled_agreement == 0


async def test_reviewer_cannot_earn_credit_by_suppressing_timeout_cancellation():
    class Reviewer:
        async def review(self, *args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return AnswerReviewDecision(status="supported")

    report = await evaluate_reviews(Reviewer(), fixture([True]), timeout_s=1)
    assert report.observations[0].error == "review_timeout"
    assert report.metrics.approved_acceptable == 0


async def test_suppressed_external_cancellation_still_cancels_evaluation():
    entered = asyncio.Event()

    class Reviewer:
        async def review(self, *args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return AnswerReviewDecision(status="supported")

    task = asyncio.create_task(evaluate_reviews(Reviewer(), fixture([True])))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_split_and_category_denominators_remain_separate_across_repeats():
    raw = fixture([True, False]).model_dump(mode="json")
    raw["cases"][1].update(split="held_out", category="conflict")

    class Reviewer:
        async def review(self, *args):
            return AnswerReviewDecision(status="supported")

    report = await evaluate_reviews(Reviewer(), ReviewFixture.model_validate_json(json.dumps(raw)), repeats=2)
    assert report.metrics.approval_precision == 0.5
    assert report.by_split["development"].approval_precision == 1
    assert report.by_split["held_out"].approval_precision == 0
    assert report.by_category["conflict"].approved_unacceptable == 2
    assert report.by_category["conflict"].false_approval_rate == 1
    assert report.by_category["source_support"].false_approval_rate is None


@pytest.mark.parametrize("options", [{"repeats": True}, {"repeats": 0}, {"repeats": 21},
                                     {"timeout_s": float("nan")}, {"timeout_s": 181}])
async def test_invalid_limits_fail_before_provider_use(options):
    class Reviewer:
        async def review(self, *args):
            pytest.fail("invalid limits must not invoke inference")

    with pytest.raises(ValueError):
        await evaluate_reviews(Reviewer(), fixture([True]), **options)


@pytest.mark.parametrize("changes", [
    {"expected_acceptable": "false"}, {"sources": []},
    {"sources": [{"id": "chunk:1", "text": "one"}, {"id": "chunk:1", "text": "two"}]},
    {"answer": "é" * 33000}, {"question": " "},
])
def test_fixture_rejects_ambiguous_labels_duplicate_sources_and_unbounded_input(changes):
    raw = fixture([True]).model_dump(mode="json")
    raw["cases"][0].update(changes)
    with pytest.raises(ValueError):
        ReviewFixture.model_validate_json(json.dumps(raw))


def test_original_development_fixture_contains_both_labels_and_known_failure_categories():
    cases = load_fixture(Path(__file__).parent / "fixtures" / "answer_review" / "v1.json").cases
    assert {case.expected_acceptable for case in cases} == {True, False}
    assert {case.split for case in cases} == {"development"}
    assert {"conflict_preserved", "conflict_erased", "missed_answer", "justified_abstention"} <= {case.id for case in cases}
    for category in {case.category for case in cases}:
        assert {case.expected_acceptable for case in cases if case.category == category} == {True, False}


def test_cli_reports_real_adapter_results_without_labels_or_credentials_in_output(tmp_path, monkeypatch, capsys):
    from scone_memory.providers import answer_reviewer
    from scone_memory.testing import answer_review_evaluation

    fixture_path, output = tmp_path / "fixture.json", tmp_path / "report.json"
    fixture_path.write_text(fixture([True]).model_dump_json())
    received = []

    class Reviewer:
        def __init__(self, *args, **kwargs):
            received.append((args, kwargs))

        async def review(self, question, answer, evidence, evidence_ids):
            assert "expected_acceptable" not in evidence
            return AnswerReviewDecision(status="supported")

    monkeypatch.setattr(answer_reviewer, "SelfHostedAnswerReviewer", Reviewer)
    monkeypatch.setenv("SCONE_ANSWER_REVIEW_API_KEY", "PRIVATE-review-key")
    monkeypatch.setattr("sys.argv", ["review-eval", "--fixture", str(fixture_path), "--output", str(output),
        "--model", "installed-model", "--endpoint", "http://127.0.0.1:11434/v1"])
    answer_review_evaluation.main()
    report = json.loads(output.read_text())
    assert report["metrics"]["approval_precision"] == 1
    assert report["model"] == "installed-model"
    assert report['quote_mode'] == 'text'
    assert received == [(("http://127.0.0.1:11434/v1", "installed-model"),
                        {"api_key": "PRIVATE-review-key", "timeout": 20, 'quote_mode':'text'})]
    assert "PRIVATE" not in output.read_text() + capsys.readouterr().out
    assert "Mira works on Platform." not in output.read_text()
    with pytest.raises(ValueError, match="output must be a new file"):
        answer_review_evaluation.main()
    assert len(received) == 1
