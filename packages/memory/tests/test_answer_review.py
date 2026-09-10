"""Bounded answer review adopts only a supported, source-checked revision."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from pydantic import ValidationError

from scone_memory.realtime.answer_review import (AnswerIssue, AnswerReviewDecision, AnswerReviewError,
    AnswerReviewLimits, review_answer)

QUESTION = "Who owns Aster?"
DRAFT = "Aster is owned by Beacon."
EVIDENCE = "Aster is owned by Cedar."
IDS = ("fact:1", "chunk:2", "link:3")
ReviewFn = Callable[[str, str, str, tuple[str, ...]], Awaitable[AnswerReviewDecision]]


class Reviewer:
    def __init__(self, callback: ReviewFn) -> None:
        self.callback = callback
        self.calls: list[str] = []

    async def review(self, question: str, answer: str, evidence: str,
                     evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        assert question == QUESTION and evidence == EVIDENCE and evidence_ids == IDS
        self.calls.append(answer)
        return await self.callback(question, answer, evidence, evidence_ids)


async def supported(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
    return AnswerReviewDecision(status="supported")


def issue(quote: str = "Beacon") -> AnswerIssue:
    return AnswerIssue(code="unsupported_claim", answer_quote=quote, evidence_ids=("fact:1",))


async def test_supported_original_is_unchanged_and_accuracy_is_not_claimed() -> None:
    reviewer = Reviewer(supported)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS)
    assert result.answer == DRAFT and reviewer.calls == [DRAFT]
    assert result.receipt.status == "supported" and result.receipt.rounds == 1
    assert not result.receipt.revised and result.receipt.verified_accuracy is False
    assert result.receipt.source_status == "unchecked" and result.receipt.errors == ()
    assert "Beacon" not in result.receipt.model_dump_json()


async def test_revision_is_adopted_only_after_second_supported_review_and_four_source_checks() -> None:
    events: list[str] = []

    async def validate() -> bool:
        events.append("source")
        return True

    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        events.append("review")
        return (AnswerReviewDecision(status="needs_revision", issues=(issue(),), revised_answer=EVIDENCE)
                if answer == DRAFT else AnswerReviewDecision(status="supported"))

    reviewer = Reviewer(assess)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS, validate_evidence=validate)
    assert reviewer.calls == [DRAFT, EVIDENCE] and result.answer == EVIDENCE
    assert result.receipt.status == "supported" and result.receipt.rounds == 2 and result.receipt.revised
    assert result.receipt.source_status == "retained" and result.receipt.issue_codes == ("unsupported_claim",)
    assert events == ["source", "review", "source", "source", "review", "source"]


@pytest.mark.parametrize("second", ["uncertain", "needs_revision", "failure", "invalid"])
async def test_unconfirmed_revision_never_replaces_original(second: str) -> None:
    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        if answer == DRAFT:
            return AnswerReviewDecision(status="needs_revision", issues=(issue(),), revised_answer=EVIDENCE)
        if second == "failure":
            raise AnswerReviewError("review_provider_failed")
        if second == "invalid":
            return AnswerReviewDecision.model_construct(status="supported", issues=(issue("not in revision"),))
        if second == "needs_revision":
            return AnswerReviewDecision(status="needs_revision", issues=(issue("Cedar"),), revised_answer="third version")
        return AnswerReviewDecision(status="uncertain")

    reviewer = Reviewer(assess)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS)
    assert result.answer == DRAFT and len(reviewer.calls) == 2 and not result.receipt.revised
    assert result.receipt.status == ("unavailable" if second in {"failure", "invalid"} else second)


async def test_one_round_budget_does_not_adopt_unreviewed_revision() -> None:
    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        return AnswerReviewDecision(status="needs_revision", issues=(issue(),), revised_answer=EVIDENCE)

    reviewer = Reviewer(assess)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS, limits=AnswerReviewLimits(max_rounds=1))
    assert result.answer == DRAFT and len(reviewer.calls) == 1 and result.receipt.status == "needs_revision"


@pytest.mark.parametrize("invalid", ["quote", "unknown_id", "duplicate_id", "empty_quote", "issue_list", "nested_mutation", "wrong_status"])
async def test_invalid_constructed_review_is_rejected_without_repairs(invalid: str) -> None:
    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        bad = issue()
        if invalid == "quote":
            bad = issue("not an exact answer span")
        elif invalid == "unknown_id":
            bad = AnswerIssue(code="unsupported_claim", answer_quote="Beacon", evidence_ids=("fact:999",))
        elif invalid == "duplicate_id":
            bad = AnswerIssue.model_construct(code="unsupported_claim", answer_quote="Beacon", evidence_ids=("fact:1", "fact:1"))
        elif invalid == "empty_quote":
            bad = AnswerIssue.model_construct(code="contradiction", answer_quote="", evidence_ids=())
        elif invalid == "nested_mutation":
            bad = bad.model_copy(update={"code": "private injected code"})
        return AnswerReviewDecision.model_construct(status="unknown" if invalid == "wrong_status" else "needs_revision",
            issues=[bad] if invalid == "issue_list" else (bad,), revised_answer=EVIDENCE)

    result = await review_answer(Reviewer(assess), QUESTION, DRAFT, EVIDENCE, IDS)
    assert result.answer == DRAFT and result.receipt.status == "unavailable"
    assert result.receipt.errors == ("invalid_review",)
    assert "private injected code" not in result.receipt.model_dump_json()


@pytest.mark.parametrize("when", [1, 2, 3, 4])
@pytest.mark.parametrize("failure", ["stale", "exception", "wrong_type"])
async def test_source_failure_stops_review_and_preserves_original(when: int, failure: str) -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        if checks != when:
            return True
        if failure == "exception":
            raise RuntimeError("private source detail")
        if failure == "wrong_type":
            from typing import cast
            return cast(bool, 1)
        return False

    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        return (AnswerReviewDecision(status="needs_revision", issues=(issue(),), revised_answer=EVIDENCE)
                if answer == DRAFT else AnswerReviewDecision(status="supported"))

    reviewer = Reviewer(assess)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS, validate_evidence=validate)
    assert result.answer == DRAFT and result.receipt.status == "unavailable"
    assert result.receipt.source_status == ("stale" if failure == "stale" else "unavailable")
    assert len(reviewer.calls) == (0 if when == 1 else 1 if when < 4 else 2)
    assert "private source detail" not in result.receipt.model_dump_json()


async def test_review_failure_requires_a_fresh_post_failure_source_check() -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    async def fails(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        raise AnswerReviewError("review_provider_failed")

    result = await review_answer(Reviewer(fails), QUESTION, DRAFT, EVIDENCE, IDS, validate_evidence=validate)
    assert checks == 2 and result.receipt.source_status == "stale"
    assert result.receipt.status == "unavailable" and "review_provider_failed" in result.receipt.errors


async def test_review_timeout_recovers_only_a_fresh_source_verdict() -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        return True

    async def blocked(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        await asyncio.Event().wait()
        return AnswerReviewDecision(status="supported")

    result = await review_answer(Reviewer(blocked), QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=validate, limits=AnswerReviewLimits(timeout_s=1.0))
    assert result.answer == DRAFT and result.receipt.status == "unavailable"
    assert checks == 2 and result.receipt.source_status == "retained"
    assert "review_timeout" in result.receipt.errors


@pytest.mark.parametrize("stage", ["source", "review"])
async def test_external_cancellation_propagates(stage: str) -> None:
    entered = asyncio.Event()

    async def validate() -> bool:
        if stage == "source":
            entered.set()
            await asyncio.Event().wait()
        return True

    async def blocked(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        entered.set()
        await asyncio.Event().wait()
        return AnswerReviewDecision(status="supported")

    task = asyncio.create_task(review_answer(Reviewer(blocked), QUESTION, DRAFT, EVIDENCE, IDS, validate_evidence=validate))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("options", [{"max_rounds": 3}, {"timeout_s": float("nan")}, {"timeout_s": True},
    {"max_answer_bytes": 511}, {"max_evidence_bytes": 262145}, {"max_rounds": "2"}])
def test_limits_are_strict_and_bounded(options: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnswerReviewLimits.model_validate(options)


@pytest.mark.parametrize("field,value", [
    ("question", ""), ("question", "q" * 8001), ("question", True),
    ("answer", " "), ("answer", "é" * 257), ("answer", b"draft"),
    ("evidence", "é" * 257), ("evidence", {}),
    ("evidence_ids", ["fact:1"]), ("evidence_ids", ("fact:0",)),
    ("evidence_ids", ("fact:1", "fact:1")), ("evidence_ids", ("fact:1\n",)),
    ("evidence_ids", ("fact:" + "1" * 64,)), ("evidence_ids", tuple(f"fact:{i+1}" for i in range(257))),
])
async def test_input_boundaries_reject_before_callbacks_or_review(field: str, value: object) -> None:
    from typing import cast

    values: dict[str, object] = {"question": QUESTION, "answer": DRAFT, "evidence": EVIDENCE, "evidence_ids": IDS}
    values[field] = value
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        return True

    reviewer = Reviewer(supported)
    with pytest.raises(ValueError):
        await review_answer(reviewer, cast(str, values["question"]), cast(str, values["answer"]),
            cast(str, values["evidence"]), cast(tuple[str, ...], values["evidence_ids"]), validate_evidence=validate,
            limits=AnswerReviewLimits(max_answer_bytes=512, max_evidence_bytes=512))
    assert checks == 0 and reviewer.calls == []


@pytest.mark.parametrize("revision", ["", "  ", "é" * 257])
async def test_revision_byte_limit_and_empty_revision_fail_without_second_call(revision: str) -> None:
    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        return AnswerReviewDecision.model_construct(status="needs_revision", issues=(issue(),), revised_answer=revision)

    reviewer = Reviewer(assess)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS, limits=AnswerReviewLimits(max_answer_bytes=512))
    assert result.answer == DRAFT and reviewer.calls == [DRAFT]
    assert result.receipt.status == "unavailable" and result.receipt.errors == ("invalid_review",)


@pytest.mark.parametrize("reason", ["review_timeout", "review_provider_failed", "invalid_review", "private provider detail", None, []])
async def test_typed_failures_preserve_only_safe_codes_and_revalidate_sources(reason: object) -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        return True

    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        error = AnswerReviewError("invalid_review")
        if reason is None:
            del error.reason
        else:
            setattr(error, "reason", reason)
        raise error

    result = await review_answer(Reviewer(assess), QUESTION, DRAFT, EVIDENCE, IDS, validate_evidence=validate)
    assert checks == 2 and result.receipt.source_status == "retained"
    assert result.answer == DRAFT and result.receipt.status == "unavailable"
    expected = reason if type(reason) is str and reason in {"review_timeout", "review_provider_failed", "invalid_review"} else "invalid_review"
    assert result.receipt.errors == (expected,)
    assert "private provider detail" not in result.receipt.model_dump_json()


async def test_untyped_provider_exception_is_sanitized_and_sources_rechecked() -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        return True

    async def fails(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        raise RuntimeError("private endpoint and request")

    result = await review_answer(Reviewer(fails), QUESTION, DRAFT, EVIDENCE, IDS, validate_evidence=validate)
    assert result.receipt.errors == ("review_provider_failed",) and checks == 2
    assert result.receipt.source_status == "retained" and "private" not in result.receipt.model_dump_json()


@pytest.mark.parametrize("stage", ["review", "source"])
async def test_late_swallowed_cancellation_cannot_produce_supported_output(stage: str) -> None:
    async def validate() -> bool:
        if stage == "source":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return True
        return True

    async def late(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return AnswerReviewDecision(status="supported")
        return AnswerReviewDecision(status="supported")

    result = await review_answer(Reviewer(late), QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=validate, limits=AnswerReviewLimits(timeout_s=1.0))
    assert result.answer == DRAFT and result.receipt.status == "unavailable"
    assert result.receipt.source_status == ("unavailable" if stage == "source" else "retained")
    assert "review_timeout" in result.receipt.errors


async def test_second_review_uses_same_original_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from scone_memory.realtime import answer_review

    class Clock:
        value = 0.0
        def monotonic(self) -> float:
            return self.value

    clock = Clock()
    monkeypatch.setattr(answer_review, "time", clock)

    async def validate() -> bool:
        return True

    async def assess(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        if answer == DRAFT:
            clock.value = 0.5
            return AnswerReviewDecision(status="needs_revision", issues=(issue(),), revised_answer=EVIDENCE)
        clock.value = 1.01
        return AnswerReviewDecision(status="supported")

    reviewer = Reviewer(assess)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS,
        limits=AnswerReviewLimits(timeout_s=1.0), validate_evidence=validate)
    assert reviewer.calls == [DRAFT, EVIDENCE] and result.answer == DRAFT
    assert result.receipt.status == "unavailable" and result.receipt.source_status == "unavailable"
    assert "review_timeout" in result.receipt.errors


@pytest.mark.parametrize("data", [
    {"status": "supported", "issues": (issue(),)},
    {"status": "supported", "revised_answer": EVIDENCE},
    {"status": "needs_revision"},
    {"status": "needs_revision", "issues": (issue(),), "revised_answer": ""},
    {"status": "needs_revision", "issues": (issue(),), "revised_answer": "   "},
    {"status": "uncertain", "revised_answer": EVIDENCE},
    {"status": "uncertain", "issues": tuple(issue() for _ in range(9))},
])
def test_decision_schema_invariants(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnswerReviewDecision.model_validate(data)


def test_issue_quote_and_id_schema_invariants() -> None:
    assert AnswerIssue(code="incomplete_answer", answer_quote="").evidence_ids == ()
    with pytest.raises(ValidationError):
        AnswerIssue(code="broken_path", answer_quote="")
    with pytest.raises(ValidationError):
        AnswerIssue(code="unsupported_claim", answer_quote="x" * 2001)
    with pytest.raises(ValidationError):
        AnswerIssue(code="contradiction", answer_quote="quote", evidence_ids=tuple(f"fact:{i+1}" for i in range(33)))


@pytest.mark.parametrize("failure", ["stale", "exception", "timeout"])
async def test_model_timeout_recovery_reports_failed_source_validation(failure: str) -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        if checks == 1:
            return True
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "exception":
            raise RuntimeError("private recovery store error")
        return False

    async def blocked(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        await asyncio.Event().wait()
        return AnswerReviewDecision(status="supported")

    result = await review_answer(Reviewer(blocked), QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=validate, limits=AnswerReviewLimits(timeout_s=1.0))
    assert checks == 2 and result.answer == DRAFT and result.receipt.status == "unavailable"
    assert result.receipt.source_status == ("stale" if failure == "stale" else "unavailable")
    assert "review_timeout" in result.receipt.errors
    assert {"stale": "stale_evidence", "exception": "source_validation_failed", "timeout": "source_validation_timeout"}[failure] in result.receipt.errors
    assert "private recovery store error" not in result.receipt.model_dump_json()


async def test_cancellation_during_reserved_validation_propagates() -> None:
    entered = asyncio.Event()
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        if checks == 2:
            entered.set()
            await asyncio.Event().wait()
        return True

    async def blocked(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        await asyncio.Event().wait()
        return AnswerReviewDecision(status="supported")

    task = asyncio.create_task(review_answer(Reviewer(blocked), QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=validate, limits=AnswerReviewLimits(timeout_s=1.0)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert checks == 2


async def test_source_stage_timeout_does_not_consume_reserve_with_another_attempt() -> None:
    checks = 0

    async def blocked() -> bool:
        nonlocal checks
        checks += 1
        await asyncio.Event().wait()
        return True

    reviewer = Reviewer(supported)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=blocked, limits=AnswerReviewLimits(timeout_s=1.0))
    assert checks == 1 and reviewer.calls == []
    assert result.receipt.source_status == "unavailable" and result.receipt.status == "unavailable"


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "20"])
async def test_external_deadline_is_strictly_validated(deadline: object) -> None:
    from typing import cast

    reviewer = Reviewer(supported)
    with pytest.raises(ValueError, match="deadline"):
        await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS, deadline=cast(float, deadline))
    assert reviewer.calls == []


async def test_expired_external_deadline_makes_no_calls() -> None:
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        return True

    reviewer = Reviewer(supported)
    result = await review_answer(reviewer, QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=validate, deadline=-1.0)
    assert checks == 0 and reviewer.calls == [] and result.answer == DRAFT
    assert result.receipt.status == "unavailable" and result.receipt.source_status == "unavailable"


async def test_short_remaining_budget_gets_its_own_reserve_without_extending_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from scone_memory.realtime import answer_review

    class Clock:
        value = 10.0
        def monotonic(self) -> float:
            return self.value

    clock = Clock()
    monkeypatch.setattr(answer_review, "time", clock)
    checks = 0

    async def validate() -> bool:
        nonlocal checks
        checks += 1
        if checks == 2:
            assert clock.value == 10.3
            clock.value = 10.41
        return True

    async def late(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        clock.value = 10.3  # 0.4s remaining => 0.3s workflow plus 0.1s reserve.
        return AnswerReviewDecision(status="supported")

    result = await review_answer(Reviewer(late), QUESTION, DRAFT, EVIDENCE, IDS,
        validate_evidence=validate, deadline=10.4)
    assert checks == 2 and result.answer == DRAFT and result.receipt.status == "unavailable"
    assert result.receipt.source_status == "unavailable" and "source_validation_timeout" in result.receipt.errors


async def test_later_external_deadline_cannot_extend_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from scone_memory.realtime import answer_review

    class Clock:
        value = 0.0
        def monotonic(self) -> float:
            return self.value

    clock = Clock()
    monkeypatch.setattr(answer_review, "time", clock)

    async def late(question: str, answer: str, evidence: str, evidence_ids: tuple[str, ...]) -> AnswerReviewDecision:
        clock.value = 1.01
        return AnswerReviewDecision(status="supported")

    result = await review_answer(Reviewer(late), QUESTION, DRAFT, EVIDENCE, IDS,
        limits=AnswerReviewLimits(timeout_s=1.0), deadline=100.0)
    assert result.answer == DRAFT and result.receipt.status == "unavailable"
    assert result.receipt.source_status == "unchecked" and result.receipt.errors == ("review_timeout",)
