"""Optional bounded review of a draft against caller-supplied evidence.

Review is a model judgment, not verified answer accuracy. A revision is adopted
only after a second supported judgment and every configured source check. The
caller owns the reviewer lifecycle and decides whether to withhold unsupported
answers. Stale or unavailable source status must be handled independently.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import math
import re
import time
from typing import Annotated, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .answer_requirements import AnswerRequirements, validated_requirements

EvidenceId = Annotated[str, Field(pattern=r"^(chunk|fact|link):[1-9][0-9]*$", max_length=64)]
IssueCode = Literal["unsupported_claim", "contradiction", "incomplete_answer", "broken_path"]
ReviewStatus = Literal["supported", "needs_revision", "uncertain", "unavailable"]
SourceStatus = Literal["unchecked", "retained", "stale", "unavailable"]
ErrorCode = Literal["review_timeout", "review_provider_failed", "invalid_review",
                    "source_validation_failed", "source_validation_timeout", "stale_evidence",
                    "answer_format_rejected"]
REVIEW_FAILURE_REASONS = frozenset({"review_timeout", "review_provider_failed", "invalid_review"})
MAX_QUESTION_BYTES = 8000
MAX_EVIDENCE_IDS = 256
_ID = re.compile(r"(?:chunk|fact|link):[1-9][0-9]*")


class AnswerReviewError(ValueError):
    """Content-free adapter failure; arbitrary provider messages are discarded."""
    def __init__(self, reason: str) -> None:
        super().__init__("answer review failed")
        self.reason = reason if type(reason) is str and reason in REVIEW_FAILURE_REASONS else "invalid_review"


class AnswerIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    code: IssueCode
    answer_quote: str = Field(max_length=2000)
    evidence_ids: tuple[EvidenceId, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def check_issue(self) -> AnswerIssue:
        if not self.answer_quote and self.code != "incomplete_answer":
            raise ValueError("only incomplete_answer may omit the answer quote")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("issue evidence IDs must be unique")
        return self


class AnswerReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Literal["supported", "needs_revision", "uncertain"]
    issues: tuple[AnswerIssue, ...] = Field(default=(), max_length=8)
    revised_answer: str | None = Field(default=None, max_length=64000)

    @model_validator(mode="after")
    def check_decision(self) -> AnswerReviewDecision:
        if self.revised_answer is not None and not self.revised_answer.strip():
            raise ValueError("revised_answer must be nonblank")
        if self.status == "supported" and (self.issues or self.revised_answer is not None):
            raise ValueError("supported must not include issues or a revision")
        if self.status == "needs_revision" and not self.issues:
            raise ValueError("needs_revision requires at least one issue")
        if self.status == "uncertain" and self.revised_answer is not None:
            raise ValueError("uncertain must not include a revision")
        return self


class AnswerReviewer(Protocol):
    async def review(self, question: str, answer: str, evidence: str,
                     evidence_ids: tuple[str, ...]) -> AnswerReviewDecision: ...


class RequirementsAnswerReviewer(AnswerReviewer, Protocol):
    async def review_with_requirements(self, question: str, answer: str, evidence: str,
        evidence_ids: tuple[str, ...], requirements: AnswerRequirements) -> AnswerReviewDecision: ...


def require_contextual_reviewer(reviewer: AnswerReviewer, requirements: AnswerRequirements | None) -> None:
    if requirements is not None and not callable(getattr(reviewer, 'review_with_requirements', None)):
        raise ValueError('answer requirements need a reviewer implementing review_with_requirements')


class AnswerReviewLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    timeout_s: float = Field(default=20.0, ge=1, le=180, allow_inf_nan=False)
    max_answer_bytes: int = Field(default=64000, ge=512, le=128000)
    max_evidence_bytes: int = Field(default=128000, ge=512, le=262144)
    max_rounds: int = Field(default=2, ge=1, le=2)


class AnswerReviewReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: ReviewStatus = "unavailable"
    rounds: int = Field(default=0, ge=0, le=2)
    revised: bool = False
    issue_codes: tuple[IssueCode, ...] = ()
    errors: tuple[ErrorCode, ...] = ()
    verified_accuracy: Literal[False] = False
    source_status: SourceStatus = "unchecked"
    format_status: Literal['unchecked', 'satisfied', 'rejected'] = 'unchecked'


class ReviewedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    answer: str
    receipt: AnswerReviewReceipt


class _StopReview(Exception):
    """Stop without adopting a proposed answer; safe diagnostics are on the run."""


def _text(value: str, name: str, maximum: int, *, nonempty: bool) -> None:
    if type(value) is not str or (nonempty and not value.strip()):
        raise ValueError(f"{name} must be a {'nonblank ' if nonempty else ''}string")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds its UTF-8 byte limit")


def _decision(value: AnswerReviewDecision, answer: str, ids: tuple[str, ...],
              limits: AnswerReviewLimits) -> AnswerReviewDecision:
    if not isinstance(value, AnswerReviewDecision):
        raise ValueError("review decision required")
    data = dict(vars(value))
    issues = data.get("issues", ())
    if type(issues) is not tuple or len(issues) > 8:
        raise ValueError("bounded issue tuple required")
    # Nested instances can also bypass Pydantic with model_construct/model_copy.
    data["issues"] = tuple(dict(vars(issue)) if isinstance(issue, AnswerIssue) else issue for issue in issues)
    decision = AnswerReviewDecision.model_validate(data, strict=True)
    for issue in decision.issues:
        if issue.answer_quote and issue.answer_quote not in answer:
            raise ValueError("issue quote does not match the reviewed draft")
        if not set(issue.evidence_ids).issubset(ids):
            raise ValueError("issue cites unavailable evidence")
    if decision.revised_answer is not None:
        _text(decision.revised_answer, "revision", limits.max_answer_bytes, nonempty=True)
    return decision


class _ReviewRun:
    def __init__(self, reviewer: AnswerReviewer, question: str, answer: str, evidence: str,
                 ids: tuple[str, ...], limits: AnswerReviewLimits,
                 validator: Callable[[], Awaitable[bool]] | None, deadline: float | None,
                 requirements: AnswerRequirements | None) -> None:
        self.reviewer, self.question, self.original = reviewer, question, answer
        self.evidence, self.ids, self.limits, self.validator = evidence, ids, limits, validator
        self.requirements = requirements
        now = time.monotonic()
        self.hard_deadline = min(now + limits.timeout_s, deadline) if deadline is not None else now + limits.timeout_s
        reserve = min(1.0, max(0.0, self.hard_deadline - now) / 4) if validator is not None else 0.0
        self.deadline = self.hard_deadline - reserve
        self.rounds = 0
        self.issue_codes: list[IssueCode] = []
        self.errors: list[ErrorCode] = []
        self.source_status: SourceStatus = "unchecked" if validator is None else "unavailable"
        self.stage: Literal["source", "review"] = "source"

    def check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise asyncio.TimeoutError()

    def error(self, code: ErrorCode) -> None:
        if code not in self.errors:
            self.errors.append(code)

    def result(self, *, status: ReviewStatus = "unavailable", answer: str | None = None) -> ReviewedAnswer:
        selected = self.original if answer is None else answer
        format_status: Literal['unchecked', 'satisfied', 'rejected'] = 'unchecked'
        if self.requirements is not None:
            format_status = 'satisfied' if self.requirements.accepts(selected) else 'rejected'
            if format_status == 'rejected':
                self.error('answer_format_rejected')
                if status == 'supported':
                    status = 'needs_revision'
        return ReviewedAnswer(answer=selected, receipt=AnswerReviewReceipt(status=status, rounds=self.rounds,
            revised=selected != self.original, issue_codes=tuple(self.issue_codes), errors=tuple(self.errors),
            source_status=self.source_status, format_status=format_status))

    async def sources(self) -> None:
        self.check_deadline()
        if self.validator is None:
            return
        self.stage = "source"
        self.source_status = "unavailable"
        try:
            retained = await self.validator()
            self.check_deadline()
            if type(retained) is not bool:
                raise ValueError("source validator must return bool")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.error("source_validation_timeout")
            raise
        except Exception:
            self.error("source_validation_failed")
            raise _StopReview() from None
        if not retained:
            self.source_status = "stale"
            self.error("stale_evidence")
            raise _StopReview()
        self.source_status = "retained"

    async def assess(self, answer: str) -> AnswerReviewDecision:
        self.stage = "review"
        self.rounds += 1
        # The pre-call verdict cannot authorize output after any model await.
        if self.validator is not None:
            self.source_status = "unavailable"
        decision: AnswerReviewDecision | None = None
        try:
            if self.requirements is None:
                response = await self.reviewer.review(self.question, answer, self.evidence, self.ids)
            else:
                response = await cast(RequirementsAnswerReviewer, self.reviewer).review_with_requirements(
                    self.question, answer, self.evidence, self.ids, self.requirements)
            self.check_deadline()
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.error("review_timeout")
        except AnswerReviewError as error:
            reason = getattr(error, "reason", None)
            if type(reason) is str and reason == "review_timeout":
                self.error("review_timeout")
            elif type(reason) is str and reason == "review_provider_failed":
                self.error("review_provider_failed")
            else:
                self.error("invalid_review")
        except Exception:
            self.error("review_provider_failed")
        else:
            try:
                decision = _decision(response, answer, self.ids, self.limits)
            except Exception:
                self.error("invalid_review")
        if decision is not None:
            for issue in decision.issues:
                if issue.code not in self.issue_codes:
                    self.issue_codes.append(issue.code)
        # Even a failed review must be followed by current source verification
        # when time remains. Otherwise source_status stays unavailable.
        await self.sources()
        if decision is None:
            raise _StopReview()
        return decision

    async def execute(self) -> ReviewedAnswer:
        current = self.original
        for round_number in range(self.limits.max_rounds):
            await self.sources()
            decision = await self.assess(current)
            self.check_deadline()
            if decision.status == "supported":
                return self.result(status="supported", answer=current)
            if (decision.status != "needs_revision" or decision.revised_answer is None
                    or round_number + 1 == self.limits.max_rounds):
                return self.result(status=decision.status)
            if self.requirements is not None and not self.requirements.accepts(decision.revised_answer):
                self.error('answer_format_rejected')
                return self.result(status='needs_revision')
            current = decision.revised_answer
        return self.result()

    async def recover_sources_after_timeout(self) -> ReviewedAnswer:
        """Use the reserved window solely to revalidate the original sources."""
        self.deadline = self.hard_deadline
        self.source_status = "unavailable"
        remaining = self.hard_deadline - time.monotonic()
        if remaining <= 0:
            self.error("source_validation_timeout")
            return self.result()
        try:
            await asyncio.wait_for(self.sources(), timeout=remaining)
            self.check_deadline()
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.source_status = "unavailable"
            self.error("source_validation_timeout")
        except _StopReview:
            pass
        except Exception:
            self.source_status = "unavailable"
            self.error("source_validation_failed")
        return self.result()


async def review_answer(reviewer: AnswerReviewer, question: str, answer: str, evidence: str,
                        evidence_ids: tuple[str, ...], *, limits: AnswerReviewLimits | None = None,
                        validate_evidence: Callable[[], Awaitable[bool]] | None = None,
                        deadline: float | None = None,
                        requirements: AnswerRequirements | None = None) -> ReviewedAnswer:
    """Review once and optionally confirm one revision under one deadline.

    Inputs are strictly validated before any callback or model invocation.
    Provider/decision failures return the original draft with an unavailable
    receipt; external cancellation propagates. Source callbacks run before and
    after each review, including failed reviews when the deadline permits.
    With source validation, up to one second (one quarter of short budgets) is
    reserved inside the same hard deadline for a final check after model timeout.
    This never retries review or adopts an unconfirmed revision. Source-check
    timeouts themselves do not trigger another validation attempt.
    An optional absolute monotonic deadline can shorten the configured budget
    to include earlier caller work; it cannot extend that budget.
    Callers must treat stale/unavailable sources independently of review policy.
    Explicit requirements are shared with a contextual reviewer. Invalid proposed
    revisions are rejected before confirmation; format_status describes the
    returned text, including fallback drafts. Callers must withhold rejected
    formats even under report policy. Instructions themselves are not verified.
    """
    if limits is not None and not isinstance(limits, AnswerReviewLimits):
        raise ValueError("limits must be AnswerReviewLimits")
    fixed = AnswerReviewLimits.model_validate(dict(vars(limits or AnswerReviewLimits())), strict=True)
    _text(question, "question", MAX_QUESTION_BYTES, nonempty=True)
    _text(answer, "answer", fixed.max_answer_bytes, nonempty=True)
    _text(evidence, "evidence", fixed.max_evidence_bytes, nonempty=False)
    if (type(evidence_ids) is not tuple or len(evidence_ids) > MAX_EVIDENCE_IDS
            or any(type(value) is not str or len(value) > 64 or _ID.fullmatch(value) is None for value in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)):
        raise ValueError("evidence IDs must be a bounded unique tuple of valid source IDs")
    if validate_evidence is not None and not callable(validate_evidence):
        raise ValueError("validate_evidence must be callable")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    requirements = validated_requirements(requirements)
    require_contextual_reviewer(reviewer, requirements)
    run = _ReviewRun(reviewer, question, answer, evidence, evidence_ids, fixed, validate_evidence, deadline, requirements)
    try:
        return await asyncio.wait_for(run.execute(), timeout=max(0.0, run.deadline - time.monotonic()))
    except asyncio.CancelledError:
        raise
    except _StopReview:
        return run.result()
    except asyncio.TimeoutError:
        run.error("review_timeout")
        if validate_evidence is not None:
            run.source_status = "unavailable"
            if run.stage == "review":
                return await run.recover_sources_after_timeout()
            if run.stage == "source":
                run.error("source_validation_timeout")
        return run.result()
    except Exception:
        run.error("invalid_review")
        if validate_evidence is not None:
            run.source_status = "unavailable"
        return run.result()
