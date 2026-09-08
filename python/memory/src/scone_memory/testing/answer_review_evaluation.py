"""Evaluate first-pass reviewer judgments against explicit draft-quality labels.

No retrieval or answer generation runs here. Fixture labels never reach the
reviewer. An uncertain or failed review is not counted as a correct rejection.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import time
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..realtime.answer_review import (
    AnswerReviewer, AnswerReviewError, AnswerReviewLimits, EvidenceId, IssueCode,
    ReviewStatus, _decision as validate_decision,
)

MAX_FIXTURE_BYTES = 8_000_000


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReviewSource(_Record):
    id: EvidenceId
    text: str = Field(min_length=1, max_length=128000)


class ReviewCase(_Record):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    category: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    split: Literal["development", "held_out"]
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1, max_length=64000)
    sources: tuple[ReviewSource, ...] = Field(min_length=1, max_length=256)
    expected_acceptable: bool

    def evidence(self) -> str:
        return json.dumps([source.model_dump() for source in self.sources], ensure_ascii=False, separators=(",", ":"))

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.question.strip() or len(self.question.encode()) > 8000:
            raise ValueError("question requires 1..8000 UTF-8 bytes")
        if not self.answer.strip() or len(self.answer.encode()) > 64000:
            raise ValueError("answer requires 1..64000 UTF-8 bytes")
        if any(not source.text.strip() for source in self.sources):
            raise ValueError("source text must be nonblank")
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique within a case")
        if len(self.evidence().encode()) > 128000:
            raise ValueError("serialized source evidence exceeds 128000 UTF-8 bytes")
        return self


class ReviewFixture(_Record):
    schema_version: Literal[1]
    cases: tuple[ReviewCase, ...] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("case IDs must be unique")
        return self


class ReviewObservation(_Record):
    case_id: str
    category: str
    split: Literal["development", "held_out"]
    repeat: int
    expected_acceptable: bool
    status: ReviewStatus
    error: Literal["review_timeout", "review_provider_failed", "invalid_review"] | None = None
    issue_codes: tuple[IssueCode, ...] = ()
    revision_proposed: bool = False
    elapsed_ms: float


class ReviewMetrics(_Record):
    count: int
    approved_acceptable: int
    approved_unacceptable: int
    rejected_acceptable: int
    rejected_unacceptable: int
    abstained: int
    failed: int
    approval_precision: float | None
    false_approval_rate: float | None
    acceptable_answer_recall: float | None
    decision_coverage: float | None
    labeled_agreement: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None


class ReviewReport(_Record):
    schema_version: Literal[1] = 1
    scope: Literal["first_pass_draft_review"] = "first_pass_draft_review"
    fixture_sha256: str
    metrics: ReviewMetrics
    by_split: dict[str, ReviewMetrics]
    by_category: dict[str, ReviewMetrics]
    observations: tuple[ReviewObservation, ...]
    verified_general_accuracy: Literal[False] = False


def load_fixture(path: Path) -> ReviewFixture:
    with path.open("rb") as source:
        raw = source.read(MAX_FIXTURE_BYTES + 1)
    if len(raw) > MAX_FIXTURE_BYTES:
        raise ValueError("review fixture exceeds 8 MB")
    return ReviewFixture.model_validate_json(raw)


def summarize(rows: tuple[ReviewObservation, ...]) -> ReviewMetrics:
    def count(status: str, expected: bool) -> int:
        return sum(row.status == status and row.expected_acceptable == expected for row in rows)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    approved_good, approved_bad = count("supported", True), count("supported", False)
    rejected_good, rejected_bad = count("needs_revision", True), count("needs_revision", False)
    acceptable = sum(row.expected_acceptable for row in rows)
    latencies = sorted(row.elapsed_ms for row in rows)
    return ReviewMetrics(count=len(rows), approved_acceptable=approved_good, approved_unacceptable=approved_bad,
        rejected_acceptable=rejected_good, rejected_unacceptable=rejected_bad,
        abstained=sum(row.status == "uncertain" for row in rows), failed=sum(row.status == "unavailable" for row in rows),
        approval_precision=ratio(approved_good, approved_good + approved_bad),
        false_approval_rate=ratio(approved_bad, len(rows) - acceptable),
        acceptable_answer_recall=ratio(approved_good, acceptable),
        decision_coverage=ratio(approved_good + approved_bad + rejected_good + rejected_bad, len(rows)),
        labeled_agreement=ratio(approved_good + rejected_bad, len(rows)),
        latency_p50_ms=median(latencies) if latencies else None,
        latency_p95_ms=latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None)


async def _observe(reviewer: AnswerReviewer, case: ReviewCase, repeat: int, timeout_s: float) -> ReviewObservation:
    started = time.perf_counter()
    status: ReviewStatus = "unavailable"
    error: Literal["review_timeout", "review_provider_failed", "invalid_review"] | None = None
    issue_codes: tuple[IssueCode, ...] = ()
    revision_proposed = False
    ids = tuple(source.id for source in case.sources)
    try:
        async with asyncio.timeout(timeout_s) as budget:
            decision = await reviewer.review(case.question, case.answer, case.evidence(), ids)
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError()
        if budget.expired() or time.perf_counter() - started >= timeout_s:
            raise TimeoutError()
    except TimeoutError:
        error = "review_timeout"
    except AnswerReviewError as failure:
        error = ("review_timeout" if failure.reason == "review_timeout" else
                 "review_provider_failed" if failure.reason == "review_provider_failed" else "invalid_review")
    except Exception:
        error = "review_provider_failed"
    else:
        try:
            validated = validate_decision(decision, case.answer, ids, AnswerReviewLimits())
            status = validated.status
            issue_codes = tuple(issue.code for issue in validated.issues)
            revision_proposed = validated.revised_answer is not None
        except (TypeError, ValueError):
            error = "invalid_review"
    return ReviewObservation(case_id=case.id, category=case.category, split=case.split, repeat=repeat,
        expected_acceptable=case.expected_acceptable, status=status, error=error, issue_codes=issue_codes,
        revision_proposed=revision_proposed, elapsed_ms=round((time.perf_counter() - started) * 1000, 3))


async def evaluate_reviews(reviewer: AnswerReviewer, fixture: ReviewFixture, *, repeats: int = 1,
                           timeout_s: float = 20.0) -> ReviewReport:
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be an integer in 1..20")
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 180:
        raise ValueError("timeout must be finite in 1..180 seconds")
    # Revalidate nested instances that callers could construct without validation.
    snapshot = ReviewFixture.model_validate_json(fixture.model_dump_json())
    rows = tuple([await _observe(reviewer, case, repeat, timeout_s)
                  for repeat in range(1, repeats + 1) for case in snapshot.cases])
    return ReviewReport(fixture_sha256=hashlib.sha256(snapshot.model_dump_json().encode()).hexdigest(),
        metrics=summarize(rows), observations=rows,
        by_split={split: summarize(tuple(row for row in rows if row.split == split)) for split in sorted({row.split for row in rows})},
        by_category={category: summarize(tuple(row for row in rows if row.category == category)) for category in sorted({row.category for row in rows})})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Explicit operator-controlled OpenAI-compatible endpoint")
    parser.add_argument("--model", required=True, help="An already installed reviewer model")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    fixture = load_fixture(args.fixture)
    if args.output.exists() or args.output.resolve() == args.fixture.resolve():
        raise ValueError("output must be a new file distinct from the fixture")
    from ..providers.answer_reviewer import SelfHostedAnswerReviewer

    reviewer = SelfHostedAnswerReviewer(args.endpoint, args.model,
        api_key=os.environ.get("SCONE_ANSWER_REVIEW_API_KEY") or None, timeout=args.timeout)
    report = asyncio.run(evaluate_reviews(reviewer, fixture, repeats=args.repeats, timeout_s=args.timeout))
    payload = report.model_dump(mode="json") | {"model": args.model, "timeout_s": args.timeout, "repeats": args.repeats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(report.metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
