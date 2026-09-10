"""Explicit review decisions produce search projections; originals are untouched.

Persist a trusted report alongside its revision before exposing review actions.
The digest detects stale inputs; it is not a signature for untrusted client-
constructed reports. Downstream index writers must retain these source offsets
instead of joining disjoint segments into an invented contiguous passage.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .types import CopiedSpan, DocumentRevision, DuplicateReport, SemanticMatch

ReviewAction = Literal["keep", "suppress_copied_passages", "exclude_document"]
DEFAULT_DOCUMENTATION_URL = "https://github.com/ProjectScone/ProjectScone/blob/main/packages/memory/docs/document-deduplication.md"


@dataclass(frozen=True)
class RetainedPassage:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SearchProjection:
    scope: str
    source_id: str
    revision: str
    input_sha256: str
    action: ReviewAction
    retained: tuple[RetainedPassage, ...]
    excluded_bytes: int
    document_bytes: int


@dataclass(frozen=True)
class ReviewSource:
    source_id: str
    revision: str
    chunk_id: str | None
    source_start: int
    source_end: int


@dataclass(frozen=True)
class DuplicateNotification:
    code: str
    scope: str
    source_id: str
    revision: str
    input_sha256: str
    requires_review: bool
    message: str
    documentation_url: str
    copied_fraction: float
    semantic_candidates: int
    sources: tuple[ReviewSource, ...]
    available_actions: tuple[ReviewAction, ...]


def review_notification(
    report: DuplicateReport, *, documentation_url: str = DEFAULT_DOCUMENTATION_URL,
) -> DuplicateNotification | None:
    if not report.requires_review:
        return None
    actions: tuple[ReviewAction, ...] = (
        ("keep", "suppress_copied_passages", "exclude_document")
        if report.matches else ("keep", "exclude_document")
    )
    matches: tuple[CopiedSpan | SemanticMatch, ...] = (*report.matches, *report.semantic_matches)
    references = tuple(dict.fromkeys(
        ReviewSource(match.source_id, match.revision, match.chunk_id, match.source_start, match.source_end)
        for match in matches
    ))
    message = (
        f"This document has {report.copied_fraction:.1%} copied text coverage by UTF-8 bytes"
        f" and {len(report.semantic_matches)} possible paraphrase matches. "
        "Review the source passages and documentation before choosing what appears in search. "
        "The original document is retained."
    )
    return DuplicateNotification(
        "document_duplicate_review", report.scope, report.source_id, report.revision,
        report.input_sha256, True, message, documentation_url, report.copied_fraction,
        len(report.semantic_matches), references, actions,
    )


def _validated_ranges(document: DocumentRevision, report: DuplicateReport) -> tuple[bytes, list[tuple[int, int]]]:
    raw = document.text.encode("utf-8")
    if (
        document.source_id != report.source_id or document.revision != report.revision
        or document.scope != report.scope or document.byte_offset != report.byte_offset
        or len(raw) != report.document_bytes or hashlib.sha256(raw).hexdigest() != report.input_sha256
    ):
        raise ValueError("duplicate report does not match this document revision and content")
    ranges: list[tuple[int, int]] = []
    for match in report.matches:
        start, end = match.start - document.byte_offset, match.end - document.byte_offset
        if not 0 <= start < end <= len(raw):
            raise ValueError("copied range lies outside the original document")
        try:
            raw[start:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("copied range splits a UTF-8 character") from exc
        ranges.append((start, end))
    return raw, ranges


def _retained_ranges(length: int, excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    retained: list[tuple[int, int]] = []
    previous = 0
    for start, end in sorted(excluded):
        if start > previous:
            retained.append((previous, start))
        previous = max(previous, end)
    if previous < length:
        retained.append((previous, length))
    return retained


def project_document(
    document: DocumentRevision, report: DuplicateReport, *, action: ReviewAction = "keep",
) -> SearchProjection:
    """Return reviewed search segments with their unchanged original offsets.

    ``suppress_copied_passages`` acts only on measured lexical copies. Semantic
    similarity never identifies bytes safe to remove. ``exclude_document`` must
    be explicitly selected and still does not delete any source or index row.
    """
    if action not in ("keep", "suppress_copied_passages", "exclude_document"):
        raise ValueError("unknown duplicate review action")
    raw, copied = _validated_ranges(document, report)
    if action == "exclude_document":
        ranges: list[tuple[int, int]] = []
    else:
        ranges = _retained_ranges(len(raw), copied if action == "suppress_copied_passages" else [])
    passages = tuple(RetainedPassage(start + document.byte_offset, end + document.byte_offset,
                                    raw[start:end].decode("utf-8")) for start, end in ranges)
    excluded_bytes = len(raw) - sum(end - start for start, end in ranges)
    return SearchProjection(document.scope, document.source_id, document.revision, report.input_sha256,
                            action, passages, excluded_bytes, len(raw))
