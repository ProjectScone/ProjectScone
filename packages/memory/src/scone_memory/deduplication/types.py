"""Bounded, storage-neutral duplicate analysis contracts.

All ranges are half-open UTF-8 byte offsets in the original source. Providers
must authorize candidate access and scope their indexes before returning rows.
A score is cosine similarity from the named embedder, never copied coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol


@dataclass(frozen=True)
class DocumentRevision:
    scope: str
    source_id: str
    revision: str
    text: str
    byte_offset: int = 0
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.scope or not self.source_id or not self.revision:
            raise ValueError("document scope, source_id and revision are required")
        if self.byte_offset < 0:
            raise ValueError("document byte_offset must be nonnegative")


@dataclass(frozen=True)
class PassageEmbedding:
    start: int
    end: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SemanticCandidate:
    source: DocumentRevision
    query_index: int
    score: float
    embedder_id: str


@dataclass(frozen=True)
class CandidateBatch:
    documents: tuple[DocumentRevision, ...] = ()
    semantic_matches: tuple[SemanticCandidate, ...] = ()
    truncated: bool = False


class CandidateProvider(Protocol):
    async def candidates(
        self, document: DocumentRevision, *,
        query_embeddings: tuple[PassageEmbedding, ...],
        embedder_id: str | None, limit: int,
    ) -> CandidateBatch:
        """Empty embeddings request lexical candidates; otherwise ANN candidates.

        Return at most ``limit`` rows per collection and mark ``truncated`` if
        any candidate limit, approximate search, or partial index limits recall.
        Never re-embed the corpus: use stored vectors from ``embedder_id``.
        """
        ...


@dataclass(frozen=True)
class DeduplicationConfig:
    overlap_threshold: float = 0.25
    semantic_enabled: bool = True
    semantic_threshold: float = 0.90
    min_match_chars: int = 24
    max_candidates: int = 32
    max_document_bytes: int = 262_144
    max_candidate_bytes: int = 2_097_152
    max_match_operations: int = 1_000_000
    semantic_passage_chars: int = 2048
    max_semantic_passages: int = 32
    embedding_cache_size: int = 256
    max_embedding_concurrency: int = 4
    timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        for name in ("overlap_threshold", "semantic_threshold"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in ("min_match_chars", "max_candidates", "max_document_bytes",
                     "max_candidate_bytes", "max_match_operations",
                     "semantic_passage_chars", "max_semantic_passages", "max_embedding_concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_cache_size < 0:
            raise ValueError("embedding_cache_size must be nonnegative")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")


@dataclass(frozen=True)
class CopiedSpan:
    start: int
    end: int
    source_id: str
    revision: str
    source_start: int
    source_end: int
    chunk_id: str | None
    kind: Literal["exact", "normalized", "copied"]


@dataclass(frozen=True)
class SemanticMatch:
    start: int
    end: int
    source_id: str
    revision: str
    source_start: int
    source_end: int
    chunk_id: str | None
    score: float
    embedder_id: str


@dataclass(frozen=True)
class DeduplicationMetrics:
    total_ms: float = 0.0
    lexical_ms: float = 0.0
    embedding_ms: float = 0.0
    semantic_search_ms: float = 0.0
    candidates_checked: int = 0
    embedded_passages: int = 0
    embedding_cache_hits: int = 0


@dataclass(frozen=True)
class DuplicateReport:
    source_id: str
    revision: str
    copied_fraction: float
    copied_bytes: int
    document_bytes: int
    matches: tuple[CopiedSpan, ...]
    semantic_matches: tuple[SemanticMatch, ...]
    requires_review: bool
    complete: bool
    semantic_status: Literal["disabled", "unavailable", "complete", "partial", "failed"]
    limitations: tuple[str, ...]
    metrics: DeduplicationMetrics
    input_sha256: str
    scope: str
    byte_offset: int
