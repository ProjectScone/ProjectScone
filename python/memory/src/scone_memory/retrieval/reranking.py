"""Optional bounded reranking of retained, authorized original passages.

Adapters are supplied by the host. This module imports no model/backend and
never treats rank scores as confidence. Candidate/page separation and pruning
before reranking follow the architecture inspected in reference/ragflow; this
implementation is original Scone code, with no RAGFlow code or dependency.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Protocol

from ..core.errors import InvalidInput
from ..core.models import Chunk, Episode, RerankTrace
from ..core.ports import TextFilter

MAX_CANDIDATE_LIMIT = 1000
MAX_RERANK_LIMIT = 128
MAX_RERANK_BYTES = 256000
MAX_RERANK_TIMEOUT = 10.0


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: int
    episode_id: int
    text: str
    source: str | None
    created_at: str
    baseline_score: float
    similarity: float | None
    lanes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RerankScore:
    chunk_id: int
    score: float


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: tuple[RerankCandidate, ...]) -> Sequence[RerankScore]: ...


@dataclass(frozen=True)
class RerankOutcome:
    ordered_ids: tuple[int, ...]
    scores: dict[int, float]
    trace: RerankTrace
    failure: str | None = None


class InvalidRerankOutput(ValueError):
    """A ranker did not return exactly one finite score per supplied chunk."""


def validate_candidate_limit(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or not 1 <= value <= MAX_CANDIDATE_LIMIT):
        raise InvalidInput(f"candidate_limit must be an integer from 1 to {MAX_CANDIDATE_LIMIT}")
    return value


def validate_rerank_options(limit: int, max_bytes: int, timeout: float) -> None:
    if type(limit) is not int or not 1 <= limit <= MAX_RERANK_LIMIT:
        raise InvalidInput(f"rerank_limit must be an integer from 1 to {MAX_RERANK_LIMIT}")
    if type(max_bytes) is not int or not 512 <= max_bytes <= MAX_RERANK_BYTES:
        raise InvalidInput(f"rerank_max_bytes must be an integer from 512 to {MAX_RERANK_BYTES}")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= MAX_RERANK_TIMEOUT:
        raise InvalidInput(f"rerank_timeout must be finite and greater than zero, at most {MAX_RERANK_TIMEOUT}")


def _payload_bytes(query: str, candidates: Sequence[RerankCandidate]) -> int:
    return len(json.dumps({"query": query, "candidates": [asdict(candidate) for candidate in candidates]},
                          ensure_ascii=False, separators=(",", ":")).encode())


def _scores(values: Sequence[RerankScore], candidates: Sequence[RerankCandidate]) -> dict[int, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != len(candidates):
        raise InvalidRerankOutput()
    expected = {candidate.chunk_id for candidate in candidates}
    scores: dict[int, float] = {}
    for index in range(len(candidates)):
        value = values[index]
        if (not isinstance(value, RerankScore) or type(value.chunk_id) is not int
                or value.chunk_id not in expected or value.chunk_id in scores
                or isinstance(value.score, bool) or not isinstance(value.score, (int, float))):
            raise InvalidRerankOutput()
        try:
            score = float(value.score)
        except (ValueError, OverflowError):
            raise InvalidRerankOutput() from None
        if not math.isfinite(score):
            raise InvalidRerankOutput()
        scores[value.chunk_id] = score
    return scores


async def rerank_candidates(reranker: Reranker, query: str, candidates: Sequence[RerankCandidate], *,
                            limit: int, max_bytes: int, timeout: float) -> RerankOutcome:
    validate_rerank_options(limit, max_bytes, timeout)
    started = time.perf_counter()
    selected: list[RerankCandidate] = []
    payload_bytes = 0
    for candidate in candidates[:limit]:
        size = _payload_bytes(query, [*selected, candidate])
        if size <= max_bytes:
            selected.append(candidate)
            payload_bytes = size
    trace = RerankTrace(status="empty", ordering="fusion", candidates_considered=len(candidates),
                        candidates_sent=len(selected), candidates_omitted=len(candidates) - len(selected),
                        payload_bytes=payload_bytes, duration_ms=0)
    if not selected:
        trace.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        return RerankOutcome((), {}, trace)
    try:
        values = await asyncio.wait_for(reranker.rerank(query, tuple(selected)), timeout=timeout)
        scores = _scores(values, selected)
        # Python's stable sort retains the original candidate order for ties.
        ordered = tuple(candidate.chunk_id for candidate in sorted(selected, key=lambda item: -scores[item.chunk_id]))
        trace.status, trace.ordering = "applied", "rerank"
        return RerankOutcome(ordered, scores, trace)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        trace.status = "failed"
        return RerankOutcome((), {}, trace, type(error).__name__)
    finally:
        trace.duration_ms = round((time.perf_counter() - started) * 1000, 3)


def candidate_is_retained(chunk: "Chunk", episode: "Episode", space: str, scope: "TextFilter") -> bool:
    """Check every authorization/scope dimension before exposing original text."""
    if chunk.space != space or episode.space != space or chunk.episode_id != episode.episode_id:
        return False
    if not set(scope.tags).issubset(episode.tags) or any(episode.metadata.get(key) != value for key, value in scope.where.items()):
        return False
    if scope.conditions is not None and not scope.conditions.matches(episode.metadata):
        return False
    if scope.kind is not None and episode.kind != scope.kind:
        return False
    if scope.source_prefix is not None and (episode.source is None or not episode.source.startswith(scope.source_prefix)):
        return False
    if scope.since is not None and episode.created_at < scope.since:
        return False
    if scope.until is not None and episode.created_at > scope.until:
        return False
    if scope.as_of is not None and (episode.created_at > scope.as_of or chunk.created_at > scope.as_of):
        return False
    return chunk.start >= 0 and chunk.end >= chunk.start and episode.content.encode()[chunk.start:chunk.end] == chunk.text.encode()
