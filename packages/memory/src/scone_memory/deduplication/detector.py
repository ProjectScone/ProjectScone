"""Read-only copied-content and optional embedding-based paraphrase analysis."""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import OrderedDict
from dataclasses import replace
from time import perf_counter
from typing import Literal

from ..core.ports import Embedder
from ..embedders.hash import HashEmbedder
from .lexical import CopiedSpanMatcher, covered_bytes
from .types import (
    CandidateProvider, CopiedSpan, DeduplicationConfig, DeduplicationMetrics,
    DocumentRevision, DuplicateReport, PassageEmbedding, SemanticMatch,
)

CacheKey = tuple[str, str, str, str, int, str]
SemanticStatus = Literal["disabled", "unavailable", "complete", "partial", "failed"]


class DocumentDuplicateDetector:
    """Inspect bounded candidates without changing documents or their indexes.

    Semantic matches are review candidates, not proof of equivalence. Disabling
    semantics avoids embedding and ANN calls. The bounded LRU stores only query
    vectors; candidate retrieval always runs again so changes in access rights
    or indexed revisions cannot be hidden by cached search results.
    """

    def __init__(self, config: DeduplicationConfig | None = None, *, embedder: Embedder | None = None) -> None:
        self.config = config or DeduplicationConfig()
        self.embedder = embedder
        self._embeddings: OrderedDict[CacheKey, tuple[float, ...]] = OrderedDict()
        self._batch_locks: dict[CacheKey, tuple[asyncio.Lock, int]] = {}
        self._embedding_slots = asyncio.Semaphore(self.config.max_embedding_concurrency)
        self._cache_epoch = 0

    def clear_cache(self, *, scope: str | None = None) -> None:
        """Drop retained vectors, for example when a scope is forgotten."""
        self._cache_epoch += 1
        if scope is None:
            self._embeddings.clear()
            return
        for key in list(self._embeddings):
            if key[0] == scope:
                del self._embeddings[key]

    async def inspect(self, document: DocumentRevision, provider: CandidateProvider) -> DuplicateReport:
        started = perf_counter()
        document_bytes = len(document.text.encode("utf-8"))
        if document_bytes > self.config.max_document_bytes:
            raise ValueError("document byte limit exceeded")
        matches, limitations, checked = await self._lexical(document, provider)
        lexical_ms = (perf_counter() - started) * 1000
        semantic, status, semantic_limits, metrics = await self._semantic(document, provider)
        limitations.extend(semantic_limits)
        copied = covered_bytes(matches)
        fraction = copied / document_bytes if document_bytes else 0.0
        return DuplicateReport(
            document.source_id, document.revision, fraction, copied, document_bytes,
            tuple(matches), semantic, fraction >= self.config.overlap_threshold or bool(semantic),
            not limitations, status, tuple(dict.fromkeys(limitations)),
            replace(metrics, lexical_ms=lexical_ms, total_ms=(perf_counter() - started) * 1000,
                    candidates_checked=checked),
            hashlib.sha256(document.text.encode()).hexdigest(), document.scope, document.byte_offset,
        )

    async def _lexical(self, document: DocumentRevision, provider: CandidateProvider) -> tuple[list[CopiedSpan], list[str], int]:
        if not document.text.strip():
            return [], [], 0
        batch = await asyncio.wait_for(provider.candidates(document, query_embeddings=(), embedder_id=None,
                                                           limit=self.config.max_candidates), self.config.timeout_seconds)
        limitations: list[str] = []
        if batch.truncated or len(batch.documents) > self.config.max_candidates:
            limitations.append("candidate_limit")
        matcher = CopiedSpanMatcher(document, self.config.min_match_chars, self.config.max_match_operations)
        matches: list[CopiedSpan] = []
        checked, candidate_bytes = 0, 0
        seen: set[tuple[str, str, str | None, int, str]] = set()
        for source in batch.documents[:self.config.max_candidates]:
            self._check_scope(document, source)
            if source.source_id == document.source_id and source.revision == document.revision:
                continue
            size = len(source.text.encode("utf-8"))
            candidate_bytes += size
            if size > self.config.max_document_bytes or candidate_bytes > self.config.max_candidate_bytes:
                limitations.append("candidate_byte_limit")
                continue
            identity = (source.source_id, source.revision, source.chunk_id, source.byte_offset, source.text)
            if identity in seen:
                continue
            seen.add(identity)
            checked += 1
            matches.extend(matcher.compare(source))
            if matcher.truncated:
                limitations.append("lexical_work_limit")
                break
        return matches, limitations, checked

    @staticmethod
    def _check_scope(document: DocumentRevision, source: DocumentRevision) -> None:
        if source.scope != document.scope:
            raise ValueError("candidate scope differs from document scope")

    def _passages(self, document: DocumentRevision) -> tuple[list[tuple[int, int, str]], bool]:
        passages: list[tuple[int, int, str]] = []
        offsets = [document.byte_offset]
        for char in document.text:
            offsets.append(offsets[-1] + len(char.encode("utf-8")))
        for paragraph in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", document.text, re.DOTALL):
            for start in range(paragraph.start(), paragraph.end(), self.config.semantic_passage_chars):
                if len(passages) == self.config.max_semantic_passages:
                    return passages, True
                end = min(start + self.config.semantic_passage_chars, paragraph.end())
                passages.append((offsets[start], offsets[end], document.text[start:end]))
        return passages, False

    async def _vectors(self, document: DocumentRevision, passages: list[tuple[int, int, str]], embedder: Embedder) -> tuple[tuple[PassageEmbedding, ...], int, int]:
        # Identical concurrent requests share a batch; unrelated misses can run
        # concurrently under the configured provider admission limit.
        batch_key = (document.scope, document.source_id, document.revision, embedder.id,
                     embedder.dim, hashlib.sha256(document.text.encode()).hexdigest())
        lock, users = self._batch_locks.get(batch_key, (asyncio.Lock(), 0))
        self._batch_locks[batch_key] = (lock, users + 1)
        try:
            async with lock:
                return await self._cached_vectors(document, passages, embedder)
        finally:
            _, users = self._batch_locks[batch_key]
            if users == 1:
                del self._batch_locks[batch_key]
            else:
                self._batch_locks[batch_key] = (lock, users - 1)

    async def _cached_vectors(self, document: DocumentRevision, passages: list[tuple[int, int, str]], embedder: Embedder) -> tuple[tuple[PassageEmbedding, ...], int, int]:
        vectors: dict[CacheKey, tuple[float, ...]] = {}
        pending: dict[CacheKey, str] = {}
        keys: list[CacheKey] = []
        hits = 0
        for _, _, text in passages:
            key = (document.scope, document.source_id, document.revision, embedder.id, embedder.dim,
                   hashlib.sha256(text.encode()).hexdigest())
            keys.append(key)
            if key in self._embeddings:
                vectors[key] = self._embeddings[key]
                self._embeddings.move_to_end(key)
                hits += 1
            else:
                pending[key] = text
        if pending:
            epoch = self._cache_epoch
            async with self._embedding_slots:
                returned = await embedder.embed(list(pending.values()))
            if len(returned) != len(pending):
                raise ValueError("embedder returned an unexpected batch length")
            for key, vector in zip(pending, returned):
                frozen = tuple(vector)
                if len(frozen) != embedder.dim or not all(math.isfinite(value) for value in frozen) or not any(frozen):
                    raise ValueError("embedder returned an invalid vector")
                vectors[key] = frozen
                if self.config.embedding_cache_size and self._cache_epoch == epoch:
                    self._embeddings[key] = frozen
                    while len(self._embeddings) > self.config.embedding_cache_size:
                        self._embeddings.popitem(last=False)
        return tuple(PassageEmbedding(start, end, vectors[key]) for (start, end, _), key in zip(passages, keys)), len(pending), hits

    async def _semantic(self, document: DocumentRevision, provider: CandidateProvider) -> tuple[tuple[SemanticMatch, ...], SemanticStatus, list[str], DeduplicationMetrics]:
        metrics = DeduplicationMetrics()
        if not self.config.semantic_enabled:
            return (), "disabled", [], metrics
        embedder = self.embedder
        if embedder is None or isinstance(embedder, HashEmbedder):
            return (), "unavailable", ["semantic_embedder_unavailable"], metrics
        passages, truncated = self._passages(document)
        if not passages:
            return (), "complete", [], metrics
        limitations = ["semantic_passage_limit"] if truncated else []
        started = perf_counter()
        try:
            vectors, embedded, hits = await asyncio.wait_for(self._vectors(document, passages, embedder), self.config.timeout_seconds)
            metrics = replace(metrics, embedding_ms=(perf_counter() - started) * 1000, embedded_passages=embedded, embedding_cache_hits=hits)
            started = perf_counter()
            batch = await asyncio.wait_for(provider.candidates(document, query_embeddings=vectors, embedder_id=embedder.id,
                                                               limit=self.config.max_candidates), self.config.timeout_seconds)
            metrics = replace(metrics, semantic_search_ms=(perf_counter() - started) * 1000)
        except Exception:
            return (), "failed", limitations + ["semantic_processing_failed"], metrics
        if batch.truncated or len(batch.semantic_matches) > self.config.max_candidates:
            limitations.append("semantic_candidate_limit")
        matches: list[SemanticMatch] = []
        seen: set[tuple[str, str, str | None, int, int]] = set()
        examined_sources: set[tuple[str, str, str | None, int, str]] = set()
        candidate_bytes = 0
        for candidate in batch.semantic_matches[:self.config.max_candidates]:
            source = candidate.source
            self._check_scope(document, source)
            if candidate.embedder_id != embedder.id or not math.isfinite(candidate.score) or not -1 <= candidate.score <= 1:
                raise ValueError("semantic candidate has an invalid score or embedder identity")
            if not 0 <= candidate.query_index < len(vectors):
                raise ValueError("semantic candidate query_index is out of bounds")
            if source.source_id == document.source_id and source.revision == document.revision:
                continue
            if candidate.score < self.config.semantic_threshold or not source.text.strip():
                continue
            source_bytes = len(source.text.encode())
            source_identity = (source.source_id, source.revision, source.chunk_id, source.byte_offset, source.text)
            if source_identity not in examined_sources:
                examined_sources.add(source_identity)
                candidate_bytes += source_bytes
            if source_bytes > self.config.max_document_bytes or candidate_bytes > self.config.max_candidate_bytes:
                limitations.append("semantic_candidate_byte_limit")
                continue
            identity = (source.source_id, source.revision, source.chunk_id, source.byte_offset, candidate.query_index)
            if identity in seen:
                continue
            seen.add(identity)
            query = vectors[candidate.query_index]
            matches.append(SemanticMatch(query.start, query.end, source.source_id, source.revision,
                                         source.byte_offset, source.byte_offset + source_bytes,
                                         source.chunk_id, candidate.score, candidate.embedder_id))
        return tuple(matches), "partial" if limitations else "complete", limitations, metrics
