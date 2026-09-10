"""Duplicate review backed by the framework's existing scoped search indexes."""
from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ..core.models import Episode
from ..core.ports import TextFilter
from ..core.validation import check_space
from ..memory.engine import MemoryEngine
from .detector import DocumentDuplicateDetector
from .types import (
    CandidateBatch, DeduplicationConfig, DocumentRevision, DuplicateReport,
    PassageEmbedding, SemanticCandidate,
)


class MemoryCandidateProvider:
    """Bounded lexical/ANN searches; no corpus scan or corpus re-embedding.

    Candidate completeness is always partial: top-k lexical/ANN retrieval is
    not a proof that every copied passage in the memory space was examined.
    The caller must authorize the space before using this native interface.
    """

    def __init__(self, memory: MemoryEngine) -> None:
        self.memory = memory

    async def candidates(
        self, document: DocumentRevision, *,
        query_embeddings: tuple[PassageEmbedding, ...],
        embedder_id: str | None, limit: int,
    ) -> CandidateBatch:
        check_space(document.scope)
        if not 1 <= limit <= 256:
            raise ValueError("candidate limit must be between 1 and 256")
        if query_embeddings:
            return await self._semantic(document, query_embeddings, embedder_id, limit)
        queries = self._queries(document.text)
        rows = await asyncio.gather(*(
            self.memory.documents.search_text(document.scope, query, limit, TextFilter())
            for query in queries
        ))
        # Interleave rankings so the first query cannot consume the whole budget.
        ids = list(dict.fromkeys(
            row[index][0] for index in range(limit) for row in rows if index < len(row)
        ))[:limit]
        sources = await self._sources(document.scope, ids)
        return CandidateBatch(documents=tuple(sources.values()), truncated=True)

    @staticmethod
    def _queries(text: str) -> tuple[str, ...]:
        if not text.strip():
            return ()
        width, count = 256, min(8, max(1, (len(text) + 255) // 256))
        last = max(0, len(text) - width)
        offsets = [round(last * index / max(1, count - 1)) for index in range(count)]
        return tuple(dict.fromkeys(text[start:start + width] for start in offsets))

    async def _semantic(
        self, document: DocumentRevision, embeddings: tuple[PassageEmbedding, ...],
        embedder_id: str | None, limit: int,
    ) -> CandidateBatch:
        if embedder_id != self.memory.embedder.id:
            raise ValueError("query embedder differs from the memory index embedder")
        if len(embeddings) > 256:
            raise ValueError("semantic query batch exceeds 256 passages")
        semaphore = asyncio.Semaphore(4)

        async def search(passage: PassageEmbedding) -> list[tuple[int, float]]:
            if len(passage.vector) != self.memory.embedder.dim:
                raise ValueError("query vector dimension differs from memory index")
            async with semaphore:
                return await self.memory.vectors.search(document.scope, passage.vector, limit)

        rows = await asyncio.gather(*(search(passage) for passage in embeddings))
        ranked = sorted(
            ((score, index, chunk_id) for index, row in enumerate(rows) for chunk_id, score in row),
            reverse=True,
        )[:limit]
        sources = await self._sources(document.scope, list(dict.fromkeys(row[2] for row in ranked)))
        matches = tuple(
            SemanticCandidate(sources[chunk_id], index, score, embedder_id)
            for score, index, chunk_id in ranked if chunk_id in sources
        )
        return CandidateBatch(semantic_matches=matches, truncated=True)

    async def _sources(self, space: str, ids: Sequence[int]) -> dict[int, DocumentRevision]:
        if not ids:
            return {}
        chunks = await self.memory.documents.get_chunks(space, ids)
        requested = set(ids)
        for chunk in chunks:
            if chunk.space != space or chunk.chunk_id not in requested:
                raise ValueError("candidate chunk escaped the requested scope")

        semaphore = asyncio.Semaphore(4)

        async def load(episode_id: int) -> tuple[int, Episode | None]:
            async with semaphore:
                return episode_id, await self.memory.documents.get_episode(space, episode_id)

        episodes = dict(await asyncio.gather(*(load(episode_id) for episode_id in
                                               dict.fromkeys(chunk.episode_id for chunk in chunks))))
        result: dict[int, DocumentRevision] = {}
        for chunk in chunks:
            episode = episodes[chunk.episode_id]
            if episode is None:
                continue  # A concurrent forget may remove the owner after search.
            if episode.space != space:
                raise ValueError("candidate episode escaped the requested scope")
            raw = episode.content.encode("utf-8")
            if not 0 <= chunk.start <= chunk.end <= len(raw) or raw[chunk.start:chunk.end] != chunk.text.encode("utf-8"):
                raise ValueError("candidate chunk does not match its retained original")
            result[chunk.chunk_id] = DocumentRevision(
                space, f"episode:{episode.episode_id}", episode.content_hash,
                chunk.text, chunk.start, str(chunk.chunk_id),
            )
        return result


class MemoryDuplicateInspector:
    """Reusable review service; inspection never mutates originals or indexes.

    Keep one inspector per configured engine to reuse its bounded query-vector
    cache. Reports carry source episode/chunk references and UTF-8 byte spans.
    """

    def __init__(self, memory: MemoryEngine, config: DeduplicationConfig | None = None) -> None:
        self.memory = memory
        self.provider = MemoryCandidateProvider(memory)
        self.detector = DocumentDuplicateDetector(config, embedder=memory.embedder)

    async def inspect(self, document: DocumentRevision) -> DuplicateReport:
        check_space(document.scope)
        return await self.detector.inspect(document, self.provider)

    async def inspect_episode(self, space: str, episode_id: int) -> DuplicateReport:
        check_space(space)
        episode = await self.memory.documents.get_episode(space, episode_id)
        if episode is None or episode.space != space:
            raise ValueError("document episode not found in this space")
        return await self.inspect(DocumentRevision(
            space, f"episode:{episode.episode_id}", episode.content_hash, episode.content,
        ))

    def clear_cache(self, *, scope: str | None = None) -> None:
        self.detector.clear_cache(scope=scope)
