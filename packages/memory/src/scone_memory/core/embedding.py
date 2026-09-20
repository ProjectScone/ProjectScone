"""Optional query encoding without breaking existing document embedders."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .ports import Embedder


@runtime_checkable
class QueryEmbedder(Protocol):
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


async def embed_queries(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    if isinstance(embedder, QueryEmbedder):
        return await embedder.embed_queries(texts)
    return await embedder.embed(texts)
