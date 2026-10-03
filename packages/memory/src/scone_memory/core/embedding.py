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


@runtime_checkable
class QueryCacheText(Protocol):
    """Explicit document-encoding input equivalent to a query encoding."""
    def query_cache_text(self, text: str) -> str | None: ...


def query_cache_text(embedder: Embedder, text: str) -> str | None:
    if isinstance(embedder, QueryCacheText):
        return embedder.query_cache_text(text)
    # Custom query encoders cannot reuse document vectors without an explicit
    # equivalence contract, even when their input strings match.
    return None if isinstance(embedder, QueryEmbedder) else text
