"""Fill the embedding cache before ranking, so a large run is not bound by one CPU.

A LongMemEval-M item holds about 480 sessions; embedding them for Scone and
LlamaIndex takes minutes per item on this machine's CPU, and a 100-item
sample shares almost no sessions between items. So the ranking runs twice:

1. a dry run with :class:`Recorder` in the model's place, which keeps every
   text each system asks to embed (documents and queries apart) and answers
   with placeholder vectors;
2. those texts are embedded concurrently by a hosted copy of the same model
   and written to the cache under the keys ``CachedEmbedder`` reads;
3. the real ranking runs over ``CachedEmbedder(HostedTwin)`` and is served
   from the cache.

Which texts get embedded does not depend on the vectors (chunking happens
before embedding, and the query is the question), so the dry run asks for
exactly the texts the real run asks for. A text it missed is not lost: the
real run sends it to the hosted model, and the cache record counts it.

:class:`HostedTwin` carries the local model's id, width, window and
tokenizer, so both systems chunk exactly as they would locally and the cache
entries sit under the local model's name. Vectors come from the hosted copy,
which agreed with local ``bge-base-en-v1.5`` at cosine 0.99999 on the texts
checked before this was written.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from collections.abc import Callable, Sequence
from typing import Protocol


class _Local(Protocol):
    id: str
    dim: int
    max_input_tokens: int | None

    def count_tokens(self, text: str) -> int: ...


class _Remote(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HostedTwin:
    """The local model as both systems see it; the vectors come from a hosted copy.

    No ``embed_queries``: the local model has none, so a query is embedded as a
    document is, and the twin must not differ from it there."""

    def __init__(self, local: _Local, remote: _Remote) -> None:
        self.id = local.id
        self.dim = local.dim
        self.max_input_tokens = local.max_input_tokens
        self.count_tokens: Callable[[str], int] = local.count_tokens
        self._remote = remote

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._remote.embed(texts)


def placeholder(text: str, dim: int) -> list[float]:
    """A fixed unit vector for ``text``: never zero, never the same for two texts."""
    raw: list[float] = []
    counter = 0
    while len(raw) < dim:
        digest = hashlib.sha256(f'{counter}:{text}'.encode()).digest()
        raw.extend(value / 2**31 - 1.0 for value in struct.unpack('>8I', digest))
        counter += 1
    norm = math.sqrt(sum(x * x for x in raw[:dim])) or 1.0
    return [x / norm for x in raw[:dim]]


class Recorder:
    """Stands in for the model during the dry run and keeps what it was asked."""

    def __init__(self, local: _Local) -> None:
        self.id = local.id
        self.dim = local.dim
        self.max_input_tokens = local.max_input_tokens
        self.count_tokens: Callable[[str], int] = local.count_tokens
        self.documents: set[str] = set()
        self.queries: set[str] = set()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents.update(texts)
        return [placeholder(text, self.dim) for text in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.queries.update(texts)
        return [placeholder(text, self.dim) for text in texts]


class _Cache(Protocol):
    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]: ...

    def keep(self, vectors: dict[str, Sequence[float]], dim: int) -> None: ...


async def fill(cache: _Cache, remote: _Remote, identity: str, dim: int, texts: Sequence[str], *,
               batch: int = 96, concurrency: int = 12, progress: Callable[[int, int], None] | None = None) -> int:
    """Embeds the texts the cache lacks under ``identity``; returns how many it sent."""
    from scone_memory.ingestion.embedding_cache import cache_key

    by_key = {cache_key(identity, dim, text): text for text in texts}
    held = set(cache.take(list(by_key), dim))
    wanted = [(key, text) for key, text in by_key.items() if key not in held]
    gate = asyncio.Semaphore(concurrency)
    sent = 0

    async def one(chunk: Sequence[tuple[str, str]]) -> None:
        nonlocal sent
        async with gate:
            vectors = await remote.embed([text for _, text in chunk])
        if len(vectors) != len(chunk) or any(len(v) != dim for v in vectors):
            raise ValueError(f'the hosted model returned {len(vectors)} vectors for {len(chunk)} texts, or the wrong width')
        cache.keep({key: vector for (key, _), vector in zip(chunk, vectors, strict=True)}, dim)
        sent += len(chunk)
        if progress:
            progress(sent, len(wanted))

    await asyncio.gather(*(one(wanted[i:i + batch]) for i in range(0, len(wanted), batch)))
    return sent
