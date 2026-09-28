from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import pytest

from .prefill import HostedTwin, Recorder, fill, placeholder


class _Local:
    id = 'bge-base-en-v1.5'
    dim = 4
    max_input_tokens: int | None = 512

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _Remote:
    def __init__(self, width: int = 4) -> None:
        self.calls: list[list[str]] = []
        self.width = width

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0, 0.0, 1.0][:self.width] for t in texts]


class _Cache:
    def __init__(self) -> None:
        self.store: dict[str, list[float]] = {}

    def take(self, keys: Sequence[str], dim: int) -> dict[str, list[float]]:
        return {k: self.store[k] for k in keys if k in self.store}

    def keep(self, vectors: dict[str, Sequence[float]], dim: int) -> None:
        self.store.update({k: list(v) for k, v in vectors.items()})


def test_placeholders_are_unit_length_and_differ_by_text() -> None:
    a, b = placeholder('one', 8), placeholder('two', 8)
    assert math.isclose(sum(x * x for x in a), 1.0) and a != b and a == placeholder('one', 8)


def test_the_twin_wears_the_local_models_identity_and_asks_the_hosted_copy() -> None:
    remote = _Remote()
    twin = HostedTwin(_Local(), remote)
    assert (twin.id, twin.dim, twin.max_input_tokens, twin.count_tokens('a b')) == ('bge-base-en-v1.5', 4, 512, 2)
    assert not hasattr(twin, 'embed_queries')  # the local model has none; a query must embed as a document does
    assert asyncio.run(twin.embed(['abc'])) == [[3.0, 0.0, 0.0, 1.0]] and remote.calls == [['abc']]


def test_the_recorder_keeps_documents_and_queries_apart() -> None:
    recorder = Recorder(_Local())
    asyncio.run(recorder.embed(['d1', 'd2']))
    asyncio.run(recorder.embed_queries(['q1']))
    assert recorder.documents == {'d1', 'd2'} and recorder.queries == {'q1'}


def test_fill_sends_only_what_the_cache_lacks_under_the_identity_given() -> None:
    from scone_memory.ingestion.embedding_cache import cache_key

    cache, remote = _Cache(), _Remote()
    cache.keep({cache_key('m', 4, 'held'): [0.0, 0.0, 0.0, 1.0]}, 4)
    sent = asyncio.run(fill(cache, remote, 'm', 4, ['held', 'new', 'new', 'other'], batch=1))
    assert sent == 2 and sorted(t for call in remote.calls for t in call) == ['new', 'other']
    assert cache.store[cache_key('m', 4, 'new')] == [3.0, 0.0, 0.0, 1.0]
    assert cache_key('m:query-cache', 4, 'new') not in cache.store


def test_fill_refuses_vectors_of_the_wrong_width() -> None:
    with pytest.raises(ValueError):
        asyncio.run(fill(_Cache(), _Remote(width=3), 'm', 4, ['x']))
