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
    sent, shortened = asyncio.run(fill(cache, remote, 'm', 4, ['held', 'new', 'new', 'other'], batch=1))
    assert sent == 2 and shortened == 0 and sorted(t for call in remote.calls for t in call) == ['new', 'other']
    assert cache.store[cache_key('m', 4, 'new')] == [3.0, 0.0, 0.0, 1.0]
    assert cache_key('m:query-cache', 4, 'new') not in cache.store


def test_fill_refuses_vectors_of_the_wrong_width() -> None:
    with pytest.raises(ValueError):
        asyncio.run(fill(_Cache(), _Remote(width=3), 'm', 4, ['x']))


class _Encoding:
    def __init__(self, text: str) -> None:
        words, position = text.split(' '), 0
        self.offsets = [(0, 0)]  # the leading special token
        for word in words:
            self.offsets.append((position, position + len(word)))
            position += len(word) + 1
        self.offsets.append((0, 0))  # the trailing special token


class _Counter:
    def encode(self, text: str) -> _Encoding:
        return _Encoding(text)


class _WindowedLocal(_Local):
    max_input_tokens: int | None = 5  # two special tokens and three words

    def __init__(self) -> None:
        self._counter = _Counter()

    def count_tokens(self, text: str) -> int:
        return len(text.split(' ')) + 2


def test_clipping_keeps_exactly_the_words_the_local_window_reads() -> None:
    from .prefill import clip_to_window

    local = _WindowedLocal()
    assert clip_to_window(local, 'one two three') == 'one two three'
    assert clip_to_window(local, 'one two three four five') == 'one two three'
    twin_remote = _Remote()
    asyncio.run(HostedTwin(local, twin_remote).embed(['one two three four']))
    assert twin_remote.calls == [['one two three']]


class _Refusing(_Remote):
    """Refuses any batch holding a text longer than ten characters, as the hosted copy refuses an over-long text."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any(len(t) > 10 for t in texts):
            raise RuntimeError('embedding server returned 400')
        return await super().embed(texts)


def test_a_refused_text_is_found_shortened_and_counted_while_its_batch_mates_embed_whole() -> None:
    from scone_memory.ingestion.embedding_cache import cache_key

    cache, remote = _Cache(), _Refusing()
    sent, shortened = asyncio.run(fill(cache, remote, 'm', 4, ['ok', 'fine', 'x' * 20], batch=3))
    assert (sent, shortened) == (3, 1)
    assert cache.store[cache_key('m', 4, 'ok')][0] == 2.0  # embedded whole
    assert cache.store[cache_key('m', 4, 'x' * 20)][0] <= 10.0  # keyed as asked, embedded shortened


def test_a_text_refused_at_every_length_stops_the_fill() -> None:
    class _Never(_Remote):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError('embedding server returned 400')

    with pytest.raises(RuntimeError, match='even cut'):
        asyncio.run(fill(_Cache(), _Never(), 'm', 4, ['anything']))
