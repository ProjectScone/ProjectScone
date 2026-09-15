"""The hash embedder remembers short tokens' slots, and only short tokens'.

Its slot cache is shared by every HashEmbedder in the process and is freed
only by ``_slot.cache_clear()``, so what one entry can hold decides what the
cache can hold. A token has no length limit of its own: a hex dump or a line
of minified text is one token as long as its chunk. These tests hold the
cache to tokens of at most 64 characters, and hold every vector, cached or
not, to the hash written out the slow way.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random

import pytest

from scone_memory.embedders import hash as hashing
from scone_memory.embedders.hash import HashEmbedder
from scone_memory.retrieval.lexical import tokenize


def hashed_the_slow_way(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        vec[int.from_bytes(digest[:4], "little") % dim] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0 else [v / norm for v in vec]


def hex_tokens(count: int, chars: int, seed: int) -> list[str]:
    chooser = random.Random(seed)
    return [f"{chooser.getrandbits(4 * chars):0{chars}x}" for _ in range(count)]


@pytest.fixture(autouse=True)
def empty_cache():
    hashing._slot.cache_clear()
    yield
    hashing._slot.cache_clear()


def test_tokens_longer_than_64_characters_are_hashed_but_not_remembered():
    tokens = hex_tokens(200, 800, seed=1) + hex_tokens(50, 65, seed=2)
    texts = [" ".join(tokens[i:i + 25]) for i in range(0, len(tokens), 25)]
    assert all(len(t) > 64 for text in texts for t in tokenize(text))

    vectors = asyncio.run(HashEmbedder().embed(texts))

    assert hashing._slot.cache_info().currsize == 0
    assert vectors == [hashed_the_slow_way(text, 256) for text in texts]


def test_tokens_of_64_characters_or_fewer_are_remembered():
    tokens = hex_tokens(30, 64, seed=3) + ["cache", "slot", "hash"]
    text = " ".join(tokens + tokens[:5])

    vector = asyncio.run(HashEmbedder().embed([text]))[0]

    assert hashing._slot.cache_info().currsize == len(tokens)
    assert vector == hashed_the_slow_way(text, 256)


def test_mixed_lengths_give_the_slow_hash_at_every_dimension():
    chooser = random.Random(4)
    words = hex_tokens(40, 64, seed=5) + hex_tokens(40, 65, seed=6) + hex_tokens(10, 700, seed=7)
    words += ["memory", "recall", "bucket", "sign", "token"]
    texts = [" ".join(chooser.choices(words, k=60)) for _ in range(30)]
    for dim in (7, 64, 256, 1024):
        vectors = asyncio.run(HashEmbedder(dim).embed(texts))
        assert vectors == [hashed_the_slow_way(text, dim) for text in texts]
