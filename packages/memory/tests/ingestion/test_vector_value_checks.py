"""What an embedder returns is judged value by value, the same way on every path.

The checks on a vector's values run for every chunk ever embedded, so they
take a quicker route when every value is a plain float or int. These tests
hold both routes to the rule written out once, value by value: numbers
only, never a bool, never a NaN or an infinity, a number too large for a
float refused rather than raised.
"""

from __future__ import annotations

from enum import IntEnum
import importlib.util
import math

import pytest

from scone_memory.backends.validation import validate_vector
from scone_memory.embedders.hash import HashEmbedder
from scone_memory.ingestion.vectors import validated_vectors
from scone_memory.retrieval.lexical import tokenize


class Float(float):
    pass


class Level(IntEnum):
    HIGH = 3


def accepted_by_the_rule(vector: list[object]) -> bool:
    try:
        return all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) for value in vector)  # type: ignore[arg-type]
    except OverflowError:
        return False


CASES: list[list[object]] = [
    [0.0, 1.5, -2.25],
    [1, 2, 3],
    [1, 2.5, -3],
    [0.0, True],
    [False, 1.0],
    [1.0, math.nan],
    [math.inf, 1.0],
    [1.0, -math.inf],
    [1.0, "2.0"],
    [None, 1.0],
    [10 ** 400, 1.0],
    [Float(1.0), 2.0],
    [Float(math.nan), 2.0],
    [Level.HIGH, 1.0],
    [1.0, 1 + 2j],
]


@pytest.mark.parametrize("vector", CASES, ids=[repr(case) for case in CASES])
def test_an_embedding_response_is_judged_as_the_rule_says(vector):
    if accepted_by_the_rule(vector):
        [kept] = validated_vectors([vector], 1, len(vector))
        assert kept == vector and kept is not vector
    else:
        with pytest.raises(ValueError, match="finite numeric values"):
            validated_vectors([vector], 1, len(vector))


@pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="numpy not installed")
def test_numpy_scalars_are_judged_as_the_rule_says():
    import numpy

    for vector in ([numpy.float64(1.0), 2.0], [numpy.float64("nan"), 2.0], [numpy.float32(1.0), 2.0]):
        if accepted_by_the_rule(vector):
            assert validated_vectors([vector], 1, 2) == [vector]
        else:
            with pytest.raises(ValueError):
                validated_vectors([vector], 1, 2)


def test_an_index_refuses_what_is_not_finite_and_nothing_else():
    validate_vector([0.0, 1, True, Float(2.0)], 4)
    for vector in ([0.0, math.nan], [math.inf, 0.0], [0.0, Float(-math.inf)]):
        with pytest.raises(ValueError, match="finite"):
            validate_vector(vector, 2)
    with pytest.raises(TypeError):
        validate_vector([0.0, "1"], 2)


def plain_hash_vector(text: str, dim: int) -> list[float]:
    """The embedder written out one token at a time."""
    import hashlib

    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        vec[int.from_bytes(digest[:4], "little") % dim] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0 else [v / norm for v in vec]


async def test_the_hash_embedder_makes_the_vector_each_token_adds_up_to():
    texts = [
        "the the the cat cat sat",
        "Zürich Zurich zürich naïve",
        "東京タワーは東京にある",
        "",
        "and the of",  # stopwords only: a zero vector
        " ".join(f"word{index % 37}" for index in range(2000)),
    ]
    for dim in (8, 256):
        embedder = HashEmbedder(dim)
        assert await embedder.embed(texts) == [plain_hash_vector(text, dim) for text in texts]


async def test_a_vocabulary_wider_than_the_slot_cache_still_makes_the_same_vectors():
    from scone_memory.embedders import hash as hashing

    size = hashing._SLOT_CACHE_SIZE
    words = [f"w{index}" for index in range(size + 5000)]
    texts = [" ".join(words[start:start + 500]) for start in range(0, len(words), 500)]
    hashing._slot.cache_clear()
    embedder = HashEmbedder(64)
    first = await embedder.embed(texts)
    # The cache holds no more than its size; the earliest words have been
    # dropped and are hashed again, to the same bucket and sign.
    assert hashing._slot.cache_info().currsize == size
    assert await embedder.embed(texts[:2]) == first[:2]
    assert first == [plain_hash_vector(text, 64) for text in texts]
