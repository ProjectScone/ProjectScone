from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import math
from operator import mul
import unicodedata
from typing import Sequence

from ..retrieval.lexical import TOKENIZER_VERSION, tokenize


class HashEmbedder:
    """Feature hashing with signed buckets, L2-normalised.

    Two texts that share tokens share buckets, so cosine tracks lexical
    overlap. Unrelated texts land near-orthogonal, which is why tests
    that need a refusal must not rely on this embedder to trigger it.

    The id names the tokenizer version and the Unicode tables it read. A
    vector is only its hashed tokens, so vectors hashed under different
    token rules are not comparable and must not pass for the same embedder.
    Rebuilding costs nothing but time, so a store it wrote under other
    rules is re-embedded when it opens.
    """

    #: Local, free and deterministic: see memory.vector_identity.
    cheap_to_rebuild = True

    def __init__(self, dim: int = 256) -> None:
        self.id = f"hash-{dim}-t{TOKENIZER_VERSION}-u{unicodedata.unidata_version}"
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        # Each distinct token once, with its count: every bucket holds a
        # whole number, which a float keeps exactly in any order of adding.
        for token, count in Counter(tokenize(text)).items():
            bucket, sign = _slot(token, self.dim)
            vec[bucket] += sign * count
        norm = math.sqrt(sum(map(mul, vec, vec)))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


#: How many distinct tokens' buckets and signs are remembered, most recent
#: first. This sizes a cache, not a result: a token that has fallen out is
#: hashed again, which costs time and never changes a vector, and
#: ``_slot.cache_info()`` says how full it is.
_SLOT_CACHE_SIZE = 65_536


@lru_cache(maxsize=_SLOT_CACHE_SIZE)
def _slot(token: str, dim: int) -> tuple[int, float]:
    """The bucket a token lands in and the sign it adds there."""
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little") % dim, 1.0 if digest[4] & 1 else -1.0
