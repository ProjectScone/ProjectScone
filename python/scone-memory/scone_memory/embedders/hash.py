from __future__ import annotations

import hashlib
import math
from typing import Sequence

from ..lexical import tokenize


class HashEmbedder:
    """Feature hashing with signed buckets, L2-normalised.

    Two texts that share tokens share buckets, so cosine tracks lexical
    overlap. Unrelated texts land near-orthogonal, which is why tests
    that need a refusal must not rely on this embedder to trigger it.
    """

    def __init__(self, dim: int = 256) -> None:
        self.id = f"hash-{dim}"
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
