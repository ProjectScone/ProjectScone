from __future__ import annotations

import hashlib
import math
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
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
