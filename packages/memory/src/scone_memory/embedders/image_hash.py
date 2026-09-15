"""A deterministic image embedder, for tests and for trying the image lane without a model.

It does not look at pixels. An image's vector is spread from a hash of its
bytes, so the same bytes always land in one place and different bytes land
nearly orthogonal. A text's vector comes from a map the caller supplies,
phrase to image bytes: a text naming a phrase as whole words lands on that
image (between the images, when it names several), and any other text lands
on a hash of itself. Retrieval with it proves the lane's plumbing -- its own
index, fusion, provenance, forgetting -- and says nothing about how well a
real model would see.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Mapping, Sequence

_WORDS = re.compile(r"\w+")


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORDS.findall(text.casefold()))


class HashImageEmbedder:
    """Image bytes and phrase-mapped texts into one hashed space. See the module."""

    def __init__(self, dim: int = 64, phrases: Mapping[str, bytes] | None = None, *, salt: str = "") -> None:
        if type(dim) is not int or dim < 2:
            raise ValueError("dim must be an integer of at least 2")
        self.dim = dim
        self._salt = salt.encode()
        self.id = f"hash-image-{dim}" + (f"-{salt}" if salt else "")
        self._phrases: list[tuple[tuple[str, ...], bytes]] = []
        for phrase, data in (phrases or {}).items():
            words = _words(phrase)
            if not words or not isinstance(data, bytes) or not data:
                raise ValueError(f"phrase {phrase!r} needs a word and the bytes of its image")
            self._phrases.append((words, data))

    async def embed_images(self, images: Sequence[bytes]) -> list[list[float]]:
        return [self._spread(data) for data in images]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._text(text) for text in texts]

    def _text(self, text: str) -> list[float]:
        words = _words(text)
        named = [data for phrase, data in self._phrases if _contains(words, phrase)]
        if not named:
            return self._spread(text.encode())
        total = [0.0] * self.dim
        for data in named:
            for i, value in enumerate(self._spread(data)):
                total[i] += value
        return _unit(total)

    def _spread(self, data: bytes) -> list[float]:
        seed = hashlib.blake2b(data, digest_size=32, key=self._salt[:64]).digest()
        values: list[float] = []
        block = 0
        while len(values) < self.dim:
            digest = hashlib.blake2b(seed + block.to_bytes(4, "little"), digest_size=64).digest()
            values.extend((int.from_bytes(digest[i:i + 2], "little") / 65535.0) * 2.0 - 1.0 for i in range(0, 64, 2))
            block += 1
        return _unit(values[:self.dim])


def _contains(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(words[start:start + size] == phrase for start in range(len(words) - size + 1))


def _unit(vector: list[float]) -> list[float]:
    # Hashed values spread over at least two places are never all zero, and
    # a sum of distinct images' spreads does not cancel.
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]
