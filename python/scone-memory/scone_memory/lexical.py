"""Tokenizer and BM25 for the in-process lexical lane.

Stores with their own full-text engine (MongoDB ``$text``) do not use
the scorer, but every backend shares the tokenizer so that tag and fact
matching behave the same everywhere.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i in is it its of on or
    that the this to was were what when where which who will with you your
    did do does my me we our""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.casefold()) if t not in STOPWORDS]


class Bm25:
    """Okapi BM25 over a mutable set of documents keyed by int id."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[int, Counter[str]] = {}
        self._lengths: dict[int, int] = {}
        self._df: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: int, text: str) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        tokens = tokenize(text)
        counts = Counter(tokens)
        self._docs[doc_id] = counts
        self._lengths[doc_id] = len(tokens)
        self._df.update(counts.keys())

    def remove(self, doc_id: int) -> None:
        counts = self._docs.pop(doc_id, None)
        if counts is None:
            return
        self._lengths.pop(doc_id, None)
        for term in counts:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def search(
        self, query: str, limit: int, allowed: Iterable[int] | None = None
    ) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        n = len(self._docs)
        avg_len = sum(self._lengths.values()) / n
        candidates = self._docs.keys() if allowed is None else [d for d in allowed if d in self._docs]
        scored: list[tuple[int, float]] = []
        for doc_id in candidates:
            counts = self._docs[doc_id]
            length = self._lengths[doc_id]
            score = 0.0
            for term in terms:
                tf = counts.get(term)
                if not tf:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += idf * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]
