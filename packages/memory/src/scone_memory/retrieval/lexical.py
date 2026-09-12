"""Tokenizer and BM25 for the in-process lexical lane.

Stores with their own full-text engine (MongoDB ``$text``) do not use
the scorer, but every backend shares the tokenizer so that tag and fact
matching behave the same everywhere.

A token is a run of letters, combining marks and digits in any script,
taken after NFKC normalisation and case folding, so "Zürich", "हिन्दी" and
"ＡＢＣ" arrive whole. Underscores still separate words, which keeps a
predicate such as ``works_at`` findable as "works". Scripts written without
spaces between words have no boundaries to find without a dictionary, so a
run of them is indexed as each character and each overlapping pair: a query
pair matches wherever the same two characters stand together, and a single
character still finds the longer word that holds it.

Anything that stores tokens, or values derived from them, records
``TOKENIZER_VERSION`` so that output of an older rule is never silently
mixed with output of this one.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Iterable

#: Raise whenever the tokens produced for any text change.
TOKENIZER_VERSION = 3

_ASCII_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Blocks of scripts written without spaces between words: Thai, Lao, Myanmar,
# Khmer, Hangul, the CJK radicals, ideographic marks and numerals, kana,
# bopomofo and the ideographs. Punctuation inside these blocks (the katakana
# middle dot, the ideographic full stop) is not a word character, so it still
# ends a run.
_UNSPACED = (
    "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff"
    "\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7ff"
    "\u2e80-\u2fdf\u3005-\u3007\u3021-\u3029\u3031-\u3035\u303b\u303c"
    "\u3040-\u30ff\u31f0-\u31ff\u3100-\u312f\u31a0-\u31bf"
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f"
)
_UNSPACED_CHAR = re.compile(f"[{_UNSPACED}]")
_SEGMENT = re.compile(f"[{_UNSPACED}]+|[^{_UNSPACED}]+")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i in is it its of on or
    that the this to was were what when where which who will with you your
    did do does my me we our""".split()
)


def _ranges(codes: Iterable[int]) -> str:
    spans: list[tuple[int, int]] = []
    for code in codes:
        if spans and spans[-1][1] == code - 1:
            spans[-1] = (spans[-1][0], code)
        else:
            spans.append((code, code))
    return "".join(re.escape(chr(first)) if first == last else f"{re.escape(chr(first))}-{re.escape(chr(last))}"
                   for first, last in spans)


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    """Word and grapheme patterns, built once from this Python's Unicode tables.

    ``\\w`` covers letters and digits but not combining marks, so on its own
    it would cut "हिन्दी" at every vowel sign. Marks live below U+20000 apart
    from the variation selectors at U+E0100; scanning only there keeps the
    one-off cost near ten milliseconds.
    """
    marks = _ranges(code for code in (*range(0x300, 0x20000), *range(0xE0100, 0xE01F0))
                    if unicodedata.category(chr(code)).startswith("M"))
    word = f"(?:[^\\W_]|[{marks}])"
    letter = f"(?:[^\\W\\d_]|[{marks}])"
    return (re.compile(f"{word}+(?:'{letter}+)?"),
            re.compile(f"[{marks}]"),
            re.compile(f"[{marks}]+|[^{marks}][{marks}]*"))


def _grams(run: str, mark: re.Pattern[str], graphemes: re.Pattern[str]) -> list[str]:
    # Each character and each overlapping pair: pairs rank a run that holds
    # the whole query word, single characters let a one-character query
    # find the longer word it is part of. A character is a base with its
    # marks, never half of one; most runs carry no marks, and then every
    # code point is a character.
    units: str | list[str] = graphemes.findall(run) if mark.search(run) else run
    grams: list[str] = []
    for index, unit in enumerate(units):
        grams.append(unit)
        if index + 1 < len(units):
            grams.append(unit + units[index + 1])
    return grams


def _unpossessed(token: str) -> str:
    """A possessive ending leaves a word ("alves's", "chris'" to "alves",
    "chris"), so a question about someone's things finds passages naming
    them; contractions and names with inner apostrophes keep them."""
    return token[:-2] if token.endswith("'s") else token.rstrip("'")


def tokenize(text: str) -> list[str]:
    folded = text.casefold()
    if folded.isascii():
        return [t for t in map(_unpossessed, _ASCII_TOKEN.findall(folded)) if t and t not in STOPWORDS]
    words, mark, graphemes = _patterns()
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())
    tokens: list[str] = []
    for word in words.findall(folded.replace("\u2019", "'")):
        if _UNSPACED_CHAR.search(word) is None:
            word = _unpossessed(word)
            if word and word not in STOPWORDS:
                tokens.append(word)
            continue
        for segment in _SEGMENT.findall(word):
            if _UNSPACED_CHAR.match(segment):
                tokens.extend(_grams(segment, mark, graphemes))
            elif (plain := segment.strip("'")) and plain not in STOPWORDS:
                tokens.append(plain)
    return tokens


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
