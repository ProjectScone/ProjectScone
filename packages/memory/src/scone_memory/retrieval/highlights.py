"""Where in a returned passage the question's words occur.

The lexical lane matches tokens, and its tokeniser folds case, normalises
width and splits scripts written without spaces into characters and
pairs. A reader who tokenises a passage again to show what matched gets
it wrong wherever their rules differ. So this reads the passage as it was
returned, word by word, tokenises each word with the lexical lane's own
tokeniser, and records the span of every word that yields a query term.
In a script without spaces a word is a run of characters, and the spans
are the places inside it where a query term's characters stand.

Spans are code-point offsets into the passage text exactly as returned,
and the function is meant to run last, after anything that changes a
passage's text, so they always point into what the reader holds. A
passage keeps at most ``MAX_HIGHLIGHTS`` spans; ``total`` counts them all
and ``truncated`` says the bound bit.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import NamedTuple, Protocol

from pydantic import BaseModel, Field

from .lexical import _UNSPACED_CHAR, tokenize

#: Spans kept per passage. The rest are counted, not listed.
MAX_HIGHLIGHTS = 32


class _Passage(Protocol):
    @property
    def chunk_id(self) -> int: ...

    @property
    def text(self) -> str: ...


class Shown(NamedTuple):
    """A passage as a response already holds it, for callers with dictionaries."""

    chunk_id: int
    text: str


class HighlightSpan(BaseModel):
    start: int
    end: int
    term: str


class Highlights(BaseModel):
    chunk_id: int
    #: The query's terms as the lexical lane reads them, stopwords gone.
    terms: list[str] = Field(default_factory=list)
    spans: list[HighlightSpan] = Field(default_factory=list)
    total: int = 0
    truncated: bool = False


def _word_char(char: str) -> bool:
    return char in "'’_" or unicodedata.category(char)[0] in "LNM"


def _words(text: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, char in enumerate(text):
        if _word_char(char):
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(text)))
    return runs


def _located(text: str, terms: list[str]) -> list[HighlightSpan]:
    wanted = set(terms)
    found: list[HighlightSpan] = []
    for start, end in _words(text):
        word = text[start:end]
        tokens = tokenize(word)
        hits = [token for token in dict.fromkeys(tokens) if token in wanted]
        if not hits:
            continue
        if _UNSPACED_CHAR.search(word) is None:
            found.append(HighlightSpan(start=start, end=end, term=hits[0]))
            continue
        # Characters of a script without spaces: where each matched term
        # stands. A query's characters and pairs overlap, so overlapping
        # places merge into one region named by its longest term.
        placed: list[tuple[int, int, str]] = []
        for term in hits:
            at = word.find(term)
            while at != -1:
                placed.append((start + at, start + at + len(term), term))
                at = word.find(term, at + 1)
        if not placed:
            found.append(HighlightSpan(start=start, end=end, term=hits[0]))
            continue
        placed.sort()
        merged = [placed[0]]
        for lo, hi, term in placed[1:]:
            last_lo, last_hi, last_term = merged[-1]
            if lo < last_hi:
                merged[-1] = (last_lo, max(last_hi, hi), term if len(term) > len(last_term) else last_term)
            else:
                merged.append((lo, hi, term))
        found.extend(HighlightSpan(start=lo, end=hi, term=term) for lo, hi, term in merged)
    found.sort(key=lambda span: (span.start, span.end))
    return found


def highlights(items: Sequence[_Passage], query: str) -> list[Highlights]:
    """One entry per passage, in order, with the spans where query terms occur."""
    terms = list(dict.fromkeys(tokenize(query)))
    out: list[Highlights] = []
    for item in items:
        found = _located(item.text, terms) if terms else []
        out.append(Highlights(chunk_id=item.chunk_id, terms=terms, spans=found[:MAX_HIGHLIGHTS],
                              total=len(found), truncated=len(found) > MAX_HIGHLIGHTS))
    return out
