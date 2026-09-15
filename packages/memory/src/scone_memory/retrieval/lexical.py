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
#: A character of a script written without spaces between words, for any
#: reader that must treat such text as the tokenizer does.
UNSPACED_CHAR = _UNSPACED_CHAR
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


def fold_diacritics(token: str) -> str:
    """``token`` without its combining marks ("facturación" to "facturacion"),
    so a query typed without accents finds the word, as the SQLite store's
    own tokenizer already does; the tokenizer itself is unchanged, so no
    persisted token or vector identity moves."""
    if token.isascii():
        return token
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


class Bm25:
    """Okapi BM25 over a mutable set of documents keyed by int id; terms are
    kept with their diacritics folded, matching the SQLite text lane.

    A document holding no query term and no member of a query family scores
    nothing and is never returned, so each term keeps the documents that
    hold it and a search scores only those."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[int, Counter[str]] = {}
        self._lengths: dict[int, int] = {}
        self._df: Counter[str] = Counter()
        #: The documents holding each term; a term no document holds is not kept.
        self._postings: dict[str, set[int]] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: int, text: str) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        tokens = [fold_diacritics(token) for token in tokenize(text)]
        counts = Counter(tokens)
        self._docs[doc_id] = counts
        self._lengths[doc_id] = len(tokens)
        self._df.update(counts.keys())
        for term in counts:
            self._postings.setdefault(term, set()).add(doc_id)

    def remove(self, doc_id: int) -> None:
        counts = self._docs.pop(doc_id, None)
        if counts is None:
            return
        self._lengths.pop(doc_id, None)
        for term in counts:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
            holders = self._postings[term]
            holders.discard(doc_id)
            if not holders:
                del self._postings[term]

    def search(
        self, query: str, limit: int, allowed: Iterable[int] | None = None, prefixes: Iterable[str] = (),
        exact_forms: bool = False,
    ) -> list[tuple[int, float]]:
        """Documents by BM25 over the query's terms; a prefix counts every
        vocabulary term that starts with it as one term, so a word's family
        ("bill" for billing, billed, bills) is one signal, not several.

        With ``exact_forms``, a document holding one of the query's own
        words in a family has that family's count weighed at the idf of the
        rarest such word it holds instead of the family's. The family is
        still one saturating term: a document holding only "billing" scores
        what "billing" alone would, one holding "billing" and "bills" scores
        their joint count at "billing"'s idf (the family's count, not a
        second term), and one holding only relatives scores as without it.
        The credit is the gap between the two idfs times BM25's count part,
        which reaches ``k1 + 1``: at most 2.2 times the gap, not the gap."""
        terms = [fold_diacritics(token) for token in tokenize(query)]
        folded = [fold_diacritics(prefix) for prefix in prefixes]
        families = [(prefix, [term for term in self._df if term.startswith(prefix)]) for prefix in folded]
        families = [(prefix, members) for prefix, members in families if members]
        # A query word its own family covers is scored once, through the
        # family: "billing" asked with the prefix "bill" is one signal, not
        # the word and then the family it belongs to.
        covered = {term for term in terms if any(term.startswith(prefix) for prefix, _ in families)}
        terms = [term for term in terms if term not in covered]
        # The query's own words in each family, rarest first: the most
        # specific form a document holds is the first it has.
        forms = {prefix: sorted((term for term in covered if term.startswith(prefix)), key=self._df.__getitem__)
                 if exact_forms else [] for prefix, _ in families}
        if (not terms and not families) or not self._docs:
            return []
        n = len(self._docs)
        avg_len = sum(self._lengths.values()) / n
        # Each family's count in every document holding a member, read from
        # the postings: whole numbers, so the same in any order of adding.
        family_tf: list[tuple[str, dict[int, int]]] = []
        # A family's document frequency is the documents holding any member,
        # each once: one holding "billing" and "bills" is one document, as
        # SQLite counts its rows, not two.
        family_df: dict[str, int] = {}
        matching: set[int] = set()
        for prefix, members in families:
            tfs: dict[int, int] = {}
            for member in members:
                for doc_id in self._postings[member]:
                    tfs[doc_id] = tfs.get(doc_id, 0) + self._docs[doc_id][member]
            family_tf.append((prefix, tfs))
            family_df[prefix] = len(tfs)
            matching.update(tfs)
        for term in terms:
            matching.update(self._postings.get(term, ()))
        # Every other document scores zero. The order candidates are scored
        # in is settled by the sort below, by score and then by id.
        candidates = matching if allowed is None else [d for d in allowed if d in matching]
        scored: list[tuple[int, float]] = []
        for doc_id in candidates:
            counts = self._docs[doc_id]
            length = self._lengths[doc_id]
            score = 0.0
            weighed: list[tuple[int, int]] = [(counts[term], self._df[term]) for term in terms if counts.get(term)]
            for prefix, tfs in family_tf:
                tf = tfs.get(doc_id, 0)
                if tf:
                    held = next((form for form in forms[prefix] if counts.get(form)), None)
                    weighed.append((tf, family_df[prefix] if held is None else self._df[held]))
            for tf, df in weighed:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += idf * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]
