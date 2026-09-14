"""A word's family in the lexical lane, by prefix, with no stemmer's guesses in the index.

"bills", "billing" and "billed" are one word to a reader and three to the
text lane, which finds the words a passage has and only those. The usual
answer is a stemmer over the index, and it is the wrong one here: it
rewrites what every store holds, ties the tokenizer's version to a set of
language rules, and puts a guess ("policies" and "police" as one) where a
reader cannot see it. This does less. A query term that ends in a known
English suffix also searches as a prefix of its stem -- ``bill`` for
"billing", ``invoic`` for "invoices" -- so the family is found, the index
is untouched, and every recall says which prefixes it added. A language
the rules do not know, a short word, a number, is left exactly as it was.

The rules are few and conservative on purpose: a stem must keep at least
four letters, a plural "s" after "s", "u" or "i" is not stripped, and a
doubled consonant left by a suffix is reduced ("running" to ``run``)
except where English keeps it ("billing" to ``bill``).
"""

from __future__ import annotations

from typing import Iterable, Optional

#: Suffixes tried longest first, each with the letters a stem must keep: a
#: plural or a final "e" is weak evidence of a family and keeps four, the
#: rest keep three. The first suffix that leaves a stem long enough wins.
SUFFIXES = (("ations", 3), ("ation", 3), ("ments", 3), ("ment", 3), ("ness", 3), ("ings", 3), ("ions", 3), ("ion", 3),
            ("ing", 3), ("edly", 3), ("ies", 3), ("ers", 3), ("ed", 3), ("er", 3), ("ly", 3), ("es", 4), ("s", 4), ("e", 4))
#: Doubled consonants English keeps before a suffix ("billing", "classes", "buzzing", "stuffed").
_KEPT_DOUBLES = frozenset("lszf")
#: A stem keeps at least this many letters once a doubled consonant is reduced.
MIN_STEM = 3


def stem(token: str) -> Optional[str]:
    """The prefix ``token``'s family starts with, or None when no rule applies."""
    if not token.isascii() or not token.isalpha():
        return None
    for suffix, keep in SUFFIXES:
        if not token.endswith(suffix) or len(token) - len(suffix) < keep:
            continue
        base = token[:-len(suffix)]
        if suffix == "s" and base[-1] in "sui":
            return None
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in _KEPT_DOUBLES and base[-1] not in "aeiou":
            base = base[:-1]
        if len(base) < MIN_STEM or base == token:
            return None
        return base
    return None


def prefixes(tokens: Iterable[str]) -> list[str]:
    """The distinct stem prefixes of ``tokens``, in order, without the tokens themselves."""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        found = stem(token)
        if found is not None and found != token and found not in seen:
            seen.add(found)
            out.append(found)
    return out
