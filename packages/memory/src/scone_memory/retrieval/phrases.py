"""Whether a passage holds a phrase, by whole words, whatever its case and punctuation.

A phrase and a passage are both read as words: letters, marks and digits
after NFKC normalisation and case folding, anything else a boundary. A
phrase matches where its words stand together, in order, as whole words,
so "slew ring" matches "SLEW-RING" and "art" does not match "party".
Scripts written without spaces between words have no boundaries to find
without a dictionary, so a phrase in one matches inside a run of it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

from ..core.errors import InvalidInput

#: Phrases one recall may require and exclude together.
MAX_PHRASES = 20
#: Characters in one phrase.
MAX_PHRASE_CHARS = 200

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_UNSPACED = re.compile("[฀-໿က-႟ក-៿぀-ヿ㐀-䶿一-鿿"
                       "가-퟿豈-﫿]")


def _words(text: str) -> str:
    return " ".join(_WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def holds(text: str, phrase: str) -> bool:
    """Whether ``text`` holds ``phrase`` as whole words in order."""
    wanted = _words(phrase)
    if not wanted:
        return False
    if _UNSPACED.search(wanted):
        return wanted.replace(" ", "") in _words(text).replace(" ", "")
    return f" {wanted} " in f" {_words(text)} "


def checked_phrases(require: object, exclude: object) -> tuple[list[str], list[str]]:
    """The phrases as given, refused when one is not a list of matchable strings or they contradict."""
    lists = []
    for name, phrases in (("require", require), ("exclude", exclude)):
        if not isinstance(phrases, (list, tuple)) or not all(isinstance(phrase, str) for phrase in phrases):
            raise InvalidInput(f"{name} is a list of phrases, not {phrases!r}")
        for phrase in phrases:
            if len(phrase) > MAX_PHRASE_CHARS:
                raise InvalidInput(f"a phrase is at most {MAX_PHRASE_CHARS} characters; one in {name} is {len(phrase)}")
            if not _words(phrase):
                raise InvalidInput(f"the phrase {phrase!r} in {name} has no word to match")
        lists.append(list(phrases))
    if len(lists[0]) + len(lists[1]) > MAX_PHRASES:
        raise InvalidInput(f"at most {MAX_PHRASES} phrases may be required and excluded together")
    both = sorted({_words(phrase) for phrase in lists[0]} & {_words(phrase) for phrase in lists[1]})
    if both:
        raise InvalidInput(f"{', '.join(both)} is both required and excluded, so nothing could be returned")
    return lists[0], lists[1]


def passes(text: str, require: Sequence[str], exclude: Sequence[str]) -> tuple[bool, str]:
    """Whether ``text`` holds every required phrase and no excluded one, and the rule that dropped it."""
    if not all(holds(text, phrase) for phrase in require):
        return False, "required"
    if any(holds(text, phrase) for phrase in exclude):
        return False, "excluded"
    return True, ""
