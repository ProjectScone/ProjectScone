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
from .lexical import UNSPACED_CHAR

#: Phrases one recall may require and exclude together.
MAX_PHRASES = 20
#: Characters in one phrase.
MAX_PHRASE_CHARS = 200

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text: str) -> str:
    return " ".join(_WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def holds(text: str, phrase: str) -> bool:
    """Whether ``text`` holds ``phrase`` as whole words in order."""
    return _within(_words(text), _words(phrase))


def _within(words: str, wanted: str) -> bool:
    """Whether normalised ``words`` hold normalised ``wanted``."""
    if not wanted:
        return False
    if UNSPACED_CHAR.search(wanted):
        return wanted.replace(" ", "") in words.replace(" ", "")
    return f" {wanted} " in f" {words} "


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


class Phrases:
    """Required and excluded phrases, normalised once, to check many passages against."""

    def __init__(self, require: Sequence[str], exclude: Sequence[str]) -> None:
        self.require = [_words(phrase) for phrase in require]
        self.exclude = [_words(phrase) for phrase in exclude]

    def passes(self, text: str) -> tuple[bool, str]:
        """Whether ``text`` holds every required phrase and no excluded one, and the rule that dropped it.
        The passage is normalised once, however many phrases it is checked for."""
        words = _words(text)
        if not all(_within(words, phrase) for phrase in self.require):
            return False, "required"
        if any(_within(words, phrase) for phrase in self.exclude):
            return False, "excluded"
        return True, ""
