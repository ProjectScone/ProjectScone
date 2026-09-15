"""Whether an extracted text layer is text a reader can use.

A PDF font without a usable Unicode map extracts as private-use code
points, as ``(cid:N)`` runs (the spelling some PDF readers give a glyph
they cannot name), as replacement characters or as control bytes. The
page is not empty, so nothing asked for OCR, and the characters were
indexed as the page's text. This names such text without a model: a count
of those characters among the visible ones.

The count and the share must both be reached. A handful of icon glyphs,
which fonts place in the private-use area, beside a page of prose is
not a garbled page, and neither are two icons on a page with no text.
Whitespace is not counted either way, so a sparse page of layout line
breaks and tabs is judged by the few words on it.
"""

from __future__ import annotations

import re
import unicodedata

#: The rule's name, for a receipt that must say how a page was judged.
TEXT_LAYER_VERSION = "unreadable-v1"
#: Unreadable characters, as a share of visible characters, from which a text layer is unreadable.
GARBLED_SHARE = 0.3
#: Unreadable characters a text layer must hold before it can be unreadable at all.
MIN_UNREADABLE = 8

_CID = re.compile(r"\(cid:\d+\)")


def _unreadable_character(character: str) -> bool:
    return unicodedata.category(character) in ("Co", "Cc") or character == "\ufffd"


def unreadable(text: str) -> bool:
    """Whether ``text`` is mostly characters no reader can use: private-use
    code points, ``(cid:N)`` runs, replacement characters or control bytes."""
    cited = sum(len(run) for run in _CID.findall(text))
    rest = [character for character in _CID.sub("", text) if not character.isspace()]
    visible = cited + len(rest)
    bad = cited + sum(1 for character in rest if _unreadable_character(character))
    return bad >= MIN_UNREADABLE and bad >= visible * GARBLED_SHARE
