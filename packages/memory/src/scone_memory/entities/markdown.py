"""Ledger text placed in Markdown, rendered as exactly that text.

A subject, predicate or value is something a person or a model wrote. Put
into a report or a note, it must not become a table column, a list, a
heading, a link, an image, emphasis or raw HTML. ``literal`` escapes the
characters CommonMark, GitHub tables and Obsidian read as syntax, and folds
line breaks into spaces so a name stays on its line. Callers keep the raw
value in their JSON forms; this is only for the Markdown ones.
"""

from __future__ import annotations

import re

# Read as syntax wherever they fall: code, emphasis, links and images, raw
# HTML and autolinks, tables, headings and tags, strikethrough, Obsidian
# highlights, block ids, math and comments, entity references.
_ANYWHERE = frozenset("\\`*[]<>|#~=^$%&!")
_LIST_MARKER = re.compile(r"(\d{1,9})([.)])(?=\s|$)")


def _underscore_is_syntax(text: str, index: int) -> bool:
    """An underscore between two letters or digits never opens or closes
    emphasis, so ``works_at`` stays readable; any other one might."""
    before = text[index - 1] if index else " "
    after = text[index + 1] if index + 1 < len(text) else " "
    return not (before.isalnum() and after.isalnum())


def literal(text: object) -> str:
    flat = " ".join(str(text).split())
    escaped = "".join("\\" + character
                      if character in _ANYWHERE or (character == "_" and _underscore_is_syntax(flat, index))
                      else character for index, character in enumerate(flat))
    if escaped[:1] in ("-", "+"):
        return "\\" + escaped
    return _LIST_MARKER.sub(r"\1\\\2", escaped, count=1) if _LIST_MARKER.match(escaped) else escaped
