"""Turn a message too long to search into a bounded query made of its own words.

Recall accepts at most ``MAX_QUERY`` characters. A conversation turn that
pastes a document and asks about it would otherwise fail recall and reach the
model with no memory. Nothing here paraphrases: the query is verbatim excerpts
of the message joined by newlines, and ``kept`` records exactly which
character spans were used, so a receipt can show what was searched.

Excerpts are taken in a fixed order of value. Questions come first, latest
first, since a pasted draft usually ends with what the person wants to know.
The closing and then the opening sentence follow. Remaining room goes to the
sentences whose words are rarest across the message, so a name mentioned once
outranks filler repeated throughout. When the most valuable excerpt is itself
too long, its head and tail are kept, cut at spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal

from ..core.validation import MAX_QUERY
from .lexical import tokenize

# A sentence ends at a terminator followed by space, straight after a
# full-width terminator (those scripts put no space after one), or at a line
# break.
_BOUNDARY = re.compile(r"(?<=[.!?…؟।])\s+|(?<=[。！？])|\n")
_QUESTION = re.compile(r"[?？؟][\"'”’)\]」』]*$")

Span = tuple[int, int]


@dataclass(frozen=True)
class FormulatedQuery:
    text: str
    kept: tuple[Span, ...]
    source_chars: int
    method: Literal["verbatim", "extract"]


def _trimmed(message: str, start: int, end: int) -> Span | None:
    while start < end and message[start].isspace():
        start += 1
    while end > start and message[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _sentences(message: str) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for boundary in _BOUNDARY.finditer(message):
        if (span := _trimmed(message, start, boundary.start())) is not None:
            spans.append(span)
        start = boundary.end()
    if (span := _trimmed(message, start, len(message))) is not None:
        spans.append(span)
    return spans


def _by_rarity(message: str, sentences: list[Span]) -> list[int]:
    words = [set(tokenize(message[start:end])) for start, end in sentences]
    frequency: dict[str, int] = {}
    for found in words:
        for word in found:
            frequency[word] = frequency.get(word, 0) + 1
    total = len(sentences)

    def rarity(index: int) -> float:
        found = words[index]
        if not found:
            return 0.0
        return sum(math.log(total / frequency[word]) for word in found) / math.sqrt(len(found))

    return sorted(range(total), key=lambda index: (-rarity(index), index))


def _head_and_tail(message: str, span: Span, room: int) -> list[Span]:
    start, end = span
    head_room = (room - 1) // 2
    head_end = start + head_room
    while head_end > start and not message[head_end].isspace():
        head_end -= 1
    if head_end == start:
        head_end = start + head_room
    tail_start = end - (room - 1 - (head_end - start))
    while tail_start < end and not message[tail_start - 1].isspace():
        tail_start += 1
    if tail_start == end:
        tail_start = end - (room - 1 - (head_end - start))
    kept = [piece for piece in (_trimmed(message, start, head_end), _trimmed(message, max(tail_start, head_end), end))
            if piece is not None]
    return kept


def formulate_query(message: str, *, limit: int = MAX_QUERY) -> FormulatedQuery:
    """Return ``message`` itself when it fits, otherwise ordered verbatim excerpts."""
    if len(message) <= limit:
        return FormulatedQuery(message, ((0, len(message)),), len(message), "verbatim")
    sentences = _sentences(message)
    if not sentences:
        return FormulatedQuery("", (), len(message), "extract")
    questions = [index for index in reversed(range(len(sentences)))
                 if _QUESTION.search(message[sentences[index][0]:sentences[index][1]])]
    order = [*questions, len(sentences) - 1, 0, *_by_rarity(message, sentences)]
    chosen: list[Span] = []
    used = 0
    for index in dict.fromkeys(order):
        start, end = sentences[index]
        cost = end - start + (1 if chosen else 0)
        if used + cost <= limit:
            chosen.append(sentences[index])
            used += cost
        elif not chosen:
            chosen = _head_and_tail(message, sentences[index], limit)
            break
    kept = tuple(sorted(chosen))
    return FormulatedQuery("\n".join(message[start:end] for start, end in kept), kept, len(message), "extract")
