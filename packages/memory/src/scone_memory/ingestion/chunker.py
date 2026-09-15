"""Split an episode into spans without rewriting a byte of it.

``chunk_spans`` works in code points, which is what Python slices. The
shared specification (rule 1.2) defines stored offsets as UTF-8 byte
offsets, half-open, so that a span means the same thing to the Rust
product; ``byte_spans`` converts. Cuts prefer paragraph breaks, then
sentence ends, then whitespace, and only fall back to a hard cut inside
a word when a single token is longer than the target. Short episodes
are one chunk; a chunk is never empty.

Scripts written without spaces end a sentence at a full-width stop with
nothing after it, and pause at a full-width comma or enumeration mark.
Those are sentence ends and pauses here too, so Chinese and Japanese
prose is not cut at the byte target inside a word.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TARGET = 700
MIN_CHUNK = 120


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def chunk_spans(content: str, target: int = DEFAULT_TARGET) -> list[Span]:
    if target <= 0:
        raise ValueError("target must be positive")
    text = content
    n = len(text)
    if n == 0:
        return []
    spans: list[Span] = []
    start = 0
    while start < n:
        # Skip leading whitespace so a chunk never starts with the
        # separator that ended the previous one.
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        if n - start <= target:
            spans.append(Span(start, n))
            break
        limit = start + target
        cut = _best_cut(text, start, limit)
        spans.append(Span(start, cut))
        start = cut
    return _merge_tail(spans, text)


def byte_spans(content: str, spans: list[Span]) -> list[Span]:
    """Code-point spans to UTF-8 byte spans over the same content, so
    that ``content.encode()[start:end].decode()`` equals the chunk text."""
    if content.isascii():
        return spans
    offsets = [0]
    for ch in content:
        offsets.append(offsets[-1] + len(ch.encode()))
    return [Span(offsets[s.start], offsets[s.end]) for s in spans]


def _best_cut(text: str, start: int, limit: int) -> int:
    """The latest boundary in ``(start + MIN_CHUNK, limit]`` in order of
    preference: blank line, sentence end, any whitespace; else ``limit``."""
    floor = start + MIN_CHUNK
    window = text[floor:limit]
    # A blank line is a paragraph and outranks everything. Both spellings
    # of one: `rfind("\n\n")` does not match `\r\n\r\n`, so a CRLF
    # document had no paragraphs at all as far as this could see.
    best = -1
    for marker in ("\n\n", "\r\n\r\n"):
        idx = window.rfind(marker)
        if idx != -1:
            best = max(best, idx + len(marker))
    if best != -1:
        return floor + best
    # Then a sentence end, and **then** a lone newline -- which is the
    # order this module's docstring has always described and the code
    # did not keep. A single newline ranked second, and in hard-wrapped
    # prose every line ends in the middle of a sentence: measured over
    # 40 of this project's documents, 86 of 531 boundaries landed
    # mid-sentence and 85 of those 86 were at a soft wrap. How the text
    # was typed is not a boundary in what it says.
    best = -1
    for marker in (". ", "! ", "? ", ".\t", ".\n", "!\n", "?\n",
                   ".\r\n", "!\r\n", "?\r\n"):
        idx = window.rfind(marker)
        if idx > best:
            best = idx + len(marker)
    # A full-width stop needs nothing after it; a closing quote or bracket
    # straight after it stays with its sentence.
    stop = _last_full_width(window, _FULL_STOPS)
    if stop > best:
        best = stop
    if best != -1:
        return floor + best
    # A lone newline still beats arbitrary whitespace: in a list or a
    # table it is a real boundary, and there is no sentence end to find.
    idx = window.rfind("\n")
    if idx != -1:
        return floor + idx + 1
    idx = -1
    for i in range(len(window) - 1, -1, -1):
        if window[i].isspace():
            idx = i
            break
    pause = _last_full_width(window, _FULL_PAUSES)
    if pause > idx + 1:
        return floor + pause
    if idx != -1:
        return floor + idx + 1
    return limit


#: Sentence ends in scripts written without spaces.
_FULL_STOPS = "。！？"
#: Pauses in them: a full-width comma, enumeration mark, semicolon or colon.
_FULL_PAUSES = "，、；："
#: Closing quotes and brackets that belong to the sentence before them.
_FULL_CLOSERS = "」』）】〕》\"'"


def _last_full_width(window: str, marks: str) -> int:
    """The position just after the last of ``marks`` in ``window`` and any closers
    straight after it, or -1 when there is none."""
    for index in range(len(window) - 1, -1, -1):
        if window[index] in marks:
            end = index + 1
            while end < len(window) and window[end] in _FULL_CLOSERS:
                end += 1
            return end
    return -1


def _merge_tail(spans: list[Span], text: str) -> list[Span]:
    """A trailing fragment shorter than MIN_CHUNK joins its predecessor;
    a lone tiny chunk is a worse retrieval unit than a slightly long one."""
    if len(spans) < 2:
        return spans
    last = spans[-1]
    if len(text[last.start : last.end].strip()) >= MIN_CHUNK:
        return spans
    prev = spans[-2]
    return spans[:-2] + [Span(prev.start, last.end)]
