"""Split an episode into spans without rewriting a byte of it.

``chunk_spans`` works in code points, which is what Python slices. The
shared specification (rule 1.2) defines stored offsets as UTF-8 byte
offsets, half-open, so that a span means the same thing to the Rust
product; ``byte_spans`` converts. Cuts prefer paragraph breaks, then
sentence ends, then whitespace, and only fall back to a hard cut inside
a word when a single token is longer than the target. Short episodes
are one chunk; a chunk is never empty.
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
    for marker in ("\n\n", "\n"):
        idx = window.rfind(marker)
        if idx != -1:
            return floor + idx + len(marker)
    best = -1
    for marker in (". ", "! ", "? ", ".\t"):
        idx = window.rfind(marker)
        if idx > best:
            best = idx + len(marker)
    if best != -1:
        return floor + best
    idx = -1
    for i in range(len(window) - 1, -1, -1):
        if window[i].isspace():
            idx = i
            break
    if idx != -1:
        return floor + idx + 1
    return limit


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
