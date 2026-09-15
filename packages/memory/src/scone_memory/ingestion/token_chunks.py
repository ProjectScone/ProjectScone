"""Cut an episode into chunks measured in tokens, packed from whole sentences.

``chunker.chunk_spans`` measures its target in characters. A model reads
tokens, and the reference framework's splitter measures its nodes in
them: 512 by default, filled with whole sentences. This does the same
with our own spans. Sentences (``semantic_chunks.sentence_spans``) are
packed into a chunk while the chunk's count stays within the target; a
sentence is cut inside only when it alone is longer than the target, and
then at a line break before a space, and at a space before a hard cut
inside a word. Every one of those is counted in the receipt, as is every
chunk the count measured over the target, since a tokenizer need not
count a chunk as the sum of its sentences.

Tokens are counted by the function given: the embedder's own tokenizer
when it has one, and otherwise the budget's estimate (``estimated_tokens``),
which needs no model and errs high, so a chunk it keeps within 512 fits a
512-token window. The count of an empty text is what the counter adds to
every input (a model's start and end markers) and is paid once a chunk.

With ``overlap`` a chunk starts with the trailing whole pieces of the one
before it, as many as fit within that many tokens, so the stored spans of
neighbouring chunks overlap. An overlap that would leave no room for the
next piece gives way, from its front, before the chunk does.

Offsets are code points, as ``chunk_spans`` returns; ``chunker.byte_spans``
converts them. A chunk never starts or ends with whitespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

from .chunker import Span
from .embedding_budget import BUDGET_VERSION, estimated_tokens
from .semantic_chunks import sentence_spans

#: The smallest target accepted. Below it a chunk holds a few words and
#: the markers a model adds are a large part of every count.
MIN_CHUNK_TOKENS = 16


class _Piece(NamedTuple):
    start: int
    end: int
    tokens: int


@dataclass(frozen=True)
class TokenCut:
    """Where a text was cut, and what the cut had to do to keep the target."""

    spans: tuple[Span, ...]
    tokens: int
    overlap: int
    #: Which count measured: the embedder's tokenizer or the estimate.
    method: str
    sentences: int
    #: Sentences longer than the target on their own, and so cut inside.
    sentences_over_target: int
    #: Cuts made inside a word, because one word was longer than the target.
    hard_cuts: int
    #: Chunks that start with text repeated from the chunk before.
    overlapped: int
    #: Chunks the count measured over the target, whole.
    over_target: int
    #: The largest chunk, as the count measured it whole.
    largest: int

    def record(self) -> dict[str, object]:
        return {"measure": "tokens", "tokens": self.tokens, "overlap": self.overlap, "method": self.method,
                "sentences": self.sentences, "sentences_over_target": self.sentences_over_target,
                "hard_cuts": self.hard_cuts, "overlapped": self.overlapped, "over_target": self.over_target,
                "largest": self.largest}


def refused(tokens: object, overlap: object) -> str | None:
    """Why a token target and overlap cannot be used, or None when they can."""
    if type(tokens) is not int or tokens < MIN_CHUNK_TOKENS:
        return f"chunk_tokens must be an integer of at least {MIN_CHUNK_TOKENS}, got {tokens!r}"
    if type(overlap) is not int or not 0 <= overlap < tokens:
        return f"chunk_overlap_tokens must be an integer from 0 to below chunk_tokens ({tokens}), got {overlap!r}"
    return None


def token_spans(content: str, tokens: int, *, overlap: int = 0, count: Callable[[str], int] = estimated_tokens,
                method: str = BUDGET_VERSION) -> TokenCut:
    reason = refused(tokens, overlap)
    if reason is not None:
        raise ValueError(reason)
    markers = count("")
    room = tokens - markers
    if room < 1:
        raise ValueError(f"a target of {tokens} tokens leaves no room past the {markers} markers this count "
                         "adds to every input")
    pieces: list[_Piece] = []
    sentences = over = hard = 0
    for sentence in sentence_spans(content):
        end = _trimmed(content, sentence.start, sentence.end)
        sentences += 1
        size = count(content[sentence.start:end]) - markers
        if size <= room:
            pieces.append(_Piece(sentence.start, end, size))
            continue
        over += 1
        hard += _inside(content, sentence.start, end, room, count, markers, pieces)
    spans, overlapped = _packed(pieces, room, overlap)
    measured = [count(content[span.start:span.end]) for span in spans]
    return TokenCut(tuple(spans), tokens, overlap, method, sentences, over, hard, overlapped,
                    sum(1 for size in measured if size > tokens), max(measured, default=0))


def _trimmed(content: str, start: int, end: int) -> int:
    while end > start and content[end - 1].isspace():
        end -= 1
    return end


def _inside(content: str, start: int, end: int, room: int, count: Callable[[str], int], markers: int,
            pieces: list[_Piece]) -> int:
    """Append the pieces of a sentence too long for ``room``: its lines that
    fit, the words of a line that does not, and the hard-cut parts of a word
    that does not either. Returns the cuts made inside words."""
    cuts = 0
    for line_start, line_end in _parts(content, start, end, "\n"):
        size = count(content[line_start:line_end]) - markers
        if size <= room:
            pieces.append(_Piece(line_start, line_end, size))
            continue
        for word_start, word_end in _parts(content, line_start, line_end, None):
            size = count(content[word_start:word_end]) - markers
            if size <= room:
                pieces.append(_Piece(word_start, word_end, size))
                continue
            cuts += _hard(content, word_start, word_end, room, count, markers, pieces)
    return cuts


def _parts(content: str, start: int, end: int, separator: str | None) -> list[tuple[int, int]]:
    """The parts of ``content[start:end]`` between separators (any whitespace
    for None), trimmed. The range ends in a character that is not
    whitespace -- a sentence and a line are trimmed before they are split --
    so skipping whitespace always stops inside it, and no part is blank."""
    found: list[tuple[int, int]] = []
    index = start
    while index < end:
        while content[index].isspace():
            index += 1
        part = index
        while index < end and not (content[index].isspace() if separator is None else content[index] == separator):
            index += 1
        found.append((part, _trimmed(content, part, index)))
    return found


def _hard(content: str, start: int, end: int, room: int, count: Callable[[str], int], markers: int,
          pieces: list[_Piece]) -> int:
    """Cut one word into the longest parts that fit ``room``; each part holds
    at least one character, so the cut always moves. Returns the cuts made."""
    cuts = 0
    at = start
    while at < end:
        low, high = at + 1, end
        # Invariant: a part ending at `low` is taken (it fits, or it is the
        # one character that must be taken); one ending past `high` is not.
        while low < high:
            middle = (low + high + 1) // 2
            if count(content[at:middle]) - markers <= room:
                low = middle
            else:
                high = middle - 1
        pieces.append(_Piece(at, low, count(content[at:low]) - markers))
        cuts += low < end
        at = low
    return cuts


def _packed(pieces: list[_Piece], room: int, overlap: int) -> tuple[list[Span], int]:
    """Greedy packing of pieces into spans within ``room``, each span after
    the first starting with up to ``overlap`` tokens of the one before."""
    spans: list[Span] = []
    overlapped = 0
    chunk: list[_Piece] = []
    used = carried = 0
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        if used + piece.tokens > room and len(chunk) > carried:
            spans.append(Span(chunk[0].start, chunk[-1].end))
            overlapped += carried > 0
            chunk, used = _carry(chunk, overlap)
            carried = len(chunk)
            continue
        while chunk and used + piece.tokens > room:
            used -= chunk.pop(0).tokens
            carried -= 1
        chunk.append(piece)
        used += piece.tokens
        index += 1
    if chunk:
        spans.append(Span(chunk[0].start, chunk[-1].end))
    return spans, overlapped + (carried > 0)


def _carry(chunk: list[_Piece], overlap: int) -> tuple[list[_Piece], int]:
    kept: list[_Piece] = []
    used = 0
    for piece in reversed(chunk):
        if used + piece.tokens > overlap:
            break
        kept.insert(0, piece)
        used += piece.tokens
    return kept, used
