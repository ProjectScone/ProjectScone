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
The estimate reads the text once (``_Estimate``): every sentence, line,
word and chunk is then a difference of running totals. Estimating each
one's text again took 2.6 to 3.0 times as long over the bench's
conversations.

With ``overlap`` a chunk starts with the trailing whole pieces of the one
before it, as many as fit within that many tokens, so the stored spans of
neighbouring chunks overlap. An overlap that would leave no room for the
next piece gives way, from its front, before the chunk does.

Offsets are code points, as ``chunk_spans`` returns; ``chunker.byte_spans``
converts them. A chunk never starts or ends with whitespace.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, NamedTuple

from .chunker import Span
from .embedding_budget import BUDGET_VERSION, MARKERS, PIECE, estimated_tokens, piece_tokens
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


class _Estimate:
    """The budget's estimate for any range of one text, from one reading.

    A piece of the estimate never spans whitespace, so a range whose ends
    cut no piece -- a sentence, a line, a word, a chunk of them -- holds
    exactly the pieces its text would, and its estimate is the difference
    of running totals. A range that cuts a piece (a hard cut inside a word)
    is estimated from its own text."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.totals = [0]
        for match in PIECE.finditer(content):
            self.starts.append(match.start())
            self.ends.append(match.end())
            self.totals.append(self.totals[-1] + piece_tokens(match.group()))

    def tokens(self, start: int, end: int) -> int:
        """The estimate for ``content[start:end]``, less the markers."""
        first, last = bisect_left(self.starts, start), bisect_left(self.starts, end)
        if (first and self.ends[first - 1] > start) or (last and self.ends[last - 1] > end):
            return estimated_tokens(self.content[start:end]) - MARKERS
        return self.totals[last] - self.totals[first]


def token_spans(content: str, tokens: int, *, overlap: int = 0, count: Callable[[str], int] | None = None,
                method: str = BUDGET_VERSION) -> TokenCut:
    """Spans of ``content`` packed to ``tokens``, counted by ``count`` or,
    when None, by the budget's estimate."""
    reason = refused(tokens, overlap)
    if reason is not None:
        raise ValueError(reason)
    size: Callable[[int, int], int]
    if count is None:
        markers, size = MARKERS, _Estimate(content).tokens
    else:
        markers = count("")
        size = _counted(content, count, markers)
    room = tokens - markers
    if room < 1:
        raise ValueError(f"a target of {tokens} tokens leaves no room past the {markers} markers this count "
                         "adds to every input")
    pieces: list[_Piece] = []
    sentences = over = hard = 0
    for sentence in sentence_spans(content):
        end = _trimmed(content, sentence.start, sentence.end)
        sentences += 1
        tokens_in = size(sentence.start, end)
        if tokens_in <= room:
            pieces.append(_Piece(sentence.start, end, tokens_in))
            continue
        over += 1
        hard += _inside(content, sentence.start, end, room, size, pieces)
    spans, overlapped = _packed(pieces, room, overlap)
    measured = [size(span.start, span.end) + markers for span in spans]
    return TokenCut(tuple(spans), tokens, overlap, method, sentences, over, hard, overlapped,
                    sum(1 for whole in measured if whole > tokens), max(measured, default=0))


def _counted(content: str, count: Callable[[str], int], markers: int) -> Callable[[int, int], int]:
    def size(start: int, end: int) -> int:
        return count(content[start:end]) - markers
    return size


def _trimmed(content: str, start: int, end: int) -> int:
    while end > start and content[end - 1].isspace():
        end -= 1
    return end


def _inside(content: str, start: int, end: int, room: int, size: Callable[[int, int], int],
            pieces: list[_Piece]) -> int:
    """Append the pieces of a sentence too long for ``room``: its lines that
    fit, the words of a line that does not, and the hard-cut parts of a word
    that does not either. Returns the cuts made inside words."""
    cuts = 0
    for line_start, line_end in _parts(content, start, end, "\n"):
        line = size(line_start, line_end)
        if line <= room:
            pieces.append(_Piece(line_start, line_end, line))
            continue
        for word_start, word_end in _parts(content, line_start, line_end, None):
            word = size(word_start, word_end)
            if word <= room:
                pieces.append(_Piece(word_start, word_end, word))
                continue
            cuts += _hard(word_start, word_end, room, size, pieces)
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


def _hard(start: int, end: int, room: int, size: Callable[[int, int], int], pieces: list[_Piece]) -> int:
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
            if size(at, middle) <= room:
                low = middle
            else:
                high = middle - 1
        pieces.append(_Piece(at, low, size(at, low)))
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
