"""Cut where the subject changes, not where the byte count lands.

``chunker.chunk_spans`` cuts at a size target and prefers a paragraph
break, then a sentence end, then whitespace. Every one of those is a
*typographic* signal: it is about how the text was typed, not about what
it says. Two unrelated paragraphs run together without a blank line are
one chunk, and a single argument that happens to cross the target is two.

This reads the text instead. Sentences are embedded, adjacent blocks are
compared, and a cut is made where the two sides stop being about the same
thing. The size target still **bounds** a chunk -- meaning decides where
to cut, never whether to -- so a long passage on one subject is still
split.

Opt-in, for the same reason structure-aware chunking is: cut positions
decide what chunks exist and stored offsets are part of the shared
specification, so it is a caller's choice and not ours to make for a
space that already holds chunks cut another way.

**Why not the reference's rule.** ``llama_index``'s semantic splitter
embeds each sentence together with its neighbours (``buffer_size=1``) and
cuts where the distance between consecutive embeddings exceeds the 95th
percentile. Both halves misbehave:

- A window that straddles a boundary contains *both* subjects, so the
  true boundary looks more similar than its neighbours do. Measured on
  two paragraphs about different things, the real boundary scored 0.2372
  and a false one inside the second paragraph scored 0.2602 -- the
  windowing inverted them. Blocks are compared here on either side of a
  candidate cut, never across it.
- A percentile always has something above it. Text that is entirely
  about one subject still has a most-distant sentence pair, so a
  percentile rule cuts it anyway. The rule here can return no cuts at
  all, which is the correct answer for a single coherent passage.

What is used instead is a valley test: a cut needs similarity that is
lower than the text on *both* sides of it (Hearst's depth score), and
deep enough to stand out from the rest of that text. Measured with
``HashEmbedder`` over the tests' fixtures: a true subject change scored
1.002, a passage on one subject peaked at 0.161, and the same passage
repeated four times -- where a one-sided rule invents a boundary at every
repeat -- peaked at 0.155. The threshold is derived from each text's own
depths, so it does not assume the scale of any one embedder's distances.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from ..core.ports import Embedder
from .chunker import DEFAULT_TARGET, Span, chunk_spans

#: Words that are always followed by a name, so the stop after them
#: closes a title and not a sentence. Deliberately short: a word listed
#: here can never end a sentence again, and being wrong in that direction
#: silently glues two chunks together.
TITLES = frozenset(
    "dr mr mrs ms miss prof rev hon sr jr st capt sgt lt col gen".split()
)

#: Sentences compared on each side of a candidate cut. One sentence is
#: too noisy -- with a single sentence either side, repeated text scored
#: 0.553 at boundaries that are not there, against 0.155 with two.
BLOCK = 2

#: How far above a text's own mean depth a valley must be, in standard
#: deviations, before it is a boundary. Higher cuts less.
SENSITIVITY = 1.5


@dataclass(frozen=True)
class Boundary(Span):
    """A chunk, and whether meaning or size ended it.

    ``subject_changed`` is false for every chunk when the embedder's
    distances are too flat for any valley to stand out. That is a real
    outcome and has to be visible: without it, semantic chunking would
    quietly degrade to size chunking under a name that says otherwise.
    """

    subject_changed: bool = False


async def semantic_spans(
    content: str,
    embedder: Embedder,
    target: int = DEFAULT_TARGET,
    sensitivity: float = SENSITIVITY,
    block: int = BLOCK,
) -> list[Boundary]:
    """Spans over ``content``, cut where its subject changes.

    Costs one embedding call for the whole episode, batched over its
    sentences. Offsets are code points, as ``chunk_spans`` returns;
    ``chunker.byte_spans`` converts for storage.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    sentences = _sentences(content)
    if not sentences:
        return []
    if len(sentences) == 1:
        only = sentences[0]
        return _by_size(content, only.start, only.end, target)
    vectors = await embedder.embed([content[s.start : s.end].strip() for s in sentences])
    similar = _block_similarity(vectors, block)
    return _assemble(content, sentences, _valleys(similar, sensitivity), target)


def _sentences(text: str) -> list[Span]:
    """Sentence spans, each running to the start of the next so that the
    separating whitespace belongs to something and nothing is lost."""
    spans: list[Span] = []
    start: int | None = None
    index, size = 0, len(text)
    while index < size:
        if start is None:
            if text[index].isspace():
                index += 1
                continue
            start = index
        if text[index] in ".!?":
            after = index + 1
            # Closing punctuation belongs to the sentence that ends: a
            # quotation or a bracket after the stop is not a new one.
            while after < size and text[after] in ".!?\"')]":
                after += 1
            if after >= size or text[after].isspace():
                while after < size and text[after].isspace():
                    after += 1
                if _ends_a_sentence(text, index, after):
                    spans.append(Span(start, after))
                    start, index = None, after
                    continue
        index += 1
    if start is not None:
        spans.append(Span(start, size))
    return spans


def _ends_a_sentence(text: str, stop: int, after: int) -> bool:
    """Whether the stop at ``stop`` closes a sentence or sits inside one.

    Every cut this module makes lands on a sentence boundary, so a stop
    read wrongly is a cut in the wrong place -- and the damage runs in
    both directions. Splitting `J. Anderson` puts a boundary where the
    text has none; refusing to split after a word that really did end a
    sentence hides one. Only `!` and `?` are unambiguous.

    Three rules, all conservative:

    - an **initial** -- a single letter before the stop -- is part of a
      name. A sentence ending in a lone letter (``option A.``) is read as
      continuing, which is the rarer mistake of the two;
    - a **title** is followed by whoever it belongs to;
    - a stop followed by a **lower-case** word is inside a sentence, which
      covers `i.e.`, `e.g.` and `etc.` without listing them, and without
      claiming they can never end one.
    """
    if text[stop] != ".":
        return True
    edge = stop
    while edge > 0 and (text[edge - 1].isalpha() or text[edge - 1] == "."):
        edge -= 1
    word = text[edge:stop].replace(".", "")
    if len(word) == 1 or word.lower() in TITLES:
        return False
    return not (after < len(text) and text[after].islower())


def _block_similarity(vectors: list[list[float]], block: int) -> list[float]:
    """How alike the ``block`` sentences before each gap are to the
    ``block`` after it. Never averaged across the gap being judged."""
    return [
        _cosine(
            _centre(vectors[max(0, index + 1 - block) : index + 1]),
            _centre(vectors[index + 1 : index + 1 + block]),
        )
        for index in range(len(vectors) - 1)
    ]


def _valleys(similar: list[float], sensitivity: float) -> set[int]:
    """Gaps where similarity dips below both sides and stands out.

    Requiring a rise on **both** sides is what keeps a boundary from
    being invented at the last gap, where there is nothing to the right
    to disagree with, and in evenly repeating text, where every gap looks
    like every other one.

    It has a cost worth stating plainly: **the first and last gaps can
    never be boundaries**, because each has a side with nothing on it. A
    subject that changes immediately after the opening sentence is not
    found here. That is the deliberate trade -- and cutting there would
    leave a one-sentence chunk, which `chunker.MIN_CHUNK` exists to avoid
    for the same reason. Fewer than three gaps means no interior gap at
    all, so a text of three sentences or fewer is never cut by meaning.
    """
    if len(similar) < 3:
        return set()
    drops: list[tuple[float, float]] = []
    for index, here in enumerate(similar):
        left, at = here, index
        while at > 0 and similar[at - 1] >= left:
            left, at = similar[at - 1], at - 1
        right, at = here, index
        while at < len(similar) - 1 and similar[at + 1] >= right:
            right, at = similar[at + 1], at + 1
        drops.append((left - here, right - here))
    depths = [left + right for left, right in drops]
    cutoff = statistics.fmean(depths) + sensitivity * statistics.pstdev(depths)
    return {
        index
        for index, (left, right) in enumerate(drops)
        if left > 0 and right > 0 and left + right >= cutoff
    }


def _assemble(
    text: str, sentences: list[Span], cuts: set[int], target: int
) -> list[Boundary]:
    """Sentences gathered into chunks, ended by a cut or by the target."""
    spans: list[Boundary] = []
    index, count = 0, len(sentences)
    while index < count:
        start = sentences[index].start
        last = index
        while (
            last + 1 < count
            and last not in cuts
            and sentences[last + 1].end - start <= target
        ):
            last += 1
        end = sentences[last].end
        if last == index and end - start > target:
            spans.extend(_by_size(text, start, end, target))
        else:
            spans.append(Boundary(start, end, last in cuts and last + 1 < count))
        index = last + 1
    return spans


def _by_size(text: str, start: int, end: int, target: int) -> list[Boundary]:
    """One sentence longer than the target, cut the old way. Meaning has
    nothing to say inside a single sentence, and a target that a long
    sentence could ignore would not be a bound at all."""
    if end - start <= target:
        return [Boundary(start, end, False)]
    inner = chunk_spans(text[start:end], target)
    if not inner:
        return [Boundary(start, end, False)]
    spans: list[Boundary] = []
    edge = start
    for position, span in enumerate(inner):
        stop = end if position == len(inner) - 1 else start + span.end
        spans.append(Boundary(edge, stop, False))
        edge = stop
    return spans


def _centre(vectors: list[list[float]]) -> list[float]:
    return [sum(axis) / len(vectors) for axis in zip(*vectors)]


def _cosine(left: list[float], right: list[float]) -> float:
    scale = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / scale if scale else 0.0
