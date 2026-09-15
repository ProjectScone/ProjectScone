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

**A second pass, when asked.** ``semantic_cut(merge_threshold=...)`` then
walks the chunks in order and joins a chunk to the one before it when
the two are at least that alike, and together still fit the target --
the second half of the reference's double-merging splitter. The valley
test is relative, so text that is about one subject all the way through
can still be cut at its deepest dip; the merge threshold is the
*absolute* check it deliberately does not make. That is also why the
pass is off unless a threshold is given: a cosine threshold does assume
an embedder's scale, and one that suits hashed tokens will not suit a
neural model. Because the first pass packs sentences up to the target,
the only chunks that can fit together are ones a valley ended -- this
undoes cuts, it never adds one.

Similarity is compared between the mean of each side's sentence vectors,
which the first pass already has, so the second pass costs no embedding
call and gives the same answer every time. The reference embeds every
joined text again, and also reaches one or two chunks past a dissimilar
neighbour to merge across it; neither is done here -- merging across a
chunk would pull in a chunk that failed the very test.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from ..core.ports import Embedder
from .chunker import DEFAULT_TARGET, MIN_CHUNK, Span, chunk_spans
from .vectors import validated_vectors

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


def widest(target: int = DEFAULT_TARGET) -> int:
    """The largest chunk ``semantic_spans`` can return for ``target``.

    A bound that lives only in prose is a bound nobody can check, and
    this one was described as ``target`` while the code allowed more. A
    final fragment shorter than ``MIN_CHUNK`` joins its predecessor
    instead of standing alone, so the real ceiling is that much higher.
    """
    return target + MIN_CHUNK


@dataclass(frozen=True)
class Boundary(Span):
    """A chunk, and whether meaning or size ended it.

    ``subject_changed`` is false for every chunk when the embedder's
    distances are too flat for any valley to stand out. That is a real
    outcome and has to be visible: without it, semantic chunking would
    quietly degrade to size chunking under a name that says otherwise.
    """

    subject_changed: bool = False


def merge_refused(threshold: object) -> str | None:
    """Why ``threshold`` cannot be a merge threshold, or None when it can.

    A similarity above 0 and at most 1. At or below 0 every pair of chunks
    that is not outright opposed would merge, which is size chunking under
    a semantic name; above 1 nothing ever merges, which is the pass turned
    off while the receipt says it ran. A bool is refused although Python
    counts it as a number: ``True`` is a switch, not a similarity.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return f"semantic_merge_threshold must be a number, not {type(threshold).__name__}"
    if not 0 < threshold <= 1:
        return f"semantic_merge_threshold must be a similarity above 0 and at most 1, got {threshold!r}"
    return None


@dataclass(frozen=True)
class SemanticCut:
    """A semantic cut, and what its second pass did.

    ``groups`` is how many chunks the first pass made. Each place two of
    them were not joined is counted once: ``stopped_by_similarity`` when
    they were less alike than the threshold, ``stopped_by_size`` when they
    were alike enough and would not fit the target together -- the count
    that says the bound bit. Two chunks that fail both are counted as
    unalike, since that alone kept them apart.
    """

    spans: list[Boundary]
    #: None when no second pass was asked for.
    merge_threshold: float | None = None
    groups: int = 0
    merges: int = 0
    stopped_by_similarity: int = 0
    stopped_by_size: int = 0

    def record(self) -> dict[str, object] | None:
        """The receipt's counts, or None when there was no second pass."""
        if self.merge_threshold is None:
            return None
        return {"merge_threshold": self.merge_threshold, "groups": self.groups, "merges": self.merges,
                "stopped_by_similarity": self.stopped_by_similarity, "stopped_by_size": self.stopped_by_size}


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

    A chunk is at most ``target + MIN_CHUNK``, not ``target``. A final
    fragment shorter than ``MIN_CHUNK`` joins its predecessor rather than
    standing alone as a poor retrieval unit -- ``chunker`` does that
    deliberately and this inherits it. Saying "the target bounds a chunk"
    was wrong by up to ``MIN_CHUNK`` bytes, and a bound that does not
    bind should not be described as one.
    """
    return (await semantic_cut(content, embedder, target, sensitivity=sensitivity, block=block)).spans


async def semantic_cut(
    content: str,
    embedder: Embedder,
    target: int = DEFAULT_TARGET,
    *,
    merge_threshold: float | None = None,
    sensitivity: float = SENSITIVITY,
    block: int = BLOCK,
) -> SemanticCut:
    """``semantic_spans``, with the second pass when ``merge_threshold``
    is given, and the counts that say what that pass did.

    Merged chunks are bounded by ``target`` exactly as the first pass
    packs sentences, so a merge never makes a chunk the first pass could
    not have made for its length. Still one embedding call.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if merge_threshold is not None:
        reason = merge_refused(merge_threshold)
        if reason is not None:
            raise ValueError(reason)
    sentences = sentence_spans(content)
    if not sentences:
        return SemanticCut([], merge_threshold)
    if len(sentences) == 1 and merge_threshold is None:
        # Nothing to compare, so no embedding call. With a second pass the
        # sentence is embedded after all: pieces of one sentence cut by
        # size are the bound biting, and the receipt has to be able to say so.
        only = sentences[0]
        return SemanticCut(_bounded(content, _by_size(content, only.start, only.end, target), target))
    asked = [content[s.start : s.end].strip() for s in sentences]
    # Snapshot before the await, not after. Read afterwards, `embedder.dim`
    # is whatever it became, so a response is validated against the width
    # the embedder ended up claiming rather than the one it was asked
    # under -- and a 2-wide embedder that answers 3-wide passes.
    identity, width = embedder.id, embedder.dim
    # Checked exactly as every other embedding call in ingestion is.
    # Unchecked, an embedder returning nothing yields one span over the
    # whole text with `subject_changed=False` -- indistinguishable from a
    # genuinely coherent passage, so a broken provider would arrive
    # disguised as a finding about the text.
    vectors = validated_vectors(await embedder.embed(asked), len(asked), width)
    if embedder.id != identity or embedder.dim != width:
        raise ValueError('embedding identity changed while chunking')
    similar = _block_similarity(vectors, block)
    groups = _assemble(content, sentences, _valleys(similar, sensitivity), target)
    if merge_threshold is None:
        return SemanticCut(_bounded(content, [span for span, _, _ in groups], target))
    spans, (merges, unalike, too_long) = _merged(groups, vectors, merge_threshold, target)
    return SemanticCut(_bounded(content, spans, target), merge_threshold, len(groups), merges, unalike, too_long)


def _merged(
    groups: list[tuple[Boundary, int, int]], vectors: list[list[float]], threshold: float, target: int
) -> tuple[list[Boundary], tuple[int, int, int]]:
    """Each chunk joined to the one before it while the two are at least
    ``threshold`` alike and fit ``target`` together.

    A group is a span and the first and last sentence it holds; pieces of
    one sentence cut by size hold the same one. A joined chunk is compared
    by the mean of *all* its sentences, not by the chunk it started as nor
    the one it last took in, so a run of merges cannot drift from where it
    began one small step at a time. It keeps whether its own end is a
    change of subject: that is the end it now has.

    Returns the spans and (merges, stopped by similarity, stopped by size).
    ``groups`` is never empty: a text with a sentence has a chunk.
    """
    joined: list[Boundary] = []
    merges = unalike = too_long = 0
    span, first, last = groups[0]
    for following, start, end in groups[1:]:
        alike = _cosine(_centre(vectors[first : last + 1]), _centre(vectors[start : end + 1])) >= threshold
        if alike and following.end - span.start <= target:
            span, last = Boundary(span.start, following.end, following.subject_changed), end
            merges += 1
            continue
        if alike:
            too_long += 1
        else:
            unalike += 1
        joined.append(span)
        span, first, last = following, start, end
    joined.append(span)
    return joined, (merges, unalike, too_long)


def _bounded(text: str, spans: list[Boundary], target: int) -> list[Boundary]:
    """Enforce ``widest(target)`` on the spans, rather than assert it.

    A span can only exceed the ceiling by reaching across whitespace.
    ``chunk_spans`` bounds the text it puts in a chunk, but its tail merge
    joins a short ending to its predecessor *whatever lies between them*
    -- so `'one.' + 2000 spaces + 'two.' + 2000 spaces + 'three.'` came
    back as a 2011-character span holding twelve characters of text. The
    fix is to undo that reach at the gap it crossed, which is exactly the
    state before the merge.

    A bound that the code enforces is worth more than one the prose
    claims, and this module has now got that wrong twice.
    """
    ceiling = widest(target)
    settled: list[Boundary] = []
    for span in spans:
        settled.extend(_unreach(text, span, ceiling))
    return settled


def _unreach(text: str, span: Boundary, ceiling: int) -> list[Boundary]:
    """One span, split at its longest run of whitespace until it fits."""
    if span.end - span.start <= ceiling:
        return [span]
    body = text[span.start : span.end]
    widest_start, widest_end, index = -1, -1, 0
    while index < len(body):
        if not body[index].isspace():
            index += 1
            continue
        run = index
        while run < len(body) and body[run].isspace():
            run += 1
        if run - index > widest_end - widest_start:
            widest_start, widest_end = index, run
        index = run
    if widest_start < 0:
        # No whitespace to give back: `chunk_spans` bounds text by the
        # target, so this is unreachable -- and returning the span whole
        # is the honest answer rather than cutting a word in half.
        return [span]
    left = Boundary(span.start, span.start + widest_start, False)
    right = Boundary(span.start + widest_end, span.end, span.subject_changed)
    return _unreach(text, left, ceiling) + _unreach(text, right, ceiling)


def sentence_spans(text: str) -> list[Span]:
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
                if ends_a_sentence(text, index, after):
                    spans.append(Span(start, after))
                    start, index = None, after
                    continue
        index += 1
    if start is not None:
        spans.append(Span(start, size))
    return spans


def ends_a_sentence(text: str, stop: int, after: int) -> bool:
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

    Computed in two passes rather than by walking outward from every
    gap. The outward walk re-reads a flat run once per gap, which is
    quadratic -- measured at 1000 gaps 0.059s, 2000 0.198s, 4000 0.780s
    -- and a plateau is both the worst case and the likeliest one, since
    a repetitive or zero-vector stream produces exactly that. The
    recurrence is exact, not an approximation: walking left from ``i``
    continues from ``i - 1`` with the same running maximum whenever
    ``similar[i - 1] >= similar[i]``, which is the definition of
    ``left[i - 1]``.

    It has a cost worth stating plainly: **the first and last gaps can
    never be boundaries**, because each has a side with nothing on it. A
    subject that changes immediately after the opening sentence is not
    found here. That is the deliberate trade -- and cutting there would
    leave a one-sentence chunk, which `chunker.MIN_CHUNK` exists to avoid
    for the same reason. Fewer than three gaps means no interior gap at
    all, so a text of three sentences or fewer is never cut by meaning.
    """
    size = len(similar)
    if size < 3:
        return set()
    highest_left = [similar[0]] * size
    for index in range(1, size):
        highest_left[index] = (highest_left[index - 1]
                               if similar[index - 1] >= similar[index] else similar[index])
    highest_right = [similar[-1]] * size
    for index in range(size - 2, -1, -1):
        highest_right[index] = (highest_right[index + 1]
                                if similar[index + 1] >= similar[index] else similar[index])
    drops = [(highest_left[index] - here, highest_right[index] - here)
             for index, here in enumerate(similar)]
    depths = [left + right for left, right in drops]
    cutoff = statistics.fmean(depths) + sensitivity * statistics.pstdev(depths)
    return {
        index
        for index, (left, right) in enumerate(drops)
        if left > 0 and right > 0 and left + right >= cutoff
    }


def _assemble(
    text: str, sentences: list[Span], cuts: set[int], target: int
) -> list[tuple[Boundary, int, int]]:
    """Sentences gathered into chunks, ended by a cut or by the target,
    each with the first and last sentence it holds."""
    spans: list[tuple[Boundary, int, int]] = []
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
            spans.extend((piece, index, index) for piece in _by_size(text, start, end, target))
        else:
            spans.append((Boundary(start, end, last in cuts and last + 1 < count), index, last))
        index = last + 1
    return spans


def _by_size(text: str, start: int, end: int, target: int) -> list[Boundary]:
    """One sentence longer than the target, cut the old way. Meaning has
    nothing to say inside a single sentence, and a target that a long
    sentence could ignore would not be a bound at all.

    Each span is exactly what ``chunk_spans`` chose. An earlier version
    stretched the last one out to ``end`` so that the spans would *tile*
    the text -- and an unbounded run of whitespace then rode into a chunk
    with it, which is how ``'x' * 700 + ' ' * 1000`` became a single
    1700-character span against a ceiling of 820. Whitespace between
    chunks belongs to nobody; ``chunk_spans`` has always left it behind,
    and the invariant that actually holds is that nothing **but**
    whitespace falls between two chunks.
    """
    if end - start <= widest(target):
        return [Boundary(start, end, False)]
    inner = chunk_spans(text[start:end], target)
    if not inner:
        return [Boundary(start, end, False)]
    return [Boundary(start + span.start, start + span.end, False) for span in inner]


def _centre(vectors: list[list[float]]) -> list[float]:
    return [sum(axis) / len(vectors) for axis in zip(*vectors)]


def _cosine(left: list[float], right: list[float]) -> float:
    scale = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / scale if scale else 0.0
