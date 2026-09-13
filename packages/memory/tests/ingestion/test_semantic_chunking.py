"""Cutting where the subject changes, not where the byte count lands.

`llama_index`'s `node_parser` holds two splitters whose whole idea is
that a boundary belongs where adjacent sentences stop being about the
same thing. Ours cut at a size target, preferring a paragraph and then a
sentence end -- all of which are *typographic* signals. None of them look
at what the text says.

This is opt-in for the same reason structure-aware chunking is: cut
positions decide what chunks exist and stored offsets are part of the
shared specification, so it is a caller's choice and not ours to make for
an existing space.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder
from scone_memory.ingestion.semantic_chunks import semantic_spans

pytestmark = pytest.mark.asyncio

ONE = ("The cat sat on the mat. The cat was grey and the mat was red. "
       "The cat had sat on that mat every morning for a year. ")
OTHER = ("Quarterly revenue rose eleven percent. Revenue in the northern region "
         "rose fastest of all. Revenue guidance for the next quarter was raised. ")


async def test_a_span_is_exactly_the_bytes_of_its_source():
    """Whatever it cuts, it never rewrites: the spans are offsets into
    the text as given and cover every non-space character once."""
    text = ONE + OTHER
    spans = await semantic_spans(text, HashEmbedder(), target=200)
    covered = [0] * len(text)
    for span in spans:
        for index in range(span.start, span.end):
            covered[index] += 1
    assert all(n == 1 for index, n in enumerate(covered) if not text[index].isspace())
    assert all(n <= 1 for n in covered)
    # Anything falling between two chunks is whitespace. The earlier
    # version of this demanded they *tile* -- end exactly where the next
    # begins -- which is a stronger invariant than `chunk_spans` itself
    # keeps, since it skips the separator between chunks. Holding the
    # stronger line here is what forced an unbounded whitespace run to be
    # re-attached to a chunk, and that is what made the size ceiling
    # false. The real claim is that nothing but whitespace is uncovered.
    assert all(not text[a.end:b.start].strip() for a, b in zip(spans, spans[1:]))
    assert "".join(text[s.start:s.end] for s in spans).replace(" ", "") \
        == text.replace(" ", "").replace("\n", "")


async def test_it_cuts_where_the_subject_changes():
    """Two paragraphs about different things, with no blank line and no
    size pressure at the junction: a size-based cutter has no reason to
    cut there and this one does."""
    text = ONE + OTHER
    spans = await semantic_spans(text, HashEmbedder(), target=400)
    assert len(spans) == 2, [text[s.start:s.end] for s in spans]
    first = text[spans[0].start:spans[0].end]
    assert "cat" in first and "revenue" not in first.lower(), first


async def test_one_subject_is_not_cut_for_no_reason():
    """The other half: text that stays on one subject and fits the target
    is one chunk. A splitter that cuts at every sentence would pass the
    test above and be useless."""
    spans = await semantic_spans(ONE, HashEmbedder(), target=400)
    assert len(spans) == 1, [ONE[s.start:s.end] for s in spans]


async def test_the_size_target_still_bounds_a_chunk():
    """Meaning decides *where* to cut, never whether to. A long passage
    on one subject is still cut, or the target would be a suggestion.

    What this pins is that a long passage **is** cut: replacing the
    target with a hundred times itself fails here.

    It does not pin the ceiling. This fixture is sentences short enough
    that no tail-merge happens, so it passes just as well against
    `widest(target) == target` -- a mutation proof says so. The ceiling
    is pinned by `test_one_long_sentence_keeps_the_same_bound`, which
    reaches the junction this one cannot. Recorded because a test named
    for a bound that does not exercise it is how the prose came to
    promise a bound the code did not keep.
    """
    from scone_memory.ingestion.semantic_chunks import widest

    text = ONE * 12
    spans = await semantic_spans(text, HashEmbedder(), target=300)
    assert len(spans) > 1
    assert all(s.end - s.start <= widest(300) for s in spans), [s.end - s.start for s in spans]


async def test_one_long_sentence_keeps_the_same_bound():
    """The path with no sentence boundary to cut at. A single sentence
    one byte over the target is one chunk, because splitting it would
    leave a one-byte tail; `target + MIN_CHUNK` still holds."""
    from scone_memory.ingestion.semantic_chunks import widest

    for size in (701, 900, 1500, 4000):
        spans = await semantic_spans("x" * size, HashEmbedder(), target=700)
        assert all(s.end - s.start <= widest(700) for s in spans), \
            (size, [s.end - s.start for s in spans])
        assert sum(s.end - s.start for s in spans) == size


async def test_text_with_one_sentence_is_one_chunk():
    spans = await semantic_spans("Only a single sentence here.", HashEmbedder(), target=400)
    assert len(spans) == 1
    assert spans[0].start == 0


async def test_empty_text_is_no_chunks():
    assert await semantic_spans("", HashEmbedder(), target=400) == []


async def test_a_repeated_subject_is_not_a_boundary():
    """Four passes over the same three sentences, with no size pressure.

    Every gap here looks like every other one, so any rule that ranks
    gaps against each other finds a most-distant pair and cuts there --
    which is what a percentile threshold does, and what comparing single
    sentences does. Measured on this text, single-sentence comparison
    scores 0.553 at two gaps that are not boundaries; blocks of two peak
    at 0.155. This is the fixture `ONE` cannot be, because two gaps are
    too few for a valley to have sides.
    """
    text = ONE * 4
    spans = await semantic_spans(text, HashEmbedder(), target=2000)
    assert len(spans) == 1, [text[s.start:s.end] for s in spans]


async def test_a_chunk_says_whether_meaning_ended_it():
    """A chunker that silently found no boundaries would return exactly
    what a size chunker returns, under a name that promises otherwise.
    Each chunk carries which of the two ended it."""
    spans = await semantic_spans(ONE + OTHER, HashEmbedder(), target=400)
    assert [s.subject_changed for s in spans] == [True, False]
    alone = await semantic_spans(ONE, HashEmbedder(), target=400)
    assert alone[0].subject_changed is False


async def test_the_runtime_cuts_semantically_only_when_asked():
    """Off unless asked for, and code keeps the better cut it already has.

    There is one dispatch point on purpose. A second, synchronous one
    would let a caller reach chunking without the embedder and get
    size-cut chunks while believing otherwise -- the same silent
    degradation `subject_changed` exists to prevent.
    """
    from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex
    from scone_memory.ingestion.batch import IngestionRuntime, spans_for

    def runtime(**options):
        return IngestionRuntime(
            documents=InMemoryDocumentStore(), vectors=InMemoryVectorIndex(),
            embedder=HashEmbedder(), clock=lambda: "2026-09-12T00:00:00Z",
            chunk_target=400, embed_text=lambda episode, text: text,
            emit=None, **options)  # type: ignore[arg-type]

    prose = ONE + OTHER
    assert len(await spans_for(runtime(), prose, "notes.txt")) == 1
    asked = await spans_for(runtime(semantic_aware=True), prose, "notes.txt")
    assert len(asked) == 2, [prose[s.start:s.end] for s in asked]

    code = "import os\n\n\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    assert ([(s.start, s.end) for s in await spans_for(runtime(semantic_aware=True), code, "m.py")]
            == [(s.start, s.end) for s in await spans_for(runtime(), code, "m.py")]), \
        "code must still be cut at its declarations"


async def test_an_engine_asked_for_semantic_chunks_stores_them():
    """End to end: the flag reaches what is actually written, and the
    same text without it is stored as one chunk."""
    from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

    async def stored(**options):
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                    HashEmbedder(), chunk_target=400, **options).open()
        try:
            return (await engine.remember("s", ONE + OTHER, source="notes.txt")).chunks
        finally:
            await engine.close()

    assert await stored() == 1
    assert await stored(semantic_aware=True) == 2


async def test_a_stop_inside_a_name_or_an_abbreviation_is_not_a_boundary():
    """Every cut this module makes lands on a sentence boundary, so a
    wrong boundary is a wrong cut. A title, an initial, and a stop
    followed by a lower-case word are all mid-sentence."""
    from scone_memory.ingestion.semantic_chunks import _sentences

    text = "Dr. J. Anderson joined in May. Revenue rose, i.e. it grew. Fine."
    assert [text[s.start:s.end] for s in _sentences(text)] == [
        "Dr. J. Anderson joined in May. ",
        "Revenue rose, i.e. it grew. ",
        "Fine.",
    ]


async def test_byte_spans_of_a_semantic_cut_name_their_own_text():
    """Cuts are code-point offsets; stored offsets are UTF-8 bytes.

    The conversion is `chunker.byte_spans`, and the junction that breaks
    it is a cut with multi-byte characters before it -- so the fixture
    has accented prose and CRLF line endings on both sides of a real
    subject change. Asserted on the bytes themselves, never on lengths:
    comparing lengths here is a tautology.
    """
    from scone_memory.ingestion.chunker import byte_spans

    left = ("Le chat était assis sur le tapis.\r\n"
            "Le chat était gris et le tapis était rouge.\r\n"
            "Le chat s'était assis là chaque matin pendant un an.\r\n")
    right = ("Les recettes trimestrielles ont augmenté de onze pour cent.\r\n"
             "Les recettes du nord ont augmenté le plus vite.\r\n"
             "Les prévisions de recettes ont été relevées.\r\n")
    text = left + right
    assert not text.isascii(), "the fixture must exercise the conversion"
    spans = await semantic_spans(text, HashEmbedder(), target=400)
    raw = text.encode()
    for span, converted in zip(spans, byte_spans(text, spans)):
        assert raw[converted.start:converted.end].decode() == text[span.start:span.end]


class _Broken:
    """An embedder that answers wrongly in one specific way."""

    id, dim = "broken", 4

    def __init__(self, answer):
        self._answer = answer

    async def embed(self, texts):
        return self._answer(list(texts))


@pytest.mark.parametrize("name, answer", [
    ("nothing at all", lambda texts: []),
    ("one short", lambda texts: [[1.0] * 4] * (len(texts) - 1)),
    ("one extra", lambda texts: [[1.0] * 4] * (len(texts) + 1)),
    ("widths that alternate", lambda texts: [[1.0] if i % 2 else [1.0, 0.0, 0.0, 0.0]
                                             for i in range(len(texts))]),
    ("a width it never declared", lambda texts: [[1.0, 0.0]] * len(texts)),
    ("an empty vector", lambda texts: [[]] * len(texts)),
    ("not a list", lambda texts: "vectors"),
    ("infinity", lambda texts: [[float("inf")] * 4] * len(texts)),
    ("a NaN", lambda texts: [[float("nan")] * 4] * len(texts)),
])
async def test_an_embedder_answering_wrongly_is_refused(name, answer):
    """The failure that must never be silent.

    With no check, an embedder returning nothing yields one span covering
    the whole text with `subject_changed=False` -- which is *exactly*
    what a coherent passage returns. Broken provider output would arrive
    disguised as a finding about the text. Ingestion validates every
    other embedding call this way; this one did not.
    """
    with pytest.raises(ValueError):
        await semantic_spans(ONE + OTHER, _Broken(answer), target=400)


async def test_valleys_are_found_in_passes_not_by_walking_out_from_every_gap():
    """A plateau is the worst case and the likeliest one.

    Walking outward from each gap re-reads the whole flat run every time:
    measured 1000 gaps 0.059s, 2000 0.198s, 4000 0.780s -- quadratic. A
    repetitive or zero-vector stream produces exactly that shape, and the
    input limit allows a great many sentences.
    """
    import time

    from scone_memory.ingestion.semantic_chunks import SENSITIVITY, _valleys

    flat = [0.5] * 20000
    started = time.perf_counter()
    assert _valleys(flat, SENSITIVITY) == set()
    assert time.perf_counter() - started < 5.0, "still walking outward from every gap"


async def test_the_faster_valleys_agree_with_walking_outward():
    """The refactor's real risk is changing which gaps are boundaries, so
    this compares against a literal transcription of the outward walk
    over shapes built to have plateaus, ties and edges."""
    import random

    from scone_memory.ingestion.semantic_chunks import SENSITIVITY, _valleys

    def by_walking(similar, sensitivity):
        import statistics
        if len(similar) < 3:
            return set()
        drops = []
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
        return {i for i, (left, right) in enumerate(drops)
                if left > 0 and right > 0 and left + right >= cutoff}

    rng = random.Random(20260912)
    for trial in range(300):
        length = rng.randint(0, 40)
        # Few distinct values, so plateaus and ties are common rather than rare.
        similar = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(length)]
        assert _valleys(similar, SENSITIVITY) == by_walking(similar, SENSITIVITY), similar


async def test_a_run_of_whitespace_cannot_push_a_chunk_past_the_ceiling():
    """`widest()` has to be true of the spans, not just of their text.

    `'x' * 700 + ' ' * 1000` produced a single 1700-character span,
    because the trailing whitespace was re-attached to the last chunk to
    keep the spans tiling. Whitespace between chunks belongs to nobody --
    that is what `chunk_spans` already does -- and a ceiling that a
    thousand spaces can breach is not a ceiling.
    """
    from scone_memory.ingestion.semantic_chunks import widest

    for text in ("x" * 700 + " " * 1000,
                 "x" * 700 + " " * 1000 + "y" * 700,
                 "x" + " " * 3000,
                 "one. " + " " * 2000 + "two. " + " " * 2000 + "three."):
        spans = await semantic_spans(text, HashEmbedder(), target=700)
        assert all(s.end - s.start <= widest(700) for s in spans), \
            (text[:12], [s.end - s.start for s in spans])
        assert all(not text[a.end:b.start].strip() for a, b in zip(spans, spans[1:]))
        assert "".join(text[s.start:s.end] for s in spans).replace(" ", "") \
            == text.replace(" ", "")


class _Drifting:
    """An embedder whose identity changes while the request is in flight."""

    def __init__(self, *, ident="steady", dim=4, becomes=None, widens=None):
        self.id, self.dim = ident, dim
        self._becomes, self._widens = becomes, widens

    async def embed(self, texts):
        width = self.dim if self._widens is None else self._widens
        if self._becomes is not None:
            self.id = self._becomes
        if self._widens is not None:
            self.dim = self._widens
        return [[1.0] * width for _ in texts]


@pytest.mark.parametrize("name, embedder", [
    ("its id changes mid-call", _Drifting(becomes="someone-else")),
    ("its width changes mid-call", _Drifting(dim=2, widens=3)),
])
async def test_an_embedder_that_changes_identity_mid_call_is_refused(name, embedder):
    """Read after the await, `embedder.dim` is whatever it became -- so a
    response validated against it validates against the wrong thing.
    Ingestion snapshots both before the request; this must too."""
    with pytest.raises(ValueError):
        await semantic_spans(ONE + OTHER, embedder, target=400)
