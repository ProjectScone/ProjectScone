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
    # Contiguous, not merely non-overlapping: chunks tile the text, so
    # each one begins where the last ended and no byte falls between two
    # of them. Counting characters alone cannot tell a hole from a
    # separator, and passed against spans that ended a character short.
    assert [s.end for s in spans[:-1]] == [s.start for s in spans[1:]]
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
    on one subject is still cut, or the target would be a suggestion."""
    text = ONE * 12
    spans = await semantic_spans(text, HashEmbedder(), target=300)
    assert len(spans) > 1
    assert all(s.end - s.start <= 300 * 2 for s in spans), [s.end - s.start for s in spans]


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
