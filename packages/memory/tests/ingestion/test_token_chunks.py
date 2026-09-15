"""A chunk measured in tokens, packed from whole sentences.

The length chunker cuts at 700 characters. The reference framework cuts
at 512 tokens, packing whole sentences, so one of its nodes carries more
of a question's words than one of our chunks does. ``chunk_tokens``
measures the length chunker's target in tokens instead: sentences are
packed whole up to the target, a sentence is cut inside only when it is
longer than the target on its own, and the receipt says how often that
happened. Tokens are counted by the embedder's tokenizer when it has one
and by the budget's estimate otherwise, so no model is needed.
"""

from __future__ import annotations

import math

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import token_chunks
from scone_memory.ingestion.chunker import chunk_spans
from scone_memory.ingestion.embedding_budget import BUDGET_VERSION, PIECE, TOKENIZER_VERSION, estimated_tokens
from scone_memory.ingestion.semantic_chunks import sentence_spans
from scone_memory.ingestion.token_chunks import MIN_CHUNK_TOKENS, _Estimate, refused, token_spans

SENTENCES = [
    "The harbour crane was repainted in May.",
    "Juniper keeps the calibration notes in the blue binder.",
    "The survey team found rust on the jib.",
    "Nobody signed the handover sheet that week.",
    "The Lisbon office asked for the maintenance log twice.",
    "A new winch arrives on Thursday.",
    "The yard foreman wants the old one kept as a spare.",
    "Insurance covers the scaffolding but not the paint.",
] * 3
PROSE = " ".join(SENTENCES)


def tokens(text: str) -> int:
    return estimated_tokens(text)


def texts(content: str, spans) -> list[str]:
    return [content[span.start:span.end] for span in spans]


def sentence_ends(content: str) -> set[int]:
    ends = set()
    for span in sentence_spans(content):
        end = span.end
        while end > span.start and content[end - 1].isspace():
            end -= 1
        ends.add(end)
    return ends


def test_whole_sentences_are_packed_up_to_the_target_and_no_further():
    cut = token_spans(PROSE, 48)
    chunks = texts(PROSE, cut.spans)
    assert len(chunks) > 2
    assert all(tokens(chunk) <= 48 for chunk in chunks), [tokens(chunk) for chunk in chunks]
    ends = sentence_ends(PROSE)
    assert all(span.end in ends for span in cut.spans), "a chunk ends where a sentence ends"
    # Packed, not one sentence a chunk: each chunk but the last would have
    # gone over the target with the sentence after it.
    for span, following in zip(cut.spans, cut.spans[1:]):
        next_sentence = next(s for s in sentence_spans(PROSE) if s.start == following.start)
        stretched = PROSE[span.start:next_sentence.end].rstrip()
        assert tokens(stretched) > 48, (PROSE[span.start:span.end], tokens(stretched))
    record = cut.record()
    assert record["measure"] == "tokens" and record["tokens"] == 48 and record["overlap"] == 0
    assert record["method"] == BUDGET_VERSION and record["sentences"] == len(SENTENCES)
    assert record["sentences_over_target"] == 0 and record["hard_cuts"] == 0 and record["over_target"] == 0
    assert record["largest"] == max(tokens(chunk) for chunk in chunks) <= 48


def test_every_character_lands_in_one_chunk_without_overlap():
    content = PROSE + "\n\n" + PROSE
    cut = token_spans(content, 64)
    covered = [0] * len(content)
    for span in cut.spans:
        for index in range(span.start, span.end):
            covered[index] += 1
    assert all(count == 1 for index, count in enumerate(covered) if not content[index].isspace())
    assert all(count <= 1 for count in covered)
    assert all(chunk == chunk.strip() and chunk for chunk in texts(content, cut.spans))


def test_short_text_is_one_chunk_and_empty_text_none():
    assert token_spans("", 64).spans == () and token_spans("   \n ", 64).spans == ()
    [only] = token_spans("just a line", 64).spans
    assert (only.start, only.end) == (0, len("just a line"))


def test_a_sentence_longer_than_the_target_is_cut_inside_and_the_receipt_says_so():
    """The fixture holds the junction: short sentences on both sides of one
    sentence that no chunk of the target could hold."""
    long = "The inventory lists " + ", ".join(f"crate {n} of rope and chain" for n in range(40)) + "."
    content = f"{SENTENCES[0]} {SENTENCES[1]} {long} {SENTENCES[2]} {SENTENCES[3]}"
    cut = token_spans(content, 40)
    chunks = texts(content, cut.spans)
    assert all(tokens(chunk) <= 40 for chunk in chunks)
    record = cut.record()
    assert record["sentences_over_target"] == 1 and record["hard_cuts"] == 0, record
    start = content.index(long)
    inside = [span for span in cut.spans if start < span.end < start + len(long)]
    assert inside, "the long sentence was cut inside"
    # Only that sentence: the short ones around it end their chunks whole.
    ends = sentence_ends(content)
    assert all(span.end in ends for span in cut.spans if not start <= span.end < start + len(long))
    assert content[cut.spans[0].start:cut.spans[0].end].startswith(SENTENCES[0])


def test_inside_a_long_sentence_a_line_break_is_cut_before_a_space():
    """A list with no stops is one sentence; its lines are indented and
    some carry trailing spaces, which no chunk starts or ends with."""
    lines = [f"  item {n} winch cable spool bracket" + " " * (n % 2) for n in range(30)]
    content = "\n".join(lines) + "\nend"
    cut = token_spans(content, 40)
    assert cut.record()["sentences_over_target"] == 1
    assert len(cut.spans) > 1
    assert all(span.end == len(content) or content[span.end:].lstrip(" ").startswith("\n") for span in cut.spans), \
        texts(content, cut.spans)
    assert all(chunk == chunk.strip() for chunk in texts(content, cut.spans)), texts(content, cut.spans)


def test_a_word_longer_than_the_target_is_cut_inside_and_counted():
    blob = "a" * 600
    content = f"{SENTENCES[0]} {blob} {SENTENCES[1]}"
    cut = token_spans(content, 40)
    assert all(tokens(chunk) <= 40 for chunk in texts(content, cut.spans))
    record = cut.record()
    assert record["largest"] == max(tokens(chunk) for chunk in texts(content, cut.spans))
    # Four letters a token: parts of 152 letters fit 40 less the 2 markers,
    # so 600 letters are cut three times.
    assert record["hard_cuts"] == 3 and record["sentences_over_target"] == 1, record
    start = content.index(blob)
    assert all(any(span.start <= index < span.end for span in cut.spans) for index in range(start, start + 600))
    # A part the room holds exactly is taken: 128 letters are 32 tokens.
    assert [span.end - span.start for span in token_spans(blob, 34).spans] == [128] * 4 + [88]


def test_cutting_a_long_word_reads_it_a_part_at_a_time():
    """Text written without spaces is one word to the cut, so a long document
    in such a script is hard-cut all the way through. Each part is found
    without counting far past it. A search that starts from the word's end
    counts about the rest of the word for every part, so the reading grows
    with the square of the word: this one would be read some 340 times over."""
    counts: list[int] = []

    def counted(text: str) -> int:
        counts.append(len(text))
        return estimated_tokens(text)

    word = "b" * 100_000
    cut = token_spans(word, 40, count=counted)
    # Parts of 152 letters, as above, and the last one ends with the word.
    assert [(span.start, span.end) for span in cut.spans] == [(at, min(at + 152, len(word)))
                                                                for at in range(0, len(word), 152)]
    assert cut.hard_cuts == len(cut.spans) - 1 == 657, cut.record()
    # Besides the word itself, counted as a sentence, a line and a word.
    assert max(size for size in counts if size < len(word)) <= 2 * 152
    assert sum(counts) < 40 * len(word), f"read {sum(counts) / len(word):.0f} times over"
    # A part's end in no more than twice the logarithm of its length and one
    # count, and one count more each of the part and of its chunk.
    assert len(counts) <= len(cut.spans) * (2 * math.log2(152) + 1 + 2), len(counts) / len(cut.spans)

def test_overlap_repeats_whole_trailing_sentences_of_the_chunk_before():
    cut = token_spans(PROSE, 48, overlap=16)
    starts = {span.start for span in sentence_spans(PROSE)}
    assert len(cut.spans) > 2
    for before, after in zip(cut.spans, cut.spans[1:]):
        assert after.start in starts, "an overlap starts where a sentence starts"
        assert before.start < after.start < before.end, (before, after)
        assert after.end > before.end, "every chunk carries something new"
        assert tokens(PROSE[after.start:before.end]) - tokens("") <= 16
    record = cut.record()
    assert record["overlap"] == 16 and record["overlapped"] == len(cut.spans) - 1
    assert all(tokens(chunk) <= 48 for chunk in texts(PROSE, cut.spans))
    assert token_spans(PROSE, 48).record()["overlapped"] == 0


def test_the_overlap_gives_way_when_it_and_the_next_sentence_do_not_fit():
    cut = token_spans(PROSE, 30, overlap=20)
    assert all(tokens(chunk) <= 30 for chunk in texts(PROSE, cut.spans))
    assert cut.spans[-1].end == len(PROSE)
    assert all(any(span.start <= index < span.end for span in cut.spans)
               for index, character in enumerate(PROSE) if not character.isspace()), "nothing is lost giving way"
    assert cut.record()["overlapped"] == sum(1 for before, after in zip(cut.spans, cut.spans[1:])
                                             if after.start < before.end)


def test_tokens_are_counted_by_the_tokenizer_given():
    """Its count of an empty text is what it adds to every input, paid once
    a chunk and not once a sentence."""
    def words(text: str) -> int:
        return len(text.split()) + 10

    cut = token_spans(PROSE, 30, count=words, method=TOKENIZER_VERSION)
    assert all(words(chunk) <= 30 for chunk in texts(PROSE, cut.spans))
    assert max(words(chunk) for chunk in texts(PROSE, cut.spans)) > 25, "packed to the target, not below it"
    assert cut.record()["method"] == TOKENIZER_VERSION
    assert [span.end for span in cut.spans] != [span.end for span in token_spans(PROSE, 30).spans]


def test_the_estimate_of_a_range_from_one_pass_is_the_estimate_of_its_text():
    """Without a tokenizer the text is estimated once and a range is read off
    running totals. That is the estimate only where no piece of the text is
    cut by the range's ends; anywhere else the text is estimated itself."""
    content = ("The OcrPdfOptions crane, invoice 20931_b. 港口起重机 was café-grey; "
               "iPhoneXR   NASA\nhello_world 3.14 !! ") * 3
    estimate = _Estimate(content)
    markers = tokens("")
    for start in range(0, len(content) + 1, 3):
        for end in range(start, len(content) + 1, 5):
            assert estimate.tokens(start, end) == tokens(content[start:end]) - markers, (start, end, content[start:end])


def test_a_range_is_estimated_again_only_when_its_ends_cut_a_piece(monkeypatch):
    """Reading a range off the totals is the point of the one pass. A range
    whose ends fall between pieces -- even touching one, as "crane" touches
    the comma after it -- is never estimated again; one whose end or start
    falls inside a piece always is."""
    content = "The OcrPdfOptions crane,invoice 20931_b. 港口起重机 was café-grey;iPhoneXR\nhello_world 3.14 !!"
    pieces = [(match.start(), match.end()) for match in PIECE.finditer(content)]
    estimate = _Estimate(content)
    calls: list[str] = []

    def counted(text: str) -> int:
        calls.append(text)
        return estimated_tokens(text)

    monkeypatch.setattr(token_chunks, "estimated_tokens", counted)
    touching = 0
    for start in range(len(content) + 1):
        for end in range(start, len(content) + 1):
            cut = any(first < at < last for first, last in pieces for at in (start, end))
            touching += not cut and any(at in (first, last) for first, last in pieces for at in (start, end))
            calls.clear()
            estimate.tokens(start, end)
            assert bool(calls) == cut, (start, end, content[start:end])
    assert touching > 0


def test_a_chunk_a_tokenizer_counts_over_the_target_is_counted():
    """A tokenizer need not count a chunk as the sum of its sentences. The
    receipt reports the chunks it measured over, rather than claiming a
    bound that did not hold."""
    def joined_costs_more(text: str) -> int:
        return len(text.split()) + 2 + 20 * text.count(". ")

    cut = token_spans(PROSE, 40, count=joined_costs_more)
    measured = [joined_costs_more(chunk) for chunk in texts(PROSE, cut.spans)]
    assert cut.record()["over_target"] == sum(1 for size in measured if size > 40) > 0
    assert cut.record()["largest"] == max(measured)


@pytest.mark.parametrize("target, overlap", [(MIN_CHUNK_TOKENS - 1, 0), (0, 0), (-3, 0), (64, 64), (64, 90),
                                             (64, -1), (True, 0), (64.0, 0), (64, 1.5)])
def test_a_target_or_overlap_that_cannot_work_is_refused(target, overlap):
    assert refused(target, overlap)
    with pytest.raises(ValueError):
        token_spans(PROSE, target, overlap=overlap)


def test_a_workable_target_is_not_refused():
    assert refused(MIN_CHUNK_TOKENS, 0) is None and refused(64, 63) is None


def test_a_target_the_tokenizer_markers_fill_is_refused():
    with pytest.raises(ValueError, match="markers"):
        token_spans(PROSE, MIN_CHUNK_TOKENS, count=lambda text: len(text) + MIN_CHUNK_TOKENS)


# The engine.

LONG = PROSE + "\n\n" + PROSE


async def open_engine(embedder=None, **options):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder or HashEmbedder(),
                              **options).open()


async def test_the_engine_cuts_by_tokens_when_asked_and_by_characters_otherwise():
    engine = await open_engine(chunk_tokens=64)
    try:
        added = await engine.remember("default", LONG)
        stored = await engine.documents.chunks_of("default", added.episode_id)
        expected = token_spans(LONG, 64)
        assert [(chunk.start, chunk.end) for chunk in stored] == [(s.start, s.end) for s in expected.spans]
        assert added.chunking == "length" and added.structure == expected.record()
    finally:
        await engine.close()
    plain = await open_engine()
    try:
        added = await plain.remember("default", LONG)
        stored = await plain.documents.chunks_of("default", added.episode_id)
        assert [(chunk.start, chunk.end) for chunk in stored] == [(s.start, s.end) for s in chunk_spans(LONG)]
        assert added.structure is None
    finally:
        await plain.close()


async def test_the_engine_passes_the_overlap_through():
    engine = await open_engine(chunk_tokens=64, chunk_overlap_tokens=24)
    try:
        added = await engine.remember("default", LONG)
        assert added.structure["overlap"] == 24 and added.structure["overlapped"] > 0
    finally:
        await engine.close()


class Counting(HashEmbedder):
    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 2


async def test_the_engine_counts_with_the_embedders_tokenizer_without_the_budget():
    engine = await open_engine(Counting(), chunk_tokens=40)
    try:
        added = await engine.remember("default", LONG)
        assert added.structure["method"] == TOKENIZER_VERSION
        stored = await engine.documents.chunks_of("default", added.episode_id)
        expected = token_spans(LONG, 40, count=Counting().count_tokens)
        assert [(chunk.start, chunk.end) for chunk in stored] == [(s.start, s.end) for s in expected.spans]
    finally:
        await engine.close()


class Broken(HashEmbedder):
    def count_tokens(self, text: str) -> int:
        raise RuntimeError("no tokenizer here")


async def test_an_engine_refuses_a_token_target_it_cannot_count_or_use():
    with pytest.raises(InvalidInput, match="no tokenizer here"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Broken(), chunk_tokens=64)
    with pytest.raises(InvalidInput, match="chunk_tokens"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_tokens=4)
    with pytest.raises(InvalidInput, match="overlap"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_overlap_tokens=8)
    with pytest.raises(InvalidInput, match="chunk_overlap_tokens must be"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_tokens=64, chunk_overlap_tokens=64)
    MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Broken())


@pytest.mark.parametrize("change", [{"chunk_tokens": 128}, {"chunk_overlap_tokens": 8}])
async def test_a_replacement_prepared_under_one_token_target_is_not_stored_under_another(change):
    """A space's chunks are cut one way; a replacement whose chunks were cut
    before the setting changed is refused rather than stored as if cut
    under the new one."""
    engine = await open_engine(chunk_tokens=96, chunk_overlap_tokens=4)
    embed = engine.embedder.embed

    async def change_during_embedding(texts):
        for name, value in change.items():
            setattr(engine, name, value)
        return await embed(texts)

    try:
        await engine.replace("default", Record(PROSE, dedup_key="log"))
        engine.embedder.embed = change_during_embedding
        with pytest.raises(InvalidInput, match="configuration changed"):
            await engine.replace("default", Record(LONG, dedup_key="log"))
    finally:
        await engine.close()
