"""A window counted in sentences, with both edges on sentence boundaries.

A window of 60 bytes either side of a hit starts and ends wherever the
count lands: halfway through the sentence before, halfway through the one
after. The reference's sentence window is built at ingestion, one sentence
per node, and a different size needs a re-index. Ours reads the sentences
from the episode at retrieval, so the unit and the count are the caller's,
and the window is still the episode's own bytes between two offsets.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import RecallItem
from scone_memory.retrieval.window import MAX_SENTENCES, sentence_spans, widen

pytestmark = pytest.mark.asyncio

S1 = "The survey of the harbour crane was booked for the third of May. "
S2 = "Dr. Okafor found rust on the jib and a slew ring that needed grease. "
S3 = "The yard did both in the same week, i.e. before the June inspection. "
S4 = "The crane returned to service on the first of June!"
TEXT = S1 + S2 + S3 + S4


async def _hit(content: str, needle: str):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    added = await engine.remember("default", content, source="crane.txt")
    raw = content.encode()
    start = raw.index(needle.encode())
    item = RecallItem(chunk_id=7, episode_id=added.episode_id, text=needle, score=1.0, created_at="2026-09-12",
                      start=start, end=start + len(needle.encode()))
    return engine, item


def test_sentences_end_at_stops_but_not_at_titles_initials_or_lowercase_continuations():
    spans = sentence_spans(TEXT)
    assert [TEXT[start:end] for start, end in spans] == [S1.rstrip(), S2.rstrip(), S3.rstrip(), S4]


def test_cjk_full_stops_and_blank_lines_end_sentences_too():
    text = "港口起重机已检修。发现吊臂生锈！何时完工？\n\nRefund policy\n\nWithin 30 days of purchase"
    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "港口起重机已检修。", "发现吊臂生锈！", "何时完工？", "Refund policy", "Within 30 days of purchase"]


def test_a_closing_quote_stays_with_its_sentence_and_trailing_space_with_none():
    text = 'He said "the jib is done." Then he left.   \n\nRefund policy   \n\nWithin 30 days  '
    assert [text[start:end] for start, end in sentence_spans(text)] == [
        'He said "the jib is done."', "Then he left.", "Refund policy", "Within 30 days"]


async def test_a_window_never_cuts_into_the_hit_it_widens():
    """A hit may end in the space after a sentence; its sentence ends before
    that space, and the window must still hold the whole hit."""
    engine, item = await _hit(TEXT, "needed grease. ")
    try:
        widened = await widen(engine, "default", [item], before=0, after=0, unit="sentences")
    finally:
        await engine.close()
    [one] = widened.items
    assert one.start <= item.start and one.end >= item.end
    assert one.text.startswith("Dr. Okafor") and one.text.endswith("grease. ")


async def test_a_hit_that_is_only_the_space_between_sentences_takes_the_sentence_after_it():
    content = "Refund policy\n\nWithin 30 days of purchase."
    engine, item = await _hit(content, "\n\n")
    try:
        widened = await widen(engine, "default", [item], before=0, after=0, unit="sentences")
    finally:
        await engine.close()
    assert widened.widened == 1 and widened.items[0].text == "\n\nWithin 30 days of purchase."


async def test_a_hit_inside_a_sentence_widens_to_whole_sentences_before_and_after():
    engine, item = await _hit(TEXT, "rust on the jib")
    try:
        widened = await widen(engine, "default", [item], before=1, after=1, unit="sentences")
    finally:
        await engine.close()
    [one] = widened.items
    assert one.text == S1 + S2 + S3.rstrip()
    assert TEXT.encode()[one.start:one.end] == one.text.encode(), "a window is quoted, never assembled"
    assert widened.unit == "sentences" and widened.clipped == 0 and widened.widened == 1


async def test_no_sentences_either_side_completes_the_sentences_the_hit_touches():
    engine, item = await _hit(TEXT, "grease. The yard")
    try:
        widened = await widen(engine, "default", [item], before=0, after=0, unit="sentences")
    finally:
        await engine.close()
    assert widened.items[0].text == S2 + S3.rstrip()


async def test_a_window_past_the_first_or_last_sentence_is_clipped_and_says_so():
    engine, item = await _hit(TEXT, "first of June")
    try:
        widened = await widen(engine, "default", [item], before=2, after=3, unit="sentences")
    finally:
        await engine.close()
    assert widened.items[0].text == S2 + S3 + S4
    assert widened.clipped == 1 and "start or end" in widened.why


async def test_a_sentence_longer_than_the_reach_is_capped_and_counted_apart():
    import scone_memory.retrieval.window as module

    long = "word " * 200 + "needle here " + "word " * 200 + "end."
    engine, item = await _hit(long, "needle here")
    original = module.MAX_WINDOW
    module.MAX_WINDOW = 100
    try:
        widened = await widen(engine, "default", [item], before=0, after=0, unit="sentences")
    finally:
        module.MAX_WINDOW = original
        await engine.close()
    [one] = widened.items
    assert item.start - one.start <= 100 and one.end - item.end <= 100
    assert widened.capped == 1 and widened.clipped == 0
    assert "reach" in widened.why and widened.record()["capped"] == 1


async def test_a_byte_window_is_as_it_was():
    engine, item = await _hit(TEXT, "rust on the jib")
    try:
        widened = await widen(engine, "default", [item], before=10, after=10)
    finally:
        await engine.close()
    [one] = widened.items
    assert widened.unit == "bytes" and item.start - one.start == 10 and one.end - item.end == 10


@pytest.mark.parametrize("arguments", [{"unit": "words"}, {"unit": "sentences", "before": MAX_SENTENCES + 1},
                                       {"unit": "sentences", "after": -1}, {"unit": "sentences", "before": 1.5}])
async def test_bad_sentence_windows_are_refused(arguments):
    engine, item = await _hit(TEXT, "rust")
    try:
        with pytest.raises(InvalidInput):
            await widen(engine, "default", [item], **{"before": 1, "after": 1, **arguments})
    finally:
        await engine.close()


def test_sentence_offsets_grow_with_the_episode_not_its_square():
    """Byte offsets for every sentence of an episode, found in one walk.
    Encoding the text before each sentence is quadratic: 50,000 sentences
    holding a non-ASCII letter took 15 seconds on a request path. Timed
    against itself at two sizes, so a loaded machine slows both alike:
    four times the sentences takes about four times as long in one walk
    and sixteen times as long by prefix."""
    import time

    from scone_memory.retrieval.window import byte_spans

    def took(count: int) -> float:
        content = "Gö to the hall. " * count
        spans = sentence_spans(content)
        best = float("inf")
        for _ in range(3):
            began = time.perf_counter()
            byte_spans(content, spans)
            best = min(best, time.perf_counter() - began)
        return best

    small, large = took(5_000), took(20_000)
    assert large / small < 8, (small, large)


async def test_a_window_over_a_long_non_ascii_episode_is_quoted():
    content = "Gö to the hall. " * 40_000 + "The needle is here."
    engine, item = await _hit(content, "needle")
    try:
        widened = await widen(engine, "default", [item], before=1, after=0, unit="sentences")
    finally:
        await engine.close()
    assert widened.items[0].text == "Gö to the hall. The needle is here."
    assert content.encode()[widened.items[0].start:widened.items[0].end] == widened.items[0].text.encode()


async def test_a_window_is_quoted_when_the_space_between_sentences_is_not_ascii():
    # An ideographic space in the blank line: three bytes the offsets must count.
    content = "Crane survey\n　\nThe jib had rust.　　\n\nThe needle is here."
    engine, item = await _hit(content, "needle")
    try:
        widened = await widen(engine, "default", [item], before=1, after=0, unit="sentences")
    finally:
        await engine.close()
    [one] = widened.items
    assert one.text == "The jib had rust.　　\n\nThe needle is here."
    assert content.encode()[one.start:one.end] == one.text.encode()
