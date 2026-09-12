"""A precise hit, answered with the passage around it.

Merging joins neighbours, so it needs two or more hits in one episode and
does nothing for one. The commonest shape in a real corpus is the
opposite: a single sentence matches exactly, and the sentence alone does
not answer the question because the answer is in the sentence after it.

The leading RAG framework calls this a sentence window, and does it by
embedding single sentences and keeping the window in metadata -- a
decision taken at ingestion, and a re-index to change the window size.
Ours reads the window from the episode at retrieval, so the size is the
caller's and no re-index is involved.

What it must never do is claim a window it did not read. A window cut
short by the start or end of the episode says so, and one that reaches
past what the episode holds is refused rather than padded.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.window import MAX_WINDOW, widen

pytestmark = pytest.mark.asyncio

PARAGRAPH = (
    "The survey of the harbour crane was booked for the third of May. "
    "It found rust on the jib and a slew ring that needed grease. "
    "The yard did both in the same week, and the crane returned to service on the first of June. "
    "Nobody recorded who signed the handover.")


async def memory(target=70):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=target).open()
    await engine.remember("default", PARAGRAPH, source="crane.txt")
    return engine


async def test_a_single_hit_comes_back_with_the_text_around_it():
    """The case merging cannot serve: one chunk matched, and one chunk is
    not the answer."""
    engine = await memory()
    try:
        found = await engine.recall("default", "rust jib slew ring grease", limit=1)
        assert len(found.items) == 1, "this test is about the single-hit case"
        narrow = found.items[0].text
        widened = await widen(engine, "default", found.items, before=60, after=60)
    finally:
        await engine.close()
    [one] = widened.items
    assert len(one.text) > len(narrow), (len(one.text), len(narrow))
    assert "rust on the jib" in one.text
    assert widened.widened == 1 and widened.by_chunk, widened.record()


async def test_the_window_is_the_text_the_episode_actually_holds():
    engine = await memory()
    try:
        found = await engine.recall("default", "rust jib slew", limit=1)
        widened = await widen(engine, "default", found.items, before=40, after=40)
    finally:
        await engine.close()
    [one] = widened.items
    assert one.text in PARAGRAPH, "a window is quoted, never assembled"
    assert len(one.text.encode()) == one.end - one.start, "the text is its own span"


async def test_a_window_cut_by_the_start_of_the_episode_says_so():
    """Asking for more than there is must not read as having got it."""
    engine = await memory()
    try:
        found = await engine.recall("default", "survey harbour crane booked third May", limit=1)
        widened = await widen(engine, "default", found.items, before=500, after=0)
    finally:
        await engine.close()
    assert widened.clipped == 1, widened.record()
    assert "met the start or end" in widened.why, widened.why
    assert "shorter than asked for" in widened.why, widened.why


async def test_a_window_of_nothing_is_refused():
    engine = await memory()
    try:
        found = await engine.recall("default", "rust jib", limit=1)
        with pytest.raises(InvalidInput):
            await widen(engine, "default", found.items, before=0, after=0)
        with pytest.raises(InvalidInput):
            await widen(engine, "default", found.items, before=MAX_WINDOW + 1, after=0)
    finally:
        await engine.close()


async def test_a_source_that_is_gone_is_not_widened_from_stale_text():
    """The same rule merging learned: evidence confirmed absent is dropped,
    never served from the excerpt we happen to be holding."""
    engine = await memory()
    try:
        found = await engine.recall("default", "rust jib slew", limit=1)
        await engine.forget("default", found.items[0].episode_id)
        widened = await widen(engine, "default", found.items, before=40, after=40)
    finally:
        await engine.close()
    assert widened.items == () and widened.gone == 1, widened.record()
    assert "no longer there" in widened.why, widened.why


async def test_widening_keeps_the_caller_s_ranking():
    engine = await memory()
    try:
        await engine.remember("default", "An unrelated note about bicycles and choirs.",
                              source="other.txt")
        found = await engine.recall("default", "rust jib bicycles choirs", limit=4)
        assert len(found.items) >= 2
        widened = await widen(engine, "default", found.items, before=30, after=30)
    finally:
        await engine.close()
    assert [i.chunk_id for i in widened.items] == [i.chunk_id for i in found.items]
