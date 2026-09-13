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
from scone_memory.core.errors import InvalidInput, SconeError
from scone_memory.core.models import RecallItem
from scone_memory.retrieval.window import MAX_EPISODES, MAX_WINDOW, widen

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
    # Equal lengths are not enough: the old assertion here compared only
    # sizes and passed while the span pointed at different bytes than the
    # text came from. The span has to name the text.
    assert PARAGRAPH.encode()[one.start:one.end] == one.text.encode(), (one.start, one.end)


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


async def test_a_window_lands_on_characters_not_bytes():
    """A window is measured in bytes and read as text, so an edge can fall
    inside a character.

    Built by hand rather than found in a corpus: the hit is the single
    byte "B", and one byte either side reaches into the two-byte "e-acute"
    before it. Decoding that range threw the half character away while the
    offsets kept pointing at it, so the item's own span named bytes the
    item's own text did not contain -- and `by_chunk` named a third range
    again. Any of the three could have been used to quote the source.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    content = "Aé" + "BCDEF"
    raw = content.encode()
    assert raw[2:4] == b"\xa9B", "the fixture is only interesting if byte 2 is a tail byte"
    try:
        added = await engine.remember("default", content, source="accent.txt")
        hit = RecallItem(chunk_id=7, episode_id=added.episode_id, text="B", score=1.0,
                         created_at="2026-09-12", start=3, end=4)
        widened = await widen(engine, "default", [hit], before=1, after=1)
    finally:
        await engine.close()
    [one] = widened.items
    assert raw[one.start:one.end] == one.text.encode(), (one.start, one.end, one.text)
    assert widened.by_chunk[7] == (one.start, one.end), widened.by_chunk
    assert one.text in content, "a window is quoted, never assembled"


async def test_moving_to_a_character_boundary_is_not_hitting_the_edge():
    """Both shorten a window, and a caller acts on them differently: one
    means the document ran out, the other means one byte of a character
    was dropped. Counting the second as the first would read as the
    document being smaller than it is."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    content = "Aé" + "BCDEF"
    try:
        added = await engine.remember("default", content, source="accent.txt")
        hit = RecallItem(chunk_id=7, episode_id=added.episode_id, text="B", score=1.0,
                         created_at="2026-09-12", start=3, end=4)
        widened = await widen(engine, "default", [hit], before=1, after=1)
    finally:
        await engine.close()
    assert widened.clipped == 0, "neither end of this window met either end of the episode"
    assert widened.aligned == 1, widened.record()
    assert "character" in widened.why, widened.why


async def test_the_episode_budget_counts_reads_that_failed():
    """A bound on work that only counts the work that succeeded is not a
    bound. A failed read never entered the cache, so `episodes=1` against
    a store failing every read performed one read per item -- the
    caller's budget asked for one.
    """
    engine = await memory()
    reads = 0
    real = engine.episode

    async def failing(space, episode_id):
        nonlocal reads
        reads += 1
        raise SconeError("the store is not answering")

    # Built by hand: five items in five distinct episodes is the shape
    # this bound is about, and how many chunks a corpus happens to yield
    # is not the point of the test.
    items = [RecallItem(chunk_id=index, episode_id=500 + index, text="x", score=1.0,
                        created_at="2026-09-12", start=0, end=1) for index in range(5)]
    try:
        engine.episode = failing  # type: ignore[method-assign]
        widened = await widen(engine, "default", items, before=20, after=20, episodes=1)
    finally:
        engine.episode = real  # type: ignore[method-assign]
        await engine.close()
    assert reads == 1, f"asked for 1 episode, read {reads}"
    assert widened.items == tuple(items), "items that could not be widened stand as they were"
    assert widened.unread == 1, widened.record()
    assert widened.not_read == len(items) - 1, widened.record()
    assert "budget" in widened.why, widened.why


async def test_repeated_failures_for_one_episode_are_read_once():
    """Several items from one unavailable episode are one failed read,
    and each item still has to be accounted for."""
    engine = await memory(target=40)
    reads = 0
    real = engine.episode

    async def failing(space, episode_id):
        nonlocal reads
        reads += 1
        raise SconeError("the store is not answering")

    try:
        found = await engine.recall("default", "rust jib slew crane survey", limit=5)
        items = list(found.items)
        assert len({item.episode_id for item in items}) == 1, "one episode, several chunks"
        assert len(items) >= 2
        engine.episode = failing  # type: ignore[method-assign]
        widened = await widen(engine, "default", items, before=20, after=20, episodes=4)
    finally:
        engine.episode = real  # type: ignore[method-assign]
        await engine.close()
    assert reads == 1, f"one episode, {reads} reads"
    assert widened.unread == len(items), widened.record()
    assert widened.not_read == 0, widened.record()


async def test_a_budget_beyond_the_cap_and_a_fractional_window_are_refused():
    """`MAX_EPISODES` documented a cap it did not enforce, so a caller
    could ask for 101 reads; and a fractional window passed the `< 0`
    test and failed much later as a TypeError from a byte slice."""
    engine = await memory()
    try:
        found = await engine.recall("default", "rust jib", limit=1)
        for bad in (0, -1, MAX_EPISODES + 1, 1.5, True):
            with pytest.raises(InvalidInput):
                await widen(engine, "default", found.items, before=10, after=10, episodes=bad)
        for bad in (0.5, 1.5, True, "10"):
            with pytest.raises(InvalidInput):
                await widen(engine, "default", found.items, before=bad, after=10)
            with pytest.raises(InvalidInput):
                await widen(engine, "default", found.items, before=10, after=bad)
    finally:
        await engine.close()
