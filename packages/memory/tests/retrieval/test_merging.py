"""Returning one coherent passage instead of three fragments of it.

Small chunks match precisely and read badly: three neighbouring
fragments of one paragraph are three citations to the same thought, and
they crowd out the rest of the answer. The leading frameworks fix this by
indexing a hierarchy at ingestion time and merging children into a
parent at retrieval. Ours needs no hierarchy and no re-indexing, because
every chunk already carries the byte span it came from: neighbours in one
episode are merged by reading the span that contains them.

What it must never do is pretend a merged passage is one chunk. The
caller is told exactly which chunks went into it, because a citation that
cannot be checked is worse than three that can.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.retrieval.merging import MAX_MERGED, merge_neighbours

pytestmark = pytest.mark.asyncio

PARAGRAPH = (
    "The harbour crane was repainted in May after the survey found rust on the jib. "
    "The survey also found the slew ring needed grease, which the yard did the same week. "
    "The crane went back into service on the first of June, a day later than planned. "
    "Nobody recorded who signed the handover, which caused an argument in July. "
)
OTHER = "The bicycle needed a new chain, and the choir moved to Tuesdays."


async def memory(target=90):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                chunk_target=target).open()
    await engine.remember("default", PARAGRAPH, source="crane.txt")
    await engine.remember("default", OTHER, source="other.txt")
    return engine


async def test_neighbouring_fragments_of_one_passage_come_back_as_the_passage():
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        chunks = [item for item in found.items if (item.source or "") == "crane.txt"]
        assert len(chunks) >= 2, "this corpus has to return several fragments to prove anything"
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    assert merged.merged == 1, merged.record()
    assert merged.absorbed >= 2, merged.record()
    [whole] = [item for item in merged.items if (item.source or "") == "crane.txt"]
    assert "rust on the jib" in whole.text and "needed grease" in whole.text, whole.text


async def test_the_chunks_that_went_into_a_passage_are_named():
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    [(chunk, absorbed)] = list(merged.from_chunks.items())
    assert len(absorbed) >= 2 and chunk in absorbed, merged.record()


async def test_a_lone_chunk_is_returned_untouched():
    engine = await memory()
    try:
        found = await engine.recall("default", "bicycle chain choir", limit=2)
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    assert merged.merged == 0 and merged.from_chunks == {}, merged.record()
    assert [item.chunk_id for item in merged.items] == [item.chunk_id for item in found.items]


async def test_a_merged_passage_keeps_the_best_score_of_its_parts_not_their_sum():
    """A sum would make a merged passage outrank everything by arithmetic
    rather than by relevance."""
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        best = max(item.score for item in found.items if (item.source or "") == "crane.txt")
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    [whole] = [item for item in merged.items if (item.source or "") == "crane.txt"]
    assert whole.score == pytest.approx(best), (whole.score, best)


async def test_a_span_too_long_to_merge_is_left_as_fragments():
    """The point is a readable passage, not the whole document. A cap that
    silently returned the document instead would be worse than no
    merging."""
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        merged = await merge_neighbours(engine, "default", found.items, max_merged=40)
    finally:
        await engine.close()
    assert merged.merged == 0 and "too far apart" in merged.why, merged.why


async def test_nothing_is_merged_across_two_different_episodes():
    engine = await memory()
    try:
        found = await engine.recall("default", "crane bicycle choir chain", limit=6)
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    sources = {item.source for item in merged.items}
    assert len(merged.items) >= 2 and sources == {"crane.txt", "other.txt"}, sources


async def test_the_cap_is_a_number_the_caller_can_see():
    assert MAX_MERGED > 0


async def test_the_ranking_survives_a_call_that_merges_nothing():
    """Grouping by episode to look for neighbours must not rearrange the
    answer. A ranked list is the caller's data, and reordering it silently
    is a change nobody asked for and nothing reports.

    The order is built by hand because it has to interleave: two chunks of
    one episode either side of a chunk of another is the only shape that
    catches a per-episode walk, and a corpus that happens not to produce it
    proves nothing.
    """
    items = [_item(1, episode=1, score=1.0), _item(4, episode=2, score=0.9),
             _item(2, episode=1, score=0.1)]
    engine = await memory()
    try:
        merged = await merge_neighbours(engine, "default", items, min_leaves=99)
    finally:
        await engine.close()
    assert merged.merged == 0, merged.record()
    assert [item.chunk_id for item in merged.items] == [1, 4, 2]


def _item(chunk_id, *, episode, score):
    from scone_memory.core.models import RecallItem

    return RecallItem(chunk_id=chunk_id, episode_id=episode, text=f"chunk {chunk_id}",
                      score=score, created_at="2024-01-01T00:00:00Z",
                      start=chunk_id * 10, end=chunk_id * 10 + 5)


async def test_a_merged_passage_takes_the_place_of_its_best_fragment():
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        best = max((i for i in found.items if (i.source or "") == "crane.txt"),
                   key=lambda i: i.score)
        at = [i.chunk_id for i in found.items].index(best.chunk_id)
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    assert merged.merged == 1 and merged.items[at].chunk_id == best.chunk_id, merged.record()


async def test_a_source_that_is_gone_is_not_served_from_a_stale_excerpt():
    """If the episode was forgotten, its text is deleted and the fragments
    quoting it must go with it. Returning them under a clean "nothing to
    join" reason serves deleted content and says nothing was wrong."""
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        [gone] = {item.episode_id for item in found.items if (item.source or "") == "crane.txt"}
        await engine.forget("default", gone)
        merged = await merge_neighbours(engine, "default", found.items)
    finally:
        await engine.close()
    assert all(item.episode_id != gone for item in merged.items), merged.record()
    assert merged.gone == 1 and "no longer there" in merged.why, merged.why


async def test_a_source_that_could_not_be_read_keeps_its_fragments_and_says_so():
    """An unverified read failure is not a deletion. The fragments are a
    worse answer than the whole passage and a better one than nothing, so
    they stay -- and the reason says the merge did not happen rather than
    implying there was nothing to do."""
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib slew grease", limit=5)
        broken = engine.episode

        async def refuse(space, episode_id):
            raise RuntimeError("the store is unwell")

        engine.episode = refuse
        try:
            merged = await merge_neighbours(engine, "default", found.items)
        finally:
            engine.episode = broken
    finally:
        await engine.close()
    assert merged.merged == 0 and merged.gone == 0, merged.record()
    assert merged.unread == 1 and "could not be read" in merged.why, merged.why
    assert len(merged.items) == len(found.items), "nothing is dropped for an unknown failure"


async def test_a_merge_across_a_gap_returns_more_bytes_not_fewer():
    """Neighbours are not always adjacent. Merging a chunk with one that
    starts well after it ends returns the text between them too, so the
    passage is *longer* than the fragments were. Any byte accounting has
    to get that direction right, and adjacent chunks hide it."""
    engine = await memory()
    try:
        found = await engine.recall("default", "crane survey rust jib", limit=5)
        [one] = [i for i in found.items if (i.source or "") == "crane.txt"][:1]
        pair = [one.model_copy(update={"start": 0, "end": 20, "text": "x" * 20}),
                one.model_copy(update={"chunk_id": one.chunk_id + 99, "score": 0.1,
                                       "start": 120, "end": 140, "text": "y" * 20})]
        merged = await merge_neighbours(engine, "default", pair)
    finally:
        await engine.close()
    assert merged.merged == 1 and merged.absorbed == 2, merged.record()
    [whole] = merged.items
    assert (whole.start, whole.end) == (0, 140), (whole.start, whole.end)
    assert len(whole.text.encode()) == 140, len(whole.text.encode())
    assert len(whole.text.encode()) > 40, "the gap between the fragments is in the passage"
