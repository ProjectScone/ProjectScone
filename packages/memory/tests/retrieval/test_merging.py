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
