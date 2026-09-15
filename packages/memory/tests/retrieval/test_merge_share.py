"""How much of a merged passage was retrieved, and fragments merged in clusters.

A merge reads the span from the first fragment to the last, so two short
hits far apart become a passage that is mostly text nobody retrieved. The
reference merges children into their parent only when enough of the
parent's children were retrieved; here the share of a merged passage's
bytes that retrieved chunks cover is said for every merge, and a caller
may ask for a floor under it. And one fragment too far from the rest no
longer stops the fragments beside each other from joining: an episode's
fragments are merged in clusters that each fit the cap.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import RecallItem
from scone_memory.retrieval.merging import merge_neighbours

pytestmark = pytest.mark.asyncio

TEXT = "".join(f"Line {n:03d} of the harbour crane log records one inspection.\n" for n in range(100))


async def stored(copies: int = 1):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    episodes = [(await engine.remember("default", (TEXT if n == 0 else TEXT.replace("crane", f"crane{n}")), source=f"log{n}.txt")).episode_id
                for n in range(copies)]
    return engine, episodes


def fragment(episode: int, chunk: int, start: int, end: int, score: float = 0.5) -> RecallItem:
    return RecallItem(chunk_id=chunk, episode_id=episode, text="x" * (end - start), score=score,
                      created_at="2026-09-14", start=start, end=end)


async def test_every_merged_passage_says_what_share_of_it_was_retrieved():
    engine, [episode] = await stored()
    try:
        merged = await merge_neighbours(engine, "default", [fragment(episode, 1, 0, 20, 0.9),
                                                            fragment(episode, 2, 120, 140)])
    finally:
        await engine.close()
    assert merged.merged == 1
    assert merged.shares == {1: round(40 / 140, 3)}
    assert merged.record()["shares"] == {"1": round(40 / 140, 3)}


async def test_overlapping_fragments_are_counted_once_in_the_share():
    engine, [episode] = await stored()
    try:
        merged = await merge_neighbours(engine, "default", [fragment(episode, 1, 0, 100, 0.9),
                                                            fragment(episode, 2, 50, 120),
                                                            fragment(episode, 3, 200, 240)])
    finally:
        await engine.close()
    assert merged.shares == {1: round(160 / 240, 3)}


async def test_a_fragment_inside_another_adds_nothing_to_the_share_or_the_span():
    engine, [episode] = await stored()
    try:
        merged = await merge_neighbours(engine, "default", [fragment(episode, 1, 0, 100, 0.9),
                                                            fragment(episode, 2, 10, 30)])
    finally:
        await engine.close()
    assert merged.shares == {1: 1.0}
    assert (merged.items[0].start, merged.items[0].end) == (0, 100)


async def test_fragments_given_out_of_order_are_clustered_by_where_they_sit():
    engine, [episode] = await stored()
    near = [fragment(episode, 1, 0, 60, 0.9), fragment(episode, 2, 61, 120)]
    far = fragment(episode, 3, 5_000, 5_060)
    try:
        merged = await merge_neighbours(engine, "default", [far, near[1], near[0]], max_merged=1_000)
    finally:
        await engine.close()
    assert merged.items == (far, merged.items[1]) and merged.from_chunks == {1: (1, 2)}
    assert (merged.items[1].start, merged.items[1].end) == (0, 120)


async def test_a_floor_on_the_share_leaves_a_sparse_merge_as_fragments_and_says_so():
    engine, [episode] = await stored()
    pair = [fragment(episode, 1, 0, 20, 0.9), fragment(episode, 2, 120, 140)]
    try:
        sparse = await merge_neighbours(engine, "default", pair, min_share=0.3)
        enough = await merge_neighbours(engine, "default", pair, min_share=round(40 / 140, 3))
    finally:
        await engine.close()
    assert sparse.merged == 0 and sparse.too_sparse == 1 and sparse.items == tuple(pair)
    assert "retrieved" in sparse.why and "0.3" in sparse.why
    assert enough.merged == 1 and enough.too_sparse == 0


async def test_one_fragment_too_far_from_the_rest_does_not_stop_the_others_joining():
    engine, [episode] = await stored()
    near = [fragment(episode, 1, 0, 60, 0.9), fragment(episode, 2, 61, 120)]
    far = fragment(episode, 3, 5_000, 5_060)
    try:
        merged = await merge_neighbours(engine, "default", [*near, far], max_merged=1_000)
    finally:
        await engine.close()
    assert merged.merged == 1 and merged.from_chunks == {1: (1, 2)}
    assert merged.items[1] == far and merged.too_far == 1 and "too far" in merged.why
    assert merged.items[0].text == TEXT.encode()[0:120].decode()


async def test_two_clusters_in_one_episode_are_each_merged_from_one_read():
    engine, [episode] = await stored()
    reads = 0
    reading = engine.episode

    async def counted(space, episode_id):
        nonlocal reads
        reads += 1
        return await reading(space, episode_id)

    engine.episode = counted
    try:
        merged = await merge_neighbours(engine, "default", [
            fragment(episode, 1, 0, 60, 0.9), fragment(episode, 2, 61, 120),
            fragment(episode, 3, 5_000, 5_060, 0.8), fragment(episode, 4, 5_061, 5_120)], max_merged=1_000)
    finally:
        await engine.close()
    assert merged.merged == 2 and merged.from_chunks == {1: (1, 2), 3: (3, 4)} and merged.too_far == 0
    assert reads == 1


async def test_episodes_past_the_read_budget_stand_unjoined_and_are_counted():
    engine, [first, second] = await stored(copies=2)
    try:
        merged = await merge_neighbours(engine, "default", [
            fragment(first, 1, 0, 60, 0.9), fragment(first, 2, 61, 120),
            fragment(second, 3, 0, 60), fragment(second, 4, 61, 120)], episodes=1)
    finally:
        await engine.close()
    assert merged.merged == 1 and merged.not_read == 1
    assert "budget" in merged.why and merged.record()["not_read"] == 1


@pytest.mark.parametrize("share", [-0.1, 1.1, True, "half"])
async def test_a_share_outside_zero_to_one_is_refused(share):
    engine, [episode] = await stored()
    try:
        with pytest.raises(InvalidInput):
            await merge_neighbours(engine, "default", [fragment(episode, 1, 0, 20)], min_share=share)
    finally:
        await engine.close()


async def test_a_passage_exactly_as_long_as_the_cap_is_merged():
    engine, [episode] = await stored()
    pair = [fragment(episode, 1, 0, 60, 0.9), fragment(episode, 2, 61, 120)]
    try:
        at_cap = await merge_neighbours(engine, "default", pair, max_merged=120)
        over = await merge_neighbours(engine, "default", pair, max_merged=119)
    finally:
        await engine.close()
    assert at_cap.merged == 1 and over.merged == 0 and over.too_far == 1


async def test_a_share_counts_the_chunks_retrieved_not_the_windows_around_them():
    """Widened items carry the window's span. Counting those as retrieved
    reported a merge of two windows as wholly retrieved; the spans the
    chunks were retrieved at are what the share counts."""
    engine, [episode] = await stored()
    hits = [fragment(episode, 1, 500, 590, 0.9), fragment(episode, 2, 1_090, 1_180)]
    widened = [hit.model_copy(update={"start": hit.start - 500, "end": hit.end + 500,
                                      "text": "x" * (hit.end - hit.start + 1_000)}) for hit in hits]
    try:
        merged = await merge_neighbours(engine, "default", widened, hits=hits)
        unaware = await merge_neighbours(engine, "default", widened)
    finally:
        await engine.close()
    assert merged.shares == {1: round(180 / 1_680, 3)}
    assert unaware.shares == {1: 1.0}, "without the hits, the items are all it has to count"


async def test_a_cluster_whose_span_reads_blank_does_not_stop_the_next_one():
    """Found in review: the first blank span ended the episode's loop, so a
    second cluster that would have merged stood as fragments, and a read
    that succeeded was counted as one that failed."""
    engine, [episode] = await stored()
    reading = engine.episode

    async def blank_start(space, episode_id):
        found = await reading(space, episode_id)
        return found.model_copy(update={"content": " " * 200 + found.content[200:]})

    engine.episode = blank_start
    try:
        merged = await merge_neighbours(engine, "default", [
            fragment(episode, 1, 0, 60, 0.9), fragment(episode, 2, 61, 120),
            fragment(episode, 3, 5_000, 5_060, 0.8), fragment(episode, 4, 5_061, 5_120)], max_merged=1_000)
    finally:
        await engine.close()
    assert merged.merged == 1 and merged.from_chunks == {3: (3, 4)}
    assert merged.blank == 1 and merged.unread == 0
    assert "blank" in merged.why and "could not be read" not in merged.why
