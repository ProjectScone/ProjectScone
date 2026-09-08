"""Browsing the source inventory, narrowed by what was recorded.

A page that can be searched with a filter but not browsed with one is
half a feature: the same question asked of the same memories gives a
different answer depending on which screen it was asked from.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput


@pytest.fixture
async def stocked():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    kept = []
    for n in range(60):
        status = "published" if n % 20 == 19 else "draft"
        added = await engine.remember("alpha", f"planning note {n}", metadata={"status": status})
        if status == "published":
            kept.append(added.episode_id)
    return engine, kept


async def test_browsing_can_be_narrowed_the_same_way_searching_is(stocked):
    engine, kept = stocked
    page = await engine.source_page("alpha", limit=25,
                                    conditions={"field": "status", "is": "published"})
    assert [e.episode_id for e in page.episodes] == sorted(kept, reverse=True)
    assert page.has_more is False


async def test_a_narrow_page_is_not_cut_short_by_the_rows_it_rejects(stocked):
    """Filtering one page of twenty-five and returning what survives gives
    a page of one. The walk keeps reading until the page is full or the
    space runs out, which is what makes the page size mean anything."""
    engine, _ = stocked
    page = await engine.source_page("alpha", limit=2,
                                    conditions={"field": "status", "is": "published"})
    assert len(page.episodes) == 2
    assert page.has_more is True and page.next_before is not None


async def test_the_next_page_carries_on_where_the_last_one_stopped(stocked):
    engine, kept = stocked
    first = await engine.source_page("alpha", limit=2,
                                     conditions={"field": "status", "is": "published"})
    second = await engine.source_page("alpha", limit=2, before=first.next_before,
                                      conditions={"field": "status", "is": "published"})
    seen = [e.episode_id for e in first.episodes] + [e.episode_id for e in second.episodes]
    assert seen == sorted(kept, reverse=True), "every match once, in order, across the two pages"


async def test_a_filter_that_cannot_mean_anything_is_refused(stocked):
    engine, _ = stocked
    with pytest.raises(InvalidInput):
        await engine.source_page("alpha", conditions={"field": "status"})


async def test_a_walk_that_reaches_its_bound_says_there_is_more(stocked):
    """The walk is bounded, because a filter matching nothing in a large
    space would otherwise read all of it to prove a negative. Stopping
    early has to be reported as stopping early, never as the end."""
    engine, _ = stocked
    page = await engine.source_page("alpha", limit=25,
                                    conditions={"field": "status", "is": "nothing-matches-this"})
    assert page.episodes == []
    assert page.has_more is False, "sixty rows is inside the bound, so this really is the end"
