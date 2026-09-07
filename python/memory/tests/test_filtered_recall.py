"""Searching a narrowed set of memories, in every store.

The filter has to reach the store, not run over what the store already
chose. A search narrow enough to matter looks at a handful of memories
among thousands, and cropping the best hundred to that handful returns
almost nothing while the memories that would have answered sit unranked.
"""

from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput


async def crowd(engine, space="alpha", count=30):
    """Enough near-identical drafts to fill any lane's window."""
    for n in range(count):
        await engine.remember(space, f"quarterly planning note {n}", metadata={"status": "draft"})


async def test_a_narrow_search_finds_what_it_asks_for_under_a_crowd(engine):
    await crowd(engine)
    kept = await engine.remember(
        "alpha", "quarterly planning note, the one that shipped",
        metadata={"status": "published", "priority": "8"})

    found = await engine.recall("alpha", "quarterly planning note", limit=5,
                                conditions={"field": "status", "is": "published"})

    assert [item.episode_id for item in found.items] == [kept.episode_id]


async def test_a_number_kept_as_text_still_filters_as_a_number(engine):
    low = await engine.remember("alpha", "planning note, minor", metadata={"priority": "9"})
    high = await engine.remember("alpha", "planning note, urgent", metadata={"priority": "10"})

    found = await engine.recall("alpha", "planning note", limit=5,
                                conditions={"field": "priority", "at_least": 10})

    assert [item.episode_id for item in found.items] == [high.episode_id]
    assert low.episode_id not in {item.episode_id for item in found.items}


async def test_a_memory_with_no_such_key_is_not_swept_in_by_a_negation(engine):
    """Otherwise "not a draft" quietly returns everything that was never
    given a status, which is the filter failing open."""
    await engine.remember("alpha", "planning note, unlabelled")
    published = await engine.remember("alpha", "planning note, shipped", metadata={"status": "published"})

    found = await engine.recall("alpha", "planning note", limit=5,
                                conditions={"field": "status", "is": "draft", "not": True})

    assert [item.episode_id for item in found.items] == [published.episode_id]


async def test_conditions_combine_the_way_they_are_written(engine):
    wanted = await engine.remember("alpha", "planning note for the team",
                                   metadata={"status": "published", "team": "eng"})
    await engine.remember("alpha", "planning note for the team",
                          metadata={"status": "draft", "team": "eng"})
    await engine.remember("alpha", "planning note for the team",
                          metadata={"status": "published", "team": "sales"})

    found = await engine.recall("alpha", "planning note for the team", limit=5, conditions={
        "all": [{"field": "status", "is": "published"},
                {"any": [{"field": "team", "is": "eng"}, {"field": "team", "is": "product"}]}]})

    assert [item.episode_id for item in found.items] == [wanted.episode_id]


async def test_a_filter_that_cannot_mean_anything_is_refused_before_the_search(engine):
    with pytest.raises(InvalidInput):
        await engine.recall("alpha", "anything", conditions={"field": "status"})


async def test_what_was_filtered_on_is_on_the_record(engine):
    """A recall event that does not say how it was narrowed cannot be
    read back as evidence for what the answer was drawn from."""
    await engine.remember("alpha", "planning note", metadata={"status": "published"})
    await engine.recall("alpha", "planning note", conditions={"field": "status", "is": "published"})

    events = await engine.events.query("alpha", kind="recall", limit=5)
    assert events and events[-1].payload["narrow"]["conditions"] == {"field": "status", "is": "published"}


async def test_a_filter_narrows_within_a_space_and_never_across_one(engine):
    """G13: metadata narrows, it does not grant. A condition is a way of
    asking for less, so it must never become a way of reaching something
    the caller could not already read. The space is the boundary; a
    filter that matched across it would turn a search term into a key.

    This one is a guard on the outcome and not a proof of any single
    line, and that is worth saying rather than leaving to be discovered.
    Isolation here is defended in three independent places on the SQLite
    path: the lexical query, fetching the chunks, and fetching the
    episode. Disabling any one of them on its own still leaves this
    passing, which is the point of having three. So a mutation cannot
    turn this test red, and it earns its place by catching the day a
    change removes the last of them rather than the first."""
    await engine.remember("alpha", "the quarterly plan", metadata={"status": "published"})
    theirs = await engine.remember("beta", "the quarterly plan", metadata={"status": "published"})

    found = await engine.recall("alpha", "the quarterly plan", limit=10,
                                conditions={"field": "status", "is": "published"})

    assert theirs.episode_id not in {item.episode_id for item in found.items}
    assert len(found.items) == 1
