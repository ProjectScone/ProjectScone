"""Evidence the ledger retired should not outrank what replaced it.

`fusion.demote_restated` already reorders chunks that **lexically**
restate one another, and it works: it needs a shared prefix of four words
covering 60% of the shorter text, so it catches a statement whose
replacement differs at the **end**. Measured over 40 subjects, that shape
scores MRR 1.000.

Two ordinary shapes it cannot see score 0.500 -- the retired passage
first, every time:

    person-0 works at Northwind as a staff engineer.      (retired)
    person-0 works at Brightlake as a principal engineer. (current)

The changed value sits mid-sentence, so no prefix is shared. The ledger
knows perfectly well: one fact carries `superseded_by` pointing at the
other. Nothing in the recall path ever read it.

This reads it. Order only -- nothing is dropped, because a caller may be
asking about the past, and `as_of` is honoured for free because the facts
this pairs against are already the ones holding at the boundary.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex,
                          MemoryEngine)

pytestmark = pytest.mark.asyncio

RETIRED = "person-1 works at Northwind as a staff engineer."
CURRENT = "person-1 works at Brightlake as a principal engineer."
QUERY = "where does person-1 work"


async def chain(**options):
    """One subject whose employer was stated, then replaced in different
    words, with the replacement recorded in the ledger."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder(), **options).open()
    # Dated, because `as_of` bounds the episode lane as well as fact
    # validity -- episodes written "now" fall outside a past boundary and
    # the reader gets nothing at all to order.
    old = await engine.remember("s", RETIRED, source="retired",
                                created_at="2024-01-02T00:00:00Z")
    new = await engine.remember("s", CURRENT, source="current",
                                created_at="2024-06-02T00:00:00Z")
    await engine.assert_fact("s", "person-1", "works_at", "Northwind",
                             valid_from="2024-01-01T00:00:00Z",
                             source_episode_id=old.episode_id)
    await engine.assert_fact("s", "person-1", "works_at", "Brightlake",
                             valid_from="2024-06-01T00:00:00Z",
                             source_episode_id=new.episode_id)
    return engine


async def sources(engine, **ask) -> list[str]:
    found = await engine.recall("s", QUERY, limit=5, **ask)
    return [item.source or "" for item in found.items]


async def test_the_replacement_outranks_what_it_replaced():
    """The whole point. Lexically these two share only three words of
    prefix, so `demote_restated` cannot group them and insertion order
    decides -- which puts the retired statement first."""
    engine = await chain()
    try:
        assert (await sources(engine))[0] == "current"
    finally:
        await engine.close()


async def test_the_retired_passage_is_moved_and_not_dropped():
    """A caller may be asking about the past, so this reorders and never
    removes -- the same promise `demote_restated` makes."""
    engine = await chain()
    try:
        assert sorted(await sources(engine)) == ["current", "retired"]
    finally:
        await engine.close()


async def test_a_reader_asking_about_the_past_is_not_overruled():
    """`as_of` before the replacement: the earlier statement is what held
    then, so it must not be demoted. This costs no special case -- the
    facts paired against are the ones holding at the boundary."""
    engine = await chain()
    try:
        assert (await sources(engine, as_of="2024-03-01T00:00:00Z"))[0] == "retired"
    finally:
        await engine.close()


async def test_nothing_moves_when_the_replacement_was_not_returned():
    """Demoting a retired passage below evidence that is not there would
    push down the one result the reader did get.

    Two passages come back and neither is the replacement -- the
    replacing fact was asserted with no episode behind it. The retired
    passage must keep its rank, so the fixture needs a second item;
    with only one, the length guard returns before this rule is reached
    and the test proves nothing.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    try:
        old = await engine.remember("s", RETIRED, source="retired",
                                    created_at="2024-01-02T00:00:00Z")
        await engine.remember("s", "person-1 works reasonable hours at the office.",
                              source="aside", created_at="2024-02-02T00:00:00Z")
        await engine.assert_fact("s", "person-1", "works_at", "Northwind",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=old.episode_id)
        await engine.assert_fact("s", "person-1", "works_at", "Brightlake",
                                 valid_from="2024-06-01T00:00:00Z")
        order = await sources(engine)
        assert len(order) == 2, order
        assert order[0] == "retired", order
    finally:
        await engine.close()


async def test_the_ledger_is_asked_about_the_passages_returned_not_the_query():
    """The cost has to be proportionate. What is read is the facts each
    returned episode stated -- one indexed read per episode -- so a query
    that matched no fact at all still learns whether the passages it got
    have been retired, and a query that matched ten facts of one subject
    does not read that subject's chain ten times."""
    engine = await chain()
    read: list[int] = []
    original = engine.documents.facts_for_graph

    async def counted(space, source_episode_id, limit):
        read.append(source_episode_id)
        return await original(space, source_episode_id, limit)

    engine.documents.facts_for_graph = counted  # type: ignore[method-assign]
    try:
        found = await engine.recall("s", "unrelated words about weather", limit=5)
        assert sorted(read) == sorted({item.episode_id for item in found.items}), read
        read.clear()
        found = await engine.recall("s", QUERY, limit=5)
        assert sorted(read) == sorted({item.episode_id for item in found.items}), read
    finally:
        engine.documents.facts_for_graph = original  # type: ignore[method-assign]
        await engine.close()


async def test_a_retired_passage_follows_its_own_successor_even_one_retired_in_turn():
    """Northwind was replaced by Calderon, and Calderon by Brightlake,
    whose episode does not match the query.

    An earlier version of this test expected nothing to move here, on the
    premise that both passages were retired by the same absent claim.
    The ledger says otherwise: Northwind's successor is Calderon, and the
    Calderon passage was returned. So the Calderon passage leads, both
    are marked, and nothing is compared against the Brightlake passage
    the reader did not get.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    try:
        first = await engine.remember("s", "person-1 works at Northwind as a staff engineer.",
                                      source="first", created_at="2024-01-02T00:00:00Z")
        second = await engine.remember("s", "person-1 works at Calderon as a senior engineer.",
                                       source="second", created_at="2024-02-02T00:00:00Z")
        away = await engine.remember("s", "Entirely unrelated notes about rainfall in March.",
                                     source="away", created_at="2024-03-02T00:00:00Z")
        for when, what, episode in (("2024-01-01", "Northwind", first),
                                    ("2024-02-01", "Calderon", second),
                                    ("2024-06-01", "Brightlake", away)):
            await engine.assert_fact("s", "person-1", "works_at", what,
                                     valid_from=f"{when}T00:00:00Z",
                                     source_episode_id=episode.episode_id)
        # Bounded so the last replacement falls outside what the reader
        # got: demotion runs after the result is cut to `limit`.
        found = await engine.recall("s", QUERY, limit=2)
        order = [item.source or "" for item in found.items]
        assert "away" not in order, order
        assert order == ["second", "first"], order
        assert [item.superseded for item in found.items] == [True, True]
    finally:
        await engine.close()


async def test_it_can_be_turned_off():
    """Off, the old order returns -- which is also what shows the feature
    is doing the work rather than something else in the pipeline."""
    engine = await chain(demote_superseded=False)
    try:
        assert (await sources(engine))[0] == "retired"
    finally:
        await engine.close()


# --- Saying so, not just reordering ---------------------------------------

async def test_a_retired_passage_says_that_it_is_retired():
    """Order alone serves a reader who takes the top result and stops.

    One who reads all of them, or quotes them, needs to know which claim
    the ledger has since retired -- and a passage carries no date that
    says so. The current one must not be marked, or the mark means
    nothing.
    """
    engine = await chain()
    try:
        found = await engine.recall("s", QUERY, limit=5)
        marked = {item.source: item.superseded for item in found.items}
        assert marked == {"current": False, "retired": True}, marked
    finally:
        await engine.close()


async def test_it_is_marked_even_when_its_replacement_was_not_returned():
    """Where the mark matters most, and a limitation worth pinning.

    Reordering runs **after** the result is cut to `limit`, exactly as
    the lexical rule beside it does. So at `limit=1` the replacement has
    already been discarded and the retired passage is the only thing the
    reader gets -- there is nothing left to move it below.

    The mark still lands, which is the whole reason it is computed for
    every retired passage rather than only for the reorderable ones.
    Without it this reader receives a retired claim with nothing at all
    to distinguish it from a current one.
    """
    engine = await chain()
    try:
        found = await engine.recall("s", QUERY, limit=1)
        assert [item.source for item in found.items] == ["retired"], \
            "truncation precedes reordering; if that changes, this test should"
        assert found.items[0].superseded is True
    finally:
        await engine.close()


async def test_what_held_at_the_boundary_is_not_marked_retired():
    """`as_of` before the replacement: that claim held then, and marking
    it retired would contradict the question asked."""
    engine = await chain()
    try:
        found = await engine.recall("s", QUERY, limit=5, as_of="2024-03-01T00:00:00Z")
        assert [item.superseded for item in found.items] == [False]
    finally:
        await engine.close()


async def test_nothing_is_marked_when_nothing_was_replaced():
    """The other half: the mark must be able to be absent."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    try:
        one = await engine.remember("s", RETIRED, source="only",
                                    created_at="2024-01-02T00:00:00Z")
        await engine.remember("s", "person-1 enjoys long walks by the river.",
                              source="aside", created_at="2024-02-02T00:00:00Z")
        await engine.assert_fact("s", "person-1", "works_at", "Northwind",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=one.episode_id)
        found = await engine.recall("s", QUERY, limit=5)
        assert not any(item.superseded for item in found.items)
    finally:
        await engine.close()
