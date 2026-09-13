"""What the ledger says about the passages a reader received.

`demote_superseded` used to start from the facts the **query** matched and
read each one's whole subject/predicate chain. Four things went wrong with
that, all found by review and all reproduced here first:

- a many-valued claim was moved under a coexisting value instead of its
  own successor, and moved even when the successor was not returned;
- a passage that both replaces one claim and is replaced under another
  lost its place as a replacement, leaving the older passage ahead of it;
- a fact a person had excluded from recall still drove the demotion;
- two passages cost ten chain reads of a thousand facts each.

The question was never about the query's facts. It is about the
**passages the reader got**: which claims they stated, which of those the
ledger has retired by the boundary asked about, and whether the successor
is among the passages too. So this reads the facts stated by each returned
episode -- one indexed read per episode, proportional to the result and
not to the ledger -- and pairs a retired claim with its successor by id.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.core import graph_read
from scone_memory.core.models import RecallItem
from scone_memory.retrieval.supersession import demote_superseded

pytestmark = pytest.mark.asyncio

QUERY = "where does person-1 work"
LATER = "2030-01-01T00:00:00Z"


async def engine(**options):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                              HashEmbedder(), **options).open()


async def remembered(memory, source: str, text: str, when: str):
    added = await memory.remember("s", text, source=source, created_at=f"{when}T00:00:00Z")
    return added.episode_id


async def stated(memory, subject, predicate, obj, when: str, episode=None):
    return await memory.assert_fact("s", subject, predicate, obj,
                                    valid_from=f"{when}T00:00:00Z",
                                    source_episode_id=episode)


async def order_and_marks(memory, **ask):
    found = await memory.recall("s", QUERY, limit=5, **ask)
    return ([item.source or "" for item in found.items],
            {item.source or "": item.superseded for item in found.items},
            found)


async def baseline(memory) -> list[str]:
    """The order the pipeline produces with the ledger rule switched off,
    so each test asserts a movement rather than an absolute order."""
    memory.demote_superseded = False
    try:
        found = await memory.recall("s", QUERY, limit=5)
        return [item.source or "" for item in found.items]
    finally:
        memory.demote_superseded = True


def item(chunk_id: int, episode_id: int, when: str = "2024-01-01T00:00:00Z") -> RecallItem:
    return RecallItem(chunk_id=chunk_id, episode_id=episode_id, text="", score=1.0,
                      created_at=when)


# --- many-valued predicates ------------------------------------------------

async def many_valued_employer(memory):
    """person-1 works at Northwind, and also at Brightlake; later the
    Northwind claim is stated again, which retires the first one. The
    successor of the old Northwind claim is the new Northwind claim --
    never the coexisting Brightlake one."""
    old = await remembered(memory, "northwind", "person-1 works at Northwind as a staff engineer.", "2024-01-02")
    beside = await remembered(memory, "brightlake", "person-1 works at Brightlake as an advisor.", "2024-02-02")
    again = await remembered(memory, "northwind-again", "person-1 works at Northwind as a principal engineer.", "2024-06-02")
    await stated(memory, "person-1", "works_at", "Brightlake", "2024-02-01", beside)
    # A repeated value stated from a later day restates; it is the late
    # arrival -- the June claim already in the ledger when January's is
    # stated -- that gets bounded by it, which is the supersession here.
    second = await stated(memory, "person-1", "works_at", "Northwind", "2024-06-01", again)
    first = await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
    retired = await memory.documents.get_fact("s", first.fact_id)
    assert retired is not None and retired.superseded_by == second.fact_id, \
        "the fixture needs the later Northwind claim to bound the earlier one"
    return old, beside, again


async def test_a_many_valued_claim_follows_its_own_successor_not_a_coexisting_value():
    memory = await engine(many_valued=["works_at"])
    try:
        await many_valued_employer(memory)
        before = await baseline(memory)
        order, marks, _ = await order_and_marks(memory)
        assert set(order) == {"northwind", "brightlake", "northwind-again"}, order
        assert order.index("northwind-again") < order.index("northwind"), (before, order)
        assert order.index("brightlake") == before.index("brightlake"), (before, order)
        assert marks == {"northwind": True, "brightlake": False, "northwind-again": False}, marks
    finally:
        await memory.close()


async def test_a_many_valued_claim_stays_put_when_its_successor_was_not_returned():
    """The successor is real but has no passage. Under the old rule the
    retired Northwind passage was moved below the coexisting Brightlake
    one -- a comparison the ledger never made."""
    memory = await engine(many_valued=["works_at"])
    try:
        old = await remembered(memory, "northwind", "person-1 works at Northwind.", "2024-01-02")
        beside = await remembered(memory, "brightlake", "person-1 works at Brightlake as an advisor on the side.", "2024-02-02")
        await stated(memory, "person-1", "works_at", "Brightlake", "2024-02-01", beside)
        await stated(memory, "person-1", "works_at", "Northwind", "2024-06-01")
        await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
        before = await baseline(memory)
        # The junction: the retired passage must lead, or "nothing moved"
        # is indistinguishable from "moved under the coexisting value".
        assert before == ["northwind", "brightlake"], before
        order, marks, _ = await order_and_marks(memory)
        assert order == before, (before, order)
        assert marks == {"northwind": True, "brightlake": False}, marks
    finally:
        await memory.close()


async def test_a_coexisting_value_is_never_taken_for_the_successor():
    """The same claim on fixed positions: the retired Northwind passage
    first, the coexisting Brightlake passage second, the real successor
    without a passage. Pairing by slot instead of by id would move the
    first under the second."""
    memory = await engine(many_valued=["works_at"])
    try:
        old = await remembered(memory, "northwind", "person-1 works at Northwind.", "2024-01-02")
        beside = await remembered(memory, "brightlake", "person-1 works at Brightlake as an advisor.", "2024-02-02")
        await stated(memory, "person-1", "works_at", "Brightlake", "2024-02-01", beside)
        await stated(memory, "person-1", "works_at", "Northwind", "2024-06-01")
        await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
        order, marks = await reorder(memory, [old, beside])
        assert order == [old, beside], order
        assert marks == {old: True, beside: False}, marks
    finally:
        await memory.close()


# --- exclusion --------------------------------------------------------------

async def employer_chain(memory):
    old = await remembered(memory, "retired", "person-1 works at Northwind as a staff engineer.", "2024-01-02")
    new = await remembered(memory, "current", "person-1 works at Brightlake as a principal engineer.", "2024-06-02")
    first = await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
    second = await stated(memory, "person-1", "works_at", "Brightlake", "2024-06-01", new)
    return first, second


async def test_an_excluded_claim_neither_marks_nor_moves_its_passage():
    """Exclusion suppresses a fact from recall. A fact that a person has
    said is wrong must not go on shaping what they are shown."""
    memory = await engine()
    try:
        first, _ = await employer_chain(memory)
        await memory.exclude("s", first.fact_id, "Incorrect source attribution")
        before = await baseline(memory)
        order, marks, _ = await order_and_marks(memory)
        assert order == before, (before, order)
        assert marks == {"retired": False, "current": False}, marks
    finally:
        await memory.close()


async def test_an_excluded_successor_cannot_promote_its_passage():
    """The ledger's intervals are untouched by exclusion, so the old claim
    is still retired and its passage still says so; but the excluded
    replacement is suppressed from recall and cannot pull its own passage
    ahead."""
    memory = await engine()
    try:
        _, second = await employer_chain(memory)
        await memory.exclude("s", second.fact_id, "Incorrect source attribution")
        before = await baseline(memory)
        order, marks, _ = await order_and_marks(memory)
        assert order == before, (before, order)
        assert marks == {"retired": True, "current": False}, marks
    finally:
        await memory.close()


# --- closure ----------------------------------------------------------------

async def test_a_closed_claim_is_marked_retired_and_nothing_moves():
    """Closing ends a claim without replacing it. The passage's claim no
    longer holds, which the reader must be told; there is no successor to
    order it under, so nothing moves."""
    memory = await engine()
    try:
        old = await remembered(memory, "ended", "person-1 works at Northwind as a staff engineer.", "2024-01-02")
        await remembered(memory, "aside", "person-1 works reasonable hours at the office.", "2024-02-02")
        first = await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
        await memory.close_fact("s", first.fact_id, "left the company")
        before = await baseline(memory)
        order, marks, _ = await order_and_marks(memory)
        assert order == before, (before, order)
        assert marks == {"ended": True, "aside": False}, marks
    finally:
        await memory.close()


async def test_a_claim_closed_after_the_boundary_asked_about_still_held_then():
    memory = await engine()
    try:
        old = await remembered(memory, "ended", "person-1 works at Northwind as a staff engineer.", "2024-01-02")
        first = await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
        await memory.close_fact("s", first.fact_id, "left the company")
        _, marks, _ = await order_and_marks(memory, as_of="2024-03-01T00:00:00Z")
        assert marks == {"ended": False}, marks
    finally:
        await memory.close()


# --- cost and bounds --------------------------------------------------------

async def test_the_ledger_is_read_once_per_returned_episode_and_never_by_chain():
    memory = await engine()
    try:
        await employer_chain(memory)
        by_episode: list[int] = []
        chains: list[tuple[str, str]] = []
        documents = memory.documents
        read_graph, read_chain = documents.facts_for_graph, documents.facts_for

        async def counted_graph(space, source_episode_id, limit):
            by_episode.append(source_episode_id)
            return await read_graph(space, source_episode_id, limit)

        async def counted_chain(space, subject, predicate):
            chains.append((subject, predicate))
            return await read_chain(space, subject, predicate)

        documents.facts_for_graph = counted_graph  # type: ignore[method-assign]
        documents.facts_for = counted_chain  # type: ignore[method-assign]
        found = await memory.recall("s", QUERY, limit=5)
        returned = {item.episode_id for item in found.items}
        assert len(returned) == 2, found.items
        assert sorted(by_episode) == sorted(returned), by_episode
        assert chains == [], chains
    finally:
        await memory.close()


async def test_an_episode_whose_facts_exceed_the_read_cap_is_disclosed(monkeypatch):
    """A cap that bites must say so. With the cap at one fact and an
    episode that stated two, the reader is told which episode was only
    partly read; with nothing cut, no note appears."""
    memory = await engine()
    try:
        first, _ = await employer_chain(memory)
        _, _, whole = await order_and_marks(memory)
        assert not [note for note in whole.degraded if note.startswith("supersession")], whole.degraded
        await stated(memory, "person-1", "role", "staff engineer", "2024-01-01", first.source_episode_id)
        monkeypatch.setattr(graph_read, "MAX_GRAPH_FACTS", 1)
        _, marks, cut = await order_and_marks(memory)
        notes = [note for note in cut.degraded if note.startswith("supersession")]
        assert notes and str(first.source_episode_id) in notes[0], cut.degraded
        assert marks["retired"] is True, marks
    finally:
        await memory.close()


async def test_a_store_that_cannot_read_by_episode_says_so_instead_of_guessing():
    memory = await engine()
    try:
        await employer_chain(memory)
        before = await baseline(memory)
        memory.documents.facts_for_graph = None  # type: ignore[method-assign]
        order, marks, found = await order_and_marks(memory)
        assert order == before, (before, order)
        assert not any(marks.values()), marks
        assert any(note.startswith("supersession") for note in found.degraded), found.degraded
    finally:
        await memory.close()


# --- the reordering itself, on fixed inputs ---------------------------------

async def reorder(memory, episodes: list[int], when: str = LATER):
    items = [item(index + 1, episode) for index, episode in enumerate(episodes)]
    degraded: list[str] = []
    ordered = await demote_superseded(memory.documents, "s", items, when, degraded=degraded)
    assert degraded == [], degraded
    return [entry.episode_id for entry in ordered], {entry.episode_id: entry.superseded for entry in ordered}


async def test_a_passage_that_replaces_one_claim_and_is_replaced_under_another_sits_between():
    """A: the old employer. B: the new employer and the old role. C: the
    new role. B is a replacement (of A) and replaced (by C). The old rule
    grouped B under C and forgot it led A, leaving A first."""
    memory = await engine()
    try:
        a = await remembered(memory, "a", "person-1 works at Northwind.", "2024-01-02")
        b = await remembered(memory, "b", "person-1 works at Brightlake as a staff engineer.", "2024-06-02")
        c = await remembered(memory, "c", "person-1 is now a principal engineer.", "2024-09-02")
        await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", a)
        await stated(memory, "person-1", "works_at", "Brightlake", "2024-06-01", b)
        await stated(memory, "person-1", "role", "staff engineer", "2024-06-01", b)
        await stated(memory, "person-1", "role", "principal engineer", "2024-09-01", c)
        order, marks = await reorder(memory, [a, b, c])
        assert order == [c, b, a], order
        assert marks == {a: True, b: True, c: False}, marks
    finally:
        await memory.close()


async def test_a_mutual_replacement_keeps_the_reader_order():
    """Each passage replaces a claim the other made -- one stated the
    employer late and the role early, the other the reverse. No order
    puts every replacement first, so the reader's order stands, and both
    passages say a claim of theirs has been retired."""
    memory = await engine()
    try:
        a = await remembered(memory, "a", "person-1 worked at Northwind, later a senior engineer.", "2024-01-02")
        b = await remembered(memory, "b", "person-1 was a junior engineer, later at Brightlake.", "2024-06-02")
        await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", a)
        await stated(memory, "person-1", "works_at", "Brightlake", "2024-06-01", b)
        await stated(memory, "person-1", "role", "junior engineer", "2024-01-01", b)
        await stated(memory, "person-1", "role", "senior engineer", "2024-07-01", a)
        assert (await reorder(memory, [a, b]))[0] == [a, b]
        assert (await reorder(memory, [b, a]))[0] == [b, a]
        assert (await reorder(memory, [a, b]))[1] == {a: True, b: True}
    finally:
        await memory.close()


async def test_unrelated_passages_keep_their_positions_when_a_pair_is_reordered():
    memory = await engine()
    try:
        old = await remembered(memory, "old", "person-1 works at Northwind.", "2024-01-02")
        aside = await remembered(memory, "aside", "It rained for most of March.", "2024-03-02")
        new = await remembered(memory, "new", "person-1 works at Brightlake.", "2024-06-02")
        await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
        await stated(memory, "person-1", "works_at", "Brightlake", "2024-06-01", new)
        order, _ = await reorder(memory, [old, aside, new])
        assert order == [new, aside, old], order
    finally:
        await memory.close()


async def test_two_chunks_of_one_retired_episode_both_follow_the_replacement():
    memory = await engine()
    try:
        old = await remembered(memory, "old", "person-1 works at Northwind.", "2024-01-02")
        new = await remembered(memory, "new", "person-1 works at Brightlake.", "2024-06-02")
        await stated(memory, "person-1", "works_at", "Northwind", "2024-01-01", old)
        await stated(memory, "person-1", "works_at", "Brightlake", "2024-06-01", new)
        order, marks = await reorder(memory, [old, old, new])
        assert order == [new, old, old], order
        assert marks == {old: True, new: False}, marks
    finally:
        await memory.close()
