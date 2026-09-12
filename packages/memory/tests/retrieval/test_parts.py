"""Searching each part of a question, so no part loses the blend.

The reason to split a question is that one query over two halves returns
one blend of passages, and the half whose words are commoner tends to
lose it entirely. These tests pin that claim directly: a passage for
every part, and an honest name for any part nothing answered.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.retrieval.decompose import decompose
from scone_memory.retrieval.parts import recall_parts

pytestmark = pytest.mark.asyncio

BILLING = "We reverted the billing change after the invoices came out wrong."
MEETING = "At the Thursday meeting were Priya, Tomas and the auditor from Lisbon."
NOISE = ["A note about the billing system and its invoices.",
         "More billing notes, invoices and charge codes.",
         "Billing again: invoices, charges, credits, and the billing ledger.",
         "Yet more about billing and invoices and the billing run."]
BOTH = "What did I decide about billing, and who was at the meeting?"


async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for text in [BILLING, MEETING, *NOISE]:
        await engine.remember("default", text)
    return engine


async def test_every_part_s_own_best_passage_is_in_the_answer():
    """The guarantee, and the whole reason to do this: with room for two,
    the two passages are the best each part found — not the two best of
    one part. Whether that retrieves better on a real corpus is for the
    bench to say, not for a corpus arranged to make the point."""
    engine = await memory()
    try:
        read = decompose(BOTH)
        tops = [(await engine.recall("default", part.text, limit=3)).items[0].chunk_id
                for part in read.parts]
        parted = await recall_parts(engine, "default", BOTH, limit=2)
    finally:
        await engine.close()
    assert len(set(tops)) == 2, "the parts have to want different passages for this to prove anything"
    assert sorted(item.chunk_id for item in parted.items) == sorted(tops), parted.record()


async def test_a_part_nothing_answered_is_named_rather_than_passed_over():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        parted = await recall_parts(engine, "default", BOTH, limit=3)
    finally:
        await engine.close()
    assert parted.unanswered == ("What did I decide about billing", "who was at the meeting?")
    assert "2 part" in parted.why and "nothing" in parted.why, parted.why


async def test_finding_passages_is_not_reported_as_having_answered_a_part():
    """With no floor configured nothing judges a part's evidence, and an
    empty `weak` must not read as "every part was answered". The report
    says which it is."""
    engine = await memory()
    try:
        parted = await recall_parts(engine, "default", BOTH, limit=3)
    finally:
        await engine.close()
    assert not parted.unanswered and not parted.weak and not parted.judged
    assert "not evidence" in parted.why, parted.why


async def test_a_question_asking_one_thing_is_one_search():
    engine = await memory()
    try:
        parted = await recall_parts(engine, "default", "Where can I buy salt and pepper?", limit=3)
    finally:
        await engine.close()
    assert not parted.decomposition.split
    assert len(parted.per_part) == 1 and parted.per_part[0].contributed == len(parted.items)


async def test_the_same_passage_answering_two_parts_is_returned_once_and_credited_twice():
    engine = await memory()
    try:
        asked = "What did I decide about billing, and what happened to the invoices?"
        parted = await recall_parts(engine, "default", asked, limit=4)
    finally:
        await engine.close()
    texts = [item.text for item in parted.items]
    assert len(texts) == len(set(texts)), texts
    shared = [chunk for chunk, which in parted.by_chunk.items() if len(which) > 1]
    assert shared, "both parts ask about the same passage; the credit should say so"


async def test_no_more_than_the_limit_comes_back():
    engine = await memory()
    try:
        parted = await recall_parts(engine, "default", BOTH, limit=2)
    finally:
        await engine.close()
    assert len(parted.items) == 2


async def test_a_part_counts_what_it_found_apart_from_what_it_contributed():
    """A part can find plenty and contribute little, because another part
    got there first. Reporting one number for both would hide that."""
    engine = await memory()
    try:
        parted = await recall_parts(engine, "default", BOTH, limit=2)
    finally:
        await engine.close()
    assert sum(part.contributed for part in parted.per_part) == len(parted.items)
    assert any(part.found > part.contributed for part in parted.per_part), parted.record()


async def test_an_empty_question_is_refused():
    from scone_memory.core.errors import InvalidInput

    engine = await memory()
    try:
        with pytest.raises(InvalidInput):
            await recall_parts(engine, "default", "  ", limit=3)
    finally:
        await engine.close()
