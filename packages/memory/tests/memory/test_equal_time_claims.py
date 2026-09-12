"""Two claims that begin at the same instant, and what the ledger says.

Valid time cannot separate them, so something else does: the order they
arrived in. That is a fact about the recording, not about the world, and
a reader who cannot see the difference will read arrival order as
history. The ledger says which happened, and the graph's health counts
them so somebody can go and look.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"
SAME = "2024-01-01T00:00:00Z"


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z")).open()


async def test_a_claim_superseded_at_the_very_instant_it_began_says_so():
    """It never held for any length of time. "superseded by fact 2" reads
    like a history; this one is a collision."""
    engine = await memory()
    await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=SAME)
    await engine.assert_fact(SPACE, "alice", "works_at", "Globex", valid_from=SAME)
    [first, second] = sorted(await engine.documents.list_facts(SPACE, include_closed=True),
                             key=lambda fact: fact.fact_id)
    assert first.status == "closed" and first.valid_until == first.valid_from
    assert first.closed_reason == "superseded at the same instant by fact 2, which was recorded later"
    assert second.status == "active"


async def test_an_ordinary_supersession_still_reads_as_one():
    engine = await memory()
    await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=SAME)
    await engine.assert_fact(SPACE, "alice", "works_at", "Globex", valid_from="2024-06-01T00:00:00Z")
    [first, _] = sorted(await engine.documents.list_facts(SPACE, include_closed=True),
                        key=lambda fact: fact.fact_id)
    assert first.closed_reason == "superseded by fact 2"


async def test_the_graph_health_counts_claims_that_collided():
    from scone_memory.entities.health import graph_health

    engine = await memory()
    await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=SAME)
    await engine.assert_fact(SPACE, "alice", "works_at", "Globex", valid_from=SAME)
    found = await graph_health(engine, SPACE, status="all")
    [concern] = [c for c in found.concerns if c["kind"] == "contested_instant"]
    assert concern["count"] == 1
    example = concern["examples"][0]
    assert example["subject"] == "alice" and example["predicate"] == "works_at"
    assert example["began"].startswith("2024-01-01") and len(example["claims"]) == 2
    assert "contested_instant" in found.text and "the order they arrived in" in found.text


async def test_a_restatement_of_the_same_claim_is_not_a_collision():
    """Saying the same thing twice at the same moment is agreement."""
    from scone_memory.entities.health import graph_health

    engine = await memory()
    await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=SAME)
    await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=SAME)
    found = await graph_health(engine, SPACE, status="all")
    assert not [c for c in found.concerns if c["kind"] == "contested_instant"]
