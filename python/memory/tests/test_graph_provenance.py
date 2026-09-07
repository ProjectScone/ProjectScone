"""A claim in the graph shows where it came from, or says it cannot.

Codex found the hole while building the depth view: the snapshot draws
every claim but hydrates only the episodes the event window happened to
mention, so a claim whose source is older than the window lost its
source edge silently. A claim floating with no visible source is worse
than a claim marked as having provenance outside the snapshot.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              events=InMemoryEventLog()).open()


def edges(graph, kind):
    return [(e.source, e.target) for e in graph.edges if e.kind == kind]


async def test_a_claim_shows_the_old_episode_it_came_from(engine):
    old = await engine.remember("alpha", "Ana moved to Lisbon in March", created_at="2024-01-01")
    fact = await engine.assert_fact("alpha", "ana", "moved_to", "lisbon",
                                    source_episode_id=old.episode_id, quote="Ana moved to Lisbon")
    for i in range(3):
        await engine.remember("alpha", f"something newer, number {i}")

    graph = await engine.graph("alpha")
    assert f"claim:{fact.fact_id}" in graph.nodes
    assert f"episode:{old.episode_id}" in graph.nodes, "the source is drawn even though the window never mentioned it"
    assert (f"episode:{old.episode_id}", f"claim:{fact.fact_id}") in edges(graph, "source_of")
    assert graph.provenance_omitted == 0


async def test_a_recall_that_returned_an_old_chunk_keeps_the_whole_chain(engine):
    old = await engine.remember("alpha", "Juniper points at Polaris on clear nights", created_at="2024-01-01")
    found = await engine.recall("alpha", "where does juniper point")
    assert found.items, "the recall has to return something for the chain to exist"

    graph = await engine.graph("alpha")
    chunk = f"chunk:{found.items[0].chunk_id}"
    assert chunk in graph.nodes and f"episode:{old.episode_id}" in graph.nodes
    assert (f"episode:{old.episode_id}", chunk) in edges(graph, "chunked_into")
    assert any(target == chunk for _, target in edges(graph, "returned")), "the recall still points at what it returned"


async def test_an_episode_nobody_captured_a_session_for_stands_alone(engine):
    """Never invent a session: an episode written by the API has no agent
    session, and drawing one would be a claim about how it arrived."""
    old = await engine.remember("alpha", "written straight to the API", created_at="2024-01-01")
    await engine.assert_fact("alpha", "note", "written_by", "api", source_episode_id=old.episode_id)

    graph = await engine.graph("alpha")
    assert not [n for n in graph.nodes if n.startswith("session:")], "no session existed, so none is drawn"
    assert not edges(graph, "captured"), "and nothing pretends one captured this episode"


async def test_provenance_left_out_of_a_snapshot_is_reported_rather_than_hidden(engine):
    for i in range(6):
        episode = await engine.remember("alpha", f"source number {i}", created_at=f"2024-01-0{i + 1}")
        await engine.assert_fact("alpha", f"subject{i}", "came_from", f"note{i}", source_episode_id=episode.episode_id)

    small = await engine.graph("alpha", limit=2)
    drawn = [n for n in small.nodes if n.startswith("episode:")]
    assert small.provenance_omitted > 0, "a snapshot that could not fit every source says so"
    assert len(drawn) + small.provenance_omitted == 6, "and the two numbers account for every source"
    assert small.as_dict()["provenance_omitted"] == small.provenance_omitted
