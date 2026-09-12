"""The recall evidence graph draws the things its claims are about.

It used to add a 'concept' for every capitalised word in a returned chunk,
so 'However', 'Monday' and 'Will' became nodes and 'Alice' and 'Chen' two
people. Concepts now come only from retained claims, objects only when they
name a thing, and a chunk mentions a concept only where one of its recorded
spellings appears.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.models import RecallResult
from scone_memory.entities.ids import key_id
from scone_memory.retrieval.evidence_graph import build_query_evidence_graph

NOTE = "However, Dr. Alice Chen joined Acme Robotics in May 2021. Will she call on Monday? The team uses AI."


@pytest.fixture
async def engine():
    instance = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield instance
    await instance.close()


async def graph_for(engine, claims):
    episode = await engine.remember("alpha", NOTE, source="notes/alice.md")
    facts = [await engine.assert_fact("alpha", subject, predicate, obj, source_episode_id=episode.episode_id, quote=quote)
             for subject, predicate, obj, quote in claims]
    result = await engine.recall("alpha", "Alice Chen")
    result.facts = facts
    return await build_query_evidence_graph(engine.documents, "alpha", "Alice Chen", result)


def concepts(graph):
    return {node.label: node for node in graph.nodes if node.kind == "concept"}


async def test_capitalised_words_with_no_claim_are_not_concepts(engine):
    graph = await graph_for(engine, [])
    assert concepts(graph) == {}
    assert not any(edge.kind == "mentions" for edge in graph.edges)


async def test_claims_draw_their_entities_with_the_casing_the_quote_used(engine):
    graph = await graph_for(engine, [("alice chen", "works_at", "Acme Robotics", "Dr. Alice Chen joined Acme Robotics")])
    found = concepts(graph)
    assert set(found) == {"Alice Chen", "Acme Robotics"}
    assert found["Alice Chen"].id == key_id("alpha", "alice chen")
    assert found["Alice Chen"].data["entity_id"] == key_id("alpha", "alice chen")
    relation = next(edge for edge in graph.edges if edge.kind == "relation")
    assert (relation.source, relation.target, relation.label) == (found["Alice Chen"].id, found["Acme Robotics"].id, "works_at")


async def test_a_value_object_stays_on_its_claim(engine):
    graph = await graph_for(engine, [("alice chen", "joined_on", "May 2021", "Alice Chen joined Acme Robotics in May 2021")])
    assert set(concepts(graph)) == {"Alice Chen"}
    assert not any(edge.kind == "relation" for edge in graph.edges)
    asserted = [edge for edge in graph.edges if edge.kind == "asserts"]
    assert [edge.data["role"] for edge in asserted] == ["subject"]


async def test_chunks_mention_only_recorded_spellings(engine):
    graph = await graph_for(engine, [("alice chen", "works_at", "Acme Robotics", "Dr. Alice Chen joined Acme Robotics"),
                                     ("team", "uses", "AI", "The team uses AI")])
    found = concepts(graph)
    mentioned = {edge.target for edge in graph.edges if edge.kind == "mentions"}
    assert mentioned == {found["Alice Chen"].id, found["Acme Robotics"].id, found["AI"].id, found["team"].id}
    for noise in ("However", "Monday", "Will", "Alice", "Chen", "May"):
        assert noise not in found


async def test_a_short_name_is_mentioned_only_in_its_own_case(engine):
    source = await engine.remember("alpha", "The team uses AI.", source="notes/team.md")
    fact = await engine.assert_fact("alpha", "team", "uses", "AI", source_episode_id=source.episode_id,
                                    quote="The team uses AI")
    other = await engine.remember("alpha", "we should ask ai about it", source="notes/chat.md")
    result = await engine.recall("alpha", "ask ai")
    result.items = [item for item in result.items if item.episode_id == other.episode_id]
    result.facts = [fact]
    graph = await build_query_evidence_graph(engine.documents, "alpha", "ask ai", result)
    ai = concepts(graph)["AI"]
    assert not any(edge.kind == "mentions" and edge.target == ai.id for edge in graph.edges)


async def test_the_evidence_graph_classifies_claims_as_the_knowledge_map_does(engine):
    from scone_memory.entities.project import project_entities
    episode = await engine.remember("alpha", "Alice enjoys graph theory. Bob enjoys graph theory.", source="notes/g.md")
    facts = [await engine.assert_fact("alpha", subject, "enjoys", "graph theory", source_episode_id=episode.episode_id,
                                      quote=f"{subject.title()} enjoys graph theory")
             for subject in ("alice", "bob")]
    result = await engine.recall("alpha", "graph theory")
    result.facts = facts
    graph = await build_query_evidence_graph(engine.documents, "alpha", "graph theory", result)
    drawn = {node.data["key"] for node in graph.nodes if node.kind == "concept"}
    projected = {entity.key for entity in project_entities("alpha", facts, revision=1).entities}
    assert drawn == projected == {"alice", "bob", "graph theory"}
    assert len([edge for edge in graph.edges if edge.kind == "relation"]) == 2
