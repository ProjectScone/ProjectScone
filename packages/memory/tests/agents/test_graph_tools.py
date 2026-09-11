"""The entity graph as tools a model calls through the ToolBox.

The same three reads the MCP server offers: the context around some names
or a question, one entity explained, and how two connect. Each answers with
the packet the HTTP route gives, reads only the box's space, and turns a
mistake into a result the model can read.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.tools import MEMORY_TOOLS, ToolBox
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"
NOW = "2025-06-01T00:00:00.000Z"


@pytest.fixture
async def box():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock(NOW)).open()
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "alice park", "knows", "Bob", valid_from=DAY)
    await engine.assert_fact("beta", "zed", "works_at", "Globex", valid_from=DAY)
    yield ToolBox(engine, "alpha")
    await engine.close()


def test_the_graph_tools_are_offered_with_the_others():
    assert [tool.name for tool in MEMORY_TOOLS][-5:] == ["graph_context", "explain_entity", "connect_entities",
                                                         "graph_schema", "graph_match"]


async def test_graph_context_answers_with_the_packet_the_route_gives(box):
    result = await box.run("graph_context", {"names": ["alice chen"]})
    assert result["ok"] is True and result["status"] == "prepared" and result["space"] == "alpha"
    assert "works_at Acme Robotics" in result["text"] and "based_in Lisbon" in result["text"]
    assert result["filters"] == {"status": "current", "as_of": NOW} and "reasons" in result["coverage"]


async def test_a_question_centres_on_the_entities_it_names(box):
    result = await box.run("graph_context", {"question": "where is acme robotics based?", "max_bytes": 1024})
    assert result["ok"] is True and "based_in Lisbon" in result["text"] and len(result["text"].encode()) <= 1024


async def test_explain_entity_lists_candidates_for_an_ambiguous_name(box):
    result = await box.run("explain_entity", {"name": "alice"})
    assert result["ok"] is True and result["status"] == "ambiguous" and len(result["candidates"]) == 2


async def test_connect_entities_gives_each_hop_with_its_facts(box):
    result = await box.run("connect_entities", {"source": "alice chen", "target": "lisbon"})
    assert result["ok"] is True
    assert "path: alice chen -works_at-> Acme Robotics -based_in-> Lisbon" in result["text"]


async def test_the_graph_tools_read_their_own_space_only(box):
    result = await box.run("explain_entity", {"name": "zed"})
    assert result["ok"] is True and result["status"] == "empty" and "Globex" not in result["text"]


@pytest.mark.parametrize("name, arguments, complaint", [
    ("graph_context", {}, "names or a question"),
    ("graph_context", {"names": ["x"] * 25}, "at most 24"),
    ("graph_context", {"question": "q" * 2001}, "question"),
    ("graph_context", {"names": ["alice chen"], "max_bytes": 100}, "max_bytes"),
    ("explain_entity", {"name": "x" * 201}, "200"),
    ("connect_entities", {"source": "alice chen", "target": "lisbon", "max_hops": 5}, "max_hops"),
    ("connect_entities", {"source": "", "target": "lisbon"}, "source"),
])
async def test_a_mistake_is_a_result_the_model_can_read(box, name, arguments, complaint):
    result = await box.run(name, arguments)
    assert result["ok"] is False and complaint in result["error"]


async def test_connect_entities_looks_no_further_than_max_hops(box):
    result = await box.run("connect_entities", {"source": "alice chen", "target": "lisbon", "max_hops": 1})
    assert result["ok"] is True and "no path: none within 1 hops" in result["text"]
    assert not any(line.startswith("path: ") for line in result["text"].splitlines())


async def test_explain_entity_stays_on_the_entitys_own_relations(box):
    """One hop: Alice's employer, not where her employer is based."""
    result = await box.run("explain_entity", {"name": "alice chen"})
    assert "works_at Acme Robotics" in result["text"] and "based_in Lisbon" not in result["text"]


async def test_graph_schema_says_what_the_space_could_be_asked(box):
    result = await box.run("graph_schema", {})
    by_name = {entry["predicate"]: entry for entry in result["predicates"]}
    assert result["ok"] is True and result["space"] == "alpha" and set(by_name) == {"works_at", "based_in", "knows"}
    assert result["filters"] == {"status": "current", "as_of": NOW} and result["complete"] is True
    assert (await box.run("graph_schema", {"limit": 1}))["truncated"] is True


async def test_the_whole_answer_stays_small_when_many_names_are_ambiguous():
    """max_bytes bounds the packet text; the JSON around it is bounded on its
    own: a repeated name is one lookup, candidates are capped and clipped."""
    import json

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock(NOW)).open()
    for n in range(30):
        await engine.assert_fact("alpha", f"alice {n:02d} " + "x" * 500, "works_at", "Acme", valid_from=DAY)
    result = await ToolBox(engine, "alpha").run("graph_context", {"names": ["alice"] * 24, "max_bytes": 512})
    await engine.close()
    assert result["ok"] is True and len(result["text"].encode()) <= 512 and len(result["candidates"]) == 24
    assert len(json.dumps(result).encode()) < 12_000


async def test_one_huge_predicate_cannot_make_the_schema_answer_huge():
    import json

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock(NOW)).open()
    await engine.assert_fact("alpha", "alice", "x" * 200_000, "value", valid_from=DAY)
    box = ToolBox(engine, "alpha")
    result = await box.run("graph_schema", {"limit": 1})
    small = await box.run("graph_schema", {"max_bytes": 1024})
    await engine.close()
    assert result["ok"] is True and len(json.dumps(result).encode()) < 4_000
    assert result["predicates"][0]["length"] == 200_000 and small["ok"] is True
    assert (await box.run("graph_schema", {"max_bytes": 100}))["ok"] is False


async def test_graph_context_can_seed_by_resemblance(box):
    result = await box.run("graph_context", {"question": "which robotics firm?", "similar": True})
    assert result["ok"] is True and result["coverage"]["similar"] and " similar " in result["text"]
    assert (await box.run("graph_context", {"question": "x", "similar": "yes"}))["ok"] is False
    assert (await box.run("graph_context", {"question": "x", "min_similarity": 2}))["ok"] is False


WHERE = [{"subject": "?who", "predicate": "works_at", "object": "?org"},
         {"subject": "?org", "predicate": "based_in", "object": "Lisbon"}]


async def test_graph_match_answers_a_structured_question_with_the_routes_rows(box):
    result = await box.run("graph_match", {"where": WHERE, "returns": ["?who"]})
    assert result["ok"] is True and result["status"] == "matched" and result["space"] == "alpha"
    assert result["rows"][0]["bindings"]["?who"]["key"] == "alice chen" and result["variables"] == ["?who"]
    assert result["filters"] == {"status": "current", "as_of": NOW, "together": True, "limit": 20}
    history = await box.run("graph_match", {"where": WHERE, "status": "history", "together": False, "limit": 5})
    assert history["ok"] is True and history["filters"]["status"] == "history" and history["filters"]["limit"] == 5


@pytest.mark.parametrize("arguments, complaint", [
    ({"where": "?who works_at ?org"}, "where must be a list of patterns"),
    ({"where": ["?who works_at ?org"]}, "where must be a list of patterns"),
    ({"where": WHERE * 4}, "between 1 and 6 patterns"),
    ({"where": WHERE, "returns": ["?nobody"]}, "?nobody"),
    ({"where": WHERE, "status": "someday"}, "status must be one of"),
    ({"where": WHERE, "limit": 0}, "limit"),
    ({}, "where"),
])
async def test_a_malformed_structured_question_is_a_result_the_model_can_read(box, arguments, complaint):
    result = await box.run("graph_match", arguments)
    assert result["ok"] is False and complaint in result["error"]


async def test_graph_match_reads_its_own_space_only(box):
    result = await box.run("graph_match", {"where": [{"subject": "zed", "predicate": "works_at", "object": "?org"}]})
    assert result["ok"] is True and result["status"] == "not_found"
