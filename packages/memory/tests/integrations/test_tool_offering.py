"""The few tools a turn needs, chosen from the many: by the query, with
the host's always-on ones first, and never a gate on what may run."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.integrations.tool_offering import MAX_TOOLS, NAME_BONUS, ToolIndex
from scone_memory.integrations.tools import MEMORY_TOOLS, ToolBox, ToolSpec

pytestmark = pytest.mark.asyncio


async def index():
    return await ToolIndex.build(MEMORY_TOOLS, HashEmbedder())


async def test_a_graph_question_offers_the_graph_tools_and_a_memory_question_the_memory_ones():
    tools = await index()
    graph = await tools.select("what is connected to Acme in the entity graph, and the path between two entities", limit=4)
    assert {"graph_context", "connect_entities"} <= set(graph.names), graph.record()
    assert len(graph.names) == 4 and graph.left_out and "search_memory" in graph.left_out
    memory = await tools.select("search memory for what Ana said about the harbour", limit=3)
    assert memory.names[0] == "search_memory", memory.record()
    assert memory.offered[0].words and "search" in memory.offered[0].words


async def test_the_always_on_tools_come_first_whatever_the_query():
    tools = await index()
    chosen = await tools.select("what depends on this module", limit=3, always=["add_memory", "search_memory"])
    assert chosen.names[:2] == ("add_memory", "search_memory") and len(chosen.names) == 3
    assert chosen.always == ("add_memory", "search_memory") and chosen.record()["always"] == ["add_memory", "search_memory"]
    more = await tools.select("anything", limit=1, always=["add_memory", "search_memory", "trace_memory"])
    assert more.names == ("add_memory", "search_memory", "trace_memory"), "the always-on tools are offered past the limit"


async def test_a_selection_says_its_scores_and_basis_and_is_the_same_twice():
    tools = await index()
    first = await tools.select("which entities were added since Tuesday?", limit=5)
    second = await tools.select("which entities were added since Tuesday?", limit=5)
    assert first == second and first.embedder == HashEmbedder().id
    record = first.record()
    assert [item["name"] for item in record["offered"]] == list(first.names)
    assert all(-1.0 <= item["similarity"] <= 1.0 and item["score"] >= item["similarity"] for item in record["offered"])
    assert "cosine" in record["basis"] and set(record["left_out"]) | set(first.names) == {t.name for t in MEMORY_TOOLS}
    assert first.note is None
    blank = await tools.select("zzzz qqqq", limit=2, always=["search_memory"])
    assert blank.note and "not a ranking" in blank.note and blank.names[0] == "search_memory", \
        "a query nothing resembles is said to be one, not ranked by accident"
    assert set(record["left_out_scores"]) == set(record["left_out"]) and all(item["named"] is False for item in record["offered"])


class Orthogonal:
    """An embedder whose vectors are what a test says, so the similarity
    term can be told from the word term."""

    id, dim = "orthogonal", 3

    def __init__(self, vectors):
        self.vectors = vectors

    async def embed(self, texts):
        return [list(self.vectors.get(text, [0.0, 0.0, 0.0])) for text in texts]  # unknown text resembles nothing


async def test_the_vectors_count_apart_from_the_words_and_a_tool_named_whole_counts_most():
    specs = [ToolSpec("weather", "the forecast for a place", {"type": "object", "properties": {}}),
             ToolSpec("calendar", "what is on a day", {"type": "object", "properties": {}}),
             ToolSpec("add", "put a number to another", {"type": "object", "properties": {}})]
    vectors = {"weather: the forecast for a place": [1.0, 0.0, 0.0], "calendar: what is on a day": [0.0, 1.0, 0.0],
               "add: put a number to another": [0.0, 0.0, 1.0], "rain tomorrow?": [0.9, 0.1, 0.0]}
    tools = await ToolIndex.build(specs, Orthogonal(vectors))
    by_vector = await tools.select("rain tomorrow?", limit=1)
    assert by_vector.names == ("weather",) and by_vector.offered[0].similarity > 0.9 and by_vector.offered[0].words == ()
    assert by_vector.left_out_scores[0][0] == "calendar" or by_vector.left_out_scores[1][0] == "calendar"
    named = await tools.select("use calendar for the shipping address", limit=1)
    assert named.names == ("calendar",) and named.offered[0].named is True and named.offered[0].score >= NAME_BONUS
    assert "add" in named.left_out, "the `add` in `address` does not name the add tool"
    assert "0.5 when the query names the tool whole" in named.record()["basis"]


async def test_the_bounds_are_refused_plainly():
    tools = await index()
    with pytest.raises(InvalidInput, match="query"):
        await tools.select("   ", limit=3)
    with pytest.raises(InvalidInput, match="2000"):
        await tools.select("q" * 2001)
    with pytest.raises(InvalidInput, match="limit"):
        await tools.select("q", limit=0)
    with pytest.raises(InvalidInput, match="not among these tools"):
        await tools.select("q", always=["no_such_tool"])
    with pytest.raises(InvalidInput, match="distinct"):
        await ToolIndex.build([MEMORY_TOOLS[0], MEMORY_TOOLS[0]], HashEmbedder())
    with pytest.raises(InvalidInput, match="at least one"):
        await ToolIndex.build([], HashEmbedder())
    many = [ToolSpec(f"tool_{n}", f"tool number {n}", {"type": "object", "properties": {}}) for n in range(MAX_TOOLS + 1)]
    with pytest.raises(InvalidInput, match=str(MAX_TOOLS)):
        await ToolIndex.build(many, HashEmbedder())


async def test_the_toolbox_offers_a_subset_and_still_runs_every_tool_it_holds():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        box = ToolBox(engine, "default")
        chosen = await box.offer("search memory for the harbour", limit=3)
        assert chosen.names[0] == "search_memory" and len(chosen.names) == 3
        assert [tool["function"]["name"] for tool in box.openai(chosen.names)] == list(chosen.names)
        assert [tool["name"] for tool in box.anthropic(chosen.names)] == list(chosen.names)
        assert len(box.openai()) == len(box.tools), "without names, every tool as before"
        stored = await box.run("add_memory", {"content": "The harbour closes in November."})
        assert stored["ok"], "an offer is a suggestion to the host, not a gate on what runs"
        with pytest.raises(InvalidInput, match="not in this toolbox"):
            box.openai(["no_such_tool"])
    finally:
        await engine.close()
