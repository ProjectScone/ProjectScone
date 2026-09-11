"""One tool contract, rendered for whichever API is calling.

A model can search memory, add to it and read a profile. The definitions
and the executor live in one place, so every ToolBox schema rendering offers the same tools
with the same arguments, and a mistake comes back as a result the model
can read rather than an exception that ends the turn.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.tools import MEMORY_TOOLS, ToolBox, anthropic_schema, openai_schema


@pytest.fixture
async def box():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("alpha", "Juniper points at Polaris on clear nights", tags=["stars"])
    await engine.remember("alpha", "Juniper asked about the telescope yesterday")
    await engine.remember("beta", "the neighbour next door keeps a telescope")
    return ToolBox(engine, "alpha")


def test_the_same_tools_are_offered_in_every_rendering():
    assert [t.name for t in MEMORY_TOOLS] == ["search_memory", "add_memory", "read_profile", "trace_memory",
                                            "graph_context", "explain_entity", "connect_entities", "graph_schema"]
    openai = openai_schema()
    anthropic = anthropic_schema()
    assert [t["function"]["name"] for t in openai] == [t["name"] for t in anthropic]
    assert openai[0]["type"] == "function"
    assert openai[0]["function"]["parameters"] == anthropic[0]["input_schema"], "one set of arguments, two wrappers"
    assert anthropic[0]["input_schema"]["required"] == ["query"]
    assert set(anthropic[0]["input_schema"]["properties"]) == {"query", "limit", "tags"}
    assert all(t["input_schema"]["additionalProperties"] is False for t in anthropic), "an unknown argument is a mistake"


async def test_searching_answers_with_what_was_found_and_where_it_came_from(box):
    result = await box.run("search_memory", {"query": "where does juniper point"})
    assert result["ok"] is True
    item = result["items"][0]
    assert "Polaris" in item["text"], "the passage that answers the question ranks first"
    assert item["episode_id"] > 0 and 0 < item["score"] <= 1.0 and item["created_at"]
    assert result["space"] == "alpha"
    wide = await box.run("search_memory", {"query": "juniper"})
    narrow = await box.run("search_memory", {"query": "juniper", "tags": ["stars"]})
    assert len(narrow["items"]) < len(wide["items"]), "a tag narrows the search to what carries it"


async def test_adding_returns_the_record_it_made(box):
    result = await box.run("add_memory", {"content": "Juniper is a cat", "tags": ["pets"]})
    assert result["ok"] is True and result["episode_id"] > 0 and result["outcome"] == "accepted"
    again = await box.run("add_memory", {"content": "Juniper is a cat", "tags": ["pets"]})
    assert again["outcome"] == "duplicate", "the same thing twice is one memory, and the model is told so"
    found = await box.run("search_memory", {"query": "what is juniper"})
    assert any("cat" in i["text"] for i in found["items"])


async def test_the_profile_reads_back_identity_and_recent_activity(box):
    result = await box.run("read_profile", {})
    assert result["ok"] is True and result["recent"] and "episode_id" in result["recent"][0]
    assert result["facts"] == [], "nothing has been claimed yet, and it says so rather than inventing"


async def test_a_mistake_comes_back_as_a_result_the_model_can_read(box):
    unknown = await box.run("no_such_tool", {})
    assert unknown["ok"] is False and "no_such_tool" in unknown["error"] and "search_memory" in unknown["error"]

    missing = await box.run("search_memory", {})
    assert missing["ok"] is False and "query" in missing["error"]

    extra = await box.run("search_memory", {"query": "x", "colour": "red"})
    assert extra["ok"] is False and "colour" in extra["error"]

    empty = await box.run("add_memory", {"content": "  "})
    assert empty["ok"] is False and "content" in empty["error"]

    wrong_type = await box.run("search_memory", {"query": "x", "limit": "many"})
    assert wrong_type["ok"] is False and "limit" in wrong_type["error"]


async def test_a_box_reaches_its_own_space_only(box):
    result = await box.run("search_memory", {"query": "telescope"})
    assert result["items"], "our own telescope note is here"
    assert all("next door" not in i["text"] for i in result["items"]), "the neighbour's memory is not ours"


async def test_the_box_can_be_asked_for_a_subset_and_refuses_an_unknown_name(box):
    small = ToolBox(box.engine, "alpha", tools=["search_memory"])
    assert [t["name"] for t in small.anthropic()] == ["search_memory"]
    assert (await small.run("add_memory", {"content": "x"}))["ok"] is False, "a tool that was not offered cannot be called"
    with pytest.raises(ValueError, match="read_everything"):
        ToolBox(box.engine, "alpha", tools=["read_everything"])
