"""The tree as tools an agent can call, and what it may not do with them.

A model that is given list, read and search will use them; giving it
write as well is a decision the owner makes, not one the toolbox makes
for them. Unless a filesystem policy says otherwise, the writing tool is
not offered at all — an absent tool is a clearer refusal than an error
message a model may argue with.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.filesystem import FilesystemPolicy
from scone_memory.integrations.tools import ToolBox
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"


async def box(**options) -> ToolBox:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"),
                                events=InMemoryEventLog()).open()
    note = await engine.remember(SPACE, "The Lisbon office opened in March and holds forty desks.")
    await engine.assert_fact(SPACE, "lisbon office", "opened_on", "March 2024",
                             valid_from="2024-01-01T00:00:00Z", source_episode_id=note.episode_id,
                             quote="The Lisbon office opened in March")
    return ToolBox(engine, SPACE, **options)


async def test_the_tree_is_read_only_unless_the_owner_says_otherwise():
    names = {tool.name for tool in (await box()).tools}
    assert {"list_path", "read_path", "search_paths"} <= names
    assert "write_note" not in names, "an absent tool is a clearer refusal than an error"


async def test_the_writing_tool_appears_when_the_policy_allows_it():
    tools = (await box(filesystem=FilesystemPolicy(writable=True))).tools
    assert "write_note" in {tool.name for tool in tools}


async def test_an_agent_can_walk_and_read_the_tree():
    tools = await box()
    root = await tools.run("list_path", {"path": "/"})
    assert root["ok"] and [entry["path"] for entry in root["entries"]] == [
        "/entities", "/episodes", "/facts", "/notes"]
    page = await tools.run("read_path", {"path": "/episodes/1.md"})
    assert page["ok"] and "forty desks" in page["text"] and page["of"] == "episode"


async def test_an_agent_searching_is_answered_in_paths():
    tools = await box()
    found = await tools.run("search_paths", {"query": "desks"})
    assert found["ok"] and found["hits"] and found["hits"][0]["path"].endswith(".md")


async def test_an_agent_writing_without_the_policy_is_refused_in_words_it_can_read():
    tools = await box()
    said = await tools.run("write_note", {"path": "/notes/x.md", "text": "hello"})
    assert said["ok"] is False and "no tool" in said["error"]


async def test_an_agent_can_write_a_note_when_it_is_allowed_to():
    tools = await box(filesystem=FilesystemPolicy(writable=True))
    written = await tools.run("write_note", {"path": "/notes/plan.md", "text": "Ship on Friday."})
    assert written["ok"] and written["episode_id"] and written["version"]
    back = await tools.run("read_path", {"path": "/notes/plan.md"})
    assert back["text"] == "Ship on Friday."


async def test_an_agent_told_to_write_somewhere_it_may_not_is_refused_not_obeyed():
    tools = await box(filesystem=FilesystemPolicy(writable=True))
    said = await tools.run("write_note", {"path": "/episodes/1.md", "text": "rewritten"})
    assert said["ok"] is False and "notes" in said["error"]
    page = await tools.run("read_path", {"path": "/episodes/1.md"})
    assert "forty desks" in page["text"], "nothing stored was touched"


async def test_a_path_that_tries_to_leave_is_refused_as_an_answer_not_an_exception():
    tools = await box()
    said = await tools.run("read_path", {"path": "/episodes/../../etc/passwd"})
    assert said["ok"] is False and "leave" in said["error"]
