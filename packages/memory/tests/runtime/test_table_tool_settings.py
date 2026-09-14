"""The table tools are a setting like the compute tool: a boolean, and only in tool mode."""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings
from scone_memory.runtime.conversation_tools import ConversationTools, validate_tool_settings


def test_the_flag_is_parsed_and_needs_tool_mode():
    assert Settings.from_env({}).conversations_tool_tables is False
    with pytest.raises(InvalidInput, match="SCONE_CONVERSATIONS_TOOL_TABLES"):
        Settings.from_env({"SCONE_CONVERSATIONS_TOOL_TABLES": "maybe"})
    with pytest.raises(InvalidInput, match="TOOL_TABLES requires"):
        validate_tool_settings(Settings.from_env({"SCONE_CONVERSATIONS_TOOL_TABLES": "1", "SCONE_CONVERSATIONS_TOOL_MODE": "off"}))


def test_the_tool_configuration_carries_it():
    from scone_memory.agents.evidence_loop import ToolLoopLimits

    tools = ConversationTools("native", ToolLoopLimits(), True, False, True)
    assert tools.tables is True
    with pytest.raises(ValueError):
        ConversationTools("native", ToolLoopLimits(), True, False, "yes")  # type: ignore[arg-type]


async def test_a_text_conversation_needs_a_tool_model_for_the_tables():
    from scone_memory.realtime.text import TextConversation

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with pytest.raises(ValueError, match="tool_tables"):
        TextConversation(memory, "s", "session-1", lambda: object(), tool_tables=True)
