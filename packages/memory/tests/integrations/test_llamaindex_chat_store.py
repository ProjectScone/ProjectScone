"""A LlamaIndex chat store whose conversations are Scone turns.

LangChain's chat history and the OpenAI Agents session already keep a
conversation as Scone turns: one episode per message, in order, and the
whole transcript recallable like any other memory. LlamaIndex keeps chat
in a ``BaseChatStore`` that its memory buffers read and write by key, and
had no Scone store. ``SconeChatStore`` is one: a key is a session, a plain
text message is stored as its text so recall over the conversation reads
well, and any other message (tool calls, several blocks, extra fields)
round-trips exactly through its JSON.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock  # noqa: E402

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, SyncMemoryEngine  # noqa: E402
from scone_memory.integrations.llamaindex import SconeChatStore  # noqa: E402


def sync_memory() -> SyncMemoryEngine:
    return SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()))


QUESTION = ChatMessage(role=MessageRole.USER, content="When was the crane survey booked?")
ANSWER = ChatMessage(role=MessageRole.ASSISTANT, content="The third of May.")
TOOL = ChatMessage(role=MessageRole.ASSISTANT, content="",
                   additional_kwargs={"tool_calls": [{"id": "call-1", "name": "recall", "arguments": "{\"q\": \"crane\"}"}]})


def test_messages_are_kept_per_key_in_order_and_plain_text_stays_text():
    with sync_memory() as memory:
        store = SconeChatStore(memory, "default", extra={"user_id": "mark"})
        store.set_messages("chat-1", [QUESTION, ANSWER])
        store.add_message("chat-1", TOOL)
        store.add_message("chat-2", QUESTION)
        assert store.get_messages("chat-1") == [QUESTION, ANSWER, TOOL]
        assert store.get_messages("chat-2") == [QUESTION]
        assert store.get_keys() == ["chat-1", "chat-2"]
        episodes = memory.episodes("default", {"session_id": "chat-1"})
        assert episodes[0].content == "When was the crane survey booked?" and episodes[0].metadata["user_id"] == "mark"
        assert episodes[2].metadata.get("encoding") == "json", "a tool call is kept whole"


def test_the_conversation_is_recallable_like_any_memory():
    with sync_memory() as memory:
        store = SconeChatStore(memory, "default")
        store.set_messages("chat-1", [QUESTION, ANSWER])
        found = memory.recall("default", "crane survey booked", where={"session_id": "chat-1"})
        assert found.items and found.items[0].text == "When was the crane survey booked?"


def test_a_message_with_several_blocks_round_trips_exactly():
    image = ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="What is in this?"),
                                                       ImageBlock(url="https://example.com/crane.png")])
    with sync_memory() as memory:
        store = SconeChatStore(memory, "default")
        store.add_message("chat-1", image)
        assert store.get_messages("chat-1") == [image]


def test_deleting_one_the_last_or_all_messages_returns_what_was_deleted():
    with sync_memory() as memory:
        store = SconeChatStore(memory, "default")
        store.set_messages("chat-1", [QUESTION, ANSWER, TOOL])
        assert store.delete_message("chat-1", 1) == ANSWER
        assert store.get_messages("chat-1") == [QUESTION, TOOL]
        assert store.delete_last_message("chat-1") == TOOL
        assert store.delete_message("chat-1", 5) is None and store.delete_last_message("chat-9") is None
        store.add_message("chat-1", ANSWER)
        assert store.get_messages("chat-1") == [QUESTION, ANSWER], "a message added after a deletion goes last"
        assert store.delete_messages("chat-1") == [QUESTION, ANSWER]
        assert store.get_messages("chat-1") == [] and store.delete_messages("chat-1") is None
        assert store.get_keys() == []


def test_setting_messages_replaces_the_conversation():
    with sync_memory() as memory:
        store = SconeChatStore(memory, "default")
        store.set_messages("chat-1", [QUESTION, ANSWER])
        store.set_messages("chat-1", [ANSWER])
        assert store.get_messages("chat-1") == [ANSWER]


async def test_the_async_methods_work_on_an_async_engine():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        store = SconeChatStore(engine, "default")
        await store.aset_messages("chat-1", [QUESTION])
        await store.async_add_message("chat-1", ANSWER)
        assert await store.aget_messages("chat-1") == [QUESTION, ANSWER]
        assert await store.aget_keys() == ["chat-1"]
        assert await store.adelete_last_message("chat-1") == ANSWER
        assert await store.adelete_message("chat-1", 0) == QUESTION
        assert await store.adelete_messages("chat-1") is None
        with pytest.raises(TypeError, match="SyncMemoryEngine"):
            store.get_messages("chat-1")
    finally:
        await engine.close()


def test_a_llamaindex_memory_buffer_reads_and_writes_through_it():
    from llama_index.core.memory import ChatMemoryBuffer

    with sync_memory() as memory:
        store = SconeChatStore(memory, "default")
        buffer = ChatMemoryBuffer.from_defaults(chat_store=store, chat_store_key="chat-1", token_limit=10_000)
        buffer.put(QUESTION)
        buffer.put(ANSWER)
        assert buffer.get() == [QUESTION, ANSWER]
        assert SconeChatStore(memory, "default").get_messages("chat-1") == [QUESTION, ANSWER], "kept in the memory, not the buffer"


def test_messages_read_back_in_the_order_they_were_added_whatever_the_clock_says():
    """Two writers with skewed clocks: the later message can carry the earlier time."""
    from scone_memory.testing import Clock

    clock = Clock("2025-06-01T00:00:00.000Z")
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock)) as memory:
        store = SconeChatStore(memory, "default")
        store.add_message("chat-1", QUESTION)
        clock.now = "2025-01-01T00:00:00.000Z"
        store.add_message("chat-1", ANSWER)
        assert store.get_messages("chat-1") == [QUESTION, ANSWER]


def test_replacing_a_conversation_on_sqlite_keeps_every_new_message(tmp_path):
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex

    path = tmp_path / "chat.db"

    async def build() -> MemoryEngine:  # built on the wrapper's own thread, which SQLite requires
        return MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder())

    with SyncMemoryEngine(build) as memory:
        store = SconeChatStore(memory, "default")
        store.set_messages("chat-1", [QUESTION, ANSWER])
        store.set_messages("chat-1", [ANSWER, QUESTION])
        assert store.get_messages("chat-1") == [ANSWER, QUESTION]
