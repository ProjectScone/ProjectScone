"""Framework adapters. The turn mapping and the engine changes behind them
(keyed deduplication, identity carried through export, the episodes walk)
are tested without any framework; each adapter's own test runs when its
framework is installed and is skipped, visibly, when it is not."""

from __future__ import annotations

import importlib.util
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, InvalidInput, MemoryEngine, SyncMemoryEngine
from scone_memory.engine import Record
from scone_memory.integrations.turns import Turn, item_metadata, next_seq, read_turn, turn_records
from scone_memory.models import RecallItem


async def fresh():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()


# -- engine behaviour the adapters rely on -----------------------------------


async def test_a_keyed_record_dedups_by_key_not_by_text():
    engine = await fresh()
    first, second = await engine.remember_many("default", [Record("ok", dedup_key="s1#0"), Record("ok", dedup_key="s1#1")])
    assert not first.deduplicated and not second.deduplicated and first.episode_id != second.episode_id, "the second ok is a second turn"
    [again] = await engine.remember_many("default", [Record("ok", dedup_key="s1#1")])
    assert again.deduplicated and again.episode_id == second.episode_id, "a retried write of the same key is not a third turn"
    [plain] = await engine.remember_many("default", [Record("ok")])
    assert not plain.deduplicated, "an unkeyed record is identified by its text, which no keyed record shares"
    [plain_again] = await engine.remember_many("default", [Record("ok")])
    assert plain_again.deduplicated and plain_again.episode_id == plain.episode_id
    with pytest.raises(InvalidInput, match="dedup_key"):
        await engine.remember_many("default", [Record("x", dedup_key="")])


async def test_a_dump_carries_identity_so_repeated_turns_survive_import():
    source = await fresh()
    await source.remember_many("default", [Record("ok", dedup_key="s1#0"), Record("ok", dedup_key="s1#1")])
    dump = [r async for r in source.export("default")]
    assert len({r["content_hash"] for r in dump}) == 2 and all(r["content"] == "ok" for r in dump)
    target = await fresh()
    summary = await target.import_records("default", dump)
    assert (summary.episodes, summary.deduplicated) == (2, 0), "without the identity the two turns would collapse into one"
    assert (await target.status("default")).episodes == 2


async def test_episodes_walk_filters_on_every_pair_and_orders_by_time_then_id():
    engine = await fresh()
    await engine.remember("default", "later", created_at="2025-01-02", metadata={"session_id": "a", "seq": "1"})
    await engine.remember("default", "earlier", created_at="2025-01-01", metadata={"session_id": "a", "seq": "0"})
    await engine.remember("default", "elsewhere", created_at="2025-01-01", metadata={"session_id": "b"})
    await engine.remember("default", "same instant, higher id", created_at="2025-01-02", metadata={"session_id": "a", "seq": "2"})
    turns = await engine.episodes("default", {"session_id": "a"})
    assert [e.content for e in turns] == ["earlier", "later", "same instant, higher id"]
    assert [e.content for e in await engine.episodes("default", {"session_id": "a"}, limit=2)] == ["later", "same instant, higher id"], "the newest N, still oldest first"
    assert await engine.episodes("default", {"session_id": "a", "seq": "1"}) == turns[1:2]
    assert await engine.episodes("default", {"session_id": "nobody"}) == []
    with pytest.raises(InvalidInput):
        await engine.episodes("default", {})


# -- the turn mapping --------------------------------------------------------


def test_turn_records_key_and_stamp_every_turn():
    records = turn_records("s1", [Turn("user", "hi"), Turn("tool", None, {"type": "function_call", "name": "f"})], start_seq=3, extra={"user_id": "mark"})
    assert [r.dedup_key for r in records] == ["s1#3", "s1#4"]
    assert records[0].metadata == {"user_id": "mark", "session_id": "s1", "role": "user", "seq": "3"} and records[0].content == "hi"
    assert records[1].metadata["encoding"] == "json" and json.loads(records[1].content) == {"type": "function_call", "name": "f"}
    assert all(r.kind == "conversation" and r.source == "s1" for r in records)


async def test_a_stored_turn_reads_back_as_it_went_in():
    engine = await fresh()
    payload = {"role": "assistant", "content": [{"type": "output_text", "text": "hi"}], "id": "msg_1"}
    await engine.remember_many("default", turn_records("s1", [Turn("user", "plain"), Turn("assistant", None, payload)], 0))
    [plain, structured] = await engine.episodes("default", {"session_id": "s1"})
    assert read_turn(plain) == Turn("user", "plain")
    assert read_turn(structured) == Turn("assistant", None, payload)
    assert next_seq([plain, structured]) == 2 and next_seq([]) == 0
    await engine.remember_many("default", turn_records("s1", [Turn("user", "after a gap")], 5))
    assert next_seq(await engine.episodes("default", {"session_id": "s1"})) == 6, "one past the highest position, not the count"


def test_item_metadata_keeps_provenance_and_scopes_cannot_shadow_it():
    item = RecallItem(chunk_id=7, episode_id=3, text="t", score=1.0, similarity=0.5, lanes={"vector": 1}, created_at="2025-01-01T00:00:00.000Z",
                      source="s1", tags=("a",), metadata={"user_id": "mark", "episode_id": "spoof"})
    meta = item_metadata(item)
    assert meta["episode_id"] == 3 and meta["chunk_id"] == 7 and meta["user_id"] == "mark" and meta["similarity"] == 0.5
    assert meta["lanes"] == {"vector": 1} and meta["tags"] == ["a"] and meta["source"] == "s1"


# -- LangChain ---------------------------------------------------------------

langchain = pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None, reason="langchain-core not installed")


@langchain
def test_langchain_retriever_returns_documents_with_provenance():
    from langchain_core.documents import Document

    from scone_memory.integrations.langchain import SconeRetriever

    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())) as memory:
        memory.remember("default", "the deploy runbook is in the ops wiki", metadata={"user_id": "mark"})
        memory.remember("default", "grocery list: eggs", metadata={"user_id": "ana"})
        memory.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
        retriever = SconeRetriever(memory=memory, space="default", limit=3, where={"user_id": "mark"})
        docs = retriever.invoke("where is the deploy runbook")
        assert len(docs) == 1 and isinstance(docs[0], Document) and "runbook" in docs[0].page_content
        assert docs[0].metadata["user_id"] == "mark" and docs[0].metadata["episode_id"] == 1 and docs[0].metadata["score"] == 1.0
        with_facts = SconeRetriever(memory=memory, space="default", include_facts=True).invoke("where does mark live")
        assert with_facts[0].metadata["kind"] == "facts" and "mark lives_in Lisbon" in with_facts[0].page_content


@langchain
async def test_langchain_retriever_async_path_takes_the_bare_engine():
    from scone_memory.integrations.langchain import SconeRetriever

    engine = await fresh()
    await engine.remember("default", "kubernetes upgrade failed on node 3")
    docs = await SconeRetriever(memory=engine, space="default").ainvoke("kubernetes upgrade")
    assert len(docs) == 1 and docs[0].metadata["chunk_id"] == 1
    with pytest.raises(TypeError, match="SyncMemoryEngine"):
        SconeRetriever(memory=engine, space="default").invoke("kubernetes")


@langchain
def test_langchain_history_round_trips_every_message_in_order():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from scone_memory.integrations.langchain import SconeChatMessageHistory

    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())) as memory:
        history = SconeChatMessageHistory(memory, "default", "chat-1", extra={"user_id": "mark"})
        assert history.messages == []
        sent = [
            HumanMessage(content="ok"),
            AIMessage(content="ok"),
            HumanMessage(content="ok"),  # the same text three times: three turns
            AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}]),
            ToolMessage(content="result", tool_call_id="call_1"),
            HumanMessage(content="named", name="mark", id="m-1"),  # text plus fields: must not be flattened to its text
        ]
        history.add_messages(sent[:3])
        history.add_messages(sent[3:])
        back = history.messages
        assert [m.type for m in back] == ["human", "ai", "human", "ai", "tool", "human"]
        assert [m.content for m in back] == [m.content for m in sent]
        assert back[3].tool_calls == sent[3].tool_calls and back[4].tool_call_id == "call_1"
        assert (back[5].name, back[5].id) == ("mark", "m-1")
        other = SconeChatMessageHistory(memory, "default", "chat-2")
        assert other.messages == [], "another session sees nothing"
        items = memory.recall("default", "ok", where={"user_id": "mark"}).items
        assert items and all(i.metadata["user_id"] == "mark" for i in items), "every turn carries the extra scope"
        assert sum(i.text == "ok" for i in items) == 3, "all three ok turns are recallable, not one"
        history.clear()
        assert history.messages == [] and memory.status("default").episodes == 0


@langchain
async def test_langchain_history_async_variants():
    from langchain_core.messages import HumanMessage

    from scone_memory.integrations.langchain import SconeChatMessageHistory

    engine = await fresh()
    history = SconeChatMessageHistory(engine, "default", "chat-1")
    await history.aadd_messages([HumanMessage(content="first")])
    await history.aadd_messages([HumanMessage(content="second")])
    assert [m.content for m in await history.aget_messages()] == ["first", "second"]
    await history.aclear()
    assert await history.aget_messages() == []


# -- LlamaIndex --------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("llama_index") is None, reason="llama-index-core not installed")
async def test_llamaindex_retriever_returns_scored_nodes():
    from scone_memory.integrations.llamaindex import SconeRetriever

    engine = await fresh()
    await engine.remember("default", "the deploy runbook is in the ops wiki", metadata={"user_id": "mark"})
    await engine.remember("default", "grocery list: eggs")
    nodes = await SconeRetriever(engine, "default", limit=2).aretrieve("deploy runbook")
    assert nodes and nodes[0].node.text.startswith("the deploy runbook") and nodes[0].score == 1.0
    assert nodes[0].node.metadata["user_id"] == "mark" and nodes[0].node.id_ == "scone-chunk-1"
    with SyncMemoryEngine(engine) as memory:
        again = SconeRetriever(memory, "default", limit=2).retrieve("deploy runbook")
        assert [n.node.id_ for n in again] == [n.node.id_ for n in nodes]


# -- OpenAI Agents SDK ---------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("agents") is None, reason="openai-agents not installed")
async def test_openai_agents_session_keeps_the_runner_contract():
    from agents.memory.session import Session

    from scone_memory.integrations.openai_agents import SconeSession

    engine = await fresh()
    session = SconeSession(engine, "default", "run-1", extra={"user_id": "mark"})
    assert isinstance(session, Session) and session.session_id == "run-1"
    assert await session.get_items() == [] and await session.pop_item() is None
    structured = {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{\"q\": \"x\"}"}
    await session.add_items([{"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}])
    await session.add_items([structured, {"role": "user", "content": "ok"}])
    assert await session.get_items() == [
        {"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}, structured, {"role": "user", "content": "ok"}
    ]
    assert await session.get_items(limit=2) == [structured, {"role": "user", "content": "ok"}], "the latest N, chronological"
    assert await session.pop_item() == {"role": "user", "content": "ok"}
    assert await session.pop_item() == structured
    assert len(await session.get_items()) == 2
    items = (await engine.recall("default", "ok", where={"session_id": "run-1"})).items
    assert items and all(i.metadata["user_id"] == "mark" for i in items) and sum(i.text == "ok" for i in items) == 2
    await session.clear_session()
    assert await session.get_items() == [] and (await engine.status("default")).episodes == 0
