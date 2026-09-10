"""Memory in front of any chat model, without its SDK.

Chat APIs agree on the shape of a message list, so the binding works on
that shape rather than on a vendor's client: recall what the last user
turn is about, put it in front of the model as its own message, and say
exactly what was supplied. What the model then does with it is the
model's business, and the receipt never claims otherwise.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.chat import ContextReceipt, recall_context, remember_exchange


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Juniper points at Polaris on clear nights", tags=["stars"])
    await memory.remember("alpha", "Mark takes his coffee black")
    return memory


async def test_what_the_last_question_is_about_goes_in_front_of_the_model(engine):
    messages = [{"role": "system", "content": "You are terse."},
                {"role": "user", "content": "where does juniper point?"}]
    prepared, receipt = await recall_context(engine, "alpha", messages)

    assert isinstance(receipt, ContextReceipt) and receipt.injected is True
    assert receipt.query == "where does juniper point?"
    assert receipt.episode_ids, "the receipt names what was supplied"
    assert [m["role"] for m in prepared] == ["system", "system", "user"], "instructions still lead"
    assert prepared[0] == messages[0], "the caller's own system message is untouched"
    assert "Polaris" in prepared[1]["content"]
    assert messages == [{"role": "system", "content": "You are terse."},
                        {"role": "user", "content": "where does juniper point?"}], "the caller's list is not edited"


async def test_nothing_found_means_nothing_added(engine):
    prepared, receipt = await recall_context(engine, "alpha", [{"role": "user", "content": "quantum chromodynamics"}], floor=0.9)
    assert receipt.injected is False and receipt.episode_ids == ()
    assert [m["role"] for m in prepared] == ["user"], "no empty here-is-your-memory block"


async def test_with_no_question_there_is_nothing_to_look_up(engine):
    prepared, receipt = await recall_context(engine, "alpha", [{"role": "system", "content": "You are terse."}])
    assert receipt.injected is False and receipt.query == ""
    assert prepared == [{"role": "system", "content": "You are terse."}]


async def test_the_block_is_capped_and_the_receipt_counts_what_fitted(engine):
    for i in range(12):
        await engine.remember("alpha", f"Note number {i} about juniper and the sky, with plenty of words in it")
    _, small = await recall_context(engine, "alpha", [{"role": "user", "content": "juniper"}], limit=12, budget=400)
    assert small.characters <= 400 and 0 < len(small.episode_ids) < 12, "what did not fit was not claimed"
    _, large = await recall_context(engine, "alpha", [{"role": "user", "content": "juniper"}], limit=12, budget=4000)
    assert len(large.episode_ids) > len(small.episode_ids)


async def test_the_exchange_is_remembered_as_turns_that_can_be_found_again(engine):
    written = await remember_exchange(
        engine, "alpha",
        [{"role": "system", "content": "ignored"},
         {"role": "user", "content": "what did I say about coffee?"},
         {"role": "assistant", "content": "you take it black"}],
        session="sess-9", metadata={"user_id": "mark"},
    )
    assert [w.outcome for w in written] == ["accepted", "accepted"], "the question and the answer, not the instructions"
    found = await engine.recall("alpha", "coffee", where={"user_id": "mark"})
    assert any("black" in i.text for i in found.items)
    episode = await engine.episode("alpha", written[0].episode_id)
    assert episode.source == "sess-9" and episode.metadata["role"] == "user"
    assert episode.kind == "conversation"


async def test_an_empty_or_tool_turn_is_not_remembered(engine):
    written = await remember_exchange(engine, "alpha", [
        {"role": "user", "content": "   "},
        {"role": "tool", "content": "{\"ok\": true}"},
        {"role": "assistant", "content": ""},
    ])
    assert written == [], "nothing worth keeping, so nothing kept"
