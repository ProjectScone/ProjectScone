"""A question restated by a model as the words a passage would use, and said so.

The lexical lane finds the words a passage has, and a question is not
written in them: "What happened with the bills?" misses a passage about
the billing run. A model can restate the question in a passage's words.
What must hold: the rewrite is taken only when it can be read, is not
empty, is within bounds and still shares a word with the question; in
every other case the question itself is searched and the record says
why. Nothing here is on by default, and a recall made this way always
carries the transform beside its results.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.validation import MAX_QUERY
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.retrieval.query_transforms import rewrite, rewritten_recall

pytestmark = pytest.mark.asyncio

QUESTION = "What happened with the bills?"


def reply(query: str) -> str:
    return json.dumps({"query": query})


class SlowChat:
    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(0.5)
        return reply("late")


async def test_a_readable_rewrite_that_keeps_a_word_of_the_question_is_applied():
    model = FakeChat([reply("bills  billing run invoices ")])
    asked = await rewrite(model, QUESTION)
    assert asked.applied is True and asked.query == "bills billing run invoices" and asked.reason is None
    assert asked.question == QUESTION and asked.model_calls == 1 and asked.kind == "rewrite"
    system, user = model.calls[0]
    assert QUESTION in user and "JSON" in system
    assert asked.record() == {"kind": "rewrite", "question": QUESTION, "query": "bills billing run invoices",
                              "applied": True, "reason": None, "model_calls": 1}


@pytest.mark.parametrize("answer, reason", [
    ("I would rather not.", "read"),
    (reply(""), "read"),
    (json.dumps({"query": 7}), "read"),
    (reply("invoices billing run"), "shares no word"),
    (reply("bills " + "x" * MAX_QUERY), "bound"),
])
async def test_a_rewrite_that_cannot_be_trusted_leaves_the_question_as_it_was(answer, reason):
    asked = await rewrite(FakeChat([answer]), QUESTION)
    assert asked.applied is False and asked.query == QUESTION and asked.reason is not None and reason in asked.reason
    assert asked.model_calls == 1


async def test_a_failing_or_slow_model_leaves_the_question_and_says_which():
    failed = await rewrite(FakeChat([ChatError("down")]), QUESTION)
    assert failed.applied is False and failed.reason is not None and failed.reason.startswith("model failed: ChatError")
    slow = await rewrite(SlowChat(), QUESTION, timeout_s=0.1)
    assert slow.applied is False and slow.reason is not None and "timeout" in slow.reason and slow.query == QUESTION


async def test_the_question_itself_is_checked():
    with pytest.raises(InvalidInput):
        await rewrite(FakeChat(), "   ")
    with pytest.raises(InvalidInput):
        await rewrite(FakeChat(), "x" * 9_000)


async def test_a_rewritten_recall_finds_what_the_question_could_not_and_carries_the_transform():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("s", "The billing run went out late and the invoices were wrong.")
    await engine.remember("s", "The garden shed needs a new roof before winter.")
    # The text lane matches a word's family by stem ("bills" finds
    # "billing"), so the question shares no family with the passage.
    question = "Any trouble with the accounts?"
    plain = await engine.recall("s", question, limit=2)
    assert all("text" not in item.lanes for item in plain.items), "the question's words are not the passage's"
    found, asked = await rewritten_recall(engine, FakeChat([reply("accounts billing run invoices")]), "s", question, limit=2)
    assert asked.applied and found.items[0].text.startswith("The billing run") and found.items[0].lanes.get("text") == 1
    kept, untouched = await rewritten_recall(engine, FakeChat(["nonsense"]), "s", question, limit=2)
    assert untouched.applied is False and [i.text for i in kept.items] == [i.text for i in plain.items]


async def test_the_bench_runner_can_transform_every_question_and_records_it():
    from scone_memory.bench.runner import BenchItem, run

    item = BenchItem(question_id="q1", question_type="single-session-user", question=QUESTION, question_date="",
                     sessions=[["The billing run went out late and the invoices were wrong."],
                               ["The bills for the shed roof are in the drawer."]],
                     session_ids=["a", "b"], session_dates=["", ""], answer_session_ids=["a"])

    def make_engine():
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    async def transform(question: str):
        return await rewrite(FakeChat([reply("bills billing run invoices")]), question)

    report = await run(make_engine, [item], ks=(1,), transform=transform)
    [result] = report.results
    assert result.query == "bills billing run invoices" and result.transformed is True and result.any_at(1)
    untouched = await run(make_engine, [item], ks=(1,))
    assert untouched.results[0].query is None and untouched.results[0].transformed is None
