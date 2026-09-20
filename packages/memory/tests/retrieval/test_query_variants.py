"""One question asked several ways, and the answers fused.

A question is one wording of a need, and the passage that answers it was
written in another. The leading framework's query fusion writes several
restatements with a model, searches each, and fuses the rankings; ours
does the same with two rules of its own: the question the person asked
is always searched and always counts, so a model that writes nonsense
cannot lose it, and every variant the rule refuses is recorded with the
reason it was refused.

Nothing here runs by default, and nothing here has moved a number yet:
until it does, it is an explicit call and no more.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.validation import MAX_QUERY
from scone_memory.retrieval.query_variants import MAX_VARIANTS, variants, variant_recall

pytestmark = pytest.mark.asyncio

QUESTION = "What happened with the billing run?"


def reply(*queries: str) -> str:
    return json.dumps({"queries": list(queries)})


class Chat:
    """A model that answers with what it was given, and counts its calls."""

    def __init__(self, answer: str) -> None:
        self.answer, self.calls, self.asked = answer, 0, []

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        self.asked.append(user)
        return self.answer


class Slow:
    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(0.5)
        return reply("late")


class Broken:
    async def complete(self, system: str, user: str) -> str:
        raise RuntimeError("no model here")


async def test_the_variants_a_model_writes_are_searched_beside_the_question():
    model = Chat(reply("billing run invoices", "billing errors last month"))
    written = await variants(model, QUESTION, count=2)
    assert [v.query for v in written.kept] == ["billing run invoices", "billing errors last month"]
    assert written.question == QUESTION and written.model_calls == 1
    assert written.searched == [QUESTION, "billing run invoices", "billing errors last month"], \
        "the question asked is searched too, and first"
    assert written.refused == []
    assert "Write 2 searches." in model.asked[0] and QUESTION in model.asked[0], \
        "the model is told the question and how many searches to write"


@pytest.mark.parametrize("written, reason", [
    ("", "empty"),
    ("   ", "empty"),
    ("x" * (MAX_QUERY + 1), "over the bound"),
    ("entirely unrelated wording", "shares no word"),
], ids=["empty", "blank", "too-long", "unrelated"])
async def test_a_variant_that_breaks_a_rule_is_refused_with_its_reason(written, reason):
    model = Chat(reply("billing run invoices", written))
    result = await variants(model, QUESTION, count=2)
    assert [v.query for v in result.kept] == ["billing run invoices"]
    assert len(result.refused) == 1 and reason in result.refused[0].reason
    assert result.refused[0].query == written


async def test_a_variant_that_repeats_another_is_kept_once():
    model = Chat(reply("billing run", "Billing   Run", "billing again"))
    result = await variants(model, QUESTION, count=3)
    assert [v.query for v in result.kept] == ["billing run", "billing again"]
    assert len(result.refused) == 1 and "repeats" in result.refused[0].reason


async def test_a_variant_that_repeats_the_question_is_not_searched_twice():
    model = Chat(reply("What happened with the billing run?", "billing invoices"))
    result = await variants(model, QUESTION, count=2)
    assert [v.query for v in result.kept] == ["billing invoices"]
    assert result.searched == [QUESTION, "billing invoices"]


@pytest.mark.parametrize("model, reason", [
    (Broken(), "model failed"),
    (Chat("not json at all"), "could not be read"),
    (Chat(json.dumps({"queries": "billing"})), "could not be read"),
    (Chat(json.dumps({"queries": []})), "no variants"),
], ids=["failed", "prose", "not-a-list", "empty-list"])
async def test_a_model_that_says_nothing_usable_leaves_the_question_alone(model, reason):
    result = await variants(model, QUESTION, count=3)
    assert result.kept == [] and result.searched == [QUESTION]
    assert reason in result.reason


async def test_a_model_that_takes_too_long_leaves_the_question_alone():
    result = await variants(Slow(), QUESTION, count=2, timeout_s=0.05)
    assert result.kept == [] and result.searched == [QUESTION]
    assert "timeout" in result.reason


async def test_more_variants_than_asked_for_are_cut_and_said_so():
    model = Chat(reply("billing one", "billing two", "billing three", "billing four"))
    result = await variants(model, QUESTION, count=2)
    assert [v.query for v in result.kept] == ["billing one", "billing two"]
    assert result.cut == 2


@pytest.mark.parametrize("count", [0, -1, MAX_VARIANTS + 1, True, 2.5, "2"])
async def test_a_count_outside_its_bounds_is_refused(count):
    with pytest.raises(InvalidInput, match="count"):
        await variants(Chat(reply("a")), QUESTION, count=count)


@pytest.mark.parametrize("question", ["", "   ", None, 5])
async def test_a_question_that_is_not_one_is_refused(question):
    with pytest.raises(InvalidInput):
        await variants(Chat(reply("a")), question, count=2)


# -- through the engine --------------------------------------------------------------


async def engine_with(*texts: str):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                vector_weight=0.01).open()
    for text in texts:
        await engine.remember("s", text)
    return engine


async def test_a_passage_only_a_variant_finds_is_brought_back_and_the_record_says_which():
    """The question's own words miss the passage; a variant's words find
    it. That is the whole point of asking twice."""
    engine = await engine_with("The billing run went out late and the invoices were wrong.",
                               "A shed roof before winter, and the gutters.",
                               "Notes from the Tuesday meeting about the office move.")
    try:
        model = Chat(reply("billing run invoices"))
        found, record = await variant_recall(engine, model, "s", QUESTION, count=1, limit=3)
        assert found.items, "something was found"
        assert "billing" in found.items[0].text.lower()
        assert record.searched == [QUESTION, "billing run invoices"]
        assert record.record()["searches"] == 2
    finally:
        await engine.close()


async def test_the_question_outvotes_a_single_variant_that_disagrees():
    """A variant is a guess at the question, so it argues more quietly
    than the question itself: where the two rank different passages
    first, the question's choice leads."""
    engine = await engine_with("The billing run went out late and the invoices were wrong.",
                               "Gutters: winter roof gutters, about nothing at all.")
    try:
        model = Chat(reply("billing gutters winter roof"))
        found, record = await variant_recall(engine, model, "s", "billing", count=1, limit=2)
        assert record.kept and record.kept[0].query == "billing gutters winter roof"
        alone = await engine.recall("s", "billing", limit=2)
        assert found.items[0].chunk_id == alone.items[0].chunk_id, \
            "the question's own first choice still leads after one variant argued against it"
    finally:
        await engine.close()


async def test_a_model_that_fails_still_answers_the_question_as_asked():
    engine = await engine_with("The billing run went out late and the invoices were wrong.")
    try:
        found, record = await variant_recall(engine, Broken(), "s", QUESTION, count=2, limit=2)
        assert found.items and record.kept == [] and "model failed" in record.reason
        assert record.record()["searches"] == 1, "one search, for the question itself"
    finally:
        await engine.close()
