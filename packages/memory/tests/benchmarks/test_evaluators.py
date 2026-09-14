"""Judged answer quality: what a local model says about an answer, and what it cannot.

Retrieval metrics say whether the right passages came back. They say nothing
about the answer that was written from them. These evaluators ask a local
judge four questions the reference framework also asks -- is every claim in
the answer supported by the contexts, does the answer address the question,
are the contexts relevant, is the answer correct against a reference -- and
return a judgment that says so, or says it could not judge. An unparseable
or failed judgment is never a pass, and an input over the bound is refused
rather than silently cut.
"""

from __future__ import annotations

import json

import pytest

from scone_memory.bench.evaluators import (MAX_CONTEXTS, answer_relevancy, context_relevancy, correctness,
                                           faithfulness)
from scone_memory.bench.metrics import recall_at
from scone_memory.providers.llm import ChatError, FakeChat

pytestmark = pytest.mark.asyncio

QUESTION = "Where does person-1 work?"
ANSWER = "Person-1 works at Brightlake. They started in June."
CONTEXTS = ["Person-1 works at Brightlake as a principal engineer.", "Person-1 joined Brightlake in June 2024."]


async def test_faithfulness_scores_the_share_of_claims_the_contexts_support():
    judge = FakeChat([json.dumps({"claims": [
        {"claim": "Person-1 works at Brightlake.", "supported": True, "context_index": 0},
        {"claim": "They started in June.", "supported": True, "context_index": 1},
    ]})])
    judged = await faithfulness(judge, answer=ANSWER, contexts=CONTEXTS)
    assert (judged.kind, judged.score, judged.passing, judged.verified) == ("faithfulness", 1.0, True, True)
    system, user = judge.calls[0]
    assert ANSWER in user and all(context in user for context in CONTEXTS), "the judge reads the exact texts"


async def test_one_unsupported_claim_fails_faithfulness_and_is_named():
    judge = FakeChat([json.dumps({"claims": [
        {"claim": "Person-1 works at Brightlake.", "supported": True, "context_index": 0},
        {"claim": "They started in June.", "supported": False, "context_index": None},
    ]})])
    judged = await faithfulness(judge, answer=ANSWER, contexts=CONTEXTS)
    assert judged.score == 0.5 and judged.passing is False and judged.verified
    assert any("started in June" in reason for reason in judged.reasons)


@pytest.mark.parametrize("reply", ["not json", json.dumps({"claims": "none"}), json.dumps({"claims": [{"claim": "x"}]}),
                                   json.dumps({"claims": [{"claim": "x", "supported": "yes"}]}), ""])
async def test_a_judgment_that_cannot_be_read_is_unverified_and_never_a_pass(reply):
    judged = await faithfulness(FakeChat([reply]), answer=ANSWER, contexts=CONTEXTS)
    assert judged.verified is False and judged.passing is None and judged.score is None
    assert judged.reasons and "judge" in judged.reasons[0]


async def test_a_judge_that_fails_is_unverified_not_a_fail():
    judged = await answer_relevancy(FakeChat([ChatError("down")]), question=QUESTION, answer=ANSWER)
    assert judged.verified is False and judged.passing is None


async def test_the_judge_may_wrap_its_json_in_prose_but_not_omit_it():
    judged = await answer_relevancy(FakeChat(['Sure. {"addresses": true, "reason": "names the employer"} Done.']),
                                    question=QUESTION, answer=ANSWER)
    assert judged.passing is True and judged.verified and judged.reasons == ("names the employer",)


async def test_context_relevancy_is_the_share_of_contexts_that_bear_on_the_question():
    judge = FakeChat([json.dumps({"contexts": [{"index": 0, "relevant": True}, {"index": 1, "relevant": False}]})])
    judged = await context_relevancy(judge, question=QUESTION, contexts=CONTEXTS)
    assert judged.score == 0.5 and judged.passing is True and judged.verified
    judged = await context_relevancy(FakeChat([json.dumps({"contexts": [{"index": 0, "relevant": False}, {"index": 1, "relevant": False}]})]),
                                     question=QUESTION, contexts=CONTEXTS)
    assert judged.score == 0.0 and judged.passing is False


async def test_context_relevancy_refuses_a_verdict_that_does_not_cover_every_context():
    judged = await context_relevancy(FakeChat([json.dumps({"contexts": [{"index": 0, "relevant": True}]})]),
                                     question=QUESTION, contexts=CONTEXTS)
    assert judged.verified is False


async def test_correctness_is_scored_against_the_reference_on_a_five_point_scale():
    judged = await correctness(FakeChat([json.dumps({"score": 4, "reason": "matches the employer, omits the title"})]),
                               question=QUESTION, answer=ANSWER, reference="Person-1 works at Brightlake as a principal engineer.")
    assert judged.score == 0.8 and judged.passing is True
    judged = await correctness(FakeChat([json.dumps({"score": 2, "reason": "wrong employer"})]),
                               question=QUESTION, answer="Northwind.", reference="Brightlake.")
    assert judged.score == 0.4 and judged.passing is False
    judged = await correctness(FakeChat([json.dumps({"score": 6, "reason": "x"})]), question=QUESTION, answer=ANSWER, reference="x")
    assert judged.verified is False, "a score off the scale is not a judgment"
    judged = await correctness(FakeChat([json.dumps({"score": 3, "reason": "partly"})]), question=QUESTION, answer=ANSWER,
                               reference="x", passing_score=3)
    assert judged.passing is True, "the threshold is the caller's"


async def test_inputs_over_the_bound_are_refused_not_cut():
    judge = FakeChat([])
    judged = await context_relevancy(judge, question=QUESTION, contexts=["c"] * (MAX_CONTEXTS + 1))
    assert judged.verified is False and "bound" in judged.reasons[0] and judge.calls == [], "the judge was never asked"
    judged = await faithfulness(judge, answer="x" * 200_000, contexts=CONTEXTS)
    assert judged.verified is False and "bound" in judged.reasons[0]


async def test_an_answer_with_no_claims_is_judged_and_says_so():
    judged = await faithfulness(FakeChat([json.dumps({"claims": []})]), answer="I do not know.", contexts=CONTEXTS)
    assert judged.verified and judged.score is None and judged.passing is None and "no claims" in judged.reasons[0]


def test_recall_at_k_is_the_share_of_relevant_items_in_the_top_k():
    assert recall_at(["a", "b", "c"], {"a", "c", "d"}, 2) == pytest.approx(1 / 3)
    assert recall_at(["a", "b", "c"], {"a", "c", "d"}, 3) == pytest.approx(2 / 3)
    assert recall_at([], {"a"}, 5) == 0.0
    assert recall_at(["a"], set(), 5) == 0.0
    with pytest.raises(ValueError):
        recall_at(["a"], {"a"}, 0)


async def test_pairwise_asks_in_both_orders_and_believes_only_agreement():
    from scone_memory.bench.evaluators import pairwise

    steady = FakeChat(['{"winner": "first", "reason": "A is grounded"}', '{"winner": "second", "reason": "still A"}'])
    verdict = await pairwise(steady, question="Q?", answer_a="A says", answer_b="B says", reference="the truth")
    assert (verdict.score, verdict.passing, verdict.verified, verdict.judge_calls) == (1.0, True, True, 2)
    assert verdict.reasons == ("A is grounded", "still A")
    assert "A right answer says:\nthe truth" in steady.calls[0][1] and steady.calls[1][1].index("B says") < steady.calls[1][1].index("A says")
    biased = FakeChat(['{"winner": "first"}', '{"winner": "first"}'])
    verdict = await pairwise(biased, question="Q?", answer_a="A", answer_b="B")
    assert (verdict.score, verdict.passing, verdict.verified) == (0.5, False, True)
    assert "in one order" in verdict.reasons[-1], "a judge that prefers whatever it read first is a tie, not a coin"
    b_wins = FakeChat(['{"winner": "second"}', '{"winner": "first"}'])
    assert (await pairwise(b_wins, question="Q?", answer_a="A", answer_b="B")).score == 0.0
    tie = FakeChat(['{"winner": "tie"}', '{"winner": "tie"}'])
    assert (await pairwise(tie, question="Q?", answer_a="A", answer_b="B")).score == 0.5
    unreadable = FakeChat(['{"winner": "first"}', '{"winner": "maybe"}'])
    verdict = await pairwise(unreadable, question="Q?", answer_a="A", answer_b="B")
    assert not verdict.verified and verdict.passing is None and verdict.judge_calls == 2
    assert not (await pairwise(FakeChat([ChatError("down")]), question="Q?", answer_a="A", answer_b="B")).verified
    assert "bound" in (await pairwise(FakeChat(), question="Q?", answer_a="x" * 70_000, answer_b="B")).reasons[0]


async def test_semantic_similarity_needs_no_judge_and_says_what_it_is_not():
    from scone_memory import HashEmbedder
    from scone_memory.bench.evaluators import semantic_similarity

    embedder = HashEmbedder()
    same = await semantic_similarity(embedder, answer="the harbour closes in November", reference="the harbour closes in November")
    assert same.score == pytest.approx(1.0) and same.passing and same.verified and same.judge_calls == 0
    assert "not correctness" in same.reasons[0] and embedder.id in same.reasons[0]
    other = await semantic_similarity(embedder, answer="apples grow on the ridge", reference="the harbour closes in November",
                                      passing_similarity=0.9)
    assert 0.0 <= other.score < 0.9 and other.passing is False
    assert (await semantic_similarity(embedder, answer="", reference="x")).verified is False
    with pytest.raises(ValueError):
        await semantic_similarity(embedder, answer="a", reference="b", passing_similarity=2)

    class Broken:
        id, dim = "broken", 3

        async def embed(self, texts):
            raise RuntimeError("no model")

    assert "embedder failed" in (await semantic_similarity(Broken(), answer="a", reference="b")).reasons[0]

    class NotANumber:
        id, dim = "nan", 2

        async def embed(self, texts):
            return [[float("nan"), 0.0], [1.0, 1.0]]

    corrupt = await semantic_similarity(NotANumber(), answer="a", reference="b")
    assert corrupt.verified is False and corrupt.passing is None and "non-finite" in corrupt.reasons[0], \
        "a NaN would clamp to a perfect match; it is unverified instead"
