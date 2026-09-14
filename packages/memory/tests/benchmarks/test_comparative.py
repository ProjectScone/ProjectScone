"""The same questions, the same passages, the same embedder: ours beside LlamaIndex.

The north star names LlamaIndex as the framework to beat, and a claim of
that kind is worth exactly the measurement behind it. This runs the
reference framework itself -- installed, unmodified -- over the same
frozen bench items we score ourselves with, using our embedder on both
sides and no model or reranker on either, and reports the two rankings'
recall, MRR and per-item outcome beside each other, with the configuration
both sides ran under written into the report.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.comparative import (SconeEmbedding, compare, llamaindex_session_ranking)
from scone_memory.bench.runner import BenchItem

pytestmark = pytest.mark.asyncio


def item(question_id: str, question: str, sessions: list[list[str]], answer: list[str]) -> BenchItem:
    ids = [f"{question_id}-s{index}" for index in range(len(sessions))]
    return BenchItem(question_id=question_id, question_type="single-hop", question=question, question_date="2026-09-13",
                     sessions=tuple(tuple(lines) for lines in sessions), session_ids=tuple(ids),
                     session_dates=tuple("2026-09-01T00:00:00Z" for _ in sessions),
                     answer_session_ids=tuple(f"{question_id}-s{index}" for index in answer))


ITEMS = [
    item("q1", "Where does Juniper keep the calibration notes?", [
        ["user: The quarterly revenue exceeded expectations.", "assistant: Noted."],
        ["user: Juniper keeps the calibration notes in the blue binder in the lab.", "assistant: Blue binder, lab."],
        ["user: We planted tomatoes and basil this spring.", "assistant: Lovely."],
    ], answer=[1]),
    item("q2", "What colour is the binder for the calibration notes?", [
        ["user: The binder for the calibration notes is blue.", "assistant: Blue."],
        ["user: The train leaves at nine.", "assistant: Nine."],
    ], answer=[0]),
    item("q3", "Which street is the Lisbon flat on?", [
        ["user: The weather was grey all week.", "assistant: It was."],
        ["user: The Lisbon flat is on Rua Augusta.", "assistant: Rua Augusta."],
    ], answer=[1]),
    item("q0", "A question with no evidence stored.", [
        ["user: Unrelated small talk.", "assistant: Sure."],
    ], answer=[]),
]


async def test_the_embedding_adapter_hands_llamaindex_our_vectors():
    embedder = HashEmbedder()
    adapter = SconeEmbedding(embedder)
    ours = (await embedder.embed(["calibration notes"]))[0]
    assert adapter.get_text_embedding("calibration notes") == ours
    assert await adapter.aget_query_embedding("calibration notes") == ours
    assert adapter.model_name == embedder.id


async def test_llamaindex_ranks_sessions_for_a_question_with_our_embedder():
    ranked = await llamaindex_session_ranking(ITEMS[0], HashEmbedder(), k=3)
    assert ranked[0] == "q1-s1", ranked
    assert len(ranked) == len(set(ranked)) <= 3 and set(ranked) <= {"q1-s0", "q1-s1", "q1-s2"}


async def test_a_session_cut_into_many_nodes_is_one_session_in_the_ranking():
    """Small chunks make several nodes per session; the ranking folds them
    to the session, once, in the order its best node came."""
    long = item("q9", "Where are the calibration notes kept?", [
        ["user: " + " ".join(["The calibration notes are kept in the blue binder in the lab."] * 12)],
        ["user: " + " ".join(["The garden needs watering every other day in summer."] * 12)],
    ], answer=[0])
    ranked = await llamaindex_session_ranking(long, HashEmbedder(), k=2, chunk_size=24, chunk_overlap=0)
    assert ranked == ["q9-s0", "q9-s1"], ranked


def test_the_delta_is_ours_minus_theirs_at_every_k():
    from scone_memory.bench.comparative import SideScores, side_delta

    ours = SideScores({1: 0.5, 3: 1.0}, {1: 0.25, 3: 0.75}, 0.6)
    theirs = SideScores({1: 0.75, 3: 0.5}, {1: 0.5, 3: 0.5}, 0.4)
    delta = side_delta(ours, theirs)
    assert delta.recall_any == {1: -0.25, 3: 0.5} and delta.recall_all == {1: -0.25, 3: 0.25} and delta.mrr == pytest.approx(0.2)
    assert delta.precision == {} and delta.ndcg == {}, "sides scored without them carry no delta for them: a missing number is not a zero"
    with pytest.raises(ValueError):
        side_delta(ours, SideScores({1: 0.5}, {1: 0.5}, 0.5))


def test_precision_and_ndcg_are_scored_beside_recall_and_carried_in_the_record():
    from scone_memory.bench.comparative import Comparison, SideScores, _scores, side_delta

    rankings = [(["s1", "x", "s2"], {"s1", "s2"}), (["x", "y", "z"], {"s3"})]
    scores = _scores(rankings, [1, 3])
    assert scores.recall_any == {1: 0.5, 3: 0.5} and scores.recall_all == {1: 0.0, 3: 0.5}
    assert scores.precision == {1: 0.5, 3: 0.3333}, "of the top three, two answered in one item and none in the other; four places"
    # Item one: answers at places 1 and 3, DCG 1 + 1/log2(4) against an ideal of 1 + 1/log2(3), so 0.92; item two: 0.
    assert scores.ndcg[1] == 0.5 and scores.ndcg[3] == pytest.approx(0.4599, abs=1e-4), "the second answer in third place is discounted"
    empty = _scores([], [1, 3])
    assert empty.precision == {1: 0.0, 3: 0.0} and empty.ndcg == {1: 0.0, 3: 0.0}
    delta = side_delta(scores, _scores([(["s1"], {"s1"})], [1, 3]))
    assert delta.precision[1] == pytest.approx(-0.5) and delta.ndcg[1] == pytest.approx(-0.5)
    report = Comparison(2, 0, {"scone": scores, "llamaindex": scores}, side_delta(scores, scores), ())
    record = report.record()
    assert record["sides"]["scone"]["precision"] == {"1": 0.5, "3": round(1 / 3, 4)}
    assert record["sides"]["scone"]["ndcg"]["1"] == 0.5 and record["delta"]["ndcg"] == {"1": 0.0, "3": 0.0}
    with pytest.raises(ValueError, match="precision"):
        side_delta(scores, SideScores(scores.recall_any, scores.recall_all, scores.mrr))
    with pytest.raises(ValueError, match="ndcg"):
        side_delta(scores, SideScores(scores.recall_any, scores.recall_all, scores.mrr, scores.precision, {1: 0.0}))


def test_our_passages_fold_to_distinct_sessions_as_the_references_nodes_do():
    from scone_memory.bench.comparative import distinct_sessions

    assert distinct_sessions(["a", "a", "b", "a", "c", "", "d"], 3) == ("a", "b", "c"), "a session cut in three is one"
    assert distinct_sessions([], 3) == () and distinct_sessions(["a"], 0) == ()



async def test_the_comparison_scores_both_sides_on_the_same_items_and_says_how_it_ran():
    def make_engine():
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    report = await compare(ITEMS, make_engine, HashEmbedder(), ks=(1, 3))
    assert report.n == 3 and report.excluded_without_evidence == 1
    for side in ("scone", "llamaindex"):
        scores = report.sides[side]
        assert set(scores.recall_any) == {1, 3} and 0.0 <= scores.mrr <= 1.0
        assert all(0.0 <= value <= 1.0 for value in scores.recall_any.values())
    assert report.delta.recall_any[3] == pytest.approx(report.sides["scone"].recall_any[3] - report.sides["llamaindex"].recall_any[3])
    outcome = report.per_item_at(3)
    assert outcome["wins"] + outcome["losses"] + outcome["ties"] == 3
    config = report.config
    assert config["embedder"] == HashEmbedder().id and config["llamaindex"]["splitter"] == "SentenceSplitter"
    assert config["llamaindex"]["chunk_size"] > 0 and config["llm"] is None and config["reranker"] is None
    assert config["llamaindex"]["version"], "the version of the framework we ran against is on the record"


async def test_the_report_records_and_serialises_every_item_ranking():
    def make_engine():
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    report = await compare(ITEMS[:2], make_engine, HashEmbedder(), ks=(1,))
    record = report.record()
    assert [row["question_id"] for row in record["items"]] == ["q1", "q2"]
    assert all(set(row) >= {"question_id", "scone", "llamaindex", "answer_sessions"} for row in record["items"])
    assert record["protocol"] == "comparative-retrieval-v1"


async def test_the_reference_synthesizer_writes_with_our_model_over_the_same_passages():
    """LlamaIndex's TreeSummarize runs over the passages we hand it, through
    our ChatModel port, so the two synthesizers share one local model."""
    from scone_memory.providers.llm import FakeChat
    from scone_memory.bench.comparative import llamaindex_summary

    model = FakeChat(["The launch moved to March after the audit."])
    summary = await llamaindex_summary(model, "What happened with the launch?",
                                       ["Priya moved the launch to March.", "The audit found two billing errors."])
    assert summary.text == "The launch moved to March after the audit."
    assert summary.model_calls == 1 and summary.synthesizer == "TreeSummarize"
    assert summary.cites is False, "the reference's summary carries no citations to check"
    (system, prompt), = model.calls
    assert "Priya moved the launch to March." in prompt and "What happened with the launch?" in prompt


async def test_a_reference_synthesizer_whose_model_fails_says_so_instead_of_raising():
    from scone_memory.providers.llm import ChatError, FakeChat
    from scone_memory.bench.comparative import llamaindex_summary

    summary = await llamaindex_summary(FakeChat([ChatError("down")]), "q", ["one passage"])
    assert summary.text == "" and summary.failed == "ChatError" and summary.model_calls == 1
