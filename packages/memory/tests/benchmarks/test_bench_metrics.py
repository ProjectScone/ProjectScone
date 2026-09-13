"""Rank-aware retrieval metrics.

`CAPABILITIES.md` asks us to beat the leading RAG framework, and its
evaluation surface reports hit rate, MRR, precision, recall, average
precision and NDCG. Ours reported recall alone, at two strictnesses.

Recall@k cannot see **rank**: "the answer was in the top ten" is the same
number whether it came first or tenth, so a change that moves answers up
the list looks identical to one that does nothing. Every measurement made
with recall alone was made with a blunter instrument than the framework
it was being compared against.
"""

from __future__ import annotations

from scone_memory.bench.metrics import mrr, ndcg_at, precision_at, reciprocal_rank


def test_reciprocal_rank_is_where_the_first_answer_landed():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["b", "a", "c"], {"a"}) == 0.5
    assert reciprocal_rank(["b", "c", "a"], {"a"}) == 1 / 3
    assert reciprocal_rank(["b", "c"], {"a"}) == 0.0


def test_reciprocal_rank_takes_the_first_of_several_answers():
    """With more than one relevant source the reciprocal rank is about
    the first one reached, which is what makes it a measure of how soon a
    reader finds something useful."""
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_mrr_is_the_mean_over_items():
    assert mrr([(["a"], {"a"}), (["x", "a"], {"a"})]) == 0.75
    assert mrr([]) == 0.0


def test_precision_counts_wasted_slots():
    """What recall cannot say: of what was returned, how much was worth
    returning. A system that returns everything has perfect recall and
    this number falls."""
    assert precision_at(["a", "b", "c", "d"], {"a", "b"}, 4) == 0.5
    assert precision_at(["a", "b"], {"a", "b"}, 2) == 1.0
    assert precision_at(["x", "y", "z"], {"a"}, 3) == 0.0


def test_precision_does_not_count_one_answer_twice():
    """The same source returned twice is one answer and one wasted slot,
    not two answers."""
    assert precision_at(["a", "a", "x", "y"], {"a"}, 4) == 0.25


def test_ndcg_rewards_the_earlier_of_two_identical_sets():
    """The property that makes it worth having: the same answers, ranked
    better, score higher. Recall cannot tell these two apart."""
    early = ndcg_at(["a", "b", "x", "y"], {"a", "b"}, 4)
    late = ndcg_at(["x", "y", "a", "b"], {"a", "b"}, 4)
    assert early == 1.0
    assert late < early


def test_ndcg_is_one_when_every_answer_leads():
    assert ndcg_at(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_metrics_are_defined_when_nothing_is_relevant():
    """A question with no evidence must not make a metric undefined or
    push it outside its range."""
    for value in (reciprocal_rank(["a"], set()), precision_at(["a"], set(), 1),
                  ndcg_at(["a"], set(), 1), ndcg_at(["x", "y"], {"a"}, 2)):
        assert value == 0.0
