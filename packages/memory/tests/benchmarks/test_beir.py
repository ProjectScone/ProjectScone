"""Retrieval measured on a BEIR dataset: its corpus, its queries, its graded judgements.

Every number this benchmark reported came from datasets it shaped
itself. BEIR is the common ground retrieval systems are compared on: a
corpus, queries, and relevance judged in grades. The reference runs it
through a package it does not declare, with the split fixed to test.
Here the files are read directly, any split is chosen, a judgement that
names a document or query the files do not hold is counted rather than
dropped, and a corpus cut down to fit a machine keeps every judged
document and says it was cut, because a smaller corpus scores higher.
"""

from __future__ import annotations

import io
import json
from math import log2

import pytest

from scone_memory.bench.beir import graded_ndcg_at, load_beir, reciprocal_rank_at, recall_at_graded, run_beir, sampled
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import main

DOCS = {
    "d1": ("Harbour crane", "The crane survey found rust on the jib and grease missing from the slew ring."),
    "d2": ("Crane booking", "The crane survey was booked with the harbour board for the third of May."),
    "d3": ("Canteen", "The canteen menu changed in April; soup is now served on Tuesdays."),
    "d4": ("Invoices", "Invoice 20931 was paid by the harbour office on the ninth of June."),
    "d5": ("Parking", "Parking passes for the harbour car park are renewed each winter."),
}
QUERIES = {"q1": "what was wrong with the crane jib", "q2": "when was invoice 20931 paid",
           "q3": "what does the canteen serve", "q4": "a query nobody judged"}
QRELS = [("q1", "d1", 2), ("q1", "d2", 1), ("q1", "d3", 0), ("q2", "d4", 1), ("q3", "d3", 1),
         ("q2", "d99", 1), ("q9", "d1", 1)]


def write(root, split="test", qrels=QRELS, corpus=DOCS, queries=QUERIES):
    (root / "qrels").mkdir(parents=True, exist_ok=True)
    with open(root / "corpus.jsonl", "w") as out:
        for doc_id, (title, text) in corpus.items():
            out.write(json.dumps({"_id": doc_id, "title": title, "text": text}) + "\n")
    with open(root / "queries.jsonl", "w") as out:
        for query_id, text in queries.items():
            out.write(json.dumps({"_id": query_id, "text": text}) + "\n")
    with open(root / "qrels" / f"{split}.tsv", "w") as out:
        out.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            out.write("\t".join(map(str, row)) + "\n")
    return root


def test_the_files_are_read_and_what_they_do_not_hold_is_counted(tmp_path):
    data = load_beir(write(tmp_path))
    assert set(data.corpus) == set(DOCS) and set(data.queries) == set(QUERIES)
    assert data.qrels["q1"] == {"d1": 2, "d2": 1, "d3": 0}
    assert data.unknown_documents == 1 and data.unknown_queries == 1, "d99 and q9 are named but not held"
    assert data.unjudged_queries == 1 and data.split == "test"


def test_any_split_is_read_and_a_missing_one_is_refused(tmp_path):
    write(tmp_path, split="dev", qrels=[("q1", "d1", 1)])
    assert load_beir(tmp_path, split="dev").qrels == {"q1": {"d1": 1}}
    with pytest.raises(InvalidInput, match="test.tsv"):
        load_beir(tmp_path, split="test")


@pytest.mark.parametrize("bad, said", [("q1\td1\ttwo\n", "line 2"), ("q1\td1\n", "line 2"), ("q1\td1\t-1\n", "line 2")])
def test_a_malformed_judgement_is_refused_with_its_line(tmp_path, bad, said):
    write(tmp_path)
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + bad)
    with pytest.raises(InvalidInput, match=said):
        load_beir(tmp_path)


def test_graded_ndcg_weighs_each_document_by_its_grade_against_the_best_order():
    judged = {"d1": 2, "d2": 1, "d3": 0}
    best = 2 / log2(2) + 1 / log2(3)
    assert graded_ndcg_at(["d1", "d2", "d3"], judged, 10) == pytest.approx(1.0)
    assert graded_ndcg_at(["d2", "d1"], judged, 10) == pytest.approx((1 / log2(2) + 2 / log2(3)) / best)
    assert graded_ndcg_at(["d3", "d5", "d2"], judged, 10) == pytest.approx((1 / log2(4)) / best)
    assert graded_ndcg_at(["d2", "d1"], judged, 1) == pytest.approx(1 / 2), "the ideal at 1 is the best grade alone"
    listed_worst_first = {"d3": 0, "d2": 1, "d1": 2}
    assert graded_ndcg_at(["d2", "d1"], listed_worst_first, 1) == pytest.approx(1 / 2), "the ideal is sorted by grade"
    assert recall_at_graded(["d3", "d2"], judged, 2) == pytest.approx(0.5), "grade 0 is judged not relevant"
    assert reciprocal_rank_at(["d3", "d5", "d1"], judged, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at(["d3", "d5", "d1"], judged, 2) == 0.0


def test_a_cut_corpus_keeps_every_judged_document_and_says_it_was_cut(tmp_path):
    # Filler first in the file, so keeping judged documents cannot happen by file order alone.
    many = {**{f"x{n}": ("Filler", f"Unrelated note number {n} about gardening.") for n in range(20)}, **DOCS}
    data = load_beir(write(tmp_path, corpus=many))
    cut = sampled(data, queries=2, seed=7, max_documents=6)
    judged = {doc for query in cut.data.queries for doc in cut.data.qrels.get(query, {})}
    assert len(cut.data.queries) == 2 and judged <= set(cut.data.corpus) and len(cut.data.corpus) == 6
    assert cut.corpus_documents == 25 and cut.kept_documents == 6 and cut.reduced is True
    assert "scores on a cut corpus run higher" in cut.why
    assert sampled(data, queries=2, seed=7, max_documents=6).data.queries == cut.data.queries, "a seed repeats"
    chosen = {tuple(sorted(sampled(data, queries=2, seed=seed).data.queries)) for seed in range(10)}
    assert len(chosen) > 1, "the seed chooses which queries run"


async def test_a_run_reports_graded_metrics_per_document_and_what_it_was_run_on(tmp_path):
    # One judged document long enough to be stored as many chunks.
    long = {**DOCS, "d1": ("Harbour crane", " ".join([DOCS["d1"][1]] * 40))}
    data = load_beir(write(tmp_path, corpus=long))
    report = await run_beir(sampled(data), ks=(1, 3))
    record = report.record()
    assert record["dataset"]["split"] == "test" and record["queries_run"] == 3
    assert set(record["metrics"]) == {"ndcg@1", "ndcg@3", "recall@1", "recall@3", "mrr@3", "precision@1", "precision@3"}
    assert all(0.0 <= value <= 1.0 for value in record["metrics"].values())
    assert record["dataset"]["unknown_documents"] == 1 and record["dataset"]["unjudged_queries"] == 1
    assert record["config"]["embedder"] and record["measured"] is True
    for query in record["per_query"]:
        assert len(query["ranked"]) == len(set(query["ranked"])), "a document chunked many times is ranked once"


def test_the_command_line_runs_a_beir_directory(tmp_path):
    write(tmp_path / "set")
    out = io.StringIO()
    code = main(["bench-beir", str(tmp_path / "set"), "--k", "1,3", "--json"], env={}, stdin=io.StringIO(""), out=out)
    record = json.loads(out.getvalue())
    assert code == 0 and record["queries_run"] == 3 and "ndcg@3" in record["metrics"]


async def test_a_query_longer_than_recall_takes_is_cut_at_a_word_and_counted(tmp_path):
    """Argument-retrieval sets hold whole paragraphs as queries, past the
    1,000 characters recall takes. The run scores the query as recall can
    take it, cut at a word, and says so, rather than failing after the
    corpus is stored."""
    from scone_memory.core.validation import MAX_QUERY

    long_query = "what was wrong with the crane jib " * 60
    data = load_beir(write(tmp_path, queries={**QUERIES, "q1": long_query}))
    report = await run_beir(sampled(data), ks=(1, 3))
    record = report.record()
    assert record["queries_run"] == 3 and record["queries_cut"] == 1 and record["queries_empty"] == 0
    [cut] = [query for query in record["per_query"] if query["query_id"] == "q1"]
    assert cut["cut"] is True and cut["ranked"][0] == "d1"
    assert all("cut" not in query for query in record["per_query"] if query["query_id"] != "q1")
    assert len(long_query) > MAX_QUERY and "1 queries were longer than recall takes" in report.text()


async def test_an_empty_query_retrieves_nothing_and_is_counted(tmp_path):
    data = load_beir(write(tmp_path, queries={**QUERIES, "q2": ""}))
    report = await run_beir(sampled(data), ks=(1, 3))
    record = report.record()
    [empty] = [query for query in record["per_query"] if query["query_id"] == "q2"]
    assert empty["empty"] is True and empty["ranked"] == [] and set(empty["scores"].values()) == {0.0}
    assert record["queries_empty"] == 1 and record["queries_run"] == 3 and "1 queries were empty" in report.text()


def test_the_cut_ends_at_a_word_within_the_limit():
    from scone_memory.bench.beir import fitted_query
    from scone_memory.core.validation import MAX_QUERY

    words = "harbour  " * 200
    cut = fitted_query(words)
    assert len(cut) <= MAX_QUERY and cut.endswith("harbour") and words.startswith(cut), "no space left at the end"
    exactly = "harbour " * 125  # 1,000 characters
    assert fitted_query(exactly) == exactly, "a query at the limit is taken whole"
    assert fitted_query(exactly + "x") == exactly.rstrip(), "a word ending at the limit is kept"
    assert fitted_query(exactly + "  x") == exactly.rstrip(), "a space just past the limit is the break"
    assert fitted_query("ab " + "x" * 997 + " y") == "ab " + "x" * 997, "a long last word that ends at the limit is kept"
    assert fitted_query("x" * (MAX_QUERY + 5)) == "x" * MAX_QUERY, "one long token is cut where the limit falls"
    assert fitted_query("short query") == "short query"
    assert fitted_query("a" * MAX_QUERY) == "a" * MAX_QUERY


def test_a_set_with_no_judged_query_is_refused_rather_than_scored_zero(tmp_path):
    """Every judgement names a document or a query the files do not hold, so
    there is nothing to score; a report of zeros would read as a measurement."""
    data = load_beir(write(tmp_path, qrels=[("q1", "d99", 1), ("q9", "d1", 1)]))
    with pytest.raises(InvalidInput, match="no query is judged"):
        sampled(data)
