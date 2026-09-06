"""MemoryAgentBench Conflict Resolution harness: the gold fact is the last
one carrying the answer, stale facts are the earlier ones that say the same
thing with another object, and the reader sees the top k oldest first."""

from __future__ import annotations

import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.memoryagentbench import (
    gold_fact,
    judge_ranking,
    load_conflict_resolution,
    parse_facts,
    run_conflict_resolution,
    stale_facts,
)
from scone_memory.llm import FakeChat

CONTEXT = """Here is a list of facts:
0. pesäpallo was created in the country of Finland.
1. goaltender is associated with the sport of ice hockey.
2. Nobuhiro Watsuki is famous for Rurouni Kenshin.
3. pesäpallo was created in the country of Philippines.
4. goaltender is associated with the sport of pesäpallo.
5. Ferdowsi is famous for Shahnameh.
6. Rand al'Thor was created by Ferdowsi.
7. Stephen Crane is famous for Shahnameh.
"""


def test_facts_are_the_numbered_lines_in_order_and_a_gap_is_an_error():
    facts = parse_facts(CONTEXT)
    assert facts[0] == "pesäpallo was created in the country of Finland." and len(facts) == 8
    with pytest.raises(ValueError):
        parse_facts("0. a.\n2. b.")


def test_rows_load_from_json_with_source_and_hops(tmp_path):
    rows = [
        {"context": CONTEXT, "questions": ["Which sport is goaltender associated with?"], "answers": [["pesäpallo"]],
         "metadata": {"source": "factconsolidation_sh_6k"}},
        {"context": CONTEXT, "questions": ["q"], "answers": [["a"]], "metadata": {"source": "factconsolidation_mh_6k"}},
    ]
    path = tmp_path / "cr.json"
    path.write_text(json.dumps(rows))
    items = load_conflict_resolution(path)
    assert [(i.source, i.hops, len(i.facts)) for i in items] == [
        ("factconsolidation_sh_6k", "single", 8), ("factconsolidation_mh_6k", "multi", 8)]
    assert items[0].questions == (("Which sport is goaltender associated with?", ("pesäpallo",)),)


def test_the_gold_fact_answers_the_question_and_is_the_last_such_and_stale_ones_share_its_words():
    facts = parse_facts(CONTEXT)
    # "Philippines" is only in fact 3; fact 0 says the same thing about Finland.
    assert gold_fact(facts, ["Philippines"], "In which country was pesäpallo created?") == 3
    assert stale_facts(facts, 3, ["Philippines"]) == [0]
    # "pesäpallo" appears in 0, 3 and 4; the question is about the goaltender,
    # so 4 is the gold, and the facts about pesäpallo's country are not stale
    # versions of it.
    assert gold_fact(facts, ["pesäpallo"], "Which sport is goaltender associated with?") == 4
    assert stale_facts(facts, 4, ["pesäpallo"]) == [1]
    # "Shahnameh" recurs in a later fact about someone else (a real pattern
    # in the split): the gold is the fact that shares the question's words.
    assert gold_fact(facts, ["Shahnameh"], "What is Ferdowsi famous for?") == 5
    assert gold_fact(facts, ["Shahnameh"], "What is Stephen Crane famous for?") == 7
    # With no question words to go on, the last carrier wins.
    assert gold_fact(facts, ["Shahnameh"]) == 7
    assert gold_fact(facts, ["Belgium"], "anything") is None


def test_judge_ranking_reads_gold_and_stale_off_the_ranked_sources():
    assert judge_ranking(["fact-3", "fact-0"], 3, [0]) == (True, False)
    assert judge_ranking(["fact-0", "fact-3"], 3, [0]) == (True, True)
    assert judge_ranking(["fact-0", "fact-9"], 3, [0]) == (False, True)
    assert judge_ranking(["fact-9"], 3, [0]) == (False, False)
    assert judge_ranking(["fact-3"], 3, []) == (True, False)


async def test_the_reader_sees_the_top_k_oldest_first_and_is_scored_by_substring(tmp_path):
    rows = [{"context": CONTEXT, "questions": [
        "Which sport is goaltender associated with?", "In which country was pesäpallo created?"],
        "answers": [["pesäpallo"], ["Philippines"]], "metadata": {"source": "factconsolidation_sh_6k"}}]
    path = tmp_path / "cr.json"
    path.write_text(json.dumps(rows))
    [item] = load_conflict_resolution(path)

    def make():
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    reader = FakeChat(["The sport is Pesäpallo.", "Finland"])
    report = await run_conflict_resolution(make, item, reader=reader, reader_name="fake", k=5)
    assert report.reader == "fake" and report.answered == 2 and report.correct == 1 and report.accuracy == 0.5
    assert [r.correct for r in report.results] == [True, False]
    assert report.results[0].gold == 4 and report.results[1].gold == 3 and report.results[1].stale == [0]
    # Both conflicting facts reach the reader, and the newer one comes last.
    _, user = reader.calls[1]
    listing = user.split("Question:")[0]
    assert "Finland" in listing and "Philippines" in listing
    assert listing.index("Finland") < listing.index("Philippines")
    assert all(r.gold_at_k for r in report.results), "k=5 holds every fact, so the gold is always there"

    plain = await run_conflict_resolution(make, item, k=5)
    assert plain.reader is None and plain.accuracy is None and plain.answered == 0
    assert "results" not in plain.as_dict(with_items=False)


def test_the_cli_runs_the_split_without_touching_the_configured_store(tmp_path):
    import io

    from scone_memory import cli

    rows = [{"context": CONTEXT, "questions": ["In which country was pesäpallo created?"], "answers": [["Philippines"]],
             "metadata": {"source": "factconsolidation_sh_6k"}},
            {"context": CONTEXT, "questions": ["q"], "answers": [["nothing"]], "metadata": {"source": "factconsolidation_mh_6k"}}]
    path = tmp_path / "cr.json"
    path.write_text(json.dumps(rows))
    report = tmp_path / "report.json"
    untouched = tmp_path / "configured.db"
    env = {"SCONE_EMBEDDER": "hash", "SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite", "SCONE_SQLITE_PATH": str(untouched)}
    out = io.StringIO()
    code = cli.main(["bench-conflicts", str(path), "--sources", "factconsolidation_sh_6k", "--k", "5", "--out", str(report)],
                    env=env, stdin=io.StringIO(), out=out)
    assert code == 0, out.getvalue()
    assert "factconsolidation_sh_6k" in out.getvalue() and "no reader" in out.getvalue() and "mh_6k" not in out.getvalue()
    [saved] = json.loads(report.read_text())
    assert saved["k"] == 5 and saved["results"][0]["gold"] == 3 and saved["accuracy"] is None
    assert not untouched.exists()
    assert cli.main(["bench-conflicts", str(path), "--reader"], env=env, stdin=io.StringIO(), out=io.StringIO()) == 2
