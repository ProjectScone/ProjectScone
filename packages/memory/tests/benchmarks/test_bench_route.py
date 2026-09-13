"""Measuring the rule that chooses a route.

A routing rule written down is better than one a model invents, but only
if somebody checks it. This runs the rule over a file of questions whose
answers are known and reports where each went — and, for the ones it sent
to the computer, whether the computed answer was right.

What it cannot say is whether a question that went to search would have
been better computed. That needs an answer to compare against for every
question and every route, which this file does not have, and the report
says so rather than implying the rule was vindicated.
"""

from __future__ import annotations

import io
import json

import pytest

from scone_memory.bench.route import RouteScore, run_route_bench

pytestmark = pytest.mark.asyncio


def item(qid, question, answer, sessions, dates):
    return {"question_id": qid, "question_type": "temporal-reasoning", "question": question,
            "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_sessions": [[{"role": "user", "content": text}] for text in sessions],
            "haystack_session_ids": [f"s{n}" for n in range(len(sessions))],
            "haystack_dates": dates, "answer_session_ids": ["s0"], "answer": answer}


DATASET = [
    item("q1", "How many days ago did I repaint the crane?", "9 days",
         ["I repainted the harbour crane today."], ["2023/05/23 (Tue) 09:00"]),
    item("q2", "what did we decide about the billing bug", "we reverted it",
         ["We reverted the billing change."], ["2023/05/20 (Sat) 09:00"]),
]


#: A question the temporal route answers by handing back the day's
#: passages rather than by computing anything -- status "recalled". It has
#: to be in a fixture of its own, because the split it exercises is
#: invisible on a file where every temporal question computes.
FROM_PASSAGES = [
    item("q1", "What did I do 9 days ago?", "repainted the crane",
         ["I repainted the harbour crane today."], ["2023/05/23 (Tue) 09:00"]),
]


@pytest.fixture()
def recalling(tmp_path):
    path = tmp_path / "recalled.json"
    path.write_text(json.dumps(FROM_PASSAGES), encoding="utf-8")
    return path


@pytest.fixture()
def dataset(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    return path


async def test_the_bench_says_where_every_question_went(dataset):
    scored = await run_route_bench(dataset)
    assert scored.questions == 2
    assert sum(scored.routes.values()) == 2
    assert set(scored.routes) == {"temporal", "graph", "recall"}


async def test_a_question_about_dates_is_routed_to_the_computer(dataset):
    scored = await run_route_bench(dataset)
    assert scored.routes["temporal"] >= 1


async def test_the_computed_answers_are_scored_against_the_file(dataset):
    """The file says nine days and the arithmetic says nine days, so the
    score is one. A score that could only ever be zero would look like a
    measurement and be none."""
    scored = await run_route_bench(dataset)
    assert scored.computed == 1 and scored.computed_right == 1
    assert "1 agree with the file" in scored.text()


async def test_an_answer_read_from_passages_is_not_counted_as_a_computed_one(recalling):
    """The temporal route answers two ways: it computes, or it hands back
    the day's passages. A recalled answer has no arithmetic in it, so
    `score_answer` cannot judge it -- and putting it in the `computed`
    denominator made the ratio report the computation as wrong when nothing
    had been computed at all."""
    scored = await run_route_bench(recalling)
    assert scored.routes["temporal"] == 1, scored.record()
    assert scored.recalled == 1 and scored.computed == 0, scored.record()
    assert "read from passages" in scored.text(), scored.text()
    assert "agree with the file" not in scored.text(), \
        "with nothing computed there is no score to report"


async def test_an_answer_the_scorer_cannot_judge_is_counted_apart_from_a_wrong_one(dataset):
    """"Cannot tell" and "wrong" are different facts, and adding them
    together tells a reader neither."""
    scored = await run_route_bench(dataset)
    assert scored.computed_unscored == 0, scored.record()
    assert not hasattr(scored, "_") and scored.computed_wrong == 0, scored.record()


async def test_the_report_says_what_it_cannot_say(dataset):
    scored = await run_route_bench(dataset)
    said = scored.text()
    assert "2 question(s)" in said and "temporal" in said
    assert "not say" in said, "a route not taken is not measured here"


async def test_a_limit_is_honoured(dataset):
    scored = await run_route_bench(dataset, limit=1)
    assert scored.questions == 1


async def test_the_score_is_a_record_that_can_be_kept(dataset):
    scored = await run_route_bench(dataset)
    record = scored.record()
    assert record["questions"] == 2 and record["routes"]["temporal"] >= 1
    assert record["dataset"].endswith("items.json")
    assert isinstance(scored, RouteScore)


async def test_the_bench_can_run_with_passage_merging(dataset):
    """A feature the bench cannot run is a feature nobody can judge."""
    from scone_memory.bench.runner import load_items, run

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

    async def make():
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                  HashEmbedder(), chunk_target=90).open()

    items = load_items(dataset)
    plain = await run(make, items, ks=(1, 2), limit=3, dataset=str(dataset))
    joined = await run(make, items, ks=(1, 2), limit=3, dataset=str(dataset), merge=True)
    assert plain.items == joined.items == len(items)
    # The claim under test: merging cannot invent or lose an episode, so
    # recall at the recall limit is the same. Anything else would mean it
    # was changing what came back rather than how it was packed.
    assert plain.recall_any[2] == joined.recall_any[2], (plain.recall_any, joined.recall_any)


async def test_the_bench_counts_the_bytes_it_actually_returned(tmp_path):
    """Swapping the items after merging and keeping recall's byte count
    reports a context reduction that never happened, and a measurement
    published from it would be wrong in the flattering direction.

    The fixture is built so merging actually fires and the merged span
    *crosses a gap*: the passage is then one byte longer than the
    fragments were, which is the direction a stale count hides.
    """
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.bench.runner import load_items, run

    long_session = (
        "The harbour crane was repainted in May after the survey found rust on the jib. "
        "The survey also found the slew ring needed grease, which the yard did that week. "
        "The crane returned to service on the first of June, a day later than planned.")
    path = tmp_path / "merging.json"
    path.write_text(json.dumps([item(
        "q1", "crane survey rust jib slew grease", "n/a",
        [long_session, "The bicycle needed a chain."],
        ["2023/05/20 (Sat) 09:00"] * 2)]), encoding="utf-8")

    async def make():
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                  HashEmbedder(), chunk_target=60).open()

    items = load_items(path)
    plain = await run(make, items, ks=(1, 3), limit=5, dataset=str(path))
    joined = await run(make, items, ks=(1, 3), limit=5, dataset=str(path), merge=True)

    assert plain.merge is False and joined.merge is True, \
        "a saved run whose metadata cannot say which configuration produced it is not evidence"
    [before], [after] = plain.results, joined.results
    assert len(after.retrieved_sessions) < len(before.retrieved_sessions), \
        "the fixture has to actually merge or this proves nothing"
    assert after.returned_bytes > before.returned_bytes, \
        (after.returned_bytes, before.returned_bytes)


def test_the_bench_reports_progress_even_when_asked_for_json(tmp_path, capsys):
    """A run that takes hours with no sign of life cannot be told from a
    wedged one. The JSON goes to stdout, so progress on stderr spoils
    nothing that reads it."""
    from scone_memory.runtime import cli

    path = tmp_path / "items.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    out = io.StringIO()
    code = cli.main(["bench", str(path), "--k", "1", "--json"], env={}, stdin=io.StringIO(""),
                    out=out)
    assert code == 0, out.getvalue()
    assert "2/2" in capsys.readouterr().err, "progress has to reach stderr with --json"
    assert json.loads(out.getvalue().strip().splitlines()[-1])["items"] == 2
