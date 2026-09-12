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
