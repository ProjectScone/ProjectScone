"""Scoring computed temporal answers on a dated question file.

The file is a LongMemEval-shaped one: each item carries a question, the
day it is asked, dated sessions, and the answer a person would accept.
Nothing here calls a model: the score checks the numbers an answer gives
against the numbers the expected answer holds, and for an ordering the
order the expected answer puts its events in. Questions the planner does
not read, and events it will not date, are counted apart from wrong
ones, because refusing to answer is not the same as answering badly.
"""

from __future__ import annotations

import json

from scone_memory.bench.temporal import TemporalScore, score_answer, run_temporal


ITEMS = [
    {
        "question_id": "between", "question_type": "temporal-reasoning",
        "question": "How many days passed between the time I sold baked goods and the time I ran the bake-off?",
        "question_date": "2023/04/20 (Thu) 10:12", "answer": "21 days",
        "haystack_dates": ["2023/03/11 (Sat) 09:00", "2023/04/01 (Sat) 18:30"],
        "haystack_session_ids": ["s1", "s2"], "answer_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "I sold baked goods at the market with my neighbour."}],
            [{"role": "user", "content": "I ran the bake-off at the village hall and we raised plenty."}],
        ],
    },
    {
        "question_id": "unread", "question_type": "temporal-reasoning",
        "question": "How old was I when I moved to Lisbon?", "question_date": "2023/04/20 (Thu) 10:12",
        "answer": "27", "haystack_dates": ["2023/03/11 (Sat) 09:00"], "haystack_session_ids": ["s1"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I moved to Lisbon when I was 27."}]],
    },
    {
        "question_id": "ungrounded", "question_type": "temporal-reasoning",
        "question": "How many days ago did I climb Mount Fuji?", "question_date": "2023/04/20 (Thu) 10:12",
        "answer": "9 days", "haystack_dates": ["2023/03/11 (Sat) 09:00"], "haystack_session_ids": ["s1"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I sold baked goods at the market."}]],
    },
    {
        "question_id": "lookup-right", "question_type": "temporal-reasoning",
        "question": "What did I do 40 days ago?", "question_date": "2023/04/20 (Thu) 10:12",
        "answer": "You sold baked goods at the market.",
        "haystack_dates": ["2023/03/11 (Sat) 09:00", "2023/04/01 (Sat) 18:30"],
        "haystack_session_ids": ["s1", "s2"], "answer_session_ids": ["s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "I sold baked goods at the market with my neighbour."}],
            [{"role": "user", "content": "I ran the bake-off at the village hall."}],
        ],
    },
    {
        "question_id": "lookup-elsewhere", "question_type": "temporal-reasoning",
        "question": "What did I do 40 days ago?", "question_date": "2023/04/20 (Thu) 10:12",
        "answer": "You ran the bake-off at the village hall.",
        "haystack_dates": ["2023/03/11 (Sat) 09:00", "2023/04/01 (Sat) 18:30"],
        "haystack_session_ids": ["s1", "s2"], "answer_session_ids": ["s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "I sold baked goods at the market with my neighbour."}],
            [{"role": "user", "content": "I ran the bake-off at the village hall."}],
        ],
    },
]


def test_a_number_answer_is_right_when_the_expected_answer_holds_it():
    assert score_answer({"asked": 3, "days": 21, "unit": "week"}, None, "3 weeks") is True
    assert score_answer({"asked": 21, "days": 21, "unit": "day"}, None,
                        "21 days. 22 days (including the last day) is also acceptable") is True
    assert score_answer({"asked": 22, "days": 22, "unit": "day"}, None, "21 days") is True, "the day counted either way"
    assert score_answer({"asked": 5, "days": 35, "unit": "week"}, None, "3 weeks") is False
    assert score_answer({"asked": 5, "days": 150, "unit": "month"}, None, "Five months ago") is True


def test_a_chosen_event_is_right_when_the_expected_answer_names_it():
    value = {"first": "solo trip to thailand", "date": "2023-02-02"}
    events = ("one to europe with family", "solo trip to thailand")
    assert score_answer(value, events, "The solo trip to Thailand") is True
    assert score_answer({"first": "one to europe with family", "date": "2023-02-02"}, events,
                        "The solo trip to Thailand") is False
    assert score_answer({"first": "one to europe with family", "date": "2023-02-02"}, events,
                        "Neither of those, it was something else") is False, "naming neither is not naming one"


def test_an_order_is_right_when_the_expected_answer_gives_the_same_one():
    value = {"order": [{"event": "nursery", "date": "2023-01-01"}, {"event": "baby shower", "date": "2023-02-01"}]}
    assert score_answer(value, None, "First the nursery, then the baby shower.") is True
    assert score_answer(value, None, "First the baby shower, then the nursery.") is False


async def test_a_run_counts_what_was_computed_apart_from_what_was_refused(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps(ITEMS), encoding="utf-8")
    score = await run_temporal(path)
    assert isinstance(score, TemporalScore)
    assert (score.items, score.computed, score.correct, score.wrong) == (5, 1, 1, 0)
    assert score.not_temporal == 1 and score.ungrounded == 1 and score.ambiguous == 0
    assert (score.recalled, score.recalled_right) == (2, 1), "a day's passages answer the day the question names"
    assert score.record()["questions"] == 5 and score.record()["correct_of_computed"] == 1.0
    assert "computed 1 of 5" in score.text()
