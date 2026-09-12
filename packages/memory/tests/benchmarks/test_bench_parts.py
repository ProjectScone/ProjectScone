"""Measuring what splitting a question changes, and saying when nothing.

The trap this guards is a report that compares a question with itself.
Questions the rule leaves whole are searched identically both ways, so
counting them would pad both columns and read as agreement.
"""

from __future__ import annotations

import json

import pytest

from scone_memory.bench.parts import run_parts_bench

pytestmark = pytest.mark.asyncio


def item(qid, question, sessions, answers):
    return {"question_id": qid, "question_type": "multi-session", "question": question,
            "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_sessions": [[{"role": "user", "content": text}] for text in sessions],
            "haystack_session_ids": [f"{qid}s{n}" for n in range(len(sessions))],
            "haystack_dates": ["2023/05/20 (Sat) 09:00"] * len(sessions),
            "answer_session_ids": answers, "answer": "n/a"}


SPLIT = item(
    "q1", "What did I decide about billing, and who was at the meeting?",
    ["We reverted the billing change after the invoices came out wrong.",
     "At the Thursday meeting were Priya, Tomas and the auditor from Lisbon.",
     "Unrelated: the bicycle needed a new chain.",
     "Unrelated: the choir moved to Tuesdays."],
    ["q1s0", "q1s1"])
WHOLE = item("q2", "Where can I buy salt and pepper?",
             ["The corner shop sells salt.", "Unrelated: the bicycle chain again."], ["q2s0"])


@pytest.fixture()
def dataset(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([SPLIT, WHOLE]), encoding="utf-8")
    return path


async def test_only_the_questions_the_rule_splits_are_compared(dataset):
    scored = await run_parts_bench(dataset, k=2)
    assert scored.questions == 2 and scored.split == 1, scored.record()


async def test_the_parted_column_is_measured_by_parted_recall_at_the_asked_k(dataset):
    """Both halves of this question have their evidence in the haystack, so
    parted recall at k=2 reaches both sessions and the counter is a real
    count rather than a zero nobody would notice.

    It does **not** show splitting beating the single query: on a haystack
    this small the one query reaches both sessions too, and whole_all
    equals parted_all here. Whether splitting wins is a question for a
    real corpus, which is what the command exists for."""
    scored = await run_parts_bench(dataset, k=2)
    assert scored.parted_all == 1 and scored.parted_any == 1, scored.record()
    assert scored.whole_all == scored.parted_all, (
        "if these ever differ on this fixture, the comparison claim above needs rewriting")


async def test_a_file_nothing_splits_says_so_instead_of_reporting_agreement(tmp_path):
    path = tmp_path / "whole.json"
    path.write_text(json.dumps([WHOLE]), encoding="utf-8")
    scored = await run_parts_bench(path, k=2)
    assert scored.split == 0
    assert "split none" in scored.text() and "changes nothing" in scored.text(), scored.text()


async def test_the_report_names_the_difference_in_questions(dataset):
    scored = await run_parts_bench(dataset, k=2)
    assert "question(s))" in scored.text() and f"{scored.parted_all - scored.whole_all:+d}" in scored.text()


async def test_the_limit_applies_before_anything_is_searched(dataset):
    scored = await run_parts_bench(dataset, limit=1, k=2)
    assert scored.questions == 1 and scored.split == 1
