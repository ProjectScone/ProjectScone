"""Recorded feedback replayed into ranking: paraphrases of judged questions gain, others hold.

The weight was chosen on the replay that judges half a; half b checks it.
The numbers are in benchmarks/feedback-replay-v1.results.md, and this test
holds them: a change that makes the prior cost unrelated questions, or stop
helping, or stop needing corroboration, fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scone_memory.bench.feedback_replay import CHOSEN, MAX_JUDGEMENTS, load_subjects, measure, replay, report
from scone_memory.core.errors import InvalidInput

SUBJECTS = Path(__file__).resolve().parents[2] / "benchmarks" / "feedback-replay-v1.json"


async def test_the_chosen_weight_lifts_judged_paraphrases_and_leaves_unrelated_questions_within_a_hundredth():
    recorded = {}
    for judge in ("kind", "strict"):
        runs = await measure(SUBJECTS, judge=judge, weights=(CHOSEN, 0.0002))
        recorded[judge] = [run.feedback_events for run in runs]
        for run in runs:
            assert run.mrr(CHOSEN, "judged") > run.mrr(0.0, "judged"), (judge, run.judged_half)
            assert abs(run.mrr(CHOSEN, "unrelated") - run.mrr(0.0, "unrelated")) <= 0.01, (judge, run.judged_half)
            assert run.fell[CHOSEN] == [] and run.held[CHOSEN] == 0, (judge, run.judged_half)
            assert run.mrr(0.0002, "unrelated") < run.mrr(0.0, "unrelated") - 0.01, \
                "twice the weight costs unrelated questions, as the results say"
        assert "unrelated" in report(runs)
    assert recorded["kind"] == [24, 24], "two useful judgements per judged subject"
    assert all(strict > kind for strict, kind in zip(recorded["strict"], recorded["kind"])), "and some against"


async def test_judgements_piled_on_a_passage_cost_unrelated_questions_no_more_than_two_do():
    """Six judgements per subject, from six recorded questions: the term does not grow past corroboration."""
    for judge in ("kind", "strict"):
        for run in await measure(SUBJECTS, judge=judge, judgements=6, weights=(CHOSEN,)):
            assert run.feedback_events >= 6 * 12 and run.held[CHOSEN] > 0, (judge, run.judged_half)
            assert run.mrr(CHOSEN, "judged") > run.mrr(0.0, "judged"), (judge, run.judged_half)
            assert run.mrr(CHOSEN, "unrelated") >= run.mrr(0.0, "unrelated") - 0.01, (judge, run.judged_half)
            assert run.fell[CHOSEN] == [], (judge, run.judged_half)


async def test_one_judgement_per_subject_moves_nothing():
    for run in await measure(SUBJECTS, judge="kind", judgements=1, weights=(CHOSEN, 0.0005)):
        for weight in (CHOSEN, 0.0005):
            assert run.ranks[weight] == run.ranks[0.0] and run.rose[weight] == [] and run.fell[weight] == []


def test_the_halves_split_every_sibling_pair():
    subjects = load_subjects(SUBJECTS)["subjects"]
    pairs: dict[str, set[str]] = {}
    for subject in subjects:
        if subject["siblings"]:
            pairs.setdefault(subject["siblings"], set()).add(subject["half"])
    assert len(subjects) == 24 and len(pairs) == 8 and all(halves == {"a", "b"} for halves in pairs.values())


async def test_a_subjects_file_of_another_schema_or_a_bad_replay_is_refused(tmp_path):
    other = tmp_path / "subjects.json"
    other.write_text(json.dumps({**json.loads(SUBJECTS.read_text()), "schema_version": 2}))
    with pytest.raises(InvalidInput):
        load_subjects(other)
    data = json.loads(SUBJECTS.read_text())
    data["subjects"][0]["gold"] = "A passage nobody stored."
    other.write_text(json.dumps(data))
    with pytest.raises(InvalidInput, match="answering passage"):
        load_subjects(other)
    with pytest.raises(InvalidInput):
        await replay(SUBJECTS, judged_half="c")
    for judgements in (0, MAX_JUDGEMENTS + 1):
        with pytest.raises(InvalidInput, match="judgements"):
            await replay(SUBJECTS, judged_half="a", judgements=judgements)


async def test_a_judge_who_was_not_shown_the_answer_marks_nothing_useful(tmp_path):
    lamps = [f"The blue lamp in room {number} is switched off at night." for number in range(1, 7)]
    subjects = tmp_path / "subjects.json"
    subjects.write_text(json.dumps({"schema_version": 1, "passages": [*lamps, "Zebra quartz melody."], "subjects": [
        {"id": "unseen", "half": "a", "siblings": None, "gold": "Zebra quartz melody.",
         "asked": ["blue lamp switched off at night", "is the blue lamp switched off at night"], "paraphrase": "zebra"},
        {"id": "lamp", "half": "b", "siblings": None, "gold": lamps[0],
         "asked": ["blue lamp room 1", "room 1 blue lamp"], "paraphrase": "lamp in room 1"}]}))
    kind = await replay(subjects, judged_half="a", judge="kind", weights=())
    strict = await replay(subjects, judged_half="a", judge="strict", weights=())
    assert kind.feedback_events == 0
    assert strict.feedback_events == 10, "every passage shown, on both askings, is marked against"
