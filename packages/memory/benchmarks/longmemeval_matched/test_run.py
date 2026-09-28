from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from .prompts import Session, Turn, history, is_abstention, judge_prompt, judged_correct, reader_prompt
from .run import Item, arm_sessions, load_rankings, parse_variant, report, wilson


def _raw(question_id: str = 'q1', question_type: str = 'multi-session') -> dict[str, object]:
    return {
        'question_id': question_id, 'question_type': question_type, 'question': 'Where?', 'answer': 'Paris',
        'question_date': '2023/06/01 (Thu) 10:00',
        'haystack_session_ids': ['s1', 's2', 's3'],
        'haystack_dates': ['2023/05/03 (Wed) 09:00', '2023/05/01 (Mon) 09:00', '2023/05/02 (Tue) 09:00'],
        'haystack_sessions': [[{'role': 'user', 'content': 'one'}], [{'role': 'user', 'content': 'two'}],
                              [{'role': 'user', 'content': 'three'}]],
        'answer_session_ids': ['s3'],
    }


def test_history_orders_sessions_by_date_whatever_order_they_arrive_in() -> None:
    sessions = [Session('late', '2023/05/03 (Wed) 09:00', (Turn('user', 'late'),)),
                Session('early', '2023/05/01 (Mon) 09:00', (Turn('user', ' early '),))]
    text = history(sessions)
    assert text.index('early') < text.index('late')
    assert '### Session 1:\nSession Date: 2023/05/01 (Mon) 09:00\nSession Content:\n\n\nuser: early\n' in text


def test_reader_prompt_refuses_an_empty_context() -> None:
    with pytest.raises(ValueError):
        reader_prompt([], '2023/06/01', 'Where?', cot=False)


def test_judge_uses_the_abstention_prompt_for_abs_ids_and_rejects_unknown_types() -> None:
    assert is_abstention('abc_abs') and not is_abstention('abc')
    assert 'unanswerable question' in judge_prompt('multi-session', 'abc_abs', 'Q', 'A', 'R')
    assert 'off-by-one' in judge_prompt('temporal-reasoning', 'abc', 'Q', 'A', 'R')
    with pytest.raises(ValueError):
        judge_prompt('not-a-type', 'abc', 'Q', 'A', 'R')


def test_judged_correct_keeps_upstreams_substring_rule() -> None:
    assert judged_correct('Yes.') and judged_correct('yes') and not judged_correct('No')


def test_arms_pick_every_session_the_evidence_or_the_ranked_prefix() -> None:
    item = Item(_raw())
    rankings: dict[str, dict[str, object]] = {'q1': {'scone': ['s2', 's3', 's1'], 'llamaindex': ['s1']}}
    assert [s.session_id for s in arm_sessions(item, 'full', rankings)] == ['s1', 's2', 's3']
    assert [s.session_id for s in arm_sessions(item, 'oracle', rankings)] == ['s3']
    assert [s.session_id for s in arm_sessions(item, 'scone@2', rankings)] == ['s2', 's3']
    assert [s.session_id for s in arm_sessions(item, 'llamaindex@5', rankings)] == ['s1']


def test_wilson_interval_brackets_the_rate_and_handles_zero() -> None:
    low, high = wilson(50, 100)
    assert low < 0.5 < high and round(high - low, 3) == pytest.approx(0.192, abs=0.002)
    assert wilson(0, 0) == (0.0, 0.0)


def _arms(result: dict[str, object]) -> dict[str, Any]:
    return cast(dict[str, Any], result['arms'])


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_report_scores_failures_missing_and_judge_errors_as_incorrect(tmp_path: Path) -> None:
    items = [Item(_raw('q1')), Item(_raw('q2_abs')), Item(_raw('q3')), Item(_raw('q4'))]
    _write(tmp_path / 'answers.jsonl', [
        {'arm': 'full', 'question_id': 'q1', 'completed': True, 'text': 'Paris', 'usage': {'prompt_tokens': 10}},
        {'arm': 'full', 'question_id': 'q2_abs', 'completed': True, 'text': 'not said', 'usage': {'prompt_tokens': 12}},
        {'arm': 'full', 'question_id': 'q3', 'completed': False, 'text': '', 'usage': None},
        # q4 has no answer at all
    ])
    _write(tmp_path / 'judgments.jsonl', [
        {'arm': 'full', 'question_id': 'q1', 'correct': True, 'judge_error': None},
        {'arm': 'full', 'question_id': 'q2_abs', 'correct': True, 'judge_error': 'http_500'},
    ])
    arm = _arms(report(items, ['full'], tmp_path))['full']
    assert arm['correct'] == 1 and arm['accuracy'] == 0.25
    assert arm['failed_generations'] == 1 and arm['missing_answers'] == 1 and arm['unjudged'] == 0
    assert arm['by_type']['abstention'] == {'n': 1, 'accuracy': 0.0}


def test_report_measures_evidence_coverage_at_the_arms_depth(tmp_path: Path) -> None:
    items = [Item(_raw('q1')), Item(_raw('q2'))]
    _write(tmp_path / 'rankings.jsonl', [{'question_id': 'q1', 'scone': ['s3', 's1'], 'llamaindex': []},
                                         {'question_id': 'q2', 'scone': ['s1', 's3'], 'llamaindex': []}])
    result = _arms(report(items, ['scone@1', 'scone@2'], tmp_path))
    assert result['scone@1']['all_evidence_in_context'] == 0.5
    assert result['scone@2']['all_evidence_in_context'] == 1.0


def test_variants_need_a_new_name_and_only_scone_settings() -> None:
    assert parse_variant('rr:SCONE_RERANK_LIMIT=40;SCONE_QUESTION_LANE=1') == (
        'rr', {'SCONE_RERANK_LIMIT': '40', 'SCONE_QUESTION_LANE': '1'})
    for bad in ('scone:SCONE_X=1', 'rr:', 'rr:PATH=/tmp', '2x:SCONE_X=1'):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_variant_rankings_merge_under_their_name(tmp_path: Path) -> None:
    _write(tmp_path / 'rankings.jsonl', [{'question_id': 'q1', 'scone': ['s1'], 'llamaindex': ['s2']}])
    _write(tmp_path / 'rankings-rr.jsonl', [{'question_id': 'q1', 'rr': ['s3', 's1']}])
    merged = load_rankings(tmp_path)
    assert merged['q1']['scone'] == ['s1'] and merged['q1']['rr'] == ['s3', 's1']
    item = Item(_raw('q1'))
    assert [s.session_id for s in arm_sessions(item, 'rr@1', merged)] == ['s3']


def test_a_jsonl_dataset_keeps_offsets_and_reads_histories_on_demand(tmp_path: Path) -> None:
    from .run import load, to_jsonl

    source = tmp_path / 'data.json'
    source.write_text(json.dumps([_raw('q1'), _raw('q2', 'temporal-reasoning')]), encoding='utf-8')
    target = tmp_path / 'data.jsonl'
    assert to_jsonl(source, target) == 2 and target.read_text(encoding='utf-8').count('\n') == 2
    lazy = {i.question_id: i for i in load(target, 0, 42)}
    eager = {i.question_id: i for i in load(source, 0, 42)}
    assert set(lazy) == {'q1', 'q2'}
    assert lazy['q2'].question_type == 'temporal-reasoning' and lazy['q2']._raw is None
    assert lazy['q1'].sessions == eager['q1'].sessions and lazy['q2'].raw == eager['q2'].raw
