"""Offline scoring regression and official-evaluator parity checks."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from qasper_structure.scoring import ARMS, score


def _annotation(answer: str, evidence: list[str], **changes: object) -> dict[str, object]:
    return {'answer': {'unanswerable': False, 'extractive_spans': [],
                       'free_form_answer': answer, 'yes_no': None,
                       'evidence': evidence, **changes}}


def _raw() -> dict[str, object]:
    return {
        'p1': {'abstract': 'An abstract.', 'full_text': [{'paragraphs': ['Evidence A.', 'Evidence B.']}], 'qas': [
            {'question_id': 'q1', 'answers': [_annotation('wrong', ['Evidence B.']),
                                            _annotation('red fox', ['Evidence A.'])]},
            {'question_id': 'q2', 'answers': [_annotation('', ['ignored'], unanswerable=True)]},
        ]},
        'p2': {'abstract': '', 'full_text': [{'paragraphs': ['Evidence C.']}], 'qas': [
            {'question_id': 'q3', 'answers': [_annotation('', ['Evidence C.', 'FLOAT SELECTED: figure 1'], yes_no=False)]},
        ]},
    }


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for identifier, paper, answer, evidence in (
            ('q1', 'p1', 'The RED fox!', ['Evidence A.']),
            ('q2', 'p1', 'Unanswerable', []),
            ('q3', 'p2', 'No', ['Evidence C.']),
        ):
            context = '\n'.join(evidence)
            rows.append({'id': identifier, 'paper_id': paper, 'arm': arm,
                         'completed': True, 'answer': answer, 'error': None,
                         'context_text': context, 'evidence_paragraphs': evidence,
                         'context_bytes': len(context.encode()),
                         'retrieval_ms': 1.0, 'route_ms': 0.0, 'fetch_ms': 1.0,
                         'rerank_ms': 2.0, 'generation_ms': 3.0, 'total_ms': 6.0,
                         'response': {'usage': {'prompt_tokens': 10, 'completion_tokens': 2,
                                                'cost': 0.01, 'completion_tokens_details': {'reasoning_tokens': 1}}}})
    return rows


def _write(tmp_path: Path, raw: dict[str, object], rows: list[dict[str, object]]) -> tuple[Path, Path]:
    gold = tmp_path / 'gold.json'
    observations = tmp_path / 'observations.jsonl'
    gold.write_text(json.dumps(raw))
    observations.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return gold, observations


def _arm(report: dict[str, object], name: str = 'scone_structure') -> dict[str, object]:
    return cast(dict[str, dict[str, object]], report['by_arm'])[name]


def test_annotation_maxima_and_figure_evidence_are_reported_separately(tmp_path: Path) -> None:
    report = score(*_write(tmp_path, _raw(), _rows()))
    arm = _arm(report)
    assert arm['answer_f1'] == 1
    assert arm['normalized_em'] == 1
    assert arm['evidence_f1'] == pytest.approx(8 / 9)
    assert arm['evidence_recall'] == pytest.approx(5 / 6)
    assert arm['text_evidence_f1'] == 1
    assert arm['text_evidence_recall'] == 1
    assert report['questions'] == 3
    assert report['papers'] == 2
    assert report['partial'] is False
    usage = cast(dict[str, object], arm['generation_usage'])
    assert usage['reported_observations'] == 3
    assert cast(dict[str, float], usage['totals']) == {
        'prompt_tokens': 30, 'completion_tokens': 6, 'cost': pytest.approx(.03),
        'completion_tokens_details.reasoning_tokens': 3,
    }


def test_failed_and_missing_rows_preserve_denominators_and_pairing(tmp_path: Path) -> None:
    rows = _rows()[1:]
    rows[2].update(completed=False, error='generation failed')
    paths = _write(tmp_path, _raw(), rows)
    with pytest.raises(ValueError, match='schedule'):
        score(*paths)
    report = score(*paths, allow_partial=True)
    assert report['partial'] is True
    assert report['missing'] == 1
    assert _arm(report, 'scone_flat')['answer_f1'] == pytest.approx(2 / 3)
    assert _arm(report)['answer_f1'] == pytest.approx(2 / 3)
    assert _arm(report)['failures'] == 1
    paired = cast(dict[str, dict[str, object]], report['paired'])
    comparison = paired['scone_structure_minus_llamaindex']
    assert comparison['n'] == 3
    metrics = cast(dict[str, dict[str, object]], comparison['metrics'])
    assert metrics['answer_f1']['delta'] == pytest.approx(-1 / 3)
    # The two questions in p1 always move together; p2 has no delta.
    assert metrics['answer_f1']['ci95'] == [-.5, 0.0]
    assert score(*paths, allow_partial=True) == report


@pytest.mark.parametrize('mutation', ['duplicate', 'foreign', 'paper', 'arm', 'error', 'coverage', 'paragraph', 'bytes', 'nan'])
def test_invalid_observations_rejected(tmp_path: Path, mutation: str) -> None:
    rows = _rows()
    if mutation == 'duplicate':
        rows.append(rows[0])
    elif mutation == 'foreign':
        rows[0]['id'] = 'foreign'
    elif mutation == 'paper':
        rows[0]['paper_id'] = 'p2'
    elif mutation == 'arm':
        rows[0]['arm'] = 'other'
    elif mutation == 'error':
        rows[0]['error'] = 'oops'
    elif mutation == 'coverage':
        rows[0].update(context_text='Evidence', context_bytes=8)
    elif mutation == 'paragraph':
        rows[0].update(evidence_paragraphs=['Evidence'], context_text='Evidence', context_bytes=8)
    elif mutation == 'bytes':
        rows[0]['context_bytes'] = 999
    else:
        rows[0]['total_ms'] = float('nan')
    with pytest.raises(ValueError):
        score(*_write(tmp_path, _raw(), rows), allow_partial=True)


def _official() -> tuple[ModuleType, Path]:
    directory = Path(os.environ.get('QASPER_REFERENCE_DIR', '/Users/msturman00/ProjectScone/bench-runs/qasper-structure-2026-09-24/raw'))
    script = directory / 'qasper_evaluator.py'
    if not script.is_file():
        pytest.skip('Set QASPER_REFERENCE_DIR to the downloaded unmodified official evaluator and QASPER test split')
    spec = importlib.util.spec_from_file_location('official_qasper_evaluator', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, directory


def test_synthetic_scores_match_unmodified_official_evaluator(tmp_path: Path) -> None:
    official, _ = _official()
    raw, rows = _raw(), _rows()
    report = score(*_write(tmp_path, raw, rows))
    predictions = {str(row['id']): {'answer': row['answer'], 'evidence': row['evidence_paragraphs']}
                   for row in rows if row['arm'] == 'scone_structure'}
    for text_only, field in ((False, 'evidence_f1'), (True, 'text_evidence_f1')):
        expected = official.evaluate(official.get_answers_and_evidence(raw, text_only), predictions)
        assert _arm(report)['answer_f1'] == expected['Answer F1']
        assert _arm(report)[field] == expected['Evidence F1']


def test_all_official_reference_annotations_match(tmp_path: Path) -> None:
    official, directory = _official()
    source = directory / 'qasper-test-v0.3.json'
    if not source.is_file():
        pytest.skip('Official QASPER test JSON is not available')
    raw = json.loads(source.read_text())
    gold = official.get_answers_and_evidence(raw, False)
    rows: list[dict[str, object]] = []
    predictions: dict[str, object] = {}
    template = _rows()[0]
    for paper_id, paper in raw.items():
        paragraphs = {paragraph for section in paper['full_text'] for paragraph in section['paragraphs']}
        paragraphs.add(paper['abstract'])
        for qa in paper['qas']:
            identifier = qa['question_id']
            reference = gold[identifier][-1]
            evidence = list(dict.fromkeys(text for text in reference['evidence'] if text in paragraphs))
            context = '\n'.join(evidence)
            predictions[identifier] = {'answer': reference['answer'], 'evidence': evidence}
            for arm in ARMS:
                rows.append({**template, 'id': identifier, 'paper_id': paper_id, 'arm': arm,
                             'answer': reference['answer'], 'evidence_paragraphs': evidence,
                             'context_text': context, 'context_bytes': len(context.encode())})
    report = score(*_write(tmp_path, raw, rows))
    assert report['questions'] == 1451
    assert report['papers'] == 416
    expected = official.evaluate(gold, predictions)
    assert _arm(report)['answer_f1'] == pytest.approx(expected['Answer F1'], abs=1e-14)
    assert _arm(report)['evidence_f1'] == pytest.approx(expected['Evidence F1'], abs=1e-14)


@pytest.mark.parametrize(('annotation', 'prediction', 'expected_f1', 'expected_em'), [
    (_annotation('free form', ['Evidence A.'], extractive_spans=['red', 'fox'], yes_no=True), 'red, fox', 1.0, 1.0),
    (_annotation('free form', ['Evidence A.'], yes_no=True), 'free form', 1.0, 1.0),
    (_annotation('', ['Evidence A.'], yes_no=True), 'Yes', 1.0, 1.0),
    (_annotation('ignored', ['Evidence A.'], unanswerable=True, extractive_spans=['red'], yes_no=True), 'Unanswerable', 1.0, 1.0),
    (_annotation('the', ['Evidence A.']), 'a', 0.0, 1.0),
    (_annotation('red red fox', ['Evidence A.']), 'red fox', .8, 0.0),
])
def test_official_answer_precedence_and_normalization(
    tmp_path: Path, annotation: dict[str, object], prediction: str, expected_f1: float, expected_em: float,
) -> None:
    raw: dict[str, object] = {'p1': {'abstract': '', 'full_text': [{'paragraphs': ['Evidence A.']}],
                                   'qas': [{'question_id': 'q1', 'answers': [annotation]}]}}
    rows = [{**row, 'answer': prediction} for row in _rows() if row['id'] == 'q1']
    report = score(*_write(tmp_path, raw, rows))
    assert _arm(report)['answer_f1'] == expected_f1
    assert _arm(report)['normalized_em'] == expected_em


def test_completely_missing_arm_still_has_zero_scores_and_no_latency(tmp_path: Path) -> None:
    paths = _write(tmp_path, _raw(), [])
    report = score(*paths, allow_partial=True)
    arm = _arm(report)
    assert report['missing'] == 9
    assert arm['answer_f1'] == 0
    assert arm['evidence_f1'] == 0
    assert arm['missing'] == 3
    timings = cast(dict[str, dict[str, object]], arm['timings'])
    assert timings['total_ms'] == {'count': 0, 'p50_ms': None, 'p95_ms': None}


def test_unanswerable_evidence_requires_empty_context_prediction(tmp_path: Path) -> None:
    rows = _rows()
    for row in rows:
        if row['id'] == 'q2':
            row.update(context_text='Evidence B.', context_bytes=11, evidence_paragraphs=['Evidence B.'])
    report = score(*_write(tmp_path, _raw(), rows))
    assert _arm(report)['evidence_f1'] == pytest.approx(5 / 9)
    assert _arm(report)['evidence_recall'] == .5


def test_usage_missing_is_visible_and_failed_usage_is_included(tmp_path: Path) -> None:
    rows = _rows()
    rows[3].pop('response')
    rows[4].update(completed=False, error='provider failed after billing')
    report = score(*_write(tmp_path, _raw(), rows))
    usage = cast(dict[str, object], _arm(report)['generation_usage'])
    assert usage['reported_observations'] == 2
    assert usage['missing_observations'] == 1
    assert cast(dict[str, float], usage['totals'])['prompt_tokens'] == 20
    assert cast(dict[str, int], usage['field_observation_counts'])['cost'] == 2


def test_gold_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    raw: dict[str, object] = {
        paper: {'abstract': '', 'full_text': [], 'qas': [
            {'question_id': 'same', 'answers': [_annotation('', [], unanswerable=True)]}
        ]} for paper in ('p1', 'p2')
    }
    with pytest.raises(ValueError, match='duplicate gold'):
        score(*_write(tmp_path, raw, []), allow_partial=True)


def test_original_captions_are_valid_retrieved_evidence(tmp_path: Path) -> None:
    raw = _raw()
    paper = cast(dict[str, object], raw['p1'])
    paper['figures_and_tables'] = [{'file': 'figure.png', 'caption': 'Original caption.'}]
    rows = _rows()
    rows[0].update(context_text='Original caption.', context_bytes=17,
                   evidence_paragraphs=['Original caption.'])
    report = score(*_write(tmp_path, raw, rows))
    assert report['observed'] == 9
