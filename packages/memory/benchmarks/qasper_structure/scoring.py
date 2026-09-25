"""Offline QASPER scoring. The caller must verify the frozen run's manifest first.

Answer/evidence F1 reproduce the official evaluator, including its empty-token
answer behavior and annotation precedence. Retrieval evidence consists only of
complete original paragraphs present verbatim in the packed generation context.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random
import re
import statistics
import string
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ARMS = ('scone_flat', 'scone_structure', 'llamaindex')
TIMINGS = ('retrieval_ms', 'route_ms', 'fetch_ms', 'rerank_ms', 'generation_ms', 'total_ms')
METRICS = ('answer_f1', 'normalized_em', 'evidence_f1', 'evidence_recall',
           'text_evidence_f1', 'text_evidence_recall')
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_REPLICATIONS = 2000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True, allow_inf_nan=False)


class _Answer(_StrictModel):
    unanswerable: bool
    extractive_spans: list[str]
    free_form_answer: str
    yes_no: bool | None
    evidence: list[str]


class _Annotation(_StrictModel):
    answer: _Answer


class _Question(_StrictModel):
    question_id: str
    answers: list[_Annotation] = Field(min_length=1)


class _Section(_StrictModel):
    paragraphs: list[str]


class _Figure(_StrictModel):
    caption: str


class _Paper(_StrictModel):
    abstract: str
    full_text: list[_Section]
    figures_and_tables: list[_Figure] = Field(default_factory=list)
    qas: list[_Question]


class Observation(_StrictModel):
    id: str
    paper_id: str
    arm: Literal['scone_flat', 'scone_structure', 'llamaindex']
    completed: bool
    answer: str
    error: str | None
    context_text: str
    evidence_paragraphs: list[str]
    retrieval_ms: float = Field(ge=0)
    route_ms: float = Field(ge=0)
    fetch_ms: float = Field(ge=0)
    rerank_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    context_bytes: int = Field(ge=0)
    response: dict[str, object] | None = None


class _Reference(_StrictModel):
    answer: str
    evidence: list[str]


def normalize_answer(answer: str) -> str:
    without_punctuation = ''.join(character for character in answer.lower() if character not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', without_punctuation).split())


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def paragraph_f1(prediction: list[str], reference: list[str]) -> float:
    if not prediction and not reference:
        return 1.0
    overlap = len(set(prediction) & set(reference))
    if not overlap:
        return 0.0
    precision, recall = overlap / len(prediction), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def paragraph_recall(prediction: list[str], reference: list[str]) -> float:
    # Empty-reference correctness is explicit: only empty predictions score 1.
    if not reference:
        return float(not prediction)
    return len(set(prediction) & set(reference)) / len(reference)


def _reference(annotation: _Annotation) -> _Reference:
    answer = annotation.answer
    if answer.unanswerable:
        return _Reference(answer='Unanswerable', evidence=[])
    if answer.extractive_spans:
        text = ', '.join(answer.extractive_spans)
    elif answer.free_form_answer:
        text = answer.free_form_answer
    elif answer.yes_no is not None:
        text = 'Yes' if answer.yes_no else 'No'
    else:
        raise ValueError('gold annotation does not contain an answer')
    return _Reference(answer=text, evidence=answer.evidence)


def _metrics(row: Observation | None, references: list[_Reference]) -> dict[str, float]:
    if row is None or not row.completed:
        return dict.fromkeys(METRICS, 0.0)
    evidence = row.evidence_paragraphs
    text_references = [[text for text in ref.evidence if 'FLOAT SELECTED' not in text] for ref in references]
    return {
        'answer_f1': max(token_f1(row.answer, ref.answer) for ref in references),
        'normalized_em': float(any(normalize_answer(row.answer) == normalize_answer(ref.answer) for ref in references)),
        'evidence_f1': max(paragraph_f1(evidence, ref.evidence) for ref in references),
        'evidence_recall': max(paragraph_recall(evidence, ref.evidence) for ref in references),
        'text_evidence_f1': max(paragraph_f1(evidence, ref) for ref in text_references),
        'text_evidence_recall': max(paragraph_recall(evidence, ref) for ref in text_references),
    }


def _latency(values: list[float]) -> dict[str, object]:
    return {'count': len(values), 'p50_ms': statistics.median(values) if values else None,
            'p95_ms': sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None}


def _numeric_leaves(value: object, prefix: str = '') -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError('generation usage keys must be strings')
        name = prefix + key
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, (float, int)):
            if not math.isfinite(item) or item < 0:
                raise ValueError('generation usage must be finite and nonnegative')
            result[name] = float(item)
        elif isinstance(item, dict):
            result.update(_numeric_leaves(item, name + '.'))
    return result


def _usage(rows: list[Observation]) -> dict[str, object]:
    totals: dict[str, float] = {}
    field_counts: dict[str, int] = {}
    reported = 0
    for row in rows:
        usage = row.response.get('usage') if row.response else None
        if not isinstance(usage, dict) or not usage:
            continue
        reported += 1
        for field, value in _numeric_leaves(usage).items():
            totals[field] = totals.get(field, 0.0) + value
            field_counts[field] = field_counts.get(field, 0) + 1
    return {'reported_observations': reported, 'missing_observations': len(rows) - reported,
            'totals': totals, 'field_observation_counts': field_counts,
            'policy': 'Sum reported numeric response.usage fields, including nested token/cost fields; missing usage is unknown, not zero. Includes failed observations.'}


def _validate_observation(row: Observation, paper_id: str, paragraphs: set[str]) -> None:
    if row.paper_id != paper_id:
        raise ValueError('observation has wrong paper_id: ' + row.id)
    if row.completed and row.error is not None:
        raise ValueError('completed observation contains an error: ' + row.id)
    if row.context_bytes != len(row.context_text.encode('utf-8')):
        raise ValueError('context_bytes differs from UTF-8 context length: ' + row.id)
    if len(row.evidence_paragraphs) != len(set(row.evidence_paragraphs)):
        raise ValueError('duplicate evidence paragraphs: ' + row.id)
    if any(not text or text not in paragraphs or text not in row.context_text for text in row.evidence_paragraphs):
        raise ValueError('evidence must be complete original paragraphs present in context: ' + row.id)


def _paired(
    question_papers: dict[str, str],
    values: dict[tuple[str, str], dict[str, float]],
    baseline: str,
) -> dict[str, object]:
    clusters: dict[str, list[str]] = {}
    for question, paper in question_papers.items():
        clusters.setdefault(paper, []).append(question)
    cluster_questions = list(clusters.values())
    sizes = [len(questions) for questions in cluster_questions]
    deltas = {metric: [values[(question, 'scone_structure')][metric] - values[(question, baseline)][metric]
                       for question in question_papers] for metric in METRICS}
    cluster_sums = {metric: [sum(values[(question, 'scone_structure')][metric] - values[(question, baseline)][metric]
                                for question in questions) for questions in cluster_questions] for metric in METRICS}
    rng = random.Random(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(BOOTSTRAP_REPLICATIONS):
        selection = [rng.randrange(len(sizes)) for _ in sizes]
        denominator = sum(sizes[index] for index in selection)
        for metric in METRICS:
            samples[metric].append(sum(cluster_sums[metric][index] for index in selection) / denominator)
    metrics: dict[str, object] = {}
    for metric in METRICS:
        ordered = sorted(samples[metric])
        metrics[metric] = {'delta': statistics.mean(deltas[metric]), 'ci95': [ordered[49], ordered[1949]],
                           'wins': sum(value > 0 for value in deltas[metric]),
                           'losses': sum(value < 0 for value in deltas[metric]),
                           'ties': sum(value == 0 for value in deltas[metric])}
    return {'n': len(question_papers), 'papers': len(sizes), 'metrics': metrics,
            'replications': BOOTSTRAP_REPLICATIONS, 'seed': BOOTSTRAP_SEED,
            'method': 'paired paper-cluster bootstrap; sample papers with replacement, preserve every question in each selected paper; question-weighted mean; nearest-rank 95% percentile interval'}


def score(raw_path: Path, observations_path: Path, *, allow_partial: bool = False) -> dict[str, object]:
    """Score a manifest-verified frozen run; raw answer annotations are read only here.

    Strict mode requires all question/arm rows, including explicit failures.
    Partial mode preserves the full dataset denominator with zero missing scores
    and labels results as progress only. It is never a system ranking.
    """
    papers = TypeAdapter(dict[str, _Paper]).validate_json(raw_path.read_bytes(), strict=True)
    question_papers: dict[str, str] = {}
    gold: dict[str, list[_Reference]] = {}
    paragraphs: dict[str, set[str]] = {}
    for paper_id, paper in papers.items():
        paragraphs[paper_id] = {text for section in paper.full_text for text in section.paragraphs}
        paragraphs[paper_id].add(paper.abstract)
        paragraphs[paper_id].update(figure.caption for figure in paper.figures_and_tables)
        for question in paper.qas:
            if question.question_id in gold:
                raise ValueError('duplicate gold question ID: ' + question.question_id)
            question_papers[question.question_id] = paper_id
            gold[question.question_id] = [_reference(annotation) for annotation in question.answers]
    if not gold:
        raise ValueError('empty gold dataset')
    observations: dict[tuple[str, str], Observation] = {}
    for line in observations_path.read_text().splitlines():
        if not line.strip():
            continue
        row = Observation.model_validate_json(line)
        pair = row.id, row.arm
        if pair in observations:
            raise ValueError('duplicate observation: ' + row.id + '/' + row.arm)
        if row.id not in gold:
            raise ValueError('foreign question ID: ' + row.id)
        _validate_observation(row, question_papers[row.id], paragraphs[question_papers[row.id]])
        observations[pair] = row
    planned = len(gold) * len(ARMS)
    missing = planned - len(observations)
    if missing and not allow_partial:
        raise ValueError(f'incomplete observation schedule: {missing} missing question/arm rows')
    values: dict[tuple[str, str], dict[str, float]] = {}
    scores: list[dict[str, object]] = []
    by_arm: dict[str, object] = {}
    for arm in ARMS:
        selected = [row for row in observations.values() if row.arm == arm]
        for identifier, references in gold.items():
            observation = observations.get((identifier, arm))
            metrics = _metrics(observation, references)
            values[(identifier, arm)] = metrics
            scores.append({'id': identifier, 'paper_id': question_papers[identifier], 'arm': arm,
                           'observed': observation is not None, 'completed': observation.completed if observation else False, **metrics})
        by_arm[arm] = {
            'n': len(gold), 'observed': len(selected), 'completed': sum(row.completed for row in selected),
            'failures': sum(not row.completed for row in selected), 'missing': len(gold) - len(selected),
            **{metric: statistics.mean(values[(identifier, arm)][metric] for identifier in gold) for metric in METRICS},
            'timings': {timing: _latency([getattr(row, timing) for row in selected]) for timing in TIMINGS},
            'generation_usage': _usage(selected),
        }
    return {
        'questions': len(gold), 'papers': len(set(question_papers.values())),
        'planned_observations': planned, 'observed': len(observations), 'missing': missing,
        'partial': bool(missing), 'by_arm': by_arm,
        'paired': {'scone_structure_minus_' + baseline: _paired(question_papers, values, baseline)
                   for baseline in ('scone_flat', 'llamaindex')},
        'scores': scores,
        'integrity_policy': 'Caller must verify dataset, schedule, source, manifest and frozen-run artifact integrity before scoring.',
        'failure_policy': 'Missing and failed observations score zero on every quality metric; every gold question remains in every arm and paired denominator. Observed timings and usage include failures.',
        'partial_policy': 'Incomplete results are progress only, never a system ranking; missing scores are zero with the full gold denominator.',
        'evidence_policy': 'Full-paragraph evidence: unique original paragraphs fully present verbatim in packed context. Split/clipped paragraphs receive no credit. Max over annotations independently per metric. Unanswerable annotations have empty evidence: empty prediction scores 1, nonempty prediction 0.',
        'figure_evidence_limitation': 'Primary evidence metrics include figure/table gold evidence (FLOAT SELECTED) that paragraph retrieval cannot return. Supplemental text_evidence metrics exclude these gold entries exactly as the official --text_evidence_only option does.',
        'answer_policy': 'Official token F1 with annotation precedence unanswerable, extractive spans joined with comma-space, free-form, yes/no. Maximum over annotations; empty normalized token answers score zero. Normalized exact match is supplemental.',
        'p95_method': 'nearest rank; observed rows including failures; missing rows have no timing',
    }
