"""Offline matched-arm scoring, including failures and artifact integrity checks."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from scone_memory.testing.public_qa import Document, Gold, Question, _array, _mapping, answer_score

ARMS = ('scone', 'llamaindex')
TIMINGS = ('retrieval_ms', 'rerank_ms', 'generation_ms', 'total_ms')


class Observation(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True, allow_inf_nan=False)
    id: str
    arm: Literal['scone', 'llamaindex']
    completed: bool
    answer: str
    retrieved_ids: list[str]
    context_ids: list[str]
    retrieval_ms: float = Field(ge=0)
    rerank_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    error: str | None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency(values: list[float]) -> dict[str, float | int | None]:
    return {'count': len(values), 'p50_ms': statistics.median(values) if values else None,
            'p95_ms': sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None}


def bootstrap(deltas: list[float]) -> dict[str, object]:
    rng = random.Random(20260924)
    n = len(deltas)
    if not n:
        raise ValueError('empty paired comparison')
    samples = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(2000))
    return {'delta_scone_minus_llamaindex': statistics.mean(deltas),
            'ci95': [samples[49], samples[1949]], 'replications': 2000, 'seed': 20260924,
            'method': 'paired question bootstrap; nearest-rank percentile interval'}


def _integrity(dataset: Path, run: Path, allow_partial: bool) -> dict[str, object]:
    manifest = _mapping(json.loads((run / 'manifest.json').read_bytes()))
    completion_path = run / 'completion.json'
    if completion_path.exists() or not allow_partial:
        completion = _mapping(json.loads(completion_path.read_bytes()))
        if completion.get('completed') is not True or completion.get('code_and_inputs_unchanged') is not True:
            raise ValueError('run is incomplete or inputs/code changed')
        artifacts = _mapping(completion['artifact_hashes'])
        required_artifacts = {'manifest.json', 'observations.jsonl'}
        allowed_artifacts = required_artifacts | {'attempts.jsonl', 'judgments.jsonl', 'index.json', 'embedding-calls.jsonl'}
        if not required_artifacts <= artifacts.keys():
            raise ValueError('completion lacks required artifact hashes')
        if not artifacts.keys() <= allowed_artifacts:
            raise ValueError('completion contains unsupported artifact filename')
        for name, expected in artifacts.items():
            if expected != digest(run / name):
                raise ValueError('completed artifact hash mismatch: ' + name)
    inputs = _mapping(manifest['input_hashes'])
    for name in ('dataset.json', 'corpus.jsonl', 'questions.jsonl', 'gold.jsonl'):
        if inputs.get(name) != digest(dataset / name):
            raise ValueError('input hash mismatch: ' + name)
    metadata = _mapping(json.loads((dataset / 'dataset.json').read_bytes()))
    files = _mapping(metadata['files'])
    for name in ('corpus.jsonl', 'questions.jsonl', 'gold.jsonl'):
        if files.get(name) != digest(dataset / name):
            raise ValueError('sample hash mismatch: ' + name)
    return manifest


def score(dataset: Path, run: Path, *, allow_partial: bool = False) -> dict[str, object]:
    manifest = _integrity(dataset, run, allow_partial)
    questions = [Question.model_validate_json(line) for line in (dataset / 'questions.jsonl').read_text().splitlines()]
    labels = [Gold.model_validate_json(line) for line in (dataset / 'gold.jsonl').read_text().splitlines()]
    documents = [Document.model_validate_json(line) for line in (dataset / 'corpus.jsonl').read_text().splitlines()]
    gold = {row.id: row for row in labels}
    question_map = {row.id: row for row in questions}
    document_ids = {row.id for row in documents}
    if not questions or len(question_map) != len(questions) or len(gold) != len(labels):
        raise ValueError('empty dataset or duplicate question/gold IDs')
    if set(gold) != set(question_map) or len(document_ids) != len(documents):
        raise ValueError('gold/question ID mismatch or duplicate document IDs')
    for identifier, label in gold.items():
        if label.dataset != question_map[identifier].dataset:
            raise ValueError('gold/question dataset mismatch')
        if not label.support_documents or not set(label.support_documents) <= document_ids:
            raise ValueError('gold support outside corpus')
    scheduled = [_mapping(row) for row in _array(manifest['scheduled'])]
    planned = [(row.get('id'), row.get('arm')) for row in scheduled]
    expected = {(question.id, arm) for question in questions for arm in ARMS}
    if len(planned) != len(expected) or any(pair not in expected for pair in planned) or len(set(planned)) != len(expected):
        raise ValueError('schedule must contain every question/arm exactly once')
    observation_path = run / 'observations.jsonl'
    observations = ([Observation.model_validate_json(line) for line in observation_path.read_text().splitlines()]
                    if observation_path.exists() else [])
    observed_pairs = {(row.id, row.arm) for row in observations}
    if len(observed_pairs) != len(observations) or not observed_pairs <= expected:
        raise ValueError('duplicate or unscheduled observations')
    if not allow_partial and observed_pairs != expected:
        raise ValueError('observations differ from the complete planned schedule')
    partial = observed_pairs != expected or not (run / 'completion.json').exists()
    scored: list[dict[str, object]] = []
    metrics_by_pair: dict[tuple[str, str], dict[str, float]] = {}
    for row in observations:
        if row.completed and row.error is not None:
            raise ValueError('completed observation contains an error')
        if len(row.context_ids) > 5 or len(row.retrieved_ids) > 32:
            raise ValueError('observation exceeds retrieval/context budget')
        if not set(row.context_ids + row.retrieved_ids) <= document_ids:
            raise ValueError('observation references unknown documents')
        if not set(row.context_ids) <= set(row.retrieved_ids):
            raise ValueError('context is outside retrieved candidates')
        label = gold[row.id]
        required = set(label.support_documents)
        context, candidates = set(row.context_ids[:5]), set(row.retrieved_ids[:32])
        metrics = answer_score(row.answer, label.answers, label.dataset, row.completed)
        metrics.update({
            'context_recall_at_5': len(required & context) / len(required),
            'candidate_recall_at_32': len(required & candidates) / len(required),
            'all_support_at_5': float(required <= context),
            'all_support_at_32': float(required <= candidates),
        })
        metrics_by_pair[(row.id, row.arm)] = metrics
        scored.append({**row.model_dump(), 'dataset': label.dataset, **metrics})
    groups: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    metric_names = ('em', 'f1', 'context_recall_at_5', 'candidate_recall_at_32', 'all_support_at_5', 'all_support_at_32')
    for dataset_name in ('all', 'hotpotqa', 'squad'):
        identifiers = [question.id for question in questions if dataset_name == 'all' or question.dataset == dataset_name]
        if not identifiers:
            continue
        identifier_set = set(identifiers)
        for arm in ARMS:
            selected = [row for row in observations if row.arm == arm and row.id in identifier_set]
            if not selected:
                continue
            arm_identifiers = [row.id for row in selected]
            group: dict[str, object] = {
                'arm': arm, 'dataset': dataset_name, 'n': len(selected),
                'planned': len(identifiers), 'observation_fraction': len(selected) / len(identifiers),
                'completed': sum(row.completed for row in selected),
                'failures': sum(not row.completed for row in selected),
                **{metric: statistics.mean(metrics_by_pair[(identifier, arm)][metric] for identifier in arm_identifiers)
                   for metric in metric_names},
            }
            group['timings'] = {timing: latency([getattr(row, timing) for row in selected]) for timing in TIMINGS}
            groups.append(group)
        paired_identifiers = [identifier for identifier in identifiers
                              if all((identifier, arm) in observed_pairs for arm in ARMS)]
        if not paired_identifiers:
            continue
        deltas = [metrics_by_pair[(identifier, 'scone')]['em'] - metrics_by_pair[(identifier, 'llamaindex')]['em']
                  for identifier in paired_identifiers]
        pairs.append({'dataset': dataset_name, 'n': len(paired_identifiers), 'planned': len(identifiers),
                      'scone_wins': sum(delta > 0 for delta in deltas),
                      'scone_losses': sum(delta < 0 for delta in deltas),
                      'ties': sum(delta == 0 for delta in deltas), **bootstrap(deltas)})
    return {'protocol': 'scone-llamaindex-matched-full-v1', 'planned_observations': len(planned),
            'observed': len(observations), 'partial': partial,
            'completion_fraction': len(observations) / len(planned),
            'partial_policy': 'Progress only: observed-arm denominators and fully observed pairs; incomplete results must not rank systems.',
            'groups': groups, 'paired_em': pairs, 'scores': scored,
            'failure_policy': 'failed answers count as zero; retrieval coverage uses observed IDs independently of generation success; timings include failures',
            'p95_method': 'nearest rank', 'retrieval_unit': 'source document ID; duplicate chunks deduplicated',
            'caveat': 'Pooled official development splits; topics are not independent. Question bootstrap intervals are descriptive and may understate uncertainty. No full-benchmark or universal superiority claim.'}
