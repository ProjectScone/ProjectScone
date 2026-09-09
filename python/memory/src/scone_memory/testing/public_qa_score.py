"""Offline label-based scoring; never imported by the inference runner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from .public_qa import Gold, answer_score
from .public_qa_run import MODELS, Observation, Prepared, digest, frozen_files, load_prepared, save, schedule


def coverage(prepared: Prepared, gold: Gold) -> dict[str, float | bool]:
    required = set(gold.support_documents)
    if not required:
        raise ValueError('missing source annotations')
    result: dict[str, float | bool] = {}
    for k in (5, 10):
        retrieved = set(prepared.ranked_document_ids[:k])
        result[f'document_recall_at_{k}'] = len(required & retrieved) / len(required)
        result[f'all_documents_at_{k}'] = required <= retrieved
    if gold.dataset == 'hotpotqa':
        matches = [any(s.document_id == doc and quote in s.text for s in prepared.sources)
                   for doc, quote in gold.support_sentences]
    else:
        matches = [any(s.document_id in required and answer in s.text
                       for s in prepared.sources for answer in gold.answers)]
    result['prepared_annotation_coverage'] = sum(matches) / len(matches) if matches else 0.
    result['prepared_all_annotations'] = bool(matches) and all(matches)
    return result


def latency(values: list[float]) -> dict[str, float | int | None]:
    """Median and nearest-rank p95; missing timings have an explicit count."""
    return {'count': len(values), 'p50_ms': statistics.median(values) if values else None,
            'p95_ms': sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None}


def validate_observations(prepared: list[Prepared], rows: list[Observation]) -> None:
    expected = [(r.id, r.model, r.block, r.block_first, r.request_sha256) for r in schedule(prepared)]
    actual = [(r.id, r.model, r.block, r.block_first, r.request_sha256) for r in rows]
    if actual != expected:
        raise ValueError('observations differ from the complete planned schedule')
    if any(r.status in ('running', 'unattempted') for r in rows):
        raise ValueError('run incomplete; do not rank unequal completed prefixes')
    if any(r.completed != (r.status == 'completed') for r in rows):
        raise ValueError('inconsistent completion status')


def score(root: Path) -> dict[str, object]:
    prepared = load_prepared(root)
    observations = [Observation.model_validate(r) for r in json.loads((root / 'observations.json').read_text())]
    validate_observations(prepared, observations)
    completion = json.loads((root / 'completion.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    if (not completion['terminal'] or not completion['code_and_inputs_unchanged']
        or digest(root / 'observations.json') != completion['observations_sha256']
        or digest(root / 'manifest.json') != completion['manifest_sha256']
        or manifest['files_sha256'] != frozen_files(root)):
        raise ValueError('completed experiment integrity verification failed')
    dataset = json.loads((root / 'dataset.json').read_text())
    if digest(root / 'gold.jsonl') != dataset['files_sha256']['gold.jsonl']:
        raise ValueError('gold labels changed after sampling')
    gold_rows = [Gold.model_validate_json(line) for line in (root / 'gold.jsonl').read_text().splitlines()]
    gold = {g.id: g for g in gold_rows}
    if len(gold) != len(gold_rows):
        raise ValueError('duplicate gold label ID')
    questions = {p.question.id: p for p in prepared}
    audits = {p.question.id: coverage(p, gold[p.question.id]) for p in prepared}
    scored: list[dict[str, object]] = []
    for row in observations:
        labels = gold[row.id]
        if labels.dataset != questions[row.id].question.dataset:
            raise ValueError('dataset label mismatch')
        scored.append({**row.model_dump(), 'dataset': labels.dataset,
            **answer_score(row.answer_text, labels.answers, labels.dataset, row.completed),
            **audits[row.id], 'abstained': row.completed and row.answer_text.strip() == 'INSUFFICIENT_EVIDENCE'})
    groups: list[dict[str, object]] = []
    for model in MODELS:
        for dataset_name in ('all', 'hotpotqa', 'squad'):
            selected = [r for r in scored if r['model'] == model and (dataset_name == 'all' or r['dataset'] == dataset_name)]
            n = len(selected)
            if n != (200 if dataset_name == 'all' else 100):
                raise ValueError('unexpected protocol denominator')
            metrics = ('em', 'f1', 'document_recall_at_5', 'document_recall_at_10', 'all_documents_at_5',
                       'all_documents_at_10', 'prepared_annotation_coverage', 'prepared_all_annotations')
            means = {key: sum(float(v) for r in selected if isinstance(v := r[key], (int, float))) / n for key in metrics}
            group: dict[str, object] = {'model': model, 'dataset': dataset_name, 'n': n, **means,
                'completed': sum(bool(r['completed']) for r in selected),
                'failures': sum(not r['completed'] for r in selected),
                'abstentions': sum(bool(r['abstained']) for r in selected)}
            for name, key in (('inference', 'total_ms'), ('first_token', 'first_token_ms')):
                group[name] = latency([float(v) for r in selected if isinstance(v := r[key], (int, float))])
            group['inference_block_first'] = latency([float(v) for r in selected
                if r['block_first'] and isinstance(v := r['total_ms'], (int, float))])
            group['inference_block_later'] = latency([float(v) for r in selected
                if not r['block_first'] and isinstance(v := r['total_ms'], (int, float))])
            group['context_preparation'] = latency([questions[str(r['id'])].prepare_ms for r in selected])
            group['estimated_sequential_cost'] = latency([float(v) + questions[str(r['id'])].prepare_ms for r in selected
                if isinstance(v := r['total_ms'], (int, float))])
            group['by_annotation_availability'] = [{
                'all_annotations_present': present,
                'n': len(subset := [r for r in selected if bool(r['prepared_all_annotations']) == present]),
                'em': sum(float(v) for r in subset if isinstance(v := r['em'], (int, float))) / len(subset) if subset else None,
            } for present in (True, False)]
            groups.append(group)
    return {'protocol': 'public-qa-v1', 'planned_observations': len(observations), 'groups': groups,
            'scores': scored, 'annotation_metrics_are_semantic_faithfulness': False,
            'p95_method': 'nearest rank; absent timings excluded with counts reported'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = score(args.directory)
    save(args.directory / 'scores.json', report)
    print(json.dumps(report['groups'], indent=2))


if __name__ == '__main__':
    main()
