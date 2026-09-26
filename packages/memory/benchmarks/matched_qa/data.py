"""Complete official development split export; this module is never imported by inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from scone_memory.testing.public_qa import (
    Dataset, Document, FrozenRecord, Gold, Question, _array, _document, _index,
    _mapping, _sentence, _squad, _text,
)

RAW_FILES = ('hotpot_dev_distractor_v1.json', 'squad_dev_v1.1.json')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_records(path: Path, records: Iterable[FrozenRecord]) -> None:
    path.write_text(''.join(record.model_dump_json() + '\n' for record in records))


def hotpot_with_audit(row: dict[str, object]) -> tuple[list[Document], Gold, list[dict[str, object]]]:
    """Retain document labels when official sentence indices are out of bounds."""
    identifier = 'hotpotqa:' + _text(row['_id'])
    titles: dict[str, tuple[Document, list[str]]] = {}
    for raw in _array(row['context']):
        pair = _array(raw)
        if len(pair) != 2:
            raise ValueError('invalid context pair')
        title = _text(pair[0])
        sentences = [_sentence(sentence) for sentence in _array(pair[1])]
        if title in titles:
            raise ValueError('duplicate context title')
        titles[title] = _document(title, '\n'.join(sentences)), sentences
    support_documents: set[str] = set()
    support_sentences: list[tuple[str, str]] = []
    issues: list[dict[str, object]] = []
    for raw in _array(row['supporting_facts']):
        pair = _array(raw)
        if len(pair) != 2 or type(pair[1]) is not int:
            raise ValueError('invalid support annotation')
        title, index = _text(pair[0]), pair[1]
        if title not in titles:
            raise ValueError(f'missing support title in official context: {identifier}: {title}')
        document, sentences = titles[title]
        support_documents.add(document.id)
        if not 0 <= index < len(sentences):
            issues.append({'id': identifier, 'kind': 'invalid_sentence_index', 'title': title,
                           'sentence_index': index, 'available_sentences': len(sentences),
                           'document_id': document.id, 'document_label_retained': True})
            continue
        support_sentences.append((document.id, sentences[index]))
    if not support_documents:
        raise ValueError('empty support annotations')
    gold = Gold(id=identifier, dataset='hotpotqa', answers=(_text(row['answer']),),
                support_documents=tuple(sorted(support_documents)),
                support_sentences=tuple(dict.fromkeys(support_sentences)))
    return [document for document, _ in titles.values()], gold, issues


def export(raw_dir: Path, previous_dataset: Path, output: Path) -> None:
    """Include every official question and supplied context; audit prior exposure."""
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    raw_hashes = {name: digest(raw_dir / name) for name in RAW_FILES}
    previous_manifest_path = previous_dataset / 'dataset.json'
    previous_manifest = (_mapping(json.loads(previous_manifest_path.read_bytes()))
                         if previous_manifest_path.exists() else {})
    expected = previous_manifest.get('raw_sha256')
    if expected is not None and raw_hashes != _mapping(expected):
        raise ValueError('raw source hashes differ from previous dataset')
    previously_seen: set[str] = set()
    previous_files: dict[str, object] = {}
    for name in ('queries.jsonl', 'reserved-queries.jsonl'):
        path = previous_dataset / name
        previous_questions = [Question.model_validate_json(line) for line in path.read_text().splitlines()]
        ids = [row.id for row in previous_questions]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate previous question ID')
        previously_seen.update(ids)
        previous_files[name] = {'sha256': digest(path), 'count': len(ids)}
    hotpot = [_mapping(row) for row in _array(json.loads((raw_dir / RAW_FILES[0]).read_bytes()))]
    squad: list[dict[str, object]] = []
    raw_squad = _mapping(json.loads((raw_dir / RAW_FILES[1]).read_bytes()))
    for raw_article in _array(raw_squad['data']):
        article = _mapping(raw_article)
        for raw_paragraph in _array(article['paragraphs']):
            paragraph = _mapping(raw_paragraph)
            for raw_question in _array(paragraph['qas']):
                squad.append({**_mapping(raw_question), 'title': article['title'], 'context': paragraph['context']})
    documents: dict[str, Document] = {}
    questions: list[Question] = []
    labels: list[Gold] = []
    annotation_issues: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    prior_counts: dict[str, int] = {}
    sources: tuple[tuple[Dataset, list[dict[str, object]], str], ...] = (
        ('hotpotqa', hotpot, '_id'), ('squad', squad, 'id'))
    for dataset, rows, key in sources:
        indexed = _index(rows, key)
        counts[dataset] = len(indexed)
        prior_counts[dataset] = sum(dataset + ':' + identifier in previously_seen for identifier in indexed)
        for identifier in sorted(indexed):
            row = indexed[identifier]
            if dataset == 'hotpotqa':
                contexts, gold, issues = hotpot_with_audit(row)
                annotation_issues.extend(issues)
            else:
                contexts, gold = _squad(row)
            for document in contexts:
                if document.id in documents and documents[document.id] != document:
                    raise ValueError('source digest collision')
                documents[document.id] = document
            questions.append(Question(id=gold.id, dataset=dataset, question=_text(row['question'])))
            labels.append(gold)
    output.mkdir(parents=True, exist_ok=True)
    write_records(output / 'corpus.jsonl', (documents[key] for key in sorted(documents)))
    write_records(output / 'questions.jsonl', questions)
    write_records(output / 'gold.jsonl', labels)
    manifest: dict[str, object] = {
        'protocol': 'scone-llamaindex-matched-full-v1',
        'selection': 'all official development questions, sorted by dataset then original ID; no exclusions or sampling',
        'raw_sha256': raw_hashes,
        'previous_dataset_sha256': digest(previous_manifest_path) if previous_manifest_path.exists() else None,
        'previous_files': previous_files, 'previously_seen_ids': sorted(previously_seen),
        'previously_seen_counts': prior_counts, 'previously_seen_count': sum(prior_counts.values()),
        'annotation_issues': annotation_issues,
        'annotation_issue_count': len(annotation_issues),
        'annotation_policy': 'Preserve every original answer and document support title; omit unavailable sentence text from sentence labels and audit invalid indices.',
        'dataset_counts': counts, 'questions': len(questions), 'documents': len(documents),
        'corpus_policy': 'all contexts of all questions, including HotpotQA distractors; pooled and deduplicated',
        'files': {name: digest(output / name) for name in ('corpus.jsonl', 'questions.jsonl', 'gold.jsonl')},
    }
    (output / 'dataset.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
