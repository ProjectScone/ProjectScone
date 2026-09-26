"""Offline fixtures exercise complete splits, failures, paired statistics and tamper checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from scone_memory.testing.public_qa import Document, Gold, Question, _mapping

from .data import RAW_FILES, digest, export
from .scoring import bootstrap, score


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def fixture_dataset(root: Path) -> tuple[Path, Path]:
    raw, prior = root / 'raw', root / 'prior'
    raw.mkdir(parents=True)
    prior.mkdir()
    hotpot: list[dict[str, object]] = []
    squad: list[dict[str, object]] = []
    for index in range(6):
        hotpot.append({'_id': str(index), 'question': f'Question {index}?', 'answer': 'Alpha',
                       'context': [[f'Support {index}', ['Alpha is the answer.']],
                                   [f'Distractor {index}', ['Unrelated material.']]],
                       'supporting_facts': [[f'Support {index}', 0]]})
        squad.append({'id': str(index), 'question': f'Where {index}?',
                      'answers': [{'text': 'Beta', 'answer_start': 0}]})
    save(raw / RAW_FILES[0], hotpot)
    save(raw / RAW_FILES[1], {'data': [{'title': 'Example', 'paragraphs': [{'context': 'Beta is here.', 'qas': squad}]}]})
    for filename, number in (('queries.jsonl', 0), ('reserved-queries.jsonl', 1)):
        rows = [Question(id=f'{name}:{number}', dataset=name, question='Prior?') for name in ('hotpotqa', 'squad')]
        (prior / filename).write_text(''.join(row.model_dump_json() + '\n' for row in rows))
    save(prior / 'dataset.json', {'raw_sha256': {name: digest(raw / name) for name in RAW_FILES}})
    return raw, prior


def test_export_is_complete_deterministic_and_keeps_distractors(tmp_path: Path) -> None:
    raw, prior = fixture_dataset(tmp_path)
    first, second = tmp_path / 'first', tmp_path / 'second'
    export(raw, prior, first)
    export(raw, prior, second)
    assert (first / 'dataset.json').read_bytes() == (second / 'dataset.json').read_bytes()
    questions = [Question.model_validate_json(line) for line in (first / 'questions.jsonl').read_text().splitlines()]
    assert len(questions) == 12
    assert {question.id for question in questions} == {f'{name}:{index}' for name in ('hotpotqa', 'squad') for index in range(6)}
    metadata = _mapping(json.loads((first / 'dataset.json').read_bytes()))
    assert metadata['previously_seen_count'] == 4
    assert metadata['dataset_counts'] == {'hotpotqa': 6, 'squad': 6}
    documents = [Document.model_validate_json(line) for line in (first / 'corpus.jsonl').read_text().splitlines()]
    assert len(documents) == 13
    assert sum(document.title.startswith('Distractor') for document in documents) == 6
    assert all('answer' not in json.loads(line) for line in (first / 'questions.jsonl').read_text().splitlines())


def test_export_refuses_changed_raw(tmp_path: Path) -> None:
    raw, prior = fixture_dataset(tmp_path)
    (raw / RAW_FILES[0]).write_text('[]')
    with pytest.raises(ValueError, match='source hashes'):
        export(raw, prior, tmp_path / 'changed')


def test_export_questions_and_contexts_do_not_depend_on_answers(tmp_path: Path) -> None:
    raw, prior = fixture_dataset(tmp_path)
    export(raw, prior, tmp_path / 'first')
    rows = cast(list[dict[str, object]], json.loads((raw / RAW_FILES[0]).read_bytes()))
    for row in rows:
        row['answer'] = 'Different answer'
    save(raw / RAW_FILES[0], rows)
    save(prior / 'dataset.json', {'raw_sha256': {name: digest(raw / name) for name in RAW_FILES}})
    export(raw, prior, tmp_path / 'second')
    assert (tmp_path / 'first/questions.jsonl').read_bytes() == (tmp_path / 'second/questions.jsonl').read_bytes()
    assert (tmp_path / 'first/corpus.jsonl').read_bytes() == (tmp_path / 'second/corpus.jsonl').read_bytes()


def finish(run: Path) -> None:
    save(run / 'completion.json', {'completed': True, 'code_and_inputs_unchanged': True,
         'artifact_hashes': {name: digest(run / name) for name in ('manifest.json', 'observations.jsonl')}})


def fixture_run(root: Path) -> tuple[Path, Path]:
    raw, prior = fixture_dataset(root)
    dataset, run = root / 'dataset', root / 'run'
    export(raw, prior, dataset)
    run.mkdir()
    gold = [Gold.model_validate_json(line) for line in (dataset / 'gold.jsonl').read_text().splitlines()]
    observations: list[dict[str, object]] = []
    for labels in gold:
        for arm in ('scone', 'llamaindex'):
            completed = not (labels.dataset == 'squad' and arm == 'scone')
            answer = labels.answers[0] if arm == 'scone' or labels.dataset == 'squad' else 'wrong'
            observations.append({'id': labels.id, 'arm': arm, 'completed': completed, 'answer': answer,
                'retrieved_ids': list(labels.support_documents), 'context_ids': list(labels.support_documents),
                'retrieval_ms': 1., 'rerank_ms': 2., 'generation_ms': 3., 'total_ms': 6.,
                'error': None if completed else 'deliberate failure'})
    save(run / 'manifest.json', {'input_hashes': {name: digest(dataset / name) for name in
         ('dataset.json', 'corpus.jsonl', 'questions.jsonl', 'gold.jsonl')},
         'scheduled': [{'id': row['id'], 'arm': row['arm']} for row in observations]})
    (run / 'observations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in observations))
    finish(run)
    return dataset, run


def test_scoring_counts_failure_and_pairs_all_questions(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    report = score(dataset, run)
    groups = cast(list[dict[str, object]], report['groups'])
    overall = next(group for group in groups if group['arm'] == 'scone' and group['dataset'] == 'all')
    assert overall['n'] == 12
    assert overall['failures'] == 6
    assert overall['em'] == .5
    assert overall['candidate_recall_at_32'] == 1.
    paired = cast(list[dict[str, object]], report['paired_em'])[0]
    assert paired['scone_wins'] == paired['scone_losses'] == 6
    assert paired['delta_scone_minus_llamaindex'] == 0
    assert paired['ci95'] == [-.5, .5]
    assert bootstrap([1., 0., -1.]) == bootstrap([1., 0., -1.])


def test_scoring_labels_provider_recovery(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    manifest = _mapping(json.loads((run / 'manifest.json').read_text()))
    manifest.update({'protocol': 'matched-qa-openrouter-recovery-v1',
        'recovery_provider': 'openrouter', 'recovery_jev_model': 'typesafe/jev-1.13-20260917',
        'parent_completion_sha256': 'parent-hash', 'recovery_ids': ['squad:0']})
    save(run / 'manifest.json', manifest)
    finish(run)
    report = score(dataset, run)
    assert report['protocol'] == 'matched-qa-openrouter-recovery-v1'
    provenance = _mapping(report['recovery'])
    assert provenance['provider'] == 'openrouter'
    assert provenance['parent_completion_sha256'] == 'parent-hash'
    assert provenance['question_count'] == 1


def test_scoring_preserves_nemotron_run_identity(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    manifest = _mapping(json.loads((run / 'manifest.json').read_text()))
    manifest.update({'protocol': 'matched-qa-nemotron-full-v1', 'embedding_profile': 'nemotron',
        'embedding_model': 'nvidia/nemotron-3-embed-1b:free', 'dimensions': 2048})
    save(run / 'manifest.json', manifest)
    finish(run)
    assert score(dataset, run)['protocol'] == 'matched-qa-nemotron-full-v1'


def test_rejects_tampered_artifact_and_input(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    with (run / 'observations.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='artifact hash'):
        score(dataset, run)
    finish(run)
    (dataset / 'gold.jsonl').write_text('')
    with pytest.raises(ValueError, match='input hash'):
        score(dataset, run)


def test_rejects_incomplete_schedule_even_with_updated_hash(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / 'observations.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    finish(run)
    with pytest.raises(ValueError, match='complete planned schedule'):
        score(dataset, run)


def test_rejects_false_completion_integrity(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / 'completion.json'
    metadata = _mapping(json.loads(path.read_bytes()))
    metadata['code_and_inputs_unchanged'] = False
    save(path, metadata)
    with pytest.raises(ValueError, match='inputs/code changed'):
        score(dataset, run)


def test_accepts_concurrent_completion_order(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / 'observations.jsonl'
    path.write_text('\n'.join(reversed(path.read_text().splitlines())) + '\n')
    finish(run)
    assert score(dataset, run)['partial'] is False


def test_partial_scores_mark_denominators_and_only_complete_pairs(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / 'observations.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:3]) + '\n')
    (run / 'completion.json').unlink()
    report = score(dataset, run, allow_partial=True)
    assert report['partial'] is True
    assert report['completion_fraction'] == 3 / 24
    pairs = cast(list[dict[str, object]], report['paired_em'])
    assert pairs[0]['n'] == 1
    with pytest.raises(FileNotFoundError):
        score(dataset, run)


def test_rejects_duplicate_pairs_even_in_partial_mode(tmp_path: Path) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / 'observations.jsonl'
    first = path.read_text().splitlines()[0]
    path.write_text(first + '\n' + first + '\n')
    (run / 'completion.json').unlink()
    with pytest.raises(ValueError, match='duplicate or unscheduled'):
        score(dataset, run, allow_partial=True)


def test_invalid_official_sentence_index_retains_question_answer_and_support(tmp_path: Path) -> None:
    raw, prior = fixture_dataset(tmp_path)
    rows = cast(list[dict[str, object]], json.loads((raw / RAW_FILES[0]).read_bytes()))
    rows[0]['supporting_facts'] = [['Support 0', 902]]
    save(raw / RAW_FILES[0], rows)
    save(prior / 'dataset.json', {'raw_sha256': {name: digest(raw / name) for name in RAW_FILES}})
    output = tmp_path / 'dataset'
    export(raw, prior, output)
    labels = [Gold.model_validate_json(line) for line in (output / 'gold.jsonl').read_text().splitlines()]
    label = next(row for row in labels if row.id == 'hotpotqa:0')
    documents = [Document.model_validate_json(line) for line in (output / 'corpus.jsonl').read_text().splitlines()]
    support = next(document for document in documents if document.title == 'Support 0')
    assert label.answers == ('Alpha',)
    assert label.support_documents == (support.id,)
    assert label.support_sentences == ()
    metadata = _mapping(json.loads((output / 'dataset.json').read_bytes()))
    assert metadata['questions'] == 12
    assert metadata['annotation_issue_count'] == 1
    issues = cast(list[dict[str, object]], metadata['annotation_issues'])
    assert issues[0]['sentence_index'] == 902
    assert issues[0]['document_label_retained'] is True


@pytest.mark.parametrize('filename', ['attempts.jsonl', 'judgments.jsonl', 'index.json'])
def test_rejects_tampered_extra_completion_artifact(tmp_path: Path, filename: str) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / filename
    path.write_text('{}\n')
    completion_path = run / 'completion.json'
    completion = _mapping(json.loads(completion_path.read_bytes()))
    artifacts = _mapping(completion['artifact_hashes'])
    artifacts[filename] = digest(path)
    save(completion_path, completion)
    assert score(dataset, run)['partial'] is False
    path.write_text('{"tampered": true}\n')
    with pytest.raises(ValueError, match='completed artifact hash mismatch: ' + filename):
        score(dataset, run)


@pytest.mark.parametrize('filename', ['../judgments.jsonl', '/tmp/judgments.jsonl', 'unknown.json'])
def test_rejects_unsafe_or_unknown_completion_artifact(tmp_path: Path, filename: str) -> None:
    dataset, run = fixture_run(tmp_path)
    path = run / 'completion.json'
    completion = _mapping(json.loads(path.read_bytes()))
    _mapping(completion['artifact_hashes'])[filename] = 'not-a-hash'
    save(path, completion)
    with pytest.raises(ValueError, match='unsupported artifact filename'):
        score(dataset, run)
