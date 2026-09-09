import asyncio
import json

import httpx
import pytest

from scone_memory.testing.public_qa import Gold, Question, benchmark_messages
from scone_memory.testing.public_qa_run import (
    MODELS, Observation, Prepared, Source, generate, load_prepared, request_digest, save, schedule,
)
from scone_memory.testing.public_qa_score import coverage, latency, validate_observations


def prepared(index=0):
    question = Question(id=f'squad:{index}', dataset='squad', question=f'Original question {index}?')
    request = benchmark_messages(question)
    return Prepared(question=question, request=request, request_sha256=request_digest(request),
        receipt={}, sources=(), ranked_document_ids=(), prepare_ms=1., recall_ms=1., recall_status='completed')


def test_schedule_rotates_models_and_has_one_response_per_question_model():
    rows = schedule([prepared(i) for i in range(200)])
    assert len(rows) == len({(r.id, r.model) for r in rows}) == 600
    assert [r.model for r in rows if r.block_first][:6] == [*MODELS, MODELS[1], MODELS[2], MODELS[0]]
    assert all(sum(r.model == m for r in rows) == 200 for m in MODELS)


def test_coverage_uses_ranked_chunks_without_refilling_and_checks_source_identity():
    row = prepared().model_copy(update={
        'ranked_document_ids': ('a', 'a', 'a', 'a', 'a', 'b'),
        'sources': (Source(document_id='unrelated', text='exact quote'),),
    })
    gold = Gold(id=row.question.id, dataset='hotpotqa', answers=('yes',), support_documents=('a', 'b'),
                support_sentences=(('b', 'exact quote'),))
    result = coverage(row, gold)
    assert result['document_recall_at_5'] == .5
    assert result['document_recall_at_10'] == 1
    assert result['prepared_annotation_coverage'] == 0


def test_partial_and_retried_observations_cannot_be_ranked():
    requests = [prepared()]
    rows = schedule(requests)
    with pytest.raises(ValueError, match='incomplete'):
        validate_observations(requests, rows)
    with pytest.raises(ValueError, match='schedule'):
        validate_observations(requests, rows[:-1])
    terminal = [r.model_copy(update={'status': 'timeout'}) for r in rows]
    validate_observations(requests, terminal)
    with pytest.raises(ValueError, match='schedule'):
        validate_observations(requests, [*terminal, terminal[0]])
    assert latency([1., 2., 3., 4.]) == {'count': 4, 'p50_ms': 2.5, 'p95_ms': 4.}


async def test_resume_keeps_interrupted_attempt_as_failure_without_retry_or_gold_access(tmp_path, monkeypatch):
    import scone_memory.testing.public_qa_run as runner
    import scone_memory.testing.generation_ablation as capture
    rows = [prepared(i) for i in range(200)]
    (tmp_path / 'queries.jsonl').write_text(''.join(r.question.model_dump_json() + '\n' for r in rows))
    (tmp_path / 'prepared.jsonl').write_text(''.join(r.model_dump_json() + '\n' for r in rows))
    (tmp_path / 'corpus.jsonl').write_text('')
    save(tmp_path / 'dataset.json', {'files_sha256': {name: runner.digest(tmp_path / name) for name in ('corpus.jsonl', 'queries.jsonl')}})
    save(tmp_path / 'preparation.json', {'code_sha256': {'test': 'frozen'}, 'prepared_sha256': runner.digest(tmp_path / 'prepared.jsonl')})
    # Deliberately inaccessible as a regular file; inference must never open labels.
    (tmp_path / 'gold.jsonl').mkdir()
    monkeypatch.setattr(runner, 'code_hashes', lambda: {'test': 'frozen'})
    operations = []
    def handler(request):
        operations.append(request.url.path)
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': m, 'digest': m} for m in MODELS]})
        if request.url.path == '/api/show':
            return httpx.Response(200, json={'parameters': 'num_ctx 8192\ntemperature 0'})
        return httpx.Response(200, json={'models': []})
    client = httpx.AsyncClient
    monkeypatch.setattr(runner.httpx, 'AsyncClient', lambda **kw: client(**{**kw, 'transport': httpx.MockTransport(handler)}))
    calls = []
    async def response(provider, messages, *, timeout):
        calls.append(messages[-1]['content'])
        await provider.aclose()
        if len(calls) == 1:
            raise asyncio.CancelledError()
        return {'status': 'completed', 'completed': True, 'answer_text': 'raw reply', 'total_ms': 1., 'first_token_ms': .5}
    monkeypatch.setattr(capture, 'capture_public_reply', response)
    with pytest.raises(asyncio.CancelledError):
        await generate(tmp_path, 'http://127.0.0.1:11434')
    await generate(tmp_path, 'http://127.0.0.1:11434')
    observations = [Observation.model_validate(r) for r in json.loads((tmp_path / 'observations.json').read_text())]
    assert len(calls) == 600
    assert observations[0].status == 'interrupted' and not observations[0].completed
    assert sum(r.completed for r in observations) == 599
    assert operations.count('/api/generate') == 62, 'unload twice per block, including resumed first block'
    assert all(calls.count(r.question.question) == 3 for r in rows)
    validate_observations(rows, observations)
    assert json.loads((tmp_path / 'completion.json').read_text())['terminal']


def test_changed_original_question_is_rejected(tmp_path):
    from scone_memory.testing.public_qa_run import digest
    row = prepared()
    (tmp_path / 'queries.jsonl').write_text(row.question.model_dump_json() + '\n')
    (tmp_path / 'corpus.jsonl').write_text('')
    save(tmp_path / 'dataset.json', {'files_sha256': {name: digest(tmp_path / name) for name in ('corpus.jsonl', 'queries.jsonl')}})
    changed = row.model_copy(update={'request': [{'role': 'user', 'content': 'Rewritten question'}]})
    (tmp_path / 'prepared.jsonl').write_text(changed.model_dump_json() + '\n')
    save(tmp_path / 'preparation.json', {'prepared_sha256': digest(tmp_path / 'prepared.jsonl')})
    with pytest.raises(ValueError, match='integrity'):
        load_prepared(tmp_path)


@pytest.mark.parametrize('change_question', [True, False])
def test_consistently_rehashed_edits_cannot_replace_frozen_questions_or_context(tmp_path, change_question):
    from scone_memory.testing.public_qa_run import digest
    row = prepared()
    (tmp_path / 'corpus.jsonl').write_text('')
    (tmp_path / 'queries.jsonl').write_text(row.question.model_dump_json() + '\n')
    (tmp_path / 'prepared.jsonl').write_text(row.model_dump_json() + '\n')
    save(tmp_path / 'dataset.json', {'files_sha256': {name: digest(tmp_path / name) for name in ('corpus.jsonl', 'queries.jsonl')}})
    save(tmp_path / 'preparation.json', {'prepared_sha256': digest(tmp_path / 'prepared.jsonl')})
    question = row.question.model_copy(update={'question': 'Changed question'}) if change_question else row.question
    messages = [{'role': 'system', 'content': 'Altered context'}, {'role': 'user', 'content': question.question}]
    changed = row.model_copy(update={'question': question, 'request': messages, 'request_sha256': request_digest(messages)})
    (tmp_path / 'queries.jsonl').write_text(question.model_dump_json() + '\n')
    (tmp_path / 'prepared.jsonl').write_text(changed.model_dump_json() + '\n')
    with pytest.raises(ValueError, match='changed|integrity'):
        load_prepared(tmp_path)
