from __future__ import annotations

import pytest
import httpx
from pathlib import Path
from typing import Sequence
import json

from .pipeline import Passage
from scone_memory.testing.public_qa import Question


def test_select_only_symmetric_billing_failures() -> None:
    from .recovery import recovery_ids
    rows = {(q, arm): {'completed': done, 'error': None if done else 'rerank_http_402'}
            for q, done in [('ok', True), ('failed', False)] for arm in ('scone', 'llamaindex')}
    assert recovery_ids(rows) == {'failed'}
    rows['failed', 'scone'] = {'completed': True, 'error': None}
    with pytest.raises(ValueError, match='symmetric'):
        recovery_ids(rows)
    rows['failed', 'scone'] = {'completed': False, 'error': 'http_500'}
    with pytest.raises(ValueError, match='symmetric'):
        recovery_ids(rows)


def test_restore_original_candidate_order_and_reject_mismatched_sources() -> None:
    from .recovery import restore_candidates
    a, b = Passage('a', 'A', 'alpha'), Passage('b', 'B', 'beta')
    row: dict[str, object] = {'retrieved_chunk_ids': ['b', 'a'], 'retrieved_ids': ['B', 'A']}
    assert restore_candidates(row, {'a': a, 'b': b}) == [b, a]
    row['retrieved_ids'] = ['A', 'B']
    with pytest.raises(ValueError, match='source'):
        restore_candidates(row, {'a': a, 'b': b})


def test_restore_rejects_missing_or_repeated_chunks() -> None:
    from .recovery import restore_candidates
    a = Passage('a', 'A', 'alpha')
    with pytest.raises(ValueError):
        restore_candidates({'retrieved_chunk_ids': ['absent'], 'retrieved_ids': ['A']}, {'a': a})
    with pytest.raises(ValueError):
        restore_candidates({'retrieved_chunk_ids': ['a', 'a'], 'retrieved_ids': ['A', 'A']}, {'a': a})


@pytest.mark.asyncio
async def test_provider_failure_stops_remaining_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from . import recovery
    calls: list[str] = []

    async def reject(client: httpx.AsyncClient, question: str, passages: Sequence[Passage],
                     *, provider: str) -> tuple[dict[str, float], dict[str, object], float]:
        calls.append(question)
        response = httpx.Response(402, request=httpx.Request('POST', 'https://openrouter.ai/api/alpha/decisions'))
        response.raise_for_status()
        raise AssertionError('unreachable')

    monkeypatch.setattr(recovery, 'rank', reject)
    candidates = {arm: [Passage('a', 'A', 'alpha')] for arm in ('scone', 'llamaindex')}
    jobs = [recovery.RecoveryQuestion(i, Question(id=f'q{i}', dataset='squad', question=f'Question {i}?'), candidates)
            for i in range(10)]
    await recovery.evaluate(jobs, {}, tmp_path, 2)
    assert 1 <= len(calls) <= 2
    assert (tmp_path / 'attempts.jsonl').read_text() == ''
    assert (tmp_path / 'observations.jsonl').read_text() == ''
    assert len((tmp_path / 'provider-errors.jsonl').read_text().splitlines()) == len(calls)


@pytest.mark.asyncio
async def test_successful_recovery_is_not_repeated_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from . import recovery
    rank_calls: list[str] = []
    generate_calls: list[list[dict[str, str]]] = []

    async def rank_ok(client: httpx.AsyncClient, question: str, passages: Sequence[Passage],
                      *, provider: str) -> tuple[dict[str, float], dict[str, object], float]:
        rank_calls.append(question)
        return {'a': .9}, {'request': {}, 'response': {}}, 10.0

    async def generate_ok(client: httpx.AsyncClient, request: list[dict[str, str]]) -> dict[str, object]:
        generate_calls.append(request)
        return {'completed': True, 'answer': 'alpha', 'request': request, 'error': None, 'generation_ms': 20.0}

    monkeypatch.setattr(recovery, 'rank', rank_ok)
    monkeypatch.setattr(recovery, 'generate', generate_ok)
    question = Question(id='q', dataset='squad', question='What?')
    candidates = {arm: [Passage('a', 'A', 'alpha')] for arm in ('scone', 'llamaindex')}
    original: dict[tuple[str, str], dict[str, object]] = {('q', arm): {'id': 'q', 'arm': arm,
        'completed': False, 'answer': '', 'error': 'rerank_http_402', 'retrieval_ms': 5.0,
        'retrieved_ids': ['A'], 'retrieved_chunk_ids': ['a']} for arm in ('scone', 'llamaindex')}
    jobs = [recovery.RecoveryQuestion(0, question, candidates)]
    await recovery.evaluate(jobs, original, tmp_path, 2)
    before = (tmp_path / 'observations.jsonl').read_bytes()
    await recovery.evaluate(jobs, original, tmp_path, 2)
    assert (tmp_path / 'observations.jsonl').read_bytes() == before
    assert rank_calls == ['What?']
    assert len(generate_calls) == 2
    rows = [json.loads(line) for line in before.splitlines()]
    assert all(row['completed'] and row['context_ids'] == ['A'] and row['total_ms'] == 35.0 for row in rows)
