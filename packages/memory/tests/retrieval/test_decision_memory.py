"""Decision persistence must never bypass fresh evidence or scope checks."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.providers.typesafe_evidence import TypeSafeEvidenceAssessor
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceCandidate
from scone_memory.retrieval.decision_memory import DecisionConflict, DecisionMemory, RememberedEvidenceAssessor
from scone_memory.retrieval.recall_scope import RecallScope

KEY = bytes(range(32))
QUESTION = 'Who owns Polaris?'
SCOPE = RecallScope.validated()
FIRST = (EvidenceCandidate(id='chunk:1', episode_id=1, text='Maya owns Polaris.'),)
SECOND = EvidenceCandidate(id='chunk:2', episode_id=2, text='Correction: Maya left. Leo now owns Polaris.')


def provider(calls, *, fail=False, model='jev-latest'):
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if fail:
            return httpx.Response(503)
        return httpx.Response(200, json={'model': 'jev-1.13.0',
            'usage': {'input_tokens': 100, 'output_tokens': 20},
            'answers': {key: {'type': 'noul', 'noul': .95} for key in body['questions']}})
    return TypeSafeEvidenceAssessor('https://api.typesafe.ai', model, api_key='private-key',
                                   transport=httpx.MockTransport(respond))


async def assess(wrapper, candidates=FIRST, *, space='alpha', scope=SCOPE, question=QUESTION):
    return await wrapper.assess_scoped(question, candidates, space=space, scope=scope)


async def test_restart_reuses_encrypted_judgment_without_provider(tmp_path, caplog):
    path, calls = tmp_path / 'decisions.db', []
    memory = DecisionMemory(path, key=KEY)
    expected = await assess(RememberedEvidenceAssessor(provider(calls), memory))
    reopened = DecisionMemory(path, key=KEY)
    with caplog.at_level('INFO'):
        result = await assess(RememberedEvidenceAssessor(provider(calls, fail=True), reopened))
    assert result == expected and len(calls) == 1
    saved = reopened.history('alpha', QUESTION, SCOPE).revisions[-1]
    assert saved.judgment.model == 'jev-1.13.0'
    assert saved.judgment.probabilities == {'sufficient': .95, 'chunk:1': .95}
    assert saved.revision == 1
    raw = path.read_bytes()
    assert all(value not in raw for value in [b'Maya', b'Polaris', b'alpha', b'chunk:1', b'jev-1.13.0'])
    assert path.stat().st_mode & 0o777 == 0o600
    assert any(getattr(record, 'outcome', '') == 'reused' for record in caplog.records)
    with pytest.raises(Exception): DecisionMemory(path, key=b'x' * 32)


async def test_additions_edits_removals_record_reassessment_and_keep_history(tmp_path):
    calls, memory = [], DecisionMemory(tmp_path / 'decisions.db', key=KEY)
    wrapper = RememberedEvidenceAssessor(provider(calls), memory)
    await assess(wrapper)
    await assess(wrapper, (*FIRST, SECOND))
    await assess(wrapper, (FIRST[0].model_copy(update={'text': 'Maya no longer owns Polaris.'}), SECOND))
    await assess(wrapper, (SECOND,))
    saved = memory.history('alpha', QUESTION, SCOPE).revisions
    assert len(calls) == 4
    assert [r.reason for r in saved] == ['initial'] + ['evidence_changed'] * 3
    assert [r.change for r in saved] == ['initial', 'revised', 'unchanged', 'revised']
    assert saved[-1].judgment.decision.selected_ids == ('chunk:2',)
    assert saved[0].judgment.decision.selected_ids == ('chunk:1',)


async def test_scope_space_question_and_unscoped_calls_are_isolated(tmp_path):
    calls, memory = [], DecisionMemory(tmp_path / 'decisions.db', key=KEY)
    wrapper = RememberedEvidenceAssessor(provider(calls), memory)
    await assess(wrapper)
    await assess(wrapper, space='beta')
    await assess(wrapper, scope=RecallScope.validated(where={'team': 'support'}))
    await assess(wrapper, question='Who formerly owned Polaris?')
    await wrapper.assess(QUESTION, FIRST)
    assert len(calls) == 5
    assert memory.forget_space('alpha') == 3
    assert memory.history('alpha', QUESTION, SCOPE) is None
    assert memory.history('beta', QUESTION, SCOPE) is not None


async def test_definition_expiry_and_clock_rollback_force_reassessment(tmp_path):
    calls, memory = [], DecisionMemory(tmp_path / 'decisions.db', key=KEY)
    now = [datetime.now(timezone.utc)]
    wrapper = RememberedEvidenceAssessor(provider(calls), memory, max_age_s=60, clock=lambda: now[0])
    await assess(wrapper)
    await assess(wrapper)
    now[0] += timedelta(seconds=60)
    await assess(wrapper)
    now[0] -= timedelta(seconds=120)
    await assess(wrapper)
    changed = RememberedEvidenceAssessor(provider(calls, model='jev-next'), memory, clock=lambda: now[0])
    await assess(changed)
    assert len(calls) == 4
    assert [r.reason for r in memory.history('alpha', QUESTION, SCOPE).revisions] == [
        'initial', 'expired', 'expired', 'definition_changed']


async def test_failure_cancellation_and_concurrent_revision_do_not_replace_history(tmp_path):
    calls, memory = [], DecisionMemory(tmp_path / 'decisions.db', key=KEY)
    await assess(RememberedEvidenceAssessor(provider(calls), memory))
    failing = RememberedEvidenceAssessor(provider(calls, fail=True), memory)
    with pytest.raises(ValueError): await assess(failing, (SECOND,))
    entered = asyncio.Event()
    async def wait(request):
        entered.set()
        await asyncio.Event().wait()
    waiting = RememberedEvidenceAssessor(TypeSafeEvidenceAssessor('https://api.typesafe.ai', 'jev-latest',
        api_key='private-key', transport=httpx.MockTransport(wait)), memory)
    task = asyncio.create_task(assess(waiting, (SECOND,)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    previous = memory.history('alpha', QUESTION, SCOPE).revisions[-1]
    assert previous.revision == 1
    next_revision = previous.model_copy(update={'revision': 2})
    memory.append('alpha', QUESTION, SCOPE, next_revision, expected_revision=1)
    with pytest.raises(DecisionConflict):
        memory.append('alpha', QUESTION, SCOPE, next_revision, expected_revision=1)
    assert len(memory.history('alpha', QUESTION, SCOPE).revisions) == 2


async def test_bounded_history_and_corrupt_cache_fall_back_to_fresh_provider(tmp_path):
    path, calls = tmp_path / 'decisions.db', []
    memory = DecisionMemory(path, key=KEY, history_limit=2, max_records=1)
    wrapper = RememberedEvidenceAssessor(provider(calls), memory)
    for text in ['Maya owns Polaris.', 'Leo owns Polaris.', 'Sam owns Polaris.']:
        await assess(wrapper, (FIRST[0].model_copy(update={'text': text}),))
    assert [r.revision for r in memory.history('alpha', QUESTION, SCOPE).revisions] == [2, 3]
    # Capacity exhaustion affects remembering, not an independently fresh judgment.
    assert (await assess(wrapper, space='beta')).status == 'sufficient'
    assert memory.history('beta', QUESTION, SCOPE) is None
    with sqlite3.connect(path) as db:
        db.execute('UPDATE evidence_decisions SET payload=?', (b'invalid encrypted payload',))
    assert (await assess(wrapper)).status == 'sufficient'
    assert len(calls) == 5


async def test_real_retriever_rechecks_sources_before_reusing_decisions(tmp_path):
    calls, memory = [], DecisionMemory(tmp_path / 'decisions.db', key=KEY)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    wrapper = RememberedEvidenceAssessor(provider(calls), memory)
    strategy = AdaptiveRetriever(engine, wrapper, limits=AdaptiveLimits(max_rounds=1, max_queries=1),
        empty_selection_policy='empty', evidence_policy='model_selected')
    try:
        old = await engine.remember('alpha', 'Maya owns Polaris.')
        first = await strategy.retrieve('alpha', QUESTION, scope=SCOPE)
        second = await strategy.retrieve('alpha', QUESTION, scope=SCOPE)
        assert first.recall.items and second.recall.items and len(calls) == 1
        await engine.forget('alpha', old.episode_id)
        empty = await strategy.retrieve('alpha', QUESTION, scope=SCOPE)
        assert not empty.recall.items and len(calls) == 1
        await engine.remember('alpha', 'Leo now owns Polaris.')
        revised = await strategy.retrieve('alpha', QUESTION, scope=SCOPE)
        assert revised.recall.items and len(calls) == 2
        assert all('Maya' not in c['text'] for c in calls[-1]['state']['candidates'])
    finally:
        await engine.close()


async def test_source_deleted_during_cache_read_is_checked_again(tmp_path):
    calls, memory = [], DecisionMemory(tmp_path / 'decisions.db', key=KEY)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    wrapper = RememberedEvidenceAssessor(provider(calls), memory)
    try:
        old = await engine.remember('alpha', 'Maya owns Polaris.')
        strategy = AdaptiveRetriever(engine, wrapper, limits=AdaptiveLimits(max_rounds=1, max_queries=1),
            empty_selection_policy='empty', evidence_policy='model_selected')
        await strategy.retrieve('alpha', QUESTION, scope=SCOPE)
        original = wrapper.assess_scoped
        async def delete_after_read(*args, **kwargs):
            decision = await original(*args, **kwargs)
            await engine.forget('alpha', old.episode_id)
            return decision
        wrapper.assess_scoped = delete_after_read
        result = await strategy.retrieve('alpha', QUESTION, scope=SCOPE)
        assert not result.recall.items and len(calls) == 1
    finally:
        await engine.close()
