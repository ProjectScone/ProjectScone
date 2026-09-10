"""Tool generation evaluation uses retained packets and confirmed corrections."""
import json

import pytest

from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from scone_memory.testing import generation_ablation as evaluation
from test_generation_ablation import fixture_file


@pytest.mark.parametrize('protocol', ['native', 'structured'])
@pytest.mark.parametrize('policy', ['report', 'require_supported'])
@pytest.mark.parametrize('outcome', ['corrected', 'unsupported', 'failed', 'deleted'])
async def test_tool_review_ablation_preserves_drafts_sources_and_policy(tmp_path, monkeypatch, protocol, policy, outcome):
    engines, reviews, packets = [], [], []
    original_seed = evaluation._seed

    async def seed(engine, *args):
        engines.append(engine)
        return await original_seed(engine, *args)

    class ToolModel:
        async def complete(self, messages, tools):
            packets.append(json.loads(messages[-1]['content']))
            return ToolStep(content='The cobalt probe uses 12 volts.')

    class Baseline:
        async def respond(self, messages):
            yield TextDelta('The cobalt probe uses 12 volts.')
            yield ReplyCompleted()

        async def aclose(self):
            pass

    class Reviewer:
        def __init__(self, *args, **kwargs):
            assert kwargs['quote_mode'] == 'spans'

        async def review(self, question, answer, evidence, evidence_ids):
            reviews.append((answer, evidence, evidence_ids))
            assert json.loads(evidence) == {'tool_results':packets, 'complete':False}
            assert 'EVALUATION_ONLY_CANARY' not in question + evidence
            if outcome == 'failed':
                raise RuntimeError('PRIVATE_REVIEW_ERROR')
            if outcome == 'deleted':
                await engines[0].forget('fixture', packets[0]['items'][0]['episode_id'])
                return AnswerReviewDecision(status='supported')
            if answer == 'The cobalt probe uses 18 volts.':
                return AnswerReviewDecision(status='supported')
            return AnswerReviewDecision(status='needs_revision', issues=(AnswerIssue(code='contradiction',
                answer_quote='12 volts', evidence_ids=(evidence_ids[0],)),),
                revised_answer='The cobalt probe uses 18 volts.' if outcome == 'corrected' else None)

    monkeypatch.setattr(evaluation, '_seed', seed)
    monkeypatch.setattr(evaluation, 'SelfHostedToolChat', lambda *a, **kw: ToolModel())
    monkeypatch.setattr(evaluation, 'SelfHostedStructuredToolChat', lambda *a, **kw: ToolModel())
    monkeypatch.setattr(evaluation, 'SelfHostedAnswerReviewer', Reviewer)
    report = await evaluation.run_ablation(fixture_file(tmp_path), output=tmp_path/'review.json',
        endpoint='http://127.0.0.1:11434/v1', model='installed', model_factory=Baseline,
        tool_mode=protocol, review_model='reviewer', review_quote_mode='spans', review_policy=policy)
    baseline, candidate = report['results']
    assert baseline['answer_text'] == 'The cobalt probe uses 12 volts.'
    assert 'answer_review' not in baseline
    assert candidate['draft_answer_text'] == baseline['answer_text']
    assert candidate['generation_provider_calls'] == 1
    assert len(reviews) == (2 if outcome == 'corrected' else 1)
    assert candidate['answer_review']['rounds'] == len(reviews)
    accepted = outcome == 'corrected' or (policy == 'report' and outcome in ('unsupported', 'failed'))
    assert candidate['completed'] == accepted
    assert candidate['answer_text'] == ('The cobalt probe uses 18 volts.' if outcome == 'corrected'
                                      else baseline['answer_text'] if accepted else '')
    assert candidate['successful_key_fact_coverage'] == (1 if outcome == 'corrected' else 0)
    assert candidate['output_bytes'] == len(candidate['answer_text'].encode())
    assert candidate['review_ms'] >= 0
    assert candidate['draft_total_ms'] >= 0
    assert candidate['total_ms'] >= candidate['draft_total_ms']
    assert candidate['first_token_ms'] == (candidate['total_ms'] if accepted else None)
    assert report['review_quote_mode'] == 'spans'
    assert 'PRIVATE_REVIEW_ERROR' not in json.dumps(report)
    if outcome == 'deleted':
        assert candidate['answer_review']['source_status'] == 'stale'
        assert candidate['tool_retrieval']['source_status'] == 'stale'
    else:
        assert candidate['answer_review']['source_status'] == 'retained'


async def test_reviewed_empty_turn_preserves_no_source_receipt(tmp_path):
    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder

    class ToolModel:
        async def complete(self, messages, tools):
            return ToolStep(content='The supplied evidence does not answer this question.')

    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            assert evidence_ids == ()
            return AnswerReviewDecision(status='supported')

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        case = evaluation.load_generation_fixture(fixture_file(tmp_path)).cases[0]
        row = await evaluation._tool_reply(engine, case, {}, ToolModel(), 30, True,
                                          reviewer=Reviewer(), review_policy='require_supported')
        assert row['completed'] is True
        assert row['tool_retrieval']['evidence_ids'] == []
        assert row['tool_retrieval']['source_status'] == 'none'
        assert row['answer_review']['source_status'] == 'retained'
        assert row['context_status'] == 'empty'
        assert row['evidence_coverage'] == 0
    finally:
        await engine.close()
