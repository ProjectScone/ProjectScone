"""Tool drafts cross the review and publication boundary before capture."""
import asyncio
import json

import pytest

from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision, AnswerReviewLimits
from scone_memory.realtime.text import TextConversation
from test_text_tool_answer import Script, search

DRAFT = 'Juniper uses an unsupported destination.'
REVISION = 'Juniper uses Polaris.'


@pytest.mark.parametrize('policy', ['report', 'require_supported'])
@pytest.mark.parametrize('outcome', ['supported', 'revised', 'unconfirmed', 'uncertain', 'failed', 'deleted'])
async def test_tool_review_controls_publication_and_revalidates_sources(engine, policy, outcome):
    source = await engine.remember('alpha', REVISION, metadata={'team':'blue'})
    await engine.remember('alpha', 'Juniper PRIVATE_TEAM', metadata={'team':'red'})
    await engine.remember('foreign', 'Juniper PRIVATE_SPACE', metadata={'team':'blue'})
    observed, reviewed = [], []
    model = Script(search(), ToolStep(content=DRAFT))

    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            assert observed == [] and question == 'What does Juniper use?'
            assert 'PRIVATE_' not in evidence and REVISION in evidence
            assert json.loads(evidence)['complete'] is False
            assert json.loads(evidence)['tool_results'] == [json.loads(model.requests[-1][-1]['content'])]
            assert evidence_ids
            reviewed.append(answer)
            if outcome == 'deleted':
                await engine.forget('alpha', source.episode_id)
            if outcome == 'failed':
                raise RuntimeError('PRIVATE_REVIEW_SECRET')
            if outcome in ('revised', 'unconfirmed') and len(reviewed) == 1:
                return AnswerReviewDecision(status='needs_revision', revised_answer=REVISION,
                    issues=(AnswerIssue(code='unsupported_claim', answer_quote=DRAFT, evidence_ids=evidence_ids[:1]),))
            return AnswerReviewDecision(status='uncertain' if outcome in ('uncertain', 'unconfirmed') else 'supported')

    conversation = TextConversation(engine, 'alpha', 'reviewed-tool', where={'team':'blue'},
        tool_model_factory=lambda:model, answer_reviewer=Reviewer(), review_policy=policy)
    async def observe(text): observed.append(text)
    blocked = outcome == 'deleted' or (policy == 'require_supported' and outcome in ('uncertain', 'failed', 'unconfirmed'))
    try:
        if blocked:
            with pytest.raises(RuntimeError) as error:
                await conversation.reply('What does Juniper use?', on_text=observe)
            receipt = error.value.answer_review
            assert 'PRIVATE_' not in str(error.value) + json.dumps(receipt)
            assert receipt['source_status'] == ('stale' if outcome == 'deleted' else 'retained')
            assert observed == []
            assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id':'reviewed-tool'})] == ['user']
        else:
            result = await conversation.reply('What does Juniper use?', on_text=observe)
            expected = REVISION if outcome == 'revised' else DRAFT
            assert result['text'] == expected and observed == [expected]
            assert result['answer_review']['status'] == ('unavailable' if outcome == 'failed' else
                'uncertain' if outcome in ('uncertain', 'unconfirmed') else 'supported')
            assert result['answer_review']['revised'] is (outcome == 'revised')
            assert result['answer_review']['verified_accuracy'] is False
            assert result['memory_context']['tool_retrieval']['model_calls'] == 2
            assert conversation._history[-1]['content'] == expected
            assert (await engine.episodes('alpha', {'session_id':'reviewed-tool'}))[-1].content == expected
        assert len(reviewed) == (2 if outcome in ('revised', 'unconfirmed') else 1)
    finally:
        await conversation.close()


async def test_tool_review_does_not_skip_an_empty_evidence_turn(engine):
    calls = []
    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            calls.append(json.loads(evidence))
            assert evidence_ids == ()
            return AnswerReviewDecision(status='uncertain')
    conversation = TextConversation(engine, 'alpha', 'empty-tool-review',
        tool_model_factory=lambda:Script(ToolStep(content=DRAFT)),
        answer_reviewer=Reviewer(), review_policy='require_supported')
    with pytest.raises(RuntimeError, match='did not support'):
        await conversation.reply('What does Juniper use?')
    assert calls == [{'tool_results':[], 'complete':False}]
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id':'empty-tool-review'})] == ['user']


async def test_reviewed_tool_revision_checks_history_before_observer(engine):
    await engine.remember('alpha', REVISION)
    revision = 'Polaris ' * 80
    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            if answer == revision:
                return AnswerReviewDecision(status='supported')
            return AnswerReviewDecision(status='needs_revision', revised_answer=revision,
                issues=(AnswerIssue(code='incomplete_answer', answer_quote=''),))
    conversation = TextConversation(engine, 'alpha', 'review-budget', system_prompt='Answer.',
        tool_model_factory=lambda:Script(search(), ToolStep(content=DRAFT)),
        answer_reviewer=Reviewer(), max_history_bytes=512)
    observed = []
    async def observe(text): observed.append(text)
    with pytest.raises(RuntimeError, match='history byte limit'):
        await conversation.reply('Juniper?', on_text=observe)
    assert observed == []
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id':'review-budget'})] == ['user']


async def test_tool_review_cancellation_never_publishes_a_late_verdict(engine):
    await engine.remember('alpha', REVISION)
    entered, stopped = asyncio.Event(), asyncio.Event()
    class Reviewer:
        async def review(self, *args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return AnswerReviewDecision(status='supported')
            finally:
                stopped.set()
    conversation = TextConversation(engine, 'alpha', 'review-cancel',
        tool_model_factory=lambda:Script(search(), ToolStep(content=DRAFT)), answer_reviewer=Reviewer())
    observed = []
    async def observe(text): observed.append(text)
    task = asyncio.create_task(conversation.reply('Juniper?', on_text=observe))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and observed == []
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id':'review-cancel'})] == ['user']


async def test_tool_review_evidence_budget_rejects_before_reviewer(engine):
    await engine.remember('alpha', 'Juniper ' + 'calibration ' * 100)
    class Reviewer:
        async def review(self, *args):
            pytest.fail('oversized evidence must not reach the reviewer')
    conversation = TextConversation(engine, 'alpha', 'review-evidence-budget',
        tool_model_factory=lambda:Script(search(), ToolStep(content=DRAFT)), answer_reviewer=Reviewer(),
        review_limits=AnswerReviewLimits(max_evidence_bytes=512))
    with pytest.raises(RuntimeError) as error:
        await conversation.reply('Juniper?')
    assert error.value.answer_review['source_status'] == 'unavailable'
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id':'review-evidence-budget'})] == ['user']


async def test_review_is_cancelled_at_the_original_tool_deadline(engine):
    from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits
    from scone_memory.integrations.scoped_tools import ScopedMemoryTools
    from scone_memory.retrieval.recall_scope import RecallScope
    entered, stopped = asyncio.Event(), asyncio.Event()
    class Reviewer:
        async def review(self, *args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
    conversation = TextConversation(engine, 'alpha', 'review-deadline',
        tool_model_factory=lambda:Script(ToolStep(content=DRAFT)), answer_reviewer=Reviewer(),
        review_policy='require_supported')
    result = await EvidenceToolLoop(Script(ToolStep(content=DRAFT)), ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated()),
        limits=ToolLoopLimits(timeout_s=.1)).run([{'role':'user', 'content':'Juniper?'}])
    # The reviewer has a 20s limit, but cannot extend this completed tool turn.
    async with asyncio.timeout(1):
        with pytest.raises(RuntimeError) as error:
            await conversation._review_tool_draft('Juniper?', result)
    assert entered.is_set() and stopped.is_set()
    assert error.value.answer_review['status'] == 'unavailable'


@pytest.mark.parametrize('characters', [5, 6])
async def test_reviewed_tool_answer_obeys_original_utf8_reply_limit(engine, characters):
    from scone_memory.agents.evidence_loop import ToolLoopLimits
    revision = 'é' * characters
    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            if answer == revision:
                return AnswerReviewDecision(status='supported')
            return AnswerReviewDecision(status='needs_revision', revised_answer=revision,
                issues=(AnswerIssue(code='incomplete_answer', answer_quote=''),))
    conversation = TextConversation(engine, 'alpha', 'review-tool-byte-limit',
        tool_model_factory=lambda:Script(ToolStep(content='ok')), answer_reviewer=Reviewer(),
        tool_limits=ToolLoopLimits(max_reply_bytes=10))
    observed = []
    async def observe(text): observed.append(text)
    if characters == 6:
        with pytest.raises(RuntimeError, match='byte limit'):
            await conversation.reply('Hello?', on_text=observe)
        assert observed == []
    else:
        assert (await conversation.reply('Hello?', on_text=observe))['text'] == revision
        assert observed == [revision]
