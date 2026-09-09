"""Requirements reach generation/review and gate every publication path."""
import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.realtime.answer_requirements import AnswerRequirements
from scone_memory.realtime.answer_review import AnswerReviewDecision
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation
from test_answer_requirements import Reviewer, revise
from test_text_answer_review import Model
from test_text_evidence_answer import Selector
from test_text_tool_answer import Script, search


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


@pytest.mark.parametrize('mode', ['ordinary', 'tool', 'review', 'tool_review', 'extractive', 'abstention'])
async def test_invalid_output_is_never_observed_or_captured(memory, mode):
    if mode != 'abstention':
        await memory.remember('alpha', 'Juniper uses Polaris.')
    model = Model('Polaris', '\nUnexpected explanation.')
    reviewer = Reviewer(AnswerReviewDecision(status='supported'))
    settings = {}
    if mode in ('tool', 'tool_review'):
        script = Script(search(), ToolStep(content='Polaris\nUnexpected explanation.'))
        settings['tool_model_factory'] = lambda: script
    if mode in ('review', 'tool_review'):
        settings['answer_reviewer'] = reviewer
    if mode in ('extractive', 'abstention'):
        settings.update(evidence_selector=Selector(), evidence_answer_policy='required')
    conversation = TextConversation(memory, 'alpha', 'format-rejected', lambda: model,
        answer_requirements=AnswerRequirements(format='json_object'), **settings)
    observed = []
    async def observe(text):
        observed.append(text)
    with pytest.raises(RuntimeError, match='answer format'):
        await conversation.reply('Juniper calibration?', on_text=observe)
    assert observed == []
    assert [row.metadata['role'] for row in await memory.episodes('alpha',
        {'session_id':'format-rejected'})] == ['user']
    assert conversation.closed


async def test_ordinary_generation_buffers_and_sends_requirements_on_each_turn(memory):
    model = Model('{"answer":', '"Polaris"}')
    requirements = AnswerRequirements(format='json_object', max_bytes=100,
        instructions='Use the answer field.')
    conversation = TextConversation(memory, 'alpha', 'format-valid', lambda: model,
        answer_requirements=requirements)
    observed = []
    async def observe(text):
        observed.append(text)
    first = await conversation.reply('Hello!', on_text=observe)
    await conversation.reply('Hello again!', on_text=observe)
    assert observed == ['{"answer":"Polaris"}'] * 2
    assert 'answer_review' not in first and model.closes == 2
    assert all(requirements.prompt() in request[0]['content'] for request in model.requests)
    assert len([row for row in await memory.episodes('alpha', {'session_id':'format-valid'})
                if row.metadata['role'] == 'assistant']) == 2
    await conversation.close()


@pytest.mark.parametrize('tool', [False, True])
async def test_requirements_shared_with_review_and_invalid_proposal_keeps_valid_draft(memory, tool):
    await memory.remember('alpha', 'Juniper uses Polaris.')
    requirements = AnswerRequirements(max_lines=1, max_bytes=50, instructions='Only the star name.')
    reviewer = Reviewer(revise('Polaris\nThe source explains this.'))
    model = Model('Pol', 'aris')
    script = Script(search(), ToolStep(content='Polaris'))
    settings = {'tool_model_factory':lambda:script} if tool else {}
    conversation = TextConversation(memory, 'alpha', 'format-fallback', lambda:model,
        answer_requirements=requirements, answer_reviewer=reviewer, **settings)
    observed = []
    async def observe(text):
        observed.append(text)
    result = await conversation.reply('Juniper calibration?', on_text=observe)
    assert observed == ['Polaris'] and result['text'] == 'Polaris'
    assert result['answer_review']['format_status'] == 'satisfied'
    assert result['answer_review']['errors'] == ['answer_format_rejected']
    assert result['answer_review']['source_status'] == 'retained'
    assert reviewer.calls == [('Polaris', requirements)]
    request = script.requests[0] if tool else model.requests[0]
    assert requirements.prompt() in request[0]['content']
    await conversation.close()


async def test_require_supported_still_withholds_format_valid_but_unconfirmed_draft(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.')
    reviewer = Reviewer(revise('Polaris\nExplanation'))
    conversation = TextConversation(memory, 'alpha', 'format-strict', lambda:Model('Polaris'),
        answer_reviewer=reviewer, review_policy='require_supported',
        answer_requirements=AnswerRequirements(max_lines=1))
    with pytest.raises(RuntimeError, match='did not support'):
        await conversation.reply('Juniper calibration?')
    assert [row.metadata['role'] for row in await memory.episodes('alpha',
        {'session_id':'format-strict'})] == ['user']


def test_configuration_fails_before_any_turn_for_unsupported_reviewer_or_oversized_history(memory):
    class Legacy:
        async def review(self, *args):
            pytest.fail('unexpected review')
    with pytest.raises(ValueError, match='review_with_requirements'):
        TextConversation(memory, 'alpha', 'legacy', lambda:Model(), answer_reviewer=Legacy(),
            answer_requirements=AnswerRequirements())
    with pytest.raises(ValueError, match='history byte limit'):
        TextConversation(memory, 'alpha', 'too-long', lambda:Model(), max_history_bytes=512,
            answer_requirements=AnswerRequirements(instructions='x' * 600))
    with pytest.raises(ValueError):
        TextConversation(memory, 'alpha', 'bypassed', lambda:Model(),
            answer_requirements=AnswerRequirements.model_construct(max_bytes=-1))


async def test_cancel_during_buffered_generation_never_publishes_partial_json(memory):
    entered = asyncio.Event()
    class WaitingModel(Model):
        async def respond(self, messages):
            yield TextDelta('{"answer":')
            entered.set()
            await asyncio.Event().wait()
            yield ReplyCompleted()
    model = WaitingModel()
    conversation = TextConversation(memory, 'alpha', 'cancel-format', lambda:model,
        answer_requirements=AnswerRequirements(format='json_object'))
    observed = []
    async def observe(text):
        observed.append(text)
    task = asyncio.create_task(conversation.reply('Hello!', on_text=observe))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed == [] and model.closes == 1
    assert [row.metadata['role'] for row in await memory.episodes('alpha',
        {'session_id':'cancel-format'})] == ['user']
    await conversation.close()


async def test_format_valid_revision_is_published_once_and_captured(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.')
    reviewer = Reviewer(revise('{"answer":"Polaris"}'), AnswerReviewDecision(status='supported'))
    conversation = TextConversation(memory, 'alpha', 'format-repaired', lambda:Model('Polaris'),
        answer_reviewer=reviewer, answer_requirements=AnswerRequirements(format='json_object'))
    observed = []
    async def observe(text):
        observed.append(text)
    result = await conversation.reply('Juniper calibration?', on_text=observe)
    assert observed == ['{"answer":"Polaris"}']
    assert result['answer_review']['revised'] is True
    assert result['answer_review']['format_status'] == 'satisfied'
    assert [row.content for row in await memory.episodes('alpha', {'session_id':'format-repaired'})
        if row.metadata['role'] == 'assistant'] == observed
    await conversation.close()


async def test_format_valid_reply_still_rejects_source_deleted_during_contextual_review(memory):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.')
    class DeletingReviewer(Reviewer):
        async def review_with_requirements(self, *args):
            await memory.forget('alpha', episode.episode_id)
            return AnswerReviewDecision(status='supported')
    conversation = TextConversation(memory, 'alpha', 'format-stale', lambda:Model('Polaris'),
        answer_reviewer=DeletingReviewer(), answer_requirements=AnswerRequirements(max_lines=1))
    observed = []
    async def observe(text):
        observed.append(text)
    with pytest.raises(RuntimeError, match='stale or unavailable') as error:
        await conversation.reply('Juniper calibration?', on_text=observe)
    assert error.value.answer_review['format_status'] == 'satisfied'
    assert error.value.answer_review['source_status'] == 'stale'
    assert observed == []
    assert [row.metadata['role'] for row in await memory.episodes('alpha',
        {'session_id':'format-stale'})] == ['user']


async def test_failed_review_cannot_publish_invalid_original_under_report_policy(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.')
    conversation = TextConversation(memory, 'alpha', 'format-unavailable', lambda:Model('Polaris'),
        answer_reviewer=Reviewer(RuntimeError('PRIVATE diagnostic')),
        answer_requirements=AnswerRequirements(format='json_object'))
    observed = []
    async def observe(text):
        observed.append(text)
    with pytest.raises(RuntimeError, match='answer format') as error:
        await conversation.reply('Juniper calibration?', on_text=observe)
    assert error.value.answer_review['status'] == 'unavailable'
    assert error.value.answer_review['format_status'] == 'rejected'
    assert 'PRIVATE' not in str(error.value) and observed == []
    assert [row.metadata['role'] for row in await memory.episodes('alpha',
        {'session_id':'format-unavailable'})] == ['user']
