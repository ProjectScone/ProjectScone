"""Output contracts constrain publication, independently of model judgments."""
import json

import httpx
import pytest

from scone_memory.providers.answer_reviewer import SelfHostedAnswerReviewer
from scone_memory.realtime.answer_requirements import AnswerRequirements
from scone_memory.realtime.answer_review import (
    AnswerIssue, AnswerReviewDecision, review_answer,
)


@pytest.mark.parametrize('text,accepted', [
    ('éé', True), ('ééé', False), ('a\nb', False), ('a\rb', False),
    ('a\u2028b', False), ('a\n', False), ('   ', False), ('\ud800', False),
])
def test_byte_and_line_limits_are_literal_not_rewrites(text, accepted):
    assert AnswerRequirements(max_bytes=4, max_lines=1).accepts(text) is accepted


@pytest.mark.parametrize('text,accepted', [
    ('{"answer":"Polaris"}', True), ('{}', True), (' {"x": [1, true, null]} ', True),
    ('[1]', False), ('null', False), ('"answer"', False),
    ('```json\n{}\n```', False), ('{} extra', False),
    ('{"x":1,"x":2}', False), ('{"nested":{"x":1,"x":2}}', False),
    ('{"x":NaN}', False), ('{"x":Infinity}', False), ('{"x":-Infinity}', False),
])
def test_json_object_requires_strict_json_without_duplicate_keys(text, accepted):
    assert AnswerRequirements(format='json_object').accepts(text) is accepted


@pytest.mark.parametrize('kwargs', [
    {'max_bytes':True}, {'max_bytes':0}, {'max_bytes':128001}, {'max_lines':0},
    {'max_lines':False}, {'format':'schema'}, {'instructions':'é' * 4001},
    {'instructions':'\ud800'}, {'unknown':True},
])
def test_configuration_is_strict_and_bounded(kwargs):
    with pytest.raises(ValueError):
        AnswerRequirements(**kwargs)


class Reviewer:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.calls = []

    async def review(self, *args):
        pytest.fail('requirements must not be dropped')

    async def review_with_requirements(self, question, answer, evidence, evidence_ids, requirements):
        self.calls.append((answer, requirements))
        result = next(self.decisions)
        if isinstance(result, Exception):
            raise result
        return result


def revise(answer):
    return AnswerReviewDecision(status='needs_revision', revised_answer=answer,
        issues=(AnswerIssue(code='incomplete_answer', answer_quote=''),))


async def test_invalid_revision_keeps_valid_original_without_second_model_call():
    requirements = AnswerRequirements(max_bytes=30, max_lines=1)
    reviewer = Reviewer(revise('Polaris.\nAn unwanted second line.'))
    result = await review_answer(reviewer, 'Which star?', 'Polaris', 'source', ('chunk:1',),
                                 requirements=requirements)
    assert result.answer == 'Polaris'
    assert result.receipt.status == 'needs_revision'
    assert result.receipt.format_status == 'satisfied'
    assert result.receipt.errors == ('answer_format_rejected',)
    assert result.receipt.rounds == 1 and result.receipt.revised is False
    assert len(reviewer.calls) == 1


async def test_supported_judgment_cannot_approve_invalid_format():
    reviewer = Reviewer(AnswerReviewDecision(status='supported'))
    result = await review_answer(reviewer, 'Which star?', 'Polaris\nExplanation', '', (),
                                 requirements=AnswerRequirements(max_lines=1))
    assert result.receipt.status == 'needs_revision'
    assert result.receipt.format_status == 'rejected'
    assert result.receipt.errors == ('answer_format_rejected',)
    assert result.receipt.verified_accuracy is False


async def test_invalid_original_can_be_repaired_only_after_second_supported_review():
    requirements = AnswerRequirements(format='json_object', instructions='Return the star in answer.')
    reviewer = Reviewer(revise('{"answer":"Polaris"}'), AnswerReviewDecision(status='supported'))
    result = await review_answer(reviewer, 'Which star?', 'Polaris', 'source', ('chunk:1',),
                                 requirements=requirements)
    assert result.answer == '{"answer":"Polaris"}' and result.receipt.revised
    assert result.receipt.format_status == 'satisfied'
    assert result.receipt.rounds == 2 and result.receipt.status == 'supported'
    assert [call[1] for call in reviewer.calls] == [requirements, requirements]


async def test_failed_confirmation_returns_original_with_its_own_format_status():
    reviewer = Reviewer(revise('{}'), RuntimeError('PRIVATE error'))
    result = await review_answer(reviewer, 'Question?', 'invalid JSON', '', (),
                                 requirements=AnswerRequirements(format='json_object'))
    assert result.answer == 'invalid JSON'
    assert result.receipt.status == 'unavailable' and result.receipt.format_status == 'rejected'
    assert result.receipt.errors == ('review_provider_failed', 'answer_format_rejected')


async def test_legacy_reviewer_and_bypassed_requirements_fail_before_callbacks():
    class Legacy:
        async def review(self, *args):
            pytest.fail('must fail before review')

    async def source_check():
        pytest.fail('must fail before source callback')

    with pytest.raises(ValueError, match='review_with_requirements'):
        await review_answer(Legacy(), 'Q', 'A', '', (), requirements=AnswerRequirements(),
                            validate_evidence=source_check)
    for bad in ({'max_bytes':1}, AnswerRequirements.model_construct(max_bytes=-1),
                AnswerRequirements().model_copy(update={'max_lines':False})):
        with pytest.raises(ValueError):
            await review_answer(Reviewer(), 'Q', 'A', '', (), requirements=bad,
                                validate_evidence=source_check)


async def test_self_hosted_adapter_transmits_explicit_requirements():
    requests = []
    async def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices':[{'message':{'content':json.dumps({
            'status':'supported', 'issues':[], 'revised_answer':None})}, 'finish_reason':'stop'}]})

    provider = SelfHostedAnswerReviewer('http://127.0.0.1:11434/v1', 'installed-model',
        transport=httpx.MockTransport(serve))
    requirements = AnswerRequirements(instructions='Return only the named star.', max_bytes=40, max_lines=1)
    await provider.review_with_requirements('Which star?', 'Polaris', 'source', ('chunk:1',), requirements)
    payload = json.loads(requests[0]['messages'][-1]['content'])
    assert payload['answer_requirements'] == requirements.model_dump(mode='json')
    assert 'answer_requirements' in requests[0]['messages'][0]['content']
    assert payload['answer'] == 'Polaris' and payload['evidence'] == 'source'
