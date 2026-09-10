"""Host-owned draft spans remove model quotation copying from review output."""
import json

import httpx
import pytest

from scone_memory.providers.answer_reviewer import SelfHostedAnswerReviewer
from scone_memory.realtime.answer_review import AnswerReviewError


def response(raw):
    return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(raw)}}]})


async def test_span_selection_returns_exact_draft_quote_without_source_substitution():
    draft = 'The café uses 17 watts.\nA second statement follows.'
    received = []
    def serve(request):
        body = json.loads(request.content)
        received.append(body)
        payload = json.loads(body['messages'][-1]['content'])
        assert payload['answer'] == draft
        assert payload['answer_spans'] == {'s1':'The café uses 17 watts.\n', 's2':'A second statement follows.'}
        schema = body['response_format']['json_schema']['schema']
        for branch in schema['anyOf']:
            issue = branch['properties']['issues']['items']
            assert 'answer_quote' not in issue['properties']
            assert issue['properties']['answer_span_id']['enum'] == [None,'s1','s2']
        return response({'status':'needs_revision','issues':[{'code':'contradiction',
            'answer_span_id':'s1','evidence_ids':['chunk:1']}],'revised_answer':'The café uses 19 watts.'})
    model = SelfHostedAnswerReviewer('http://localhost:11434/v1','model',quote_mode='spans',transport=httpx.MockTransport(serve))
    result = await model.review('How much power?',draft,'The café uses 19 watts.',('chunk:1',))
    assert result.issues[0].answer_quote == 'The café uses 17 watts.\n'
    assert result.revised_answer == 'The café uses 19 watts.'
    assert len(received) == 1


@pytest.mark.parametrize('issue', [
    {'code':'unsupported_claim','answer_span_id':None,'evidence_ids':[]},
    {'code':'contradiction','answer_span_id':'s99','evidence_ids':['chunk:1']},
    {'code':'contradiction','answer_span_id':1,'evidence_ids':['chunk:1']},
    {'code':'contradiction','answer_span_id':'s1','answer_quote':'forged','evidence_ids':['chunk:1']},
    {'code':'contradiction','answer_quote':'answer','evidence_ids':['chunk:1']},
    {'code':'contradiction','answer_span_id':'s1','evidence_ids':['chunk:99']},
])
async def test_invalid_span_decisions_remain_rejected(issue):
    model = SelfHostedAnswerReviewer('http://localhost:11434/v1','model',quote_mode='spans',
        transport=httpx.MockTransport(lambda request:response({'status':'needs_revision','issues':[issue],'revised_answer':None})))
    with pytest.raises(AnswerReviewError) as error:
        await model.review('question','answer','evidence',('chunk:1',))
    assert error.value.reason == 'invalid_review'


async def test_only_an_omission_can_select_no_draft_span():
    model = SelfHostedAnswerReviewer('http://localhost:11434/v1','model',quote_mode='spans',
        transport=httpx.MockTransport(lambda request:response({'status':'needs_revision','issues':[
            {'code':'incomplete_answer','answer_span_id':None,'evidence_ids':['chunk:1']}],'revised_answer':None})))
    assert (await model.review('question','answer','evidence',('chunk:1',))).issues[0].answer_quote == ''


@pytest.mark.parametrize('draft', ['é' * 32000, 'A. ' * 20000, 'x' * 128000, '\n' * 120000 + 'Tail'])
async def test_span_catalog_is_bounded_and_lossless(draft):
    def serve(request):
        payload=json.loads(json.loads(request.content)['messages'][-1]['content'])
        spans=payload['answer_spans']
        assert 1 <= len(spans) <= 128
        assert ''.join(spans.values()) == draft
        assert all(1 <= len(text) <= 2000 for text in spans.values())
        return response({'status':'supported','issues':[],'revised_answer':None})
    model=SelfHostedAnswerReviewer('http://localhost:11434/v1','model',quote_mode='spans',max_answer_bytes=128000,
        transport=httpx.MockTransport(serve))
    assert (await model.review('question',draft,'evidence',())).status == 'supported'


@pytest.mark.parametrize('mode', ['auto','',True,None])
def test_invalid_span_protocol_option_is_rejected(mode):
    with pytest.raises(ValueError,match='quote_mode'):
        SelfHostedAnswerReviewer('http://localhost:11434/v1','model',quote_mode=mode)
