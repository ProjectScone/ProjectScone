"""Field contracts cross provider, publication, review and retention boundaries."""
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolStep
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.realtime.answer_requirements import AnswerRequirements
from scone_memory.realtime.answer_review import AnswerReviewDecision
from scone_memory.realtime.text import TextConversation
from ..agents.test_answer_requirements import Reviewer, revise
from ..agents.test_evidence_tool_loop import Script, binding
from ..conversations.test_text_answer_review import Model


SCHEMA = {'$defs':{'word':{'type':'string'}}, 'properties':{'answer':{'$ref':'#/$defs/word'}},
          'required':['answer'], 'additionalProperties':False}


@pytest.mark.parametrize('exhausted', [False, True])
@pytest.mark.parametrize('answer,accepted', [({'answer':'Polaris'}, True), ({'star':'Polaris'}, False)])
async def test_provider_uses_same_schema_and_host_rejects_nonconforming_objects(engine, exhausted, answer, accepted):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    requests = []
    async def serve(request):
        requests.append(json.loads(request.content))
        value = answer if exhausted else {'action':'answer', 'answer':answer}
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop',
            'message':{'role':'assistant', 'content':json.dumps(value)}}]})
    model = SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1', 'test-model',
        transport=httpx.MockTransport(serve))
    requirements = AnswerRequirements(format='json_object', output_schema=SCHEMA)
    loop = EvidenceToolLoop(model, binding(engine), initial_search=True,
        limits=ToolLoopLimits(max_tool_calls=1 if exhausted else 4), answer_requirements=requirements)
    if accepted:
        result = await loop.run([{'role':'user', 'content':'Which star does Juniper use?'}])
        assert json.loads(result.text) == answer and await result.validate()
        assert result.source_status == 'retained' and result.verified_accuracy is False
    else:
        with pytest.raises(RuntimeError, match='answer format'):
            await loop.run([{'role':'user', 'content':'Which star does Juniper use?'}])
    assert len(requests) == 1
    schema = requests[0]['response_format']['json_schema']['schema']
    embedded = schema if exhausted else next(branch['properties']['answer'] for branch in schema['anyOf']
        if branch['properties']['action']['const'] == 'answer')
    assert embedded == requirements.output_schema
    value = answer if exhausted else {'action':'answer', 'answer':answer}
    assert Draft202012Validator(schema).is_valid(value) is accepted


async def test_provider_cannot_mutate_host_schema_to_admit_wrong_fields(engine):
    class Mutating(Script):
        async def complete_with_requirements(self, messages, tools, requirements):
            requirements.output_schema.clear()
            return ToolStep(content='{"star":"Polaris"}')

    with pytest.raises(RuntimeError, match='answer format'):
        await EvidenceToolLoop(Mutating(), binding(engine),
            answer_requirements=AnswerRequirements(format='json_object', output_schema=SCHEMA)).run(
                [{'role':'user', 'content':'Which star?'}])


async def test_schema_valid_answer_does_not_bypass_source_retention(engine):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    async def delete_then_answer():
        await engine.forget('alpha', episode.episode_id)
        return ToolStep(content='{"answer":"Polaris"}')
    with pytest.raises(RuntimeError, match='evidence changed'):
        await EvidenceToolLoop(Script(delete_then_answer), binding(engine), initial_search=True,
            answer_requirements=AnswerRequirements(format='json_object', output_schema=SCHEMA)).run(
                [{'role':'user', 'content':'Which star does Juniper use?'}])


@pytest.mark.parametrize('review', [False, True])
async def test_text_publication_waits_for_schema_validation_after_optional_review(engine, review):
    await engine.remember('alpha', 'Juniper uses Polaris.')
    reviewer = Reviewer(revise('{"answer":"Polaris"}'), AnswerReviewDecision(status='supported'))
    requirements = AnswerRequirements(format='json_object', output_schema=SCHEMA)
    settings = {'answer_reviewer':reviewer} if review else {}
    conversation = TextConversation(engine, 'alpha', 'schema-publication', lambda:Model('{"star":"Polaris"}'),
        answer_requirements=requirements, **settings)
    observed = []
    async def observe(text):
        observed.append(text)
    if review:
        result = await conversation.reply('Which star does Juniper use?', on_text=observe)
        assert result['text'] == '{"answer":"Polaris"}' and observed == [result['text']]
        assert reviewer.calls[0][1] == requirements
    else:
        with pytest.raises(RuntimeError, match='answer format'):
            await conversation.reply('Which star does Juniper use?', on_text=observe)
        assert observed == []
    saved = [row.content for row in await engine.episodes('alpha', {'session_id':'schema-publication'})
             if row.metadata['role'] == 'assistant']
    assert saved == observed
    await conversation.close()
