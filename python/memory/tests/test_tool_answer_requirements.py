"""Standalone tool turns enforce host output contracts before returning text."""
import copy
import json

import httpx
import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolStep
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.realtime.answer_requirements import AnswerRequirements
from test_evidence_tool_loop import Script, binding, search


@pytest.mark.parametrize('reply', ['Polaris', '{"star":"Polaris","star":"Sirius"}',
                                  '{"star":"Polaris"}\nExplanation'])
async def test_invalid_final_answer_is_withheld_without_retry_or_repair(engine, reply):
    model = Script(ToolStep(content=reply))
    loop = EvidenceToolLoop(model, binding(engine),
        answer_requirements=AnswerRequirements(format='json_object'))
    with pytest.raises(RuntimeError, match='answer format') as error:
        await loop.run([{'role':'user', 'content':'Return the recorded star.'}])
    assert reply not in str(error.value)
    assert len(model.requests) == 1


async def test_contract_snapshot_reaches_each_round_and_leaves_input_unchanged(engine):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    requirements = AnswerRequirements(instructions='Return only the star name.', max_lines=1)
    model = Script(search(), ToolStep(content='Polaris'))
    loop = EvidenceToolLoop(model, binding(engine), answer_requirements=requirements)
    object.__setattr__(requirements, 'instructions', 'MUTATED_AFTER_CONSTRUCTION')
    conversation = [{'role':'system', 'content':'Answer using recorded sources.'},
                    {'role':'user', 'content':'Which star does Juniper use?'}]
    original = copy.deepcopy(conversation)
    result = await loop.run(conversation)
    assert result.text == 'Polaris' and result.verified_accuracy is False
    assert result.source_status == 'retained' and await result.validate()
    assert conversation == original
    for messages, _ in model.requests:
        system = [row['content'] for row in messages if row['role'] == 'system']
        assert any('Return only the star name.' in text and '"max_lines":1' in text for text in system)
        assert 'MUTATED_AFTER_CONSTRUCTION' not in json.dumps(messages)


@pytest.mark.parametrize('requirements,reply', [
    (AnswerRequirements(max_bytes=5), '界界'),
    (AnswerRequirements(max_lines=1), 'Polaris\u2028Explanation'),
])
async def test_utf8_and_line_limits_are_enforced_in_addition_to_loop_limits(engine, requirements, reply):
    model = Script(ToolStep(content=reply))
    with pytest.raises(RuntimeError, match='answer format'):
        await EvidenceToolLoop(model, binding(engine), answer_requirements=requirements).run(
            [{'role':'user', 'content':'Which star?'}])


async def test_requirements_consume_transcript_budget_before_initial_retrieval(engine, monkeypatch):
    tools = binding(engine)
    async def unexpected(*args):
        pytest.fail('over-budget contract reached retrieval')
    monkeypatch.setattr(tools, 'prepare', unexpected)
    model = Script()
    loop = EvidenceToolLoop(model, tools, initial_search=True,
        limits=ToolLoopLimits(max_transcript_bytes=1024),
        answer_requirements=AnswerRequirements(instructions='界' * 400))
    with pytest.raises(RuntimeError, match='transcript byte limit'):
        await loop.run([{'role':'user', 'content':'Juniper?'}])
    assert model.requests == []


@pytest.mark.parametrize('requirements', [
    {'max_bytes':1}, AnswerRequirements.model_construct(max_bytes=-1),
    AnswerRequirements().model_copy(update={'max_lines':False}),
])
def test_bypassed_requirements_fail_at_construction(engine, requirements):
    with pytest.raises(ValueError):
        EvidenceToolLoop(Script(), binding(engine), answer_requirements=requirements)


@pytest.mark.parametrize('exhausted', [False, True])
@pytest.mark.parametrize('structured', [False, True])
async def test_provider_requests_preserve_contract_and_original_query(engine, exhausted, structured):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    requests = []
    answer = ' {"star":"Polaris"} '
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        content = '{"action":"answer","answer":'+answer+'}' if structured and not exhausted else answer
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop',
            'message':{'role':'assistant', 'content':content}}]})
    provider = SelfHostedStructuredToolChat if structured else SelfHostedToolChat
    model = provider('http://127.0.0.1:11434/v1', 'test-model',
        transport=httpx.MockTransport(serve))
    query = 'Which star does Juniper use?'
    result = await EvidenceToolLoop(model, binding(engine), initial_search=True,
        limits=ToolLoopLimits(max_tool_calls=1 if exhausted else 4),
        answer_requirements=AnswerRequirements(format='json_object', instructions='Use the star field.')).run(
            [{'role':'user', 'content':query}])
    assert result.text == ('{"star":"Polaris"}' if structured else answer)
    assert result.tool_calls == result.model_calls == 1
    assert await result.validate() and result.verified_accuracy is False
    body = requests[0]
    assert ('response_format' in body) is structured
    assert {'role':'user', 'content':query} in body['messages']
    if structured:
        assert body['messages'][-1] == {'role':'user', 'content':query}
    system = '\n'.join(row['content'] for row in body['messages'] if row['role'] == 'system')
    assert 'Use the star field.' in system and '"format":"json_object"' in system


async def test_format_valid_answer_still_requires_current_sources(engine):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    async def delete_then_answer():
        await engine.forget('alpha', episode.episode_id)
        return ToolStep(content='{"star":"Polaris"}')
    model = Script(search(), delete_then_answer)
    with pytest.raises(RuntimeError, match='evidence changed'):
        await EvidenceToolLoop(model, binding(engine),
            answer_requirements=AnswerRequirements(format='json_object')).run(
                [{'role':'user', 'content':'Which star does Juniper use?'}])


async def test_requirements_remain_optional_for_legacy_callers(engine):
    reply = 'Polaris\nThe recorded calibration star.'
    result = await EvidenceToolLoop(Script(ToolStep(content=reply)), binding(engine)).run(
        [{'role':'user', 'content':'Which star?'}])
    assert result.text == reply


async def test_capable_provider_receives_separate_contract_snapshot(engine):
    class Constrained(Script):
        async def complete(self, messages, tools):
            pytest.fail('format-capable provider used unconstrained path')

        async def complete_with_requirements(self, messages, tools, requirements):
            assert requirements.max_lines == 1
            object.__setattr__(requirements, 'max_lines', None)
            return ToolStep(content='Polaris\nExplanation')

    with pytest.raises(RuntimeError, match='answer format'):
        await EvidenceToolLoop(Constrained(), binding(engine),
            answer_requirements=AnswerRequirements(max_lines=1)).run(
                [{'role':'user', 'content':'Which star?'}])


@pytest.mark.parametrize('exhausted', [False, True])
async def test_object_generation_preserves_numbers_and_escaped_content(engine, exhausted):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    requests = []
    value = '{\n "measurement": 1.234567890123456789, "large": 1e400, "note": "quoted \\\" text\\nline"\n}'
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        content = value if exhausted else '{"action":"answer","answer":'+value+'}'
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop',
            'message':{'role':'assistant', 'content':content}}]})
    model = SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1', 'test-model',
        transport=httpx.MockTransport(serve))
    result = await EvidenceToolLoop(model, binding(engine), initial_search=True,
        limits=ToolLoopLimits(max_tool_calls=1 if exhausted else 4),
        answer_requirements=AnswerRequirements(format='json_object', max_lines=1)).run(
            [{'role':'user', 'content':'Report the measurement.'}])
    assert result.text == '{"measurement":1.234567890123456789,"large":1e400,"note":"quoted \\\" text\\nline"}'
    schema = requests[0]['response_format']['json_schema']['schema']
    if exhausted:
        assert schema['type'] == 'object'
    else:
        branch = next(b for b in schema['anyOf'] if b['properties']['action']['const']=='answer')
        assert branch['properties']['answer']['type'] == 'object'


async def test_object_contract_preserves_intermediate_tool_calls(engine):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team':'blue'})
    requests = []
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        content = ('{"action":"search_memory","query":"Juniper","limit":1}'
                   if len(requests) == 1 else '{"action":"answer","answer":{"star":"Polaris"}}')
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop',
            'message':{'role':'assistant', 'content':content}}]})
    model = SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1', 'test-model',
        transport=httpx.MockTransport(serve))
    result = await EvidenceToolLoop(model, binding(engine),
        answer_requirements=AnswerRequirements(format='json_object')).run(
            [{'role':'user', 'content':'Which star does Juniper use?'}])
    assert result.text == '{"star":"Polaris"}' and result.model_calls == 2
    assert result.tool_calls == 1 and result.tool_outcomes[0].name == 'search_memory'
    assert result.source_status == 'retained' and await result.validate()
