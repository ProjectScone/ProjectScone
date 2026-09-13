"""Native agent usage preserves missing provider reports and call boundaries."""
import json

import httpx
import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolStep
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from .test_evidence_tool_loop import Script, binding, search


def packet(content, usage):
    return {'choices': [{'finish_reason': 'stop', 'message': {
        'role': 'assistant', 'content': content}}], 'usage': usage}


@pytest.mark.parametrize('protocol', ['native', 'structured_action', 'structured_answer', 'structured_json'])
async def test_all_provider_paths_preserve_usage_outside_model_authored_content(protocol):
    from scone_memory.realtime.answer_requirements import AnswerRequirements
    tools = []
    content = 'Done.'
    if protocol == 'structured_action':
        tools = [{'type': 'function', 'function': {'name': 'search_memory'}}]
        content = json.dumps({'action': 'answer', 'answer': 'Done.'})
    if protocol == 'structured_json':
        content = '{"answer":"Done."}'
    counts = {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17,
              'private_metadata': 'SECRET_DO_NOT_RETAIN'}
    provider = SelfHostedToolChat if protocol == 'native' else SelfHostedStructuredToolChat
    model = provider('http://127.0.0.1:18999/v1', 'fixture', transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=packet(content, counts))))
    if protocol == 'structured_json':
        step = await model.complete_with_requirements([{'role': 'user', 'content': 'Hello'}], tools,
                                                     AnswerRequirements(format='json_object'))
    else:
        step = await model.complete([{'role': 'user', 'content': 'Hello'}], tools)
    assert step.usage.model_dump() == {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17}
    assert 'SECRET_DO_NOT_RETAIN' not in step.model_dump_json()


@pytest.mark.parametrize('usage, expected', [
    (None, (None, None, None)),
    ('private', (None, None, None)),
    ({'prompt_tokens': True, 'completion_tokens': -1, 'total_tokens': '19'}, (None, None, None)),
    ({'prompt_tokens': 1.5, 'completion_tokens': 10**9 + 1, 'total_tokens': []}, (None, None, None)),
    ({'prompt_tokens': 10, 'completion_tokens': 2}, (10, 2, None)),
    ({'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 3}, (10, 2, None)),
    ({'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 13}, (10, 2, None)),
    ({'prompt_tokens': 10, 'total_tokens': 3}, (10, None, None)),
    ({'completion_tokens': 5, 'total_tokens': 3}, (None, 5, None)),
    ({'total_tokens': 8}, (None, None, 8)),
    ({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, (0, 0, 0)),
])
async def test_optional_provider_usage_is_not_inferred_and_invalid_fields_are_unknown(usage, expected):
    model = SelfHostedToolChat('http://127.0.0.1:18999/v1', 'fixture', transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=packet('Done.', usage))))
    step = await model.complete([{'role': 'user', 'content': 'Hello'}], [])
    assert step.content == 'Done.'
    assert (step.usage.prompt_tokens, step.usage.completion_tokens, step.usage.total_tokens) == expected


async def test_multiple_rounds_report_complete_totals_without_counting_tools(engine):
    from scone_memory.agents.usage import ModelTokenUsage
    first = search().model_copy(update={'usage': ModelTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)})
    final = ToolStep(content='No evidence.', usage=ModelTokenUsage(prompt_tokens=20, completion_tokens=3, total_tokens=23))
    result = await EvidenceToolLoop(Script(first, final), binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])
    assert result.model_calls == len(result.usage.calls) == 2
    assert result.tool_calls == 1
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (30, 5, 35)
    assert result.usage.calls == (first.usage, final.usage)


async def test_legacy_model_missing_report_does_not_make_partial_totals_look_complete(engine):
    from scone_memory.agents.usage import ModelTokenUsage
    first = search().model_copy(update={'usage': ModelTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)})
    result = await EvidenceToolLoop(Script(first, ToolStep(content='Done.')), binding(engine)).run(
        [{'role': 'user', 'content': 'Juniper?'}])
    assert result.model_calls == len(result.usage.calls) == 2
    assert result.usage.prompt_tokens is result.usage.completion_tokens is result.usage.total_tokens is None
    assert result.usage.calls[0].total_tokens == 12 and result.usage.calls[1].total_tokens is None


async def test_report_is_detached_before_next_model_call(engine):
    from scone_memory.agents.usage import ModelTokenUsage
    usage = ModelTokenUsage(prompt_tokens=10)
    first = search().model_copy(update={'usage': usage})
    async def mutate():
        object.__setattr__(usage, 'prompt_tokens', 999)
        return ToolStep(content='Done.', usage=ModelTokenUsage(prompt_tokens=20))
    result = await EvidenceToolLoop(Script(first, mutate), binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])
    assert result.usage.prompt_tokens == 30
    assert result.usage.completion_tokens is None and result.usage.total_tokens is None


async def test_model_authored_usage_is_only_answer_text(engine):
    content = '{"usage":{"total_tokens":1}}'
    result = await EvidenceToolLoop(Script(ToolStep(content=content)), binding(engine)).run(
        [{'role': 'user', 'content': 'Juniper?'}])
    assert result.text == content and result.usage.total_tokens is None


async def test_host_search_is_not_a_model_call_and_catalog_reports_selected_model(engine):
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.usage import ModelTokenUsage
    models = [AgentModel(model_id, model_id, '1', lambda count=count: Script(
        ToolStep(content='Done.', usage=ModelTokenUsage(prompt_tokens=count))))
        for model_id, count in [('small', 10), ('large', 20)]]
    catalog = AgentCatalog(models=models, agents=[AgentDefinition(
        agent_id='research', instructions='Use evidence.', models=('small', 'large'), default_model='small',
        initial_search=True)])
    for selected, expected in [('large', 20), ('small', 10)]:
        result = await catalog.bind('research', model_id=selected).run('Juniper?', tools=binding(engine))
        assert result.model_id == selected and result.output.model_calls == 1
        assert len(result.output.usage.calls) == 1 and result.output.usage.prompt_tokens == expected
        assert result.output.tool_outcomes[0].origin == 'host'
        assert result.output.usage.total_tokens is None


@pytest.mark.parametrize('bad', [True, -1, 1.2, '10', 10**9 + 1])
async def test_custom_adapter_cannot_bypass_strict_usage_validation(engine, bad):
    from scone_memory.agents.usage import ModelTokenUsage
    usage = ModelTokenUsage.model_construct(prompt_tokens=bad)
    step = ToolStep.model_construct(content='Done.', calls=(), usage=usage)
    with pytest.raises(RuntimeError, match='invalid tool protocol'):
        await EvidenceToolLoop(Script(step), binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])


async def test_inconsistent_custom_adapter_total_is_protocol_error(engine):
    from scone_memory.agents.usage import ModelTokenUsage
    usage = ModelTokenUsage.model_construct(prompt_tokens=10, completion_tokens=2, total_tokens=1)
    step = ToolStep.model_construct(content='Done.', calls=(), usage=usage)
    with pytest.raises(RuntimeError, match='invalid tool protocol'):
        await EvidenceToolLoop(Script(step), binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])


def test_empty_usage_is_unknown_and_serialization_preserves_coverage():
    from scone_memory.agents.usage import ModelTokenUsage, ToolTokenUsage
    empty = ToolTokenUsage()
    assert empty.prompt_tokens is empty.completion_tokens is empty.total_tokens is None
    usage = ToolTokenUsage(calls=(ModelTokenUsage(prompt_tokens=10), ModelTokenUsage(total_tokens=15)))
    reopened = ToolTokenUsage.model_validate_json(usage.model_dump_json())
    assert reopened == usage
    assert reopened.calls[0].prompt_tokens == 10 and reopened.calls[1].total_tokens == 15
    assert reopened.prompt_tokens is reopened.total_tokens is None


def test_full_call_budget_sums_above_single_response_limit_without_overflow():
    from scone_memory.agents.usage import ModelTokenUsage, ToolTokenUsage
    call = ModelTokenUsage(prompt_tokens=10**9, completion_tokens=0, total_tokens=10**9)
    usage = ToolTokenUsage(calls=(call,) * 17)
    assert usage.prompt_tokens == usage.total_tokens == 17 * 10**9
    assert usage.completion_tokens == 0
    with pytest.raises(ValueError):
        ToolTokenUsage(calls=(call,) * 18)


async def test_invalid_custom_usage_cannot_leak_values_through_serializer_warnings(engine):
    import warnings
    from scone_memory.agents.usage import ModelTokenUsage
    usage = ModelTokenUsage.model_construct(prompt_tokens='PRIVATE_PROVIDER_VALUE')
    step = ToolStep.model_construct(content='Done.', calls=(), usage=usage)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        with pytest.raises(RuntimeError, match='^invalid tool protocol$') as failure:
            await EvidenceToolLoop(Script(step), binding(engine)).run([{'role': 'user', 'content': 'Juniper?'}])
    assert 'PRIVATE_PROVIDER_VALUE' not in str(failure.value)
    assert not emitted


async def test_usage_metadata_does_not_reduce_existing_protocol_byte_budget(engine):
    from scone_memory.agents.evidence_loop import ToolCall
    calls = tuple(ToolCall(id=f'call-{index}', name='unknown_tool', arguments={'value': 'x' * 15000})
                  for index in range(8))
    step = ToolStep(calls=calls)
    payload = {'content': '', 'calls': [call.model_dump() for call in calls]}
    size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode())
    step = step.model_copy(update={'content': 'x' * (127990 - size)})
    result = await EvidenceToolLoop(Script(step, ToolStep(content='Done.')), binding(engine)).run(
        [{'role': 'user', 'content': 'Juniper?'}])
    assert result.text == 'Done.' and result.model_calls == 2
    assert result.usage.total_tokens is None
