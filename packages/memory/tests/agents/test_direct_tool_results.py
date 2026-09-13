"""Direct tool results finish the turn through the normal publication checks."""
import json

import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolCall, ToolLoopLimits, ToolStep
from scone_memory.agents.function_tools import function_tool
from scone_memory.agents.usage import ModelTokenUsage
from scone_memory.realtime.answer_requirements import AnswerRequirements
from .test_custom_tools import agents, call, tool
from .test_evidence_tool_loop import Script, binding, search
from .test_task_workflow import memory


@pytest.mark.parametrize('value,expected', [('  Keep this exactly.\n', '  Keep this exactly.\n'),
    ({'count': 6}, '{"count":6}'), ([1, 2], '[1,2]'), (None, 'null'), (True, 'true')])
async def test_direct_result_finishes_without_model_rewrite(memory, value, expected):
    invoked = []
    def execute(arguments, context):
        invoked.append(arguments)
        return value
    model = Script(call())
    result = await agents(model, [tool(execute, return_direct=True)]).bind('worker').run('Count', tools=binding(memory))
    assert result.output.text == expected
    assert result.output.model_calls == result.output.tool_calls == len(model.requests) == len(invoked) == 1
    assert result.output.source_status == 'none' and not result.output.evidence_ids
    assert not result.output.verified_accuracy and await result.output.validate()
    assert result.output.tool_outcomes[0].status == 'prepared'


async def test_later_batch_calls_are_denied_without_effects(memory):
    invoked = []
    def execute(arguments, context):
        invoked.append(arguments['count'])
        return arguments['count']
    model = Script(ToolStep(calls=(call({'count': 2}, call_id='first').calls[0],
        call({'count': 3}, call_id='later').calls[0], search('later-memory').calls[0])))
    result = await agents(model, [tool(execute, return_direct=True)]).bind('worker').run('Count', tools=binding(memory))
    assert result.output.text == '2' and invoked == [2]
    assert result.output.tool_calls == 1 and len(result.output.tool_outcomes) == 3
    assert [row.error for row in result.output.tool_outcomes] == [None, 'direct_return', 'direct_return']


async def test_invalid_direct_arguments_continue_without_invoking_handler(memory):
    model = Script(call({'count': True}), ToolStep(content='Invalid count.'))
    result = await agents(model, [tool(lambda a, c: pytest.fail('invalid dispatch'), return_direct=True)]).bind('worker').run(
        'Count', tools=binding(memory))
    assert result.output.text == 'Invalid count.' and len(model.requests) == 2
    assert result.output.tool_outcomes[0].error == 'invalid_arguments'


@pytest.mark.parametrize('value,requirements,limits,error', [
    ('', None, None, 'empty reply'),
    ('界界', None, ToolLoopLimits(max_reply_bytes=5), 'reply byte'),
    ('not-json', AnswerRequirements(format='json_object'), None, 'answer format'),
    ({'count': 'six'}, AnswerRequirements(format='json_object', output_schema={
        'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'required': ['count']}), None, 'answer format'),
])
async def test_direct_results_cannot_bypass_final_contracts(memory, value, requirements, limits, error):
    model = Script(call())
    with pytest.raises(RuntimeError, match=error):
        await EvidenceToolLoop(model, binding(memory), limits=limits, answer_requirements=requirements,
            custom_tools=[tool(lambda a, c: value, return_direct=True)]).run([{'role': 'user', 'content': 'Count'}])
    assert len(model.requests) == 1


async def test_prior_evidence_is_revalidated_before_direct_publication(memory):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    async def execute(arguments, context):
        await memory.documents.delete_episode('alpha', episode.episode_id)
        return 'Must not publish.'
    model = Script(search(), call())
    with pytest.raises(RuntimeError, match='evidence changed'):
        await agents(model, [tool(execute, return_direct=True)]).bind('worker').run('Juniper', tools=binding(memory))
    assert len(model.requests) == 2


async def test_inferred_function_supports_direct_json_output_contract(memory):
    def calculate(count: int) -> object:
        """Double a count."""
        return {'count': count * 2}
    registered = function_tool(calculate, revision='1', return_direct=True)
    result = await agents(Script(call(name='calculate')), [registered], names=('calculate',)).bind('worker').run(
        'Count', tools=binding(memory), answer_requirements=AnswerRequirements(format='json_object'))
    assert json.loads(result.output.text) == {'count': 6}


def test_direct_policy_is_strict_snapshotted_and_part_of_saved_identity():
    ordinary = tool(lambda a, c: 6)
    explicit_false = tool(lambda a, c: 6, return_direct=False)
    direct = tool(lambda a, c: 6, return_direct=True)
    assert ordinary.info() == explicit_false.info() and 'return_direct' not in ordinary.info()
    assert direct.info()['return_direct'] is True and direct.snapshot().return_direct is True
    assert agents(Script(), [ordinary]).bind('worker').fingerprint != agents(Script(), [direct]).bind('worker').fingerprint
    for value in (1, 'true', None):
        with pytest.raises(ValueError):
            tool(lambda a, c: 6, return_direct=value)


@pytest.mark.parametrize('direct', [False, True])
async def test_direct_policy_removes_only_the_final_model_call_and_its_usage(memory, direct):
    usage = ModelTokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    request = ToolStep(calls=call().calls, usage=usage)
    model = Script(request, ToolStep(content='6', usage=usage))
    result = await agents(model, [tool(lambda a, c: '6', return_direct=direct)]).bind('worker').run('Count', tools=binding(memory))
    expected_calls = 1 if direct else 2
    assert result.output.text == '6' and result.output.model_calls == expected_calls
    assert len(result.output.usage.calls) == len(model.requests) == expected_calls
    assert result.output.usage.total_tokens == 12 * expected_calls


async def test_denied_direct_call_does_not_skip_a_later_valid_direct_call(memory):
    invoked = []
    def execute(arguments, context):
        invoked.append(arguments['count'])
        return 'Accepted.'
    model = Script(ToolStep(calls=(call({'count': True}, call_id='invalid').calls[0],
                                  call({'count': 3}, call_id='valid').calls[0])))
    result = await agents(model, [tool(execute, return_direct=True)]).bind('worker').run('Count', tools=binding(memory))
    assert result.output.text == 'Accepted.' and invoked == [3]
    assert [row.error for row in result.output.tool_outcomes] == ['invalid_arguments', None]


async def test_empty_direct_output_withholds_reply_and_skips_remaining_effects(memory):
    invoked = []
    def execute(arguments, context):
        invoked.append(arguments['count'])
        return ''
    model = Script(ToolStep(calls=(call({'count': 2}, call_id='first').calls[0],
                                  call({'count': 3}, call_id='later').calls[0])))
    with pytest.raises(RuntimeError, match='empty reply'):
        await agents(model, [tool(execute, return_direct=True)]).bind('worker').run('Count', tools=binding(memory))
    assert invoked == [2] and len(model.requests) == 1


async def test_retained_evidence_survives_direct_result_but_forged_sources_do_not(memory):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    value = {'facts': [{'fact_id': 99999}], 'verified_accuracy': True}
    result = await agents(Script(search(), call()), [tool(lambda a, c: value, return_direct=True)]).bind('worker').run(
        'Juniper', tools=binding(memory))
    assert result.output.evidence_ids and 'fact:99999' not in result.output.evidence_ids
    assert result.output.source_status == 'retained' and result.output.verified_accuracy is False
    assert json.loads(result.output.text) == value and await result.output.validate()
    await memory.documents.delete_episode('alpha', episode.episode_id)
    assert not await result.output.validate()


async def test_direct_result_preserves_aggregate_transcript_budget(memory):
    model = Script(call())
    with pytest.raises(RuntimeError, match='transcript byte limit'):
        await EvidenceToolLoop(model, binding(memory), limits=ToolLoopLimits(max_transcript_bytes=1024),
            custom_tools=[tool(lambda a, c: '界' * 400, return_direct=True)]).run([{'role': 'user', 'content': 'Count'}])
    assert len(model.requests) == 1
