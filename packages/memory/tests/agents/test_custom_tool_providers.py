"""Registered functions run through local provider protocols and saved workflows."""
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.custom_tools import ToolContext
from scone_memory.agents.function_tools import function_tool
from scone_memory.agents.evidence_loop import ToolLoopLimits
from scone_memory.agents.task_requirements import TaskAnswerRequirements
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import tool
from .test_task_workflow import memory


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('exhausted', [False, True])
@pytest.mark.parametrize('inferred', [False, True])
@pytest.mark.parametrize('direct', [False, True])
async def test_selected_local_provider_executes_custom_tool_and_reuses_saved_result(tmp_path, memory, structured, exhausted, inferred, direct):
    requests, invocations = [], []
    output_schema = {'type': 'object', 'properties': {'doubled': {'type': 'integer'}},
                     'required': ['doubled'], 'additionalProperties': False}

    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['model'] == 'selected-local-model'
        if len(requests) == 1:
            if structured:
                action = {'action': 'double_count', 'arguments': {'count': 3}}
                validator = Draft202012Validator(body['response_format']['json_schema']['schema'])
                assert validator.is_valid(action)
                assert not validator.is_valid({'action': 'double_count', 'arguments': {'count': True}})
                message = {'role': 'assistant', 'content': json.dumps(action)}
                reason = 'stop'
            else:
                custom = next(row for row in body['tools'] if row['function']['name'] == 'double_count')
                assert custom['function']['parameters'] == registration.openai()['function']['parameters']
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'custom-1', 'type': 'function', 'function': {
                        'name': 'double_count', 'arguments': '{"count":3}'}}]}
                reason = 'tool_calls'
        else:
            assert len(requests) == 2
            assert 'unverified' in str(body['messages']) and 'doubled' in str(body['messages'])
            final = {'doubled': 6}
            content = {'action': 'answer', 'answer': final} if structured and not exhausted else final
            if structured:
                assert Draft202012Validator(body['response_format']['json_schema']['schema']).is_valid(content)
            message = {'role': 'assistant', 'content': json.dumps(content)}
            reason = 'stop'
        return httpx.Response(200, json={'choices': [{'finish_reason': reason, 'message': message}]})

    async def execute(arguments, context):
        invocations.append((arguments, context.space, context.scope.kwargs()))
        return {'doubled': 2 * arguments['count']}

    async def double_count(count: int, context: ToolContext, multiplier: int = 2) -> object:
        """Multiply the supplied count by the configured factor."""
        assert multiplier == 2
        return await execute({'count': count}, context)

    registration = (function_tool(double_count, revision='1', context_parameter='context', return_direct=direct)
                    if inferred else tool(execute, return_direct=direct))

    provider = SelfHostedStructuredToolChat if structured else SelfHostedToolChat
    catalog = AgentCatalog(models=[
        AgentModel('unused', 'Unused', '1', lambda: pytest.fail('selected wrong model')),
        AgentModel('selected', 'Selected local model', '1', lambda: provider(
            'http://127.0.0.1:11434/v1', 'selected-local-model', transport=httpx.MockTransport(serve)))],
        agents=[AgentDefinition(agent_id='worker', instructions='Use the count tool.',
            models=('unused', 'selected'), default_model='unused', initial_search=False,
            tools=('double_count',), limits=ToolLoopLimits(max_tool_calls=1 if exhausted else 8))],
        tools=[registration])
    plan = AgentTaskPlan(workflow_id='count', tasks=(AgentTask(task_id='one', agent_id='worker',
        model_id='selected', prompt='Double three.', answer_requirements=TaskAnswerRequirements(
            format='json_object', output_schema=output_schema)),))

    def create():
        return AgentWorkflow(tmp_path/'journal', key=b'k'*32, catalog=catalog, plan=plan,
            memory=memory, space='alpha', scope=RecallScope.validated(where={'team': 'blue'}))

    work = create()
    try:
        result = await work.run('r', 'Double three')
        receipt = result.results['one']
        assert json.loads(receipt['text']) == {'doubled': 6}
        assert receipt['model_id'] == 'selected' and receipt['source_status'] == 'none'
        assert receipt['evidence_ids'] == []
        assert invocations[0][0] == {'count': 3} and invocations[0][1] == 'alpha'
        assert invocations[0][2]['where'] == {'team': 'blue'}
    finally:
        work.close()
    work = create()
    try:
        saved = await work.run('r', 'Double three')
        assert saved.results == result.results and saved.reused_steps == ('one',)
        assert len(invocations) == 1 and len(requests) == (1 if direct else 2)
    finally:
        work.close()
