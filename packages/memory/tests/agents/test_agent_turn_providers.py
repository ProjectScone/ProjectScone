"""Native journal resumption through both actual local provider wire protocols."""
import json

import httpx
import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolLoopLimits
from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.agents.workflow import WorkflowPausableStep, WorkflowRunner
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.realtime.answer_requirements import AnswerRequirements
from .test_custom_tools import tool
from .test_evidence_tool_loop import binding
from .test_task_workflow import memory


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('direct', [False, True])
async def test_local_selected_provider_resumes_exact_proposal_once(memory, tmp_path, structured, direct):
    requests, effects, outputs = [], [], []
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['model'] == 'selected-local-model'
        if len(requests) == 1:
            message = {'role': 'assistant', 'content': json.dumps({'action': 'double_count', 'arguments': {'count': 3}})} if structured else {
                'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'original-proposal', 'type': 'function',
                    'function': {'name': 'double_count', 'arguments': '{"count":3}'}}]}
            reason = 'stop' if structured else 'tool_calls'
        else:
            assert len(requests) == 2 and not direct
            assert 'doubled' in str(body['messages'])
            message = {'role': 'assistant', 'content': '{"doubled":6}'}
            reason = 'stop'
        return httpx.Response(200, json={'choices': [{'finish_reason': reason, 'message': message}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}})
    def execute(arguments, context):
        effects.append((arguments, context.space))
        return {'doubled': 6}
    provider = SelfHostedStructuredToolChat if structured else SelfHostedToolChat
    catalog = AgentCatalog(models=[AgentModel('default', 'Default', '1', lambda: pytest.fail('wrong model')),
        AgentModel('selected', 'Selected', '1', lambda: provider('http://127.0.0.1:11434/v1',
            'selected-local-model', transport=httpx.MockTransport(serve)))], tools=[tool(execute, return_direct=direct)],
        agents=[AgentDefinition(agent_id='worker', instructions='Double the count.', models=('default', 'selected'),
            default_model='default', initial_search=False, tools=('double_count',), limits=ToolLoopLimits(max_tool_calls=1))])
    agent = catalog.bind('worker', model_id='selected')
    async def valid(context):
        return True
    async def run(context):
        try:
            output = await agent.run('Double three', tools=binding(memory), checkpoints=context.checkpoints,
                max_new_operations=1, answer_requirements=AnswerRequirements(format='json_object'))
        except TurnJournalPaused as pause:
            return pause.pause
        outputs.append(output)
        return output.output.text
    args = {'path': tmp_path/'journal', 'key': b'k'*32, 'steps': [WorkflowPausableStep('agent', '1', run)], 'source_verifier': valid}
    for _ in range(3):
        job = WorkflowRunner(**args)
        try:
            result = await job.run('one', space='alpha', scope={}, inputs=None)
        finally:
            job.close()
        assert b'double_count' not in (tmp_path/'journal').read_bytes()
        if result.status == 'completed':
            break
    assert result.status == 'completed' and json.loads(result.results['agent']) == {'doubled': 6}
    assert effects == [({'count': 3}, 'alpha')]
    assert len(requests) == (1 if direct else 2)
    assert outputs[0].model_id == 'selected'
    assert outputs[0].output.usage.total_tokens == len(requests) * 14
