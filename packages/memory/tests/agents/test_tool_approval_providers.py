"""Exact approved calls through the actual local provider and function adapter."""

import json
import httpx
import pytest

from scone_memory.agents.approval_context import ApprovalContext
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.function_tools import function_tool
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.agents.workflow import WorkflowPausableStep, WorkflowRunner
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.realtime.answer_requirements import AnswerRequirements
from scone_memory.retrieval.recall_scope import RecallScope
from .test_evidence_tool_loop import binding
from .test_task_workflow import memory


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('direct', [False, True])
async def test_selected_local_provider_executes_only_the_approved_function(
    memory, tmp_path, structured, direct
):
    requests, effects, outputs = [], [], []

    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['model'] == 'chosen-local-model'
        if len(requests) == 1:
            if structured:
                message = {'role': 'assistant', 'content': '{"action":"multiply","arguments":{"count":3}}'}
                reason = 'stop'
            else:
                message = {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'exact-original-call',
                            'type': 'function',
                            'function': {'name': 'multiply', 'arguments': '{"count":3}'},
                        }
                    ],
                }
                reason = 'tool_calls'
        else:
            assert not direct and len(requests) == 2 and 'doubled' in str(body['messages'])
            content = '{"action":"answer","answer":{"doubled":6}}' if structured else '{"doubled":6}'
            message, reason = {'role': 'assistant', 'content': content}, 'stop'
        return httpx.Response(
            200,
            json={
                'choices': [{'message': message, 'finish_reason': reason}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
            },
        )

    def multiply(count: int, factor: int = 2) -> dict[str, int]:
        """Multiply the supplied count."""
        effects.append((count, factor))
        return {'doubled': count * factor}

    registration = function_tool(multiply, revision='1', requires_approval=True, return_direct=direct)
    provider = SelfHostedStructuredToolChat if structured else SelfHostedToolChat
    catalog = AgentCatalog(
        models=[
            AgentModel('default', 'Default', '1', lambda: pytest.fail('wrong model')),
            AgentModel(
                'selected',
                'Selected',
                '1',
                lambda: provider(
                    'http://127.0.0.1:11434/v1', 'chosen-local-model', transport=httpx.MockTransport(serve)
                ),
            ),
        ],
        tools=[registration],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='Multiply.',
                models=('default', 'selected'),
                default_model='default',
                initial_search=False,
                tools=('multiply',),
            )
        ],
    )
    agent = catalog.bind('worker', model_id='selected')
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    store = AgentApprovalStore(runs)
    saved = plans.save(
        'alpha',
        AgentTaskPlan(
            workflow_id='multiply',
            tasks=(
                AgentTask(task_id='calculate', agent_id='worker', model_id='selected', prompt='Multiply'),
            ),
        ),
        catalog=catalog,
        expected_revision=0,
    )
    runs.register(
        'alpha',
        'one',
        plan=saved,
        question='Multiply three',
        scope=RecallScope.validated(where={'team': 'blue'}),
        exclude_session_id='current',
    )
    activation = None

    async def valid(context):
        return True

    async def execute(context):
        approval = ApprovalContext(
            store, context, step_id='calculate', selection_id='calculate', activation_id=activation
        )
        try:
            result = await agent.run(
                'Multiply three',
                tools=binding(memory),
                checkpoints=context.checkpoints,
                approval=approval,
                answer_requirements=AnswerRequirements(format='json_object'),
            )
        except TurnJournalPaused as pause:
            return pause.pause
        outputs.append(result)
        return result.output.text

    async def run():
        runner = WorkflowRunner(
            tmp_path / 'workflow',
            key=b'k' * 32,
            steps=[WorkflowPausableStep('calculate', '1', execute)],
            source_verifier=valid,
        )
        try:
            return await runner.run('one', space='alpha', scope={}, inputs='Multiply three')
        finally:
            runner.close()

    try:
        assert (await run()).status == 'paused' and effects == [] and len(requests) == 1
        (record,) = store.list('alpha', 'one')
        assert record.call.model_id == 'selected' and record.call.arguments_json == '{"count":3}'
        store.decide(
            'alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1
        )
        assert (await run()).status == 'paused' and effects == [] and len(requests) == 1
        store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2})
        activation = 'continue'
        result = await run()
        assert result.status == 'completed' and json.loads(result.results['calculate']) == {'doubled': 6}
        assert effects == [(3, 2)] and len(requests) == (1 if direct else 2)
        assert (
            outputs[0].model_id == 'selected' and outputs[0].output.usage.total_tokens == len(requests) * 14
        )
        for path in ('runs', 'workflow'):
            assert b'exact-original-call' not in (tmp_path / path).read_bytes()
    finally:
        runs.close()
        plans.close()
