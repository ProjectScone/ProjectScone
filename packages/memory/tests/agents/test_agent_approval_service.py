"""Saved runs expose decisions separately from explicit owned continuation."""

import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.handoff_workflow import AgentHandoffPlan, HandoffAgent
from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import call, tool
from .test_evidence_tool_loop import Script
from .test_task_workflow import memory


@pytest.mark.parametrize('kind', ['task', 'interactive', 'handoff'])
@pytest.mark.parametrize('decision', ['approve', 'deny'])
async def test_saved_service_approval_reopens_without_reproposing(memory, tmp_path, kind, decision):
    effects = []
    final = '{"answer":"Done.","handoff_to":null}' if kind == 'handoff' else 'Done.'
    model = Script(call(), ToolStep(content=final))
    registry = AgentCatalog(
        models=[
            AgentModel('default', 'Default', '1', lambda: pytest.fail('wrong model')),
            AgentModel('selected', 'Selected', '1', lambda: model),
        ],
        tools=[tool(lambda args, ctx: effects.append(args['count']), requires_approval=True)],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='Work.',
                models=('default', 'selected'),
                default_model='default',
                initial_search=False,
                tools=('double_count',),
            )
        ],
    )
    task = AgentTask(task_id='work', agent_id='worker', model_id='selected', prompt='Work')
    if kind == 'task':
        plan = AgentTaskPlan(workflow_id='job', tasks=(task,))
    elif kind == 'handoff':
        plan = AgentHandoffPlan(
            workflow_id='job',
            root_agent='worker',
            max_handoffs=0,
            agents=(HandoffAgent(agent_id='worker', model_id='selected'),),
        )
    else:
        plan = InteractiveAgentPlan(
            kind='interactive',
            workflow_id='job',
            tasks=(
                task,
                HumanInputTask(kind='input', task_id='review', prompt='Review', depends_on=('work',)),
            ),
        )
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    plans.save('alpha', plan, catalog=registry, expected_revision=0)

    def open_service():
        return AgentRunService(
            tmp_path / 'runs',
            key=b'k' * 32,
            catalog=registry,
            plans=plans,
            memory=memory,
            scope_for=lambda _: RecallScope.validated(),
            max_parallel_tasks=2,
        )

    service = open_service()
    try:
        await service.start('alpha', 'one', workflow_id='job', plan_revision=1, question='Count')
        paused = await service.wait('alpha', 'one')
        assert paused.status == 'paused' and not paused.outcome_unknown and not paused.active_local
        assert paused.paused_steps == (('hop-01',) if kind == 'handoff' else ('work',))
        await service.aclose()
        service = open_service()
        records = await service.approvals('alpha', 'one')
        (pending,) = records
        assert pending.call.model_id == 'selected' and pending.call.arguments_json == '{"count":3}'
        assert effects == [] and len(model.requests) == 1
        decided = await service.decide_tool(
            'alpha', 'one', pending.request_id, decision=decision, actor='owner', expected_revision=1
        )
        assert decided.revision == 2
        again = await service.start('alpha', 'one', workflow_id='job', plan_revision=1, question='Count')
        assert again.status == 'paused' and not again.active_local
        assert effects == [] and len(model.requests) == 1
        admitted = await service.continue_tools(
            'alpha', 'one', continuation_id='continue', decisions={pending.request_id: 2}
        )
        assert admitted.activation.activation_id == 'continue'
        assert admitted.activation.decisions == {pending.request_id: 2}
        assert admitted.status.active_local
        complete = await service.wait('alpha', 'one')
        assert complete.status == ('awaiting_input' if kind == 'interactive' else 'completed')
        assert effects == ([3] if decision == 'approve' else []) and len(model.requests) == 2
        (history,) = await service.approvals('alpha', 'one')
        assert history.revision == 4
        replay = await service.continue_tools(
            'alpha', 'one', continuation_id='continue', decisions={pending.request_id: 2}
        )
        assert replay.activation == admitted.activation and not replay.status.active_local
        assert len(model.requests) == 2
    finally:
        await service.aclose()
        plans.close()
