"""Guarded calls require an exact decision and an explicit activation."""

from dataclasses import replace
import pytest

from scone_memory.agents.approval_context import ApprovalContext
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.agents.workflow import WorkflowPausableStep, WorkflowRunner
from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolStep
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import agents, call, tool
from .test_evidence_tool_loop import Script, binding
from .test_task_workflow import memory


@pytest.mark.parametrize('decision', ['approve', 'deny'])
@pytest.mark.parametrize('parallel', [False, True])
async def test_decision_alone_never_dispatches_and_activation_survives_reopen(
    memory, tmp_path, decision, parallel
):
    effects, outputs = [], []
    model = Script(call(), ToolStep(content='Finished.'))
    registry = agents(model, [tool(lambda args, ctx: effects.append(args['count']), requires_approval=True)])
    agent = registry.bind('worker')
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    store = AgentApprovalStore(runs)
    saved = plans.save(
        'alpha',
        AgentTaskPlan(
            workflow_id='job', tasks=(AgentTask(task_id='count', agent_id='worker', prompt='Count'),)
        ),
        catalog=registry,
        expected_revision=0,
    )
    runs.register(
        'alpha',
        'one',
        plan=saved,
        question='Count',
        scope=RecallScope.validated(where={'team': 'blue'}),
        exclude_session_id='current',
    )
    activation_id = None

    async def valid(context):
        return True

    async def execute(context):
        approval = ApprovalContext(
            store, context, step_id='count', selection_id='count', activation_id=activation_id
        )
        try:
            result = await agent.run(
                'Count', tools=binding(memory), checkpoints=context.checkpoints, approval=approval
            )
        except TurnJournalPaused as pause:
            return pause.pause
        outputs.append(result.output)
        return result.output.text

    settings = dict(
        key=b'k' * 32,
        steps=[WorkflowPausableStep('count', '1', execute)],
        source_verifier=valid,
        dependencies={'count': []} if parallel else None,
    )

    async def run_once():
        runner = WorkflowRunner(tmp_path / 'workflow', **settings)
        try:
            return await runner.run('one', space='alpha', scope={}, inputs='Count')
        finally:
            runner.close()

    try:
        assert (await run_once()).status == 'paused'
        (record,) = store.list('alpha', 'one')
        assert record.call.arguments_json == '{"count":3}' and record.revision == 1
        assert effects == [] and len(model.requests) == 1
        store.decide('alpha', 'one', record.request_id, decision=decision, actor='owner', expected_revision=1)
        assert (await run_once()).status == 'paused'
        assert effects == [] and len(model.requests) == 1
        store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2})
        assert (await run_once()).status == 'paused'
        activation_id = 'continue'
        assert (await run_once()).status == 'completed'
        assert effects == ([3] if decision == 'approve' else []) and len(model.requests) == 2
        assert store.get('alpha', 'one', record.request_id).revision == 4
        assert outputs[0].tool_outcomes[0].error == (None if decision == 'approve' else 'approval_denied')
        assert (await run_once()).status == 'completed'
        assert effects == ([3] if decision == 'approve' else []) and len(model.requests) == 2
    finally:
        runs.close()
        plans.close()


async def test_guarded_tools_refuse_missing_context_before_factory_or_model(memory):
    model = Script(call())
    agent = agents(model, [tool(lambda *_: pytest.fail('executed'), requires_approval=True)]).bind('worker')
    factories = []
    agent = replace(agent, model=replace(agent.model, factory=lambda: factories.append(1) or model))
    with pytest.raises(ValueError, match='approval'):
        await agent.run('Count', tools=binding(memory))
    with pytest.raises(ValueError, match='approval'):
        EvidenceToolLoop(model, binding(memory), custom_tools=agent.tools)
    assert factories == [] and model.requests == []


def test_approval_policy_changes_fingerprint_without_changing_legacy_shape():
    registration = tool(lambda *_: None)
    assert 'requires_approval' not in registration.info()
    guarded = replace(registration, requires_approval=True)
    assert guarded.snapshot().requires_approval and guarded.info()['requires_approval'] is True
    assert (
        agents(Script(), [registration]).bind('worker').fingerprint
        != agents(Script(), [guarded]).bind('worker').fingerprint
    )
