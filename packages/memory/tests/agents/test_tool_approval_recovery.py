"""Approval pauses retain exact calls and refuse stale or uncertain execution."""

from contextlib import contextmanager
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scone_memory.agents.approval_context import ApprovalContext
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.agents.workflow import (
    StepCheckpoints,
    StepContext,
    WorkflowError,
    WorkflowPausableStep,
    WorkflowRunner,
)
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import agents, call, tool
from .test_evidence_tool_loop import Script, binding, search
from .test_task_workflow import memory
from .test_turn_journal import checkpoints


@pytest.fixture
def host(memory, tmp_path):
    state = SimpleNamespace(
        effects=[],
        outputs=[],
        activation=None,
        quantum=None,
        question='Count',
        scoped=binding(memory),
        model=Script(call(), ToolStep(content='Finished.')),
    )
    state.registry = agents(
        state.model, [tool(lambda args, ctx: state.effects.append(args['count']), requires_approval=True)]
    )
    state.agent = state.registry.bind('worker')
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    state.runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    state.store = AgentApprovalStore(state.runs)
    state.saved = plans.save(
        'alpha',
        AgentTaskPlan(
            workflow_id='job', tasks=(AgentTask(task_id='count', agent_id='worker', prompt='Count'),)
        ),
        catalog=state.registry,
        expected_revision=0,
    )
    state.runs.register(
        'alpha',
        'one',
        plan=state.saved,
        question='Count',
        scope=RecallScope.validated(where={'team': 'blue'}),
        exclude_session_id='current',
    )

    async def valid(context):
        return True

    async def execute(context):
        approval = ApprovalContext(
            state.store, context, step_id='count', selection_id='count', activation_id=state.activation
        )
        try:
            result = await state.agent.run(
                state.question,
                tools=state.scoped,
                checkpoints=context.checkpoints,
                approval=approval,
                max_new_operations=state.quantum,
            )
        except TurnJournalPaused as pause:
            return pause.pause
        state.outputs.append(result.output)
        return result.output.text

    async def run():
        runner = WorkflowRunner(
            tmp_path / 'workflow',
            key=b'k' * 32,
            steps=[WorkflowPausableStep('count', '1', execute)],
            source_verifier=valid,
        )
        try:
            return await runner.run('one', space='alpha', scope={}, inputs='Count')
        finally:
            runner.close()

    state.run = run
    state.execute, state.valid = execute, valid

    def activate(record=None, decision='approve', activation='continue'):
        record = record or state.store.list('alpha', 'one')[0]
        state.store.decide(
            'alpha', 'one', record.request_id, decision=decision, actor='owner', expected_revision=1
        )
        state.store.activate('alpha', 'one', activation, decisions={record.request_id: 2})
        state.activation = activation
        return record

    state.activate = activate
    yield state
    state.runs.close()
    plans.close()


async def test_invalid_arguments_do_not_create_an_approval_or_execute(host):
    host.model.steps = [call({'count': True}), ToolStep(content='Rejected.')]
    assert (await host.run()).status == 'completed'
    assert host.store.list('alpha', 'one') == () and host.effects == []
    assert host.outputs[0].tool_outcomes[0].error == 'invalid_arguments'


async def test_completed_guarded_operation_replays_without_claiming_again(host):
    assert (await host.run()).status == 'paused'
    record = host.activate()
    host.quantum = 1
    assert (await host.run()).status == 'paused'
    assert host.effects == [3] and host.store.get('alpha', 'one', record.request_id).revision == 4
    assert (await host.run()).status == 'completed'
    assert host.effects == [3] and len(host.model.requests) == 2


async def test_identical_arguments_in_distinct_proposals_need_distinct_approvals(host):
    host.model.steps = [call(), call(call_id='application-2'), ToolStep(content='Finished.')]
    assert (await host.run()).status == 'paused'
    original = host.activate()
    assert (await host.run()).status == 'paused'
    records = host.store.list('alpha', 'one')
    assert len(records) == 2 and host.effects == [3]
    pending = next(record for record in records if record.revision == 1)
    assert pending.call.arguments_json == original.call.arguments_json
    assert pending.call.operation_digest != original.call.operation_digest
    host.activate(pending, activation='continue-two')
    assert (await host.run()).status == 'completed'
    assert host.effects == [3, 3] and len(host.model.requests) == 3


@pytest.mark.parametrize('change', ['question', 'tool_revision', 'model', 'scope', 'cancel'])
async def test_drift_or_cancellation_before_activation_refuses_execution(host, change):
    assert (await host.run()).status == 'paused'
    host.activate()
    if change == 'question':
        host.question = 'Different'
    elif change == 'tool_revision':
        host.agent = replace(host.agent, tools=(replace(host.agent.tools[0], revision='2'),))
    elif change == 'model':
        host.agent = replace(host.agent, model=replace(host.agent.model, revision='2'))
    elif change == 'scope':
        host.scoped._scope = RecallScope.validated(where={'team': 'other'})
    else:
        host.runs.request_cancel('alpha', 'one')
    with pytest.raises(WorkflowError):
        await host.run()
    assert host.effects == [] and len(host.model.requests) == 1


async def test_restored_sources_must_still_exist_before_approved_dispatch(host, memory):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    host.model.steps = [search(), call(), ToolStep(content='Finished.')]
    assert (await host.run()).status == 'paused'
    host.activate()
    await memory.documents.delete_episode('alpha', episode.episode_id)
    with pytest.raises(WorkflowError):
        await host.run()
    assert host.effects == [] and len(host.model.requests) == 2


async def test_lost_claim_ack_never_dispatches_or_reclaims(host):
    assert (await host.run()).status == 'paused'
    record = host.activate()
    access = host.runs._storage._access

    @contextmanager
    def lose_ack(*, write=False):
        with access(write=write) as db:
            yield db
        if write and host.store.get('alpha', 'one', record.request_id).revision == 4:
            raise WorkflowError('run_store_unavailable')

    host.runs._storage._access = lose_ack
    try:
        with pytest.raises(WorkflowError):
            await host.run()
    finally:
        host.runs._storage._access = access
    assert host.store.get('alpha', 'one', record.request_id).revision == 4
    with pytest.raises(WorkflowError, match='outcome_unknown'):
        await host.run()
    assert host.effects == [] and len(host.model.requests) == 1


@pytest.mark.parametrize(
    'change', ['inputs', 'space', 'run_id', 'selection_id', 'step_id', 'checkpoints', 'context']
)
async def test_context_cannot_be_reused_for_another_invocation_or_step(host, change):
    checkpoint, _ = checkpoints()
    context = StepContext('one', 'alpha', {}, 'Count', {}, checkpoint)
    kwargs = dict(step_id='count', selection_id='count')
    if change in ('inputs', 'space', 'run_id'):
        context = replace(context, **{change: 'other'})
    elif change in kwargs:
        kwargs[change] = 'other'
    with pytest.raises((ValueError, WorkflowError)):
        approval = ApprovalContext(host.store, context, **kwargs)
        if change == 'checkpoints':
            checkpoint, _ = checkpoints()
        elif change == 'context':
            context.scope['changed'] = True
        await host.agent.run('Count', tools=host.scoped, checkpoints=checkpoint, approval=approval)
    assert host.effects == [] and host.model.requests == []


async def test_parallel_sibling_completion_does_not_change_paused_task_identity(host, tmp_path):
    from scone_memory.agents.workflow import WorkflowStep

    async def sibling(context):
        return 'Independent sibling completed'

    runner = WorkflowRunner(
        tmp_path / 'parallel',
        key=b'k' * 32,
        steps=[WorkflowPausableStep('count', '1', host.execute), WorkflowStep('sibling', '1', sibling)],
        source_verifier=host.valid,
        dependencies={'count': [], 'sibling': []},
        max_parallel=2,
    )
    try:
        first = await runner.run('one', space='alpha', scope={}, inputs='Count')
        assert first.status == 'paused' and first.results == {'sibling': 'Independent sibling completed'}
        host.activate()
        final = await runner.run('one', space='alpha', scope={}, inputs='Count')
        assert final.status == 'completed' and host.effects == [3]
    finally:
        runner.close()


async def test_revoked_evidence_cannot_create_a_new_approval_request(host, memory):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})

    async def forget_then_propose():
        await memory.documents.delete_episode('alpha', episode.episode_id)
        return call()

    host.model.steps = [search(), forget_then_propose]
    with pytest.raises(WorkflowError):
        await host.run()
    assert host.store.list('alpha', 'one') == () and host.effects == []


async def test_revoked_checkpoint_lease_during_final_evidence_check_prevents_claim(host, memory):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    host.model.steps = [search(), call(), ToolStep(content='Finished.')]
    underlying, records = checkpoints()
    revoked = False

    def get(key):
        if revoked:
            raise WorkflowError('checkpoint_inactive')
        return underlying.get(key)

    checkpoint = StepCheckpoints(get, underlying.put)
    context = StepContext('one', 'alpha', {}, 'Count', {}, checkpoint)

    async def execute():
        approval = ApprovalContext(
            host.store, context, step_id='count', selection_id='count', activation_id=host.activation
        )
        return await host.agent.run('Count', tools=host.scoped, checkpoints=checkpoint, approval=approval)

    with pytest.raises(TurnJournalPaused):
        await execute()
    record = host.activate()
    restore = host.scoped.restore

    async def patched_restore(*args, **kwargs):
        evidence = await restore(*args, **kwargs)

        async def validate():
            nonlocal revoked
            result = await evidence.validate()
            state = json.loads(next(iter(records.values())))
            if state['events'][-1]['kind'] == 'custom' and state['events'][-1]['status'] == 'started':
                revoked = True
            return result

        return replace(evidence, _validator=validate)

    host.scoped.restore = patched_restore
    with pytest.raises(Exception):
        await execute()
    assert revoked
    assert host.effects == []
    assert host.store.get('alpha', 'one', record.request_id).revision == 3
