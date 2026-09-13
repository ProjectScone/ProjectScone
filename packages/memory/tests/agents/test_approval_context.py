"""A registered agent alone does not authorize a handoff hop or human node."""

from dataclasses import replace

import pytest

from scone_memory.agents.approval_context import ApprovalContext
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.handoff_workflow import AgentHandoffPlan, HandoffAgent, HandoffReceipt
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_workflow import AgentTaskReceipt
from scone_memory.agents.workflow import StepContext, WorkflowError
from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import agents, call, tool
from .test_evidence_tool_loop import Script, binding
from .test_task_workflow import memory
from .test_turn_journal import checkpoints


@pytest.fixture
def handoff(tmp_path):
    registry = AgentCatalog(
        models=[AgentModel('local', 'Local', '1', lambda: Script())],
        tools=[tool(lambda *_: None, requires_approval=True)],
        agents=[
            AgentDefinition(
                agent_id=name,
                instructions='Work.',
                models=('local',),
                default_model='local',
                tools=('double_count',),
            )
            for name in ('root', 'writer')
        ],
    )
    plan = AgentHandoffPlan(
        workflow_id='job',
        root_agent='root',
        max_handoffs=2,
        agents=(
            HandoffAgent(agent_id='root', model_id='local', can_handoff_to=('writer',)),
            HandoffAgent(agent_id='writer', model_id='local', can_handoff_to=()),
        ),
    )
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    saved = plans.save('alpha', plan, catalog=registry, expected_revision=0)
    runs.register(
        'alpha',
        'one',
        plan=saved,
        question='Write',
        scope=RecallScope.validated(where={'team': 'blue'}),
        exclude_session_id='current',
    )
    receipt = HandoffReceipt(
        output=AgentTaskReceipt(
            task_id='hop-01',
            agent_id='root',
            model_id='local',
            binding=saved.bindings['root'],
            depends_on=(),
            text='Notes',
            source_status='none',
            evidence_ids=(),
            evidence_packets=(),
            model_calls=1,
            tool_calls=0,
        ),
        handoff_to='writer',
    )
    checkpoint, _ = checkpoints()
    context = StepContext(
        'one', 'alpha', {}, 'Write', {'hop-01': receipt.model_dump(mode='json')}, checkpoint
    )
    yield AgentApprovalStore(runs), registry, context, receipt
    runs.close()
    plans.close()


def test_handoff_root_and_following_allowed_selection_bind(handoff, memory):
    store, registry, context, _ = handoff
    first = replace(context, completed={})
    ApprovalContext(store, first, step_id='hop-01', selection_id='root').bind(
        registry.bind('root'), binding(memory), first.checkpoints
    )
    ApprovalContext(store, context, step_id='hop-02', selection_id='writer').bind(
        registry.bind('writer'), binding(memory), context.checkpoints
    )


@pytest.mark.parametrize(
    'change', ['wrong_root', 'skip', 'wrong_agent', 'closed', 'edge', 'binding', 'model', 'dependencies']
)
def test_unrelated_registered_agent_or_changed_handoff_receipt_refuses(handoff, memory, change):
    store, registry, context, receipt = handoff
    step, selection = 'hop-02', 'writer'
    if change == 'wrong_root':
        context, step = replace(context, completed={}), 'hop-01'
    elif change == 'skip':
        step = 'hop-03'
    elif change == 'wrong_agent':
        selection = 'root'
    else:
        data = receipt.model_dump(mode='json')
        if change == 'closed':
            data['handoff_to'] = None
        elif change == 'edge':
            data['handoff_to'] = 'root'
        elif change == 'binding':
            data['output']['binding'] = 'f' * 64
        elif change == 'model':
            data['output']['model_id'] = 'other'
        else:
            data['output']['depends_on'] = ['hop-00']
        context = replace(context, completed={'hop-01': data})
    with pytest.raises((ValueError, WorkflowError)):
        ApprovalContext(store, context, step_id=step, selection_id=selection).bind(
            registry.bind(selection), binding(memory), context.checkpoints
        )
    assert store.list('alpha', 'one') == ()


@pytest.mark.parametrize(
    'change',
    [None, 'terminal', 'foreign_route', 'wrong_binding', 'wrong_model', 'wrong_dependencies', 'wrong_task'],
)
async def test_handoff_approval_binding_checks_complete_selected_prefix(memory, tmp_path, change):
    effects = []
    model = Script(call())
    registry = agents(model, [tool(lambda args, ctx: effects.append(1), requires_approval=True)])
    agent = registry.bind('worker')
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    store = AgentApprovalStore(runs)
    saved = plans.save(
        'alpha',
        AgentHandoffPlan(
            workflow_id='job',
            root_agent='worker',
            max_handoffs=1,
            agents=(HandoffAgent(agent_id='worker', can_handoff_to=('worker',)),),
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
    prefix = HandoffReceipt(
        output=AgentTaskReceipt(
            task_id='hop-01',
            agent_id='worker',
            model_id='local',
            binding=agent.fingerprint,
            depends_on=(),
            text='Continue',
            source_status='none',
            evidence_ids=(),
            evidence_packets=(),
            model_calls=1,
            tool_calls=0,
        ),
        handoff_to='worker',
    ).model_dump(mode='json')
    if change == 'terminal':
        prefix['handoff_to'] = None
    elif change == 'foreign_route':
        prefix['handoff_to'] = 'other'
    elif change == 'wrong_binding':
        prefix['output']['binding'] = '0' * 64
    elif change == 'wrong_model':
        prefix['output']['model_id'] = 'other'
    elif change == 'wrong_dependencies':
        prefix['output']['depends_on'] = ['hop-00']
    elif change == 'wrong_task':
        prefix['output']['task_id'] = 'hop-00'
    checkpoint, _ = checkpoints()
    context = StepContext('one', 'alpha', {}, 'Count', {'hop-01': prefix}, checkpoint)
    try:
        if change is None:
            approval = ApprovalContext(store, context, step_id='hop-02', selection_id='worker')
            with pytest.raises(TurnJournalPaused):
                await agent.run('Count', tools=binding(memory), checkpoints=checkpoint, approval=approval)
            (record,) = store.list('alpha', 'one')
            assert record.call.step_id == 'hop-02' and record.call.selection_id == 'worker'
            assert len(model.requests) == 1 and effects == []
        else:
            with pytest.raises(WorkflowError):
                ApprovalContext(store, context, step_id='hop-02', selection_id='worker')
            assert model.requests == [] and store.list('alpha', 'one') == ()
    finally:
        runs.close()
        plans.close()
