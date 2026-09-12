"""Human input never implies model execution or retained source evidence."""
import json

import pytest

from scone_memory.agents.input_store import AgentInputStore
from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.interactive_workflow import InteractiveAgentWorkflow
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_workflow import AgentTask
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_task_workflow import memory, catalog


@pytest.fixture
async def interactive(tmp_path, memory):
    calls = []
    agents = catalog(calls)
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    plan = InteractiveAgentPlan(kind='interactive', workflow_id='w', tasks=(
        HumanInputTask(kind='input', task_id='choose', prompt='Choose a direction'),
        AgentTask(task_id='independent', agent_id='worker', model_id='small', prompt='Independent work'),
        AgentTask(task_id='answer', agent_id='worker', model_id='large', prompt='Answer', depends_on=('choose',)),))
    saved = plans.save('alpha', plan, catalog=agents, expected_revision=0)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    request = runs.register('alpha', 'one', plan=saved, question='Original question', scope=RecallScope.validated())
    inbox = AgentInputStore(runs)
    def build():
        return InteractiveAgentWorkflow(tmp_path / 'journal', key=b'k' * 32, catalog=agents,
            request=request, memory=memory, inputs=inbox, activated=inbox.activated('alpha', 'one'))
    yield build, inbox, calls, memory
    runs.close()
    plans.close()


async def test_width_one_runs_independent_branch_then_resumes_exact_selected_model(interactive):
    build, inbox, calls, _ = interactive
    first = build()
    try:
        result = await first.run('one', 'Original question')
        assert result.status == 'awaiting_input'
        assert first.status('one', 'Original question').waiting_steps == ('choose',)
        assert set(result.results) == {'independent'}
        assert len(calls) == 1 and calls[0][0] == 'small'
        assert first.status('one', 'Original question').attempts == {'independent': 1}
        inbox.respond('alpha', 'one', 'choose', response='Take the north route', expected_revision=1)
        inbox.activate('alpha', 'one', 'continue-1', responses={'choose': 2})
        # This already-admitted runner cannot dynamically consume a later activation.
        assert (await first.run('one', 'Original question')).status == 'awaiting_input'
        assert len(calls) == 1
    finally:
        first.close()
    second = build()
    try:
        result = await second.run('one', 'Original question')
        assert result.status == 'completed' and len(calls) == 2
        assert calls[-1][0] == 'large'
        assert 'Take the north route' in json.dumps(calls[-1][1])
        assert result.results['choose']['kind'] == 'human_input'
        assert 'evidence_ids' not in result.results['choose']
        assert (await second.read_result('one', 'Original question')).results == result.results
        assert len(calls) == 2
    finally:
        second.close()


async def test_answer_without_activation_stays_waiting_after_reopen(interactive):
    build, inbox, calls, _ = interactive
    first = build()
    await first.run('one', 'Original question')
    first.close()
    inbox.respond('alpha', 'one', 'choose', response='Saved only', expected_revision=1)
    second = build()
    try:
        assert (await second.run('one', 'Original question')).status == 'awaiting_input'
        assert len(calls) == 1
    finally:
        second.close()


async def test_input_disclosure_and_consumption_recheck_space(interactive):
    build, inbox, calls, memory = interactive
    first = build()
    await first.run('one', 'Original question')
    first.close()
    inbox.respond('alpha', 'one', 'choose', response='Saved', expected_revision=1)
    inbox.activate('alpha', 'one', 'continue-1', responses={'choose': 2})
    second = build()
    await memory.delete_space('alpha')
    try:
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await second.inspect_inputs('one', 'Original question')
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await second.run('one', 'Original question')
        assert len(calls) == 1
    finally:
        second.close()
