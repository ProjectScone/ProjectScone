"""Explicit human-input plans preserve the meaning of existing model plans."""
import json

import pytest
from pydantic import ValidationError

from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.plan_store import AgentPlanStore, PlanConfigurationChanged, SavedAgentPlan
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from .test_plan_store import catalog


def plan():
    return InteractiveAgentPlan(kind='interactive', workflow_id='research', tasks=(
        AgentTask(task_id='find', agent_id='a', prompt='Find supporting evidence.'),
        HumanInputTask(kind='input', task_id='choose', prompt='Which direction should we pursue?', depends_on=('find',)),
        AgentTask(task_id='write', agent_id='a', prompt='Use the selected direction.', depends_on=('choose',)),
    ))


def test_input_tasks_do_not_select_models_and_reopen_with_exact_identity(tmp_path):
    store = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    saved = store.save('alpha', plan(), catalog=catalog(), expected_revision=0)
    assert set(saved.bindings) == {'find', 'write'}
    assert saved.plan.tasks[0].model_id == saved.plan.tasks[2].model_id == 'local'
    assert saved.plan.tasks[1].model_dump() == plan().tasks[1].model_dump()
    assert [task.task_id for task in saved.plan.ordered()] == ['find', 'choose', 'write']
    store.close()
    store = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    try:
        reopened = store.get('alpha', 'research')
        assert reopened == saved
        assert reopened.checked_plan(catalog()) == saved.plan
        with pytest.raises(PlanConfigurationChanged):
            reopened.checked_plan(catalog('changed'))
        assert store.get('beta', 'research') is None
    finally:
        store.close()


def test_existing_plan_serialization_does_not_grow_a_discriminator(tmp_path):
    old = AgentTaskPlan(workflow_id='old', tasks=(AgentTask(task_id='one', agent_id='a', prompt='Answer.'),))
    assert old.model_dump_json() == ('{"workflow_id":"old","tasks":[{"task_id":"one",'
        '"agent_id":"a","model_id":null,"prompt":"Answer.","depends_on":[]}]}')
    store = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    try:
        saved = store.save('alpha', old, catalog=catalog(), expected_revision=0)
        decoded = SavedAgentPlan.model_validate_json(saved.model_dump_json())
        assert type(decoded.plan) is AgentTaskPlan
        assert 'kind' not in json.loads(decoded.model_dump_json())['plan']
    finally:
        store.close()


@pytest.mark.parametrize('change', [
    {'prompt': ' '}, {'prompt': 'é' * 1001}, {'max_response_bytes': True},
    {'max_response_bytes': 0}, {'max_response_bytes': 4001},
    {'depends_on': ('one', 'one')}, {'depends_on': ('choose',)},
    {'task_id': 'choose\n'}, {'agent_id': 'a'}, {'kind': 'agent'},
])
def test_input_constraints_are_validated_before_storage(change):
    data = {'kind': 'input', 'task_id': 'choose', 'prompt': 'Choose a direction.', **change}
    with pytest.raises((ValidationError, ValueError)):
        HumanInputTask.model_validate(data)


def test_interactive_mode_is_explicit_and_requires_an_input_node():
    with pytest.raises(ValidationError):
        InteractiveAgentPlan(workflow_id='x', tasks=plan().tasks)
    with pytest.raises(ValidationError):
        InteractiveAgentPlan(kind='interactive', workflow_id='x', tasks=(plan().tasks[0],))
    only_input = InteractiveAgentPlan(kind='interactive', workflow_id='x', tasks=(
        HumanInputTask(kind='input', task_id='ask', prompt='What should we do?'),))
    assert only_input.ordered() == only_input.tasks


@pytest.mark.parametrize('dependencies', [('missing',), ('write',)])
def test_missing_or_cyclic_dependencies_are_rejected(dependencies):
    raw = plan().model_dump()
    raw['tasks'][1]['depends_on'] = dependencies
    with pytest.raises(ValidationError):
        InteractiveAgentPlan.model_validate(raw)
