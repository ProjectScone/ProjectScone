"""Standalone agent models retain selected LLMs and exact workflow identities."""
import pytest
from scone import SconeError
from scone.agent_models import HumanInput, ModelTask, TaskPlan, HandoffAgent, HandoffPlan, SavedPlan, RunRequest, RunStatus

PLAN = {'kind': 'interactive', 'workflow_id': 'research', 'tasks': [
    {'kind': 'input', 'task_id': 'choose', 'prompt': 'Choose', 'depends_on': [], 'max_response_bytes': 32},
    {'task_id': 'answer', 'agent_id': 'worker', 'model_id': 'careful', 'prompt': 'Answer', 'depends_on': ['choose']}]}
SAVED = {'space': 'alpha', 'revision': 1, 'plan': PLAN, 'bindings': {'answer': 'a' * 64},
         'updated_at': '2026-09-12T10:00:00Z', 'configuration_current': True}
REQUEST = {'space': 'alpha', 'run_id': 'one', 'question': 'Question', 'plan': SAVED,
           'created_at': SAVED['updated_at'], 'max_parallel': 1}
STATUS = {'space': 'alpha', 'run_id': 'one', 'workflow_id': 'research', 'plan_revision': 1,
          'created_at': SAVED['updated_at'], 'status': 'awaiting_input', 'active_local': False,
          'completed_steps': [], 'inflight': None, 'outcome_unknown': False, 'error_class': None,
          'waiting_steps': ['choose']}


def test_task_plans_validate_graph_and_keep_explicit_models():
    plan = TaskPlan('research', (HumanInput('choose', 'Choose', max_response_bytes=32),
                                ModelTask('answer', 'worker', 'careful', 'Answer', ('choose',))))
    assert plan.to_json() == PLAN
    saved = SavedPlan.from_json(SAVED, expected_space='alpha')
    assert saved.plan == plan and list(saved.bindings) == ['answer']
    with pytest.raises(TypeError):
        saved.bindings['answer'] = 'b' * 64
    with pytest.raises(SconeError):
        TaskPlan('w', (ModelTask('a', 'worker', 'careful', 'Work', ('missing',)),))
    with pytest.raises(SconeError):
        HumanInput('h', 'é' * 1001)
    with pytest.raises(SconeError):
        HumanInput('h', 'Choose', max_response_bytes=True)


def test_legacy_task_and_handoff_serialization_stays_distinct():
    plan = TaskPlan('w', (ModelTask('a', 'worker', 'careful', 'Work'),))
    assert 'kind' not in plan.to_json()
    handoff = HandoffPlan('h', 'worker', (HandoffAgent('worker', 'careful', ('worker',)),), max_handoffs=2)
    assert handoff.to_json() == {'workflow_id': 'h', 'root_agent': 'worker', 'agents': [
        {'agent_id': 'worker', 'model_id': 'careful', 'can_handoff_to': ['worker']}], 'max_handoffs': 2}


def test_saved_bindings_cannot_omit_models_or_add_human_bindings():
    for changes in ({'space': 'beta'}, {'bindings': {}}, {'bindings': {'answer': 'a' * 64, 'choose': 'b' * 64}}):
        with pytest.raises(SconeError):
            SavedPlan.from_json({**SAVED, **changes}, expected_space='alpha')


def test_run_status_matches_request_and_keeps_human_waits_out_of_inflight():
    request = RunRequest.from_json(REQUEST, expected_space='alpha', run_id='one')
    status = RunStatus.from_json(STATUS, expected_space='alpha', run_id='one')
    status.match(request)
    for changes in ({'workflow_id': 'other'}, {'completed_steps': ['missing']},
                    {'waiting_steps': ['answer']}, {'waiting_steps': [], 'inflight': 'choose', 'inflight_steps': ['choose']}):
        with pytest.raises(SconeError):
            RunStatus.from_json({**STATUS, **changes}, expected_space='alpha', run_id='one').match(request)


def test_native_utc_timestamp_spellings_bind_to_the_same_instant():
    request = RunRequest.from_json(REQUEST, expected_space='alpha', run_id='one')
    status = RunStatus.from_json({**STATUS, 'created_at': '2026-09-12T10:00:00+00:00'},
                                 expected_space='alpha', run_id='one')
    status.match(request)
