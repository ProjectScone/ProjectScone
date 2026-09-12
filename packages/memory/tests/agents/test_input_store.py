"""Input responses are scoped, encrypted and independent of run invocation bytes."""
import sqlite3

import pytest

from scone_memory.agents.input_store import AgentInputStore
from scone_memory.agents.interactive_plan import HumanInputTask, InteractiveAgentPlan
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore, _request_bytes
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_plan_store import catalog


@pytest.fixture
def inbox(tmp_path):
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    plan = InteractiveAgentPlan(kind='interactive', workflow_id='research', tasks=(
        HumanInputTask(kind='input', task_id='choose', prompt='Private question', max_response_bytes=32),
        HumanInputTask(kind='input', task_id='confirm', prompt='Another question'),))
    saved = plans.save('alpha', plan, catalog=catalog(), expected_revision=0)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32, max_runs=2)
    request = runs.register('alpha', 'one', plan=saved, question='Private original request', scope=RecallScope.validated())
    yield AgentInputStore(runs), runs, request, saved
    runs.close()
    plans.close()


def test_response_and_activation_are_separate_and_survive_reopen(inbox, tmp_path):
    inputs, runs, original, _ = inbox
    before = _request_bytes(original)
    pending = inputs.request('alpha', 'one', 'choose', context='[]')
    assert pending.revision == 1 and pending.response is None and pending.activation_id is None
    assert inputs.request('alpha', 'one', 'choose', context='[]') == pending
    answered = inputs.respond('alpha', 'one', 'choose', response='Café direction', expected_revision=1)
    assert answered.revision == 2 and answered.activation_id is None
    assert inputs.activated('alpha', 'one') == ()
    activation = inputs.activate('alpha', 'one', 'continue-1', responses={'choose': 2})
    assert activation.responses == {'choose': 2}
    assert _request_bytes(runs.get('alpha', 'one')) == before
    other_runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    try:
        reopened = AgentInputStore(other_runs)
        records = reopened.activated('alpha', 'one')
        assert len(records) == 1 and records[0].response == 'Café direction'
        assert records[0].activation_id == 'continue-1' and records[0].revision == 3
        assert reopened.activate('alpha', 'one', 'continue-1', responses={'choose': 2}) == activation
        assert reopened.respond('alpha', 'one', 'choose', response='Café direction', expected_revision=1) == records[0]
    finally:
        other_runs.close()
    assert b'Private question' not in (tmp_path / 'runs').read_bytes()
    assert b'Caf' not in (tmp_path / 'runs').read_bytes()


def test_conflicting_response_context_and_activation_are_refused(inbox):
    inputs, _, _, _ = inbox
    inputs.request('alpha', 'one', 'choose', context='[]')
    with pytest.raises(WorkflowError, match='input_request_conflict'):
        inputs.request('alpha', 'one', 'choose', context='["changed"]')
    with pytest.raises(WorkflowError, match='input_revision_conflict'):
        inputs.respond('alpha', 'one', 'choose', response='answer', expected_revision=2)
    with pytest.raises(WorkflowError, match='input_not_answered'):
        inputs.activate('alpha', 'one', 'c', responses={'choose': 1})
    inputs.respond('alpha', 'one', 'choose', response='answer', expected_revision=1)
    with pytest.raises(WorkflowError, match='input_response_conflict'):
        inputs.respond('alpha', 'one', 'choose', response='other', expected_revision=1)
    inputs.activate('alpha', 'one', 'c', responses={'choose': 2})
    with pytest.raises(WorkflowError, match='input_activation_conflict'):
        inputs.activate('alpha', 'one', 'different', responses={'choose': 2})
    with pytest.raises(WorkflowError, match='input_activation_conflict'):
        inputs.activate('alpha', 'one', 'c', responses={'confirm': 2})


def test_run_capacity_and_history_do_not_count_auxiliary_input_rows(inbox):
    inputs, runs, _, saved = inbox
    inputs.request('alpha', 'one', 'choose', context='[]')
    inputs.respond('alpha', 'one', 'choose', response='answer', expected_revision=1)
    inputs.activate('alpha', 'one', 'c', responses={'choose': 2})
    runs.register('alpha', 'two', plan=saved, question='Second', scope=RecallScope.validated())
    page = runs.list('alpha')
    assert {request.run_id for request in page.items} == {'one', 'two'}
    assert len(inputs.list('alpha', 'one')) == 1
    assert inputs.list('alpha', 'two') == ()
    with pytest.raises(WorkflowError, match='run_store_limit'):
        runs.register('alpha', 'three', plan=saved, question='Third', scope=RecallScope.validated())


def test_other_scope_unknown_prompt_and_cancellation_cannot_receive_answers(inbox, tmp_path):
    inputs, runs, _, _ = inbox
    assert inputs.get('beta', 'one', 'choose') is None
    for space, run, task in [('beta', 'one', 'choose'), ('alpha', 'missing', 'choose'), ('alpha', 'one', 'confirm')]:
        with pytest.raises(WorkflowError):
            inputs.respond(space, run, task, response='answer', expected_revision=1)
    with pytest.raises(WorkflowError, match='input_task_not_found'):
        inputs.request('alpha', 'one', 'invented', context='[]')
    inputs.request('alpha', 'one', 'choose', context='[]')
    other = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    other.request_cancel('alpha', 'one')
    other.close()
    with pytest.raises(WorkflowError, match='run_cancelled'):
        inputs.respond('alpha', 'one', 'choose', response='answer', expected_revision=1)
    assert inputs.get('alpha', 'one', 'choose').response is None


@pytest.mark.parametrize('response', ['', ' ', 'é' * 17])
def test_response_byte_limits_do_not_normalize_or_truncate(inbox, response):
    inputs, _, _, _ = inbox
    inputs.request('alpha', 'one', 'choose', context='[]')
    with pytest.raises((WorkflowError, ValueError)):
        inputs.respond('alpha', 'one', 'choose', response=response, expected_revision=1)
    assert inputs.get('alpha', 'one', 'choose').revision == 1


def test_ciphertext_cannot_move_between_prompt_identities(inbox, tmp_path):
    inputs, _, _, _ = inbox
    inputs.request('alpha', 'one', 'choose', context='[]')
    inputs.request('alpha', 'one', 'confirm', context='[]')
    db = sqlite3.connect(tmp_path / 'runs')
    rows = db.execute("SELECT token,payload FROM agent_runs WHERE token LIKE 'input:%'").fetchall()
    assert len(rows) == 2
    db.execute('UPDATE agent_runs SET payload=? WHERE token=?', (rows[0][1], rows[1][0]))
    db.commit()
    db.close()
    with pytest.raises(WorkflowError, match='integrity'):
        inputs.list('alpha', 'one')
