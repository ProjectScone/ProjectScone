"""One explicit mutation, validated acknowledgements, no hidden replay."""
import pytest
from scone import Scone, SconeError
from scone.agent_models import SavedPlan
from stub_server import StubScone
from test_agent_models import PLAN, SAVED, REQUEST, STATUS

CAPS = {'schema_version': 1, 'implementation': 'python', 'features': {
    'agents.catalog': True, 'agents.plans': True, 'agents.runs': True, 'agents.inputs': True}}


@pytest.fixture
def fixture():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.route('GET', '/v1/capabilities', 200, CAPS)
        server.route('GET', '/v1/status', 200, {'space': 'alpha'})
        yield server, client.agents(expected_space='alpha')


def test_save_explicit_model_and_human_plan_without_execution(fixture):
    server, agents = fixture
    saved = SavedPlan.from_json(SAVED, expected_space='alpha')
    server.route('PUT', '/v1/agent-plans/research', 200, SAVED)
    result = agents.save_plan(saved.plan, expected_revision=0)
    assert result == saved
    writes = [request for request in server.requests if request.method != 'GET']
    assert len(writes) == 1 and writes[0].json == {'plan': PLAN, 'expected_revision': 0}


def test_start_checks_original_request_and_never_retries_failed_admission(fixture):
    server, agents = fixture
    saved = SavedPlan.from_json(SAVED, expected_space='alpha')
    server.route('POST', '/v1/agent-runs', 202, STATUS)
    server.route('GET', '/v1/agent-runs/one/request', 200, REQUEST)
    assert agents.start('one', plan=saved, question='Question').status == 'awaiting_input'
    assert len([request for request in server.requests if request.method == 'POST']) == 1
    server.route('POST', '/v1/agent-runs', 503, {'error': 'unavailable'})
    with pytest.raises(SconeError) as error:
        agents.start('one', plan=saved, question='Question')
    assert error.value.status == 503
    assert len([request for request in server.requests if request.method == 'POST']) == 2


def test_substituted_question_is_not_acknowledged(fixture):
    server, agents = fixture
    server.route('POST', '/v1/agent-runs', 202, STATUS)
    server.route('GET', '/v1/agent-runs/one/request', 200, {**REQUEST, 'question': 'Changed'})
    with pytest.raises(SconeError, match='acknowledgement'):
        agents.start('one', plan=SavedPlan.from_json(SAVED, expected_space='alpha'), question='Question')


def test_status_and_request_reads_do_not_start_work(fixture):
    server, agents = fixture
    server.route('GET', '/v1/agent-runs/one', 200, STATUS)
    server.route('GET', '/v1/agent-runs/one/request', 200, REQUEST)
    agents.status('one').match(agents.request('one'))
    assert all(request.method == 'GET' for request in server.requests)


def test_interactive_support_does_not_replace_plan_or_run_capability(fixture):
    server, agents = fixture
    saved = SavedPlan.from_json(SAVED, expected_space='alpha')
    server.route('GET', '/v1/capabilities', 200, {**CAPS, 'features': {'agents.inputs': True}})
    with pytest.raises(SconeError, match='agents.plans'):
        agents.save_plan(saved.plan, expected_revision=0)
    with pytest.raises(SconeError, match='agents.runs'):
        agents.start('one', plan=saved, question='Question')
    assert all(request.method == 'GET' for request in server.requests)


def test_escaped_request_budget_is_rejected_before_transport(fixture):
    server, agents = fixture
    with pytest.raises(SconeError, match='byte limit'):
        agents.start('one', plan=SavedPlan.from_json(SAVED, expected_space='alpha'), question='x' + '\0' * 3999)
    assert server.requests == []
