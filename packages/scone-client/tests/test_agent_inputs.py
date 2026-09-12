"""Human replies require separate, exactly acknowledged continuation."""
import pytest
from scone import SconeError
from scone.agent_inputs import InputRecord
from test_agents import fixture
from test_agent_models import REQUEST, STATUS

PENDING = {'space': 'alpha', 'run_id': 'one', 'task_id': 'choose', 'prompt': 'Choose',
           'context': '[]', 'max_response_bytes': 32, 'revision': 1, 'response': None,
           'activation_id': None, 'created_at': '2026-09-12T10:00:00Z', 'responded_at': None}
ANSWERED = {**PENDING, 'revision': 2, 'response': 'Proceed', 'responded_at': '2026-09-12T10:01:00Z'}
ACTIVATED = {**ANSWERED, 'revision': 3, 'activation_id': 'approval'}


def setup(server, rows):
    server.route('GET', '/v1/agent-runs/one/request', 200, REQUEST)
    server.route('GET', '/v1/agent-runs/one/inputs', 200, {'space': 'alpha', 'run_id': 'one', 'items': rows})


def test_input_read_and_reply_never_continue(fixture):
    server, agents = fixture
    setup(server, [PENDING])
    pending, = agents.inputs('one')
    server.route('POST', '/v1/agent-runs/one/inputs/choose/response', 200, ANSWERED)
    answer = agents.respond(pending, response='Proceed')
    assert answer.response == 'Proceed' and answer.revision == 2
    writes = [request for request in server.requests if request.method == 'POST']
    assert len(writes) == 1 and writes[0].json == {'response': 'Proceed', 'expected_revision': 1}
    assert not writes[0].path.endswith('/continue')


@pytest.mark.parametrize('change', [{'task_id': 'answer'}, {'prompt': 'Changed'},
    {'max_response_bytes': 4000}, {'revision': True}, {'revision': 2}, {'context': '\ud800'}])
def test_input_read_refuses_unbound_or_malformed_records(fixture, change):
    server, agents = fixture
    setup(server, [{**PENDING, **change}])
    with pytest.raises(SconeError):
        agents.inputs('one')


def test_reply_refuses_changed_acknowledgement(fixture):
    server, agents = fixture
    setup(server, [PENDING])
    pending, = agents.inputs('one')
    server.route('POST', '/v1/agent-runs/one/inputs/choose/response', 200, {**ANSWERED, 'context': 'Changed'})
    with pytest.raises(SconeError, match='acknowledgement'):
        agents.respond(pending, response='Proceed')
    assert sum(request.method == 'POST' for request in server.requests) == 1


def test_continuation_requires_exact_fresh_activation_group(fixture):
    server, agents = fixture
    setup(server, [ANSWERED])
    answered, = agents.inputs('one')
    server.route('POST', '/v1/agent-runs/one/continue', 202, STATUS)
    setup(server, [ACTIVATED])
    assert agents.continue_run('one', continuation_id='approval', responses=(answered,)).run_id == 'one'
    writes = [request for request in server.requests if request.method == 'POST']
    assert len(writes) == 1 and writes[0].json == {'continuation_id': 'approval', 'responses': {'choose': 2}}
    setup(server, [{**ACTIVATED, 'activation_id': 'other'}])
    with pytest.raises(SconeError, match='activation acknowledgement'):
        agents.continue_run('one', continuation_id='approval', responses=(answered,))
    assert sum(request.method == 'POST' for request in server.requests) == 1


def test_ambiguous_continuation_is_not_replayed(fixture):
    server, agents = fixture
    setup(server, [ANSWERED])
    answered, = agents.inputs('one')
    server.route('POST', '/v1/agent-runs/one/continue', 503, {'error': 'unavailable'})
    with pytest.raises(SconeError):
        agents.continue_run('one', continuation_id='approval', responses=(answered,))
    assert sum(request.method == 'POST' for request in server.requests) == 1


def test_pending_and_duplicate_selection_refused_before_io(fixture):
    server, agents = fixture
    pending = InputRecord.from_json(PENDING, expected_space='alpha', run_id='one')
    answered = InputRecord.from_json(ANSWERED, expected_space='alpha', run_id='one')
    for selection in ((), (pending,), (answered, answered)):
        with pytest.raises(SconeError):
            agents.continue_run('one', continuation_id='approval', responses=selection)
    with pytest.raises(SconeError):
        agents.respond(pending, response='é' * 17)
    assert not server.requests


@pytest.mark.parametrize('change', [{'activation_id': 'other'}, {'response': 'Changed'}, {'revision': 2, 'activation_id': None}])
def test_continuation_rechecks_receipt_after_mutation(fixture, change):
    server, agents = fixture
    setup(server, [ANSWERED])
    answered, = agents.inputs('one')
    del server.routes[('GET', '/v1/agent-runs/one/inputs')]
    server.enqueue(200, {'space': 'alpha', 'run_id': 'one', 'items': [ANSWERED]})
    server.enqueue(200, {'space': 'alpha', 'run_id': 'one', 'items': [{**ACTIVATED, **change}]})
    server.route('POST', '/v1/agent-runs/one/continue', 202, STATUS)
    with pytest.raises(SconeError, match='activation acknowledgement'):
        agents.continue_run('one', continuation_id='approval', responses=(answered,))
    assert sum(request.method == 'POST' for request in server.requests) == 1
