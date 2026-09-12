"""Independent human-input wire and pre-mutation identity checks."""
import pytest
from scone import SconeError
from scone.agent_inputs import InputRecord
from test_agents import fixture
from test_agent_inputs import setup, PENDING, ANSWERED, ACTIVATED
from test_agent_models import STATUS


@pytest.mark.parametrize('context', ['', 'not JSON', '{}', '[{"task_id":"undeclared","text":"Foreign output"}]'])
def test_inputs_reject_context_outside_declared_dependencies(fixture, context):
    server, agents = fixture
    setup(server, [{**PENDING, 'context': context}])
    with pytest.raises(SconeError):
        agents.inputs('one')


def test_changed_saved_reply_is_refused_before_continuation_post(fixture):
    server, agents = fixture
    original = InputRecord.from_json({**ANSWERED, 'context': '[]'}, expected_space='alpha', run_id='one')
    setup(server, [{**ACTIVATED, 'context': '[]', 'response': 'Different recorded answer'}])
    server.route('POST', '/v1/agent-runs/one/continue', 202, STATUS)
    with pytest.raises(SconeError):
        agents.continue_run('one', continuation_id='approval', responses=(original,))
    assert not [request for request in server.requests if request.method == 'POST']


def test_replaced_prompt_identity_is_refused_before_response_post(fixture):
    server, agents = fixture
    original = InputRecord.from_json({**PENDING, 'context': '[]'}, expected_space='alpha', run_id='one')
    current = {**PENDING, 'context': '[]', 'created_at': '2026-09-12T10:10:00Z'}
    setup(server, [current])
    server.route('POST', '/v1/agent-runs/one/inputs/choose/response', 200,
                 {**current, 'revision': 2, 'response': 'Proceed', 'responded_at': '2026-09-12T10:11:00Z'})
    with pytest.raises(SconeError):
        agents.respond(original, response='Proceed')
    assert not [request for request in server.requests if request.method == 'POST']

