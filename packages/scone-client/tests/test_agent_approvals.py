"""Tool control receipts bind exact calls and explicit activation batches."""

from dataclasses import replace
from scone.agent_approvals import ToolApprovalRecord

import pytest

from scone import SconeError
from scone.agent_models import RunRequest, RunStatus
from test_agents import fixture, CAPS
from test_agent_models import REQUEST, STATUS

CALL = {
    'step_id': 'answer',
    'selection_id': 'answer',
    'agent_id': 'worker',
    'model_id': 'careful',
    'binding': 'a' * 64,
    'tool_name': 'send_note',
    'tool_revision': '1',
    'tool_digest': 'b' * 64,
    'arguments_json': '{"message":"Hello"}',
    'operation_digest': 'c' * 64,
}
PENDING = {
    'space': 'alpha',
    'run_id': 'one',
    'request_id': 'd' * 64,
    'call': CALL,
    'revision': 1,
    'created_at': '2026-09-12T10:00:00Z',
    'decision': None,
    'decided_by': None,
    'decided_at': None,
    'activation_id': None,
    'activated_at': None,
    'consumed_at': None,
    'decision_digest': None,
}
DECIDED = {
    **PENDING,
    'revision': 2,
    'decision': 'approve',
    'decided_by': 'key:' + 'e' * 64,
    'decided_at': '2026-09-12T10:01:00Z',
    'decision_digest': 'f' * 64,
}
PAUSED = {
    **STATUS,
    'status': 'paused',
    'waiting_steps': [],
    'completed_steps': ['choose'],
    'paused_steps': ['answer'],
}
ACTIVATION = {
    'space': 'alpha',
    'run_id': 'one',
    'activation_id': 'next',
    'decisions': {'d' * 64: 2},
    'decision_digests': {'d' * 64: 'f' * 64},
    'created_at': '2026-09-12T10:02:00Z',
}


def setup(server, rows):
    server.route(
        'GET', '/v1/capabilities', 200, {**CAPS, 'features': {**CAPS['features'], 'agents.approvals': True}}
    )
    server.route('GET', '/v1/agent-runs/one/request', 200, REQUEST)
    server.route(
        'GET', '/v1/agent-runs/one/approvals', 200, {'space': 'alpha', 'run_id': 'one', 'items': rows}
    )


def test_paused_status_keeps_completed_human_and_selected_model_separate():
    parsed = RunStatus.from_json(PAUSED, expected_space='alpha')
    parsed.match(RunRequest.from_json(REQUEST, expected_space='alpha', run_id='one'))
    assert parsed.paused_steps == ('answer',)
    assert RunStatus.from_json(STATUS, expected_space='alpha').paused_steps == ()


@pytest.mark.parametrize(
    'change',
    [
        {'paused_steps': ['choose']},
        {'paused_steps': ['missing']},
        {'paused_steps': []},
        {'paused_steps': ['answer', 'answer']},
        {'status': 'completed'},
        {'outcome_unknown': True},
        {'completed_steps': ['choose', 'answer']},
        {'completed_steps': []},
    ],
)
def test_malformed_paused_status_is_refused(change):
    with pytest.raises(SconeError):
        RunStatus.from_json({**PAUSED, **change}, expected_space='alpha').match(
            RunRequest.from_json(REQUEST, expected_space='alpha', run_id='one')
        )


def test_read_and_decision_do_not_continue(fixture):
    server, agents = fixture
    setup(server, [PENDING])
    (pending,) = agents.approvals('one')
    server.route('POST', '/v1/agent-runs/one/approvals/' + 'd' * 64 + '/decision', 200, DECIDED)
    decided = agents.decide_tool(pending, decision='approve')
    assert decided.revision == 2 and decided.call.arguments() == {'message': 'Hello'}
    writes = [value for value in server.requests if value.method == 'POST']
    assert len(writes) == 1 and writes[0].json == {'decision': 'approve', 'expected_revision': 1}
    assert pending.call.arguments() == {'message': 'Hello'}


@pytest.mark.parametrize(
    'change',
    [
        {'step_id': 'choose'},
        {'selection_id': 'choose'},
        {'model_id': 'fast'},
        {'binding': 'f' * 64},
        {'tool_name': 'search_memory'},
        {'arguments_json': '{"x":1,"x":2}'},
        {'arguments_json': '{"x":NaN}'},
        {'arguments_json': '{ "x": 1 }'},
        {'operation_digest': 'no'},
        {'arguments_json': '{"x":' + str(2**257) + '}'},
    ],
)
def test_unbound_or_noncanonical_calls_are_refused(fixture, change):
    server, agents = fixture
    setup(server, [{**PENDING, 'call': {**CALL, **change}}])
    with pytest.raises(SconeError):
        agents.approvals('one')
    assert all(value.method == 'GET' for value in server.requests)


def test_exact_activation_acknowledgement_and_no_implicit_retry(fixture):
    server, agents = fixture
    setup(server, [DECIDED])
    (decided,) = agents.approvals('one')
    path = '/v1/agent-runs/one/approval-continuations'
    server.route('POST', path, 202, {'status': PAUSED, 'activation': ACTIVATION})
    result = agents.continue_tools('one', continuation_id='next', decisions=(decided,))
    assert result.activation.activation_id == 'next' and result.status.paused_steps == ('answer',)
    with pytest.raises(TypeError):
        result.activation.decisions['d' * 64] = 1
    server.route('POST', path, 503, {'error': 'unavailable'})
    with pytest.raises(SconeError):
        agents.continue_tools('one', continuation_id='next', decisions=(decided,))
    assert sum(value.method == 'POST' for value in server.requests) == 2


@pytest.mark.parametrize(
    'change',
    [
        {'activation_id': 'other'},
        {'decisions': {'f' * 64: 2}},
        {'decisions': {'d' * 64: True}},
        {'decision_digests': {}},
        {'run_id': 'two'},
    ],
)
def test_changed_activation_acknowledgement_is_refused(fixture, change):
    server, agents = fixture
    setup(server, [DECIDED])
    (decided,) = agents.approvals('one')
    server.route(
        'POST',
        '/v1/agent-runs/one/approval-continuations',
        202,
        {'status': PAUSED, 'activation': {**ACTIVATION, **change}},
    )
    with pytest.raises(SconeError):
        agents.continue_tools('one', continuation_id='next', decisions=(decided,))
    assert sum(value.method == 'POST' for value in server.requests) == 1


def test_changed_decision_is_refused_before_continuation(fixture):
    server, agents = fixture
    setup(server, [DECIDED])
    (decided,) = agents.approvals('one')
    setup(server, [{**DECIDED, 'decision': 'deny'}])
    with pytest.raises(SconeError):
        agents.continue_tools('one', continuation_id='next', decisions=(decided,))
    assert all(value.method == 'GET' for value in server.requests)


def test_unsupported_server_is_not_mutated(fixture):
    server, agents = fixture
    with pytest.raises(SconeError, match='agents.approvals'):
        agents.approvals('one')
    assert all(value.method == 'GET' for value in server.requests)


@pytest.mark.parametrize('revision', ['.', '..'])
def test_native_tool_revision_is_metadata_not_path(revision):
    parsed = ToolApprovalRecord.from_json(
        {**PENDING, 'call': {**PENDING['call'], 'tool_revision': revision}},
        expected_space='alpha',
        run_id='one',
    )
    assert parsed.call.tool_revision == revision


def test_fabricated_record_refuses_cycle_as_typed_error_before_io(fixture):
    server, agents = fixture
    pending = ToolApprovalRecord.from_json(PENDING, expected_space='alpha', run_id='one')
    cycle = {}
    cycle['loop'] = cycle
    forged = replace(pending, call=cycle)
    with pytest.raises(SconeError):
        agents.decide_tool(forged, decision='approve')
    assert server.requests == []


from test_agent_models import REQUEST, SAVED

from scone.agent_models import RunRequest


@pytest.mark.parametrize(
    'hop,edges,allowed', [(1, [], False), (2, [], False), (2, ['worker'], True), (3, ['worker'], False)]
)
def test_handoff_call_requires_reachable_agent_at_exact_hop(hop, edges, allowed):
    plan = {
        'workflow_id': 'research',
        'root_agent': 'root',
        'max_handoffs': 2,
        'agents': [
            {'agent_id': 'root', 'model_id': 'careful', 'can_handoff_to': edges},
            {'agent_id': 'worker', 'model_id': 'careful', 'can_handoff_to': []},
        ],
    }
    request = RunRequest.from_json(
        {**REQUEST, 'plan': {**SAVED, 'plan': plan, 'bindings': {'root': 'b' * 64, 'worker': 'a' * 64}}},
        expected_space='alpha',
        run_id='one',
    )
    pending = ToolApprovalRecord.from_json(
        {**PENDING, 'call': {**PENDING['call'], 'step_id': 'hop-%02d' % hop, 'selection_id': 'worker'}},
        expected_space='alpha',
        run_id='one',
    )
    if allowed:
        pending.match(request)
    else:
        with pytest.raises(SconeError):
            pending.match(request)


@pytest.mark.parametrize('where', ['current', 'ack'])
def test_substituted_decision_digest_never_acknowledged(fixture, where):
    server, agents = fixture
    setup(server, [DECIDED])
    (decided,) = agents.approvals('one')
    if where == 'current':
        setup(server, [{**DECIDED, 'decision_digest': '0' * 64}])
    response = {'status': PAUSED, 'activation': {**ACTIVATION, 'decision_digests': {'d' * 64: '0' * 64}}}
    server.route('POST', '/v1/agent-runs/one/approval-continuations', 202, response)
    with pytest.raises(SconeError):
        agents.continue_tools('one', continuation_id='next', decisions=(decided,))
    assert sum(row.method == 'POST' for row in server.requests) == (1 if where == 'ack' else 0)
