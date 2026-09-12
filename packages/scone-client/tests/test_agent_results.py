"""Completed outputs bind to selected models and independently retained replies."""
import json

import pytest
from scone import SconeError
from test_agents import fixture
from test_agent_models import REQUEST, SAVED
from test_agent_inputs import ACTIVATED, setup

HUMAN = {'kind': 'human_input', 'task_id': 'choose', 'depends_on': [], 'text': 'Proceed',
         'activation_id': 'approval', 'response_digest': 'c'*64}
OUTPUT = {'task_id': 'answer', 'agent_id': 'worker', 'model_id': 'careful', 'binding': 'a'*64,
          'depends_on': ['choose'], 'text': 'Answer', 'source_status': 'none',
          'evidence_ids': [], 'evidence_packets': [], 'model_calls': 1, 'tool_calls': 0}
RESULT = {'space': 'alpha', 'run_id': 'one', 'status': 'completed', 'results': {'choose': HUMAN, 'answer': OUTPUT},
          'reused_steps': ['choose', 'answer']}


def serve(server, result=RESULT):
    setup(server, [ACTIVATED])
    server.route('GET', '/v1/agent-runs/one/result', 200, result)


def test_result_retains_typed_model_and_human_outputs_without_execution(fixture):
    server, agents = fixture
    serve(server)
    result = agents.result('one')
    assert result.results['answer'].model_id == 'careful'
    assert result.results['choose'].activation_id == 'approval'
    assert result.reused_steps == ('choose', 'answer')
    with pytest.raises(TypeError):
        result.results['answer'] = result.results['choose']
    assert all(row.method == 'GET' for row in server.requests)


@pytest.mark.parametrize('change', [{'model_id': 'fast'}, {'binding': 'b'*64}, {'depends_on': []},
    {'task_id': 'other'}, {'source_status': 'retained'}, {'model_calls': True}, {'tool_calls': 17}])
def test_result_refuses_substituted_model_and_invalid_counters(fixture, change):
    server, agents = fixture
    serve(server, {**RESULT, 'results': {'choose': HUMAN, 'answer': {**OUTPUT, **change}}})
    with pytest.raises(SconeError):
        agents.result('one')


@pytest.mark.parametrize('change', [{'text': 'Changed'}, {'activation_id': 'other'}, {'model_calls': 1},
    {'source_status': 'retained'}, {'evidence_ids': ['chunk:1']}])
def test_human_result_matches_fresh_reply_and_cannot_claim_model_evidence(fixture, change):
    server, agents = fixture
    serve(server, {**RESULT, 'results': {'choose': {**HUMAN, **change}, 'answer': OUTPUT}})
    with pytest.raises(SconeError):
        agents.result('one')


def test_model_output_uses_native_character_limit_for_unicode(fixture):
    server, agents = fixture
    output = {**OUTPUT, 'text': '😀'*64000}
    serve(server, {**RESULT, 'results': {'choose': HUMAN, 'answer': output}})
    assert agents.result('one').results['answer'].text == output['text']


def test_evidence_identifiers_must_come_from_retained_packets(fixture):
    server, agents = fixture
    packet = {'status': 'prepared', 'items': [{'chunk_id': 1, 'episode_id': 2, 'text': 'Source',
        'source': None, 'created_at': '2026-09-12T10:00:00Z', 'score': 1.0}]}
    output = {**OUTPUT, 'evidence_ids': ['chunk:1'], 'evidence_packets': [json.dumps(packet)], 'source_status': 'retained'}
    serve(server, {**RESULT, 'results': {'choose': HUMAN, 'answer': output}})
    parsed = agents.result('one').results['answer']
    assert parsed.evidence_packets[0].evidence_ids == ('chunk:1',)
    for change in ({'evidence_ids': ['chunk:2']}, {'evidence_packets': []},
                   {'evidence_packets': ['{"status":"prepared","status":"changed"}']},
                   {'evidence_packets': [json.dumps({**packet, 'status': 'unavailable'})]}):
        serve(server, {**RESULT, 'results': {'choose': HUMAN, 'answer': {**output, **change}}})
        with pytest.raises(SconeError):
            agents.result('one')


def test_handoff_limit_preserves_actual_hops_without_inventing_final(fixture):
    server, agents = fixture
    plan = {'workflow_id': 'handoff', 'root_agent': 'worker', 'agents': [
        {'agent_id': 'worker', 'model_id': 'careful', 'can_handoff_to': ['worker']}], 'max_handoffs': 0}
    request = {**REQUEST, 'plan': {**SAVED, 'plan': plan, 'bindings': {'worker': 'a'*64}}}
    output = {**OUTPUT, 'task_id': 'hop-01', 'depends_on': []}
    result = {'space': 'alpha', 'run_id': 'one', 'status': 'handoff_limit', 'final': None,
              'hops': [{'output': output, 'handoff_to': 'worker'}], 'reused_hops': ['hop-01']}
    server.route('GET', '/v1/agent-runs/one/request', 200, request)
    server.route('GET', '/v1/agent-runs/one/result', 200, result)
    parsed = agents.result('one')
    assert parsed.status == 'handoff_limit' and parsed.final is None
    for change in ({'final': output}, {'status': 'completed'}, {'hops': [{'output': output, 'handoff_to': 'unknown'}]}):
        server.route('GET', '/v1/agent-runs/one/result', 200, {**result, **change})
        with pytest.raises(SconeError):
            agents.result('one')
