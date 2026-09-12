"""Independent result publication and evidence boundary checks."""
import json
import pytest
from scone import SconeError
from scone.agent_evidence import EvidencePacket
from test_agents import fixture
from test_agent_results import serve


def claim(fact_id, subject, obj):
    return {'fact_id': fact_id, 'subject': subject, 'predicate': 'knows', 'object': obj,
            'origin': 'stated', 'status': 'active', 'valid_from': '2024-01-01T00:00:00Z',
            'valid_until': None, 'confidence': 1.0, 'source_episode_id': 1, 'quote': 'Retained source'}


def packet(step=None, right='Acme'):
    return {'status': 'prepared', 'claims': [claim(1,'Alice','Acme'), claim(2,right,'Bob')],
            'paths': [{'fact_ids': [1,2], 'steps': [step or {
                'from_fact': 1, 'to_fact': 2, 'kind': 'subject_object', 'direction': 'forward'}]}]}


def test_final_source_verified_result_is_last_remote_read(fixture):
    server, agents = fixture
    serve(server)
    agents.result('one')
    assert server.requests[-1].path == '/v1/agent-runs/one/result'
    assert all(row.method == 'GET' for row in server.requests)


def test_path_refuses_unrelated_subject_object_join():
    with pytest.raises(SconeError):
        EvidencePacket.from_json(json.dumps(packet(right='Globex')))


def test_path_endpoint_ids_reject_boolean_coercion():
    step={'from_fact': True, 'to_fact': 2, 'kind': 'subject_object', 'direction': 'forward'}
    with pytest.raises(SconeError):
        EvidencePacket.from_json(json.dumps(packet(step)))


def test_oversized_numeric_metadata_raises_typed_boundary_error():
    row=packet(); row['metadata_number']=10**400
    with pytest.raises(SconeError):
        EvidencePacket.from_json(json.dumps(row))


def test_empty_original_source_label_remains_valid():
    row={'status': 'prepared', 'items': [{'chunk_id': 1, 'episode_id': 1,
         'text': 'Retained source', 'source': '', 'created_at': '2024-01-01T00:00:00Z', 'score': 1.0}]}
    assert EvidencePacket.from_json(json.dumps(row)).evidence_ids == ('chunk:1',)
