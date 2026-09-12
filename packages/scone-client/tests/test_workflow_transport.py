"""Modern workflow negotiation is explicit and never performs hidden writes."""
import pytest
from scone import Scone, SconeError
from stub_server import StubScone


def test_capabilities_are_strict_and_optional_support_is_explicit():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.enqueue(200, {'schema_version': 1, 'implementation': 'python', 'features': {'agents.runs': True}})
        capabilities = client.capabilities()
        assert capabilities.supports('agents.runs')
        assert not capabilities.supports('agents.inputs')
        assert len(server.requests) == 1 and server.requests[0].method == 'GET'
        server.enqueue(200, {'schema_version': 1, 'implementation': 'python', 'features': {'agents.runs': 1}})
        with pytest.raises(SconeError):
            client.capabilities()


def test_resource_creation_is_local_and_space_binding_is_explicit():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        agents = client.agents(expected_space='alpha')
        assert agents.expected_space == 'alpha'
        assert server.requests == []
        with pytest.raises(SconeError):
            client.agents(expected_space='alpha\n')


def test_modern_mutations_refuse_unsupported_capability_before_writing():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.enqueue(200, {'schema_version': 1, 'implementation': 'rust', 'features': {}})
        with pytest.raises(SconeError, match='agents.runs'):
            client.agents(expected_space='alpha').cancel('one')
        assert [request.method for request in server.requests] == ['GET']


def test_modern_mutations_refuse_changed_space_before_writing():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.enqueue(200, {'schema_version': 1, 'implementation': 'python', 'features': {'agents.runs': True}})
        server.enqueue(200, {'space': 'beta'})
        with pytest.raises(SconeError, match='space'):
            client.agents(expected_space='alpha').cancel('one')
        assert [request.method for request in server.requests] == ['GET', 'GET']


def test_redirect_response_is_refused_even_with_successful_json_body():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.enqueue(302, {'schema_version': 1, 'implementation': 'python', 'features': {}})
        with pytest.raises(SconeError, match='redirect'):
            client.capabilities()
        assert len(server.requests) == 1


def test_transport_preserves_utf8_budget_without_ascii_expansion():
    import requests
    captured = []
    class Capture(requests.Session):
        def send(self, request, **kwargs):
            captured.append(request)
            response = requests.Response()
            response.status_code = 200
            response._content = b'{}'
            return response
    session = Capture()
    try:
        with Scone('http://127.0.0.1:7437', 'key', session=session) as client:
            client._request('POST', '/v1/agent-runs/one/inputs/choose/response',
                            json={'response': 'é' * 2000, 'expected_revision': 1})
        assert len(captured) == 1
        assert len(captured[0].body) < 8192
        assert captured[0].headers['Content-Type'] == 'application/json'
    finally:
        session.close()
