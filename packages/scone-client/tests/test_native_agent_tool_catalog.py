"""Configured native tool discovery survives restart without executing callbacks."""
import pytest
from scone import ToolChoice
from native_server import native_server


@pytest.mark.integration
def test_native_tool_catalog_is_shared_and_read_only_across_restart(tmp_path):
    expected = None
    for _ in range(2):
        with native_server(tmp_path, 'agent_tool_catalog_server.py') as client:
            assert client.capabilities().supports('agents.tools')
            choices = client.agents(expected_space='alpha').catalog()
            agents = {agent.agent_id: agent for agent in choices}
            assert agents['first'].tools == agents['second'].tools == (
                ToolChoice('inventory', '😀' * 1000, 'stock:1'),)
            assert agents['empty'].tools == ()
            raw = client._request('GET', '/v1/agents/catalog')
            assert raw['tools'] == [{'name': 'inventory', 'description': '😀' * 1000, 'revision': 'stock:1'}]
            if expected is not None: assert choices == expected
            expected = choices
            assert not (tmp_path / 'invoked').exists()
