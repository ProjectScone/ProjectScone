"""Native history HTTP replay survives independent server processes."""

import json
import time

import pytest
from scone import ModelTask, TaskPlan
from native_server import native_server


@pytest.mark.integration
def test_history_cursor_replays_after_restart_without_model_execution(tmp_path):
    path = '/v1/agent-runs/history-run/history'
    with native_server(tmp_path, 'agent_server.py') as client:
        agents = client.agents(expected_space='alpha')
        saved = agents.save_plan(
            TaskPlan('history', (ModelTask('answer', 'worker', 'careful', 'Answer'),)), expected_revision=0
        )
        agents.start('history-run', plan=saved, question='PRIVATE request')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = agents.status('history-run')
            if status.status == 'completed' and not status.active_local:
                break
            time.sleep(0.01)
        assert status.status == 'completed' and not status.active_local
        first = client._request('GET', path, params={'limit': '1'})
        assert first['items'][0]['event']['kind'] == 'collection_started'
        cursor = first['next_after']
        full = client._request('GET', path)
        assert 'PRIVATE request' not in json.dumps(full)
        assert 'careful completed the task.' not in json.dumps(full)
    with native_server(tmp_path, 'agent_server.py') as client:
        replay = client._request('GET', path, params={'after': cursor})
        assert replay['items'] == full['items'][1:]
        assert replay['items'][-1]['event']['kind'] == 'collection_finished'
        assert any(row['event'].get('model_id') == 'careful' for row in replay['items'])
        tail = client._request('GET', path, params={'after': replay['next_after']})
        assert tail['items'] == [] and tail['next_after'] == replay['next_after']
    with native_server(tmp_path, 'agent_server.py') as client:
        assert client._request('GET', path) == full
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 1 and calls[0]['model'] == 'careful'
