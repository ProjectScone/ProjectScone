"""Native history HTTP replay survives independent server processes."""

from dataclasses import asdict
import json
import time

import pytest
from scone import ModelTask, TaskPlan, ProgressEvent
from native_server import native_server


@pytest.mark.integration
@pytest.mark.parametrize('initial_search', [False, True])
def test_history_cursor_replays_after_restart_without_model_execution(tmp_path, initial_search):
    if initial_search:
        (tmp_path / 'initial-search').touch()
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
        first = agents.history('history-run', limit=1)
        assert first.items[0].event.kind == 'collection_started'
        cursor = first.next_after
        full = agents.history('history-run')
        if initial_search:
            assert any(
                isinstance(row.event, ProgressEvent) and row.event.tool_name == 'search_memory'
                for row in full.items
            )
        assert 'PRIVATE request' not in json.dumps(asdict(full))
        assert 'careful completed the task.' not in json.dumps(asdict(full))
    with native_server(tmp_path, 'agent_server.py') as client:
        replay = client.agents(expected_space='alpha').history('history-run', after=cursor)
        assert replay.items == full.items[1:]
        assert replay.items[-1].event.kind == 'collection_finished'
        assert any(
            isinstance(row.event, ProgressEvent) and row.event.model_id == 'careful' for row in replay.items
        )
        tail = client.agents(expected_space='alpha').history('history-run', after=replay.next_after)
        assert tail.items == () and tail.next_after == replay.next_after
    with native_server(tmp_path, 'agent_server.py') as client:
        assert client.agents(expected_space='alpha').history('history-run') == full
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 1 and calls[0]['model'] == 'careful'
