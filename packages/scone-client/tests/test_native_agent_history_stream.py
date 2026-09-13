"""Native live delivery read off a real socket, and resumed across a restart."""
import time

import pytest

from scone import ModelTask, TaskPlan
from native_server import native_server


@pytest.mark.integration
def test_stream_delivers_every_page_and_resumes_after_restart(tmp_path):
    with native_server(tmp_path, 'agent_server.py') as client:
        agents = client.agents(expected_space='alpha')
        saved = agents.save_plan(
            TaskPlan('history', (ModelTask('answer', 'worker', 'careful', 'Answer'),)), expected_revision=0
        )
        agents.start('stream-run', plan=saved, question='PRIVATE request')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = agents.status('stream-run')
            if status.status == 'completed' and not status.active_local:
                break
            time.sleep(0.01)
        assert status.status == 'completed'
        pages = []
        with agents.stream_history('stream-run', limit=1) as stream:
            for received in stream:
                pages.append(received)
                if received.items and received.items[-1].event.kind == 'collection_finished':
                    break
        positions = [item.position for received in pages for item in received.items]
        assert positions == list(range(1, len(positions) + 1)) and positions
        assert pages[-1].items[-1].event.kind == 'collection_finished'
        assert not any('PRIVATE request' in str(item) for received in pages for item in received.items)
        cursor = pages[0].next_after
        seen_after_first = positions[1:]
    with native_server(tmp_path, 'agent_server.py') as client:
        agents = client.agents(expected_space='alpha')
        resumed = []
        with agents.stream_history('stream-run', limit=1, after=cursor) as stream:
            for received in stream:
                resumed.append(received)
                if received.items and received.items[-1].event.kind == 'collection_finished':
                    break
        assert [item.position for received in resumed for item in received.items] == seen_after_first
