"""A step's answer read off a real socket as the model writes it."""
import time

import pytest

from scone import Ended, Gap, ModelTask, TaskPlan, Terminal, TextDelta, Withdrawn
from native_server import native_server


@pytest.mark.integration
def test_the_answer_arrives_as_it_is_written_and_the_receipt_says_the_same(tmp_path):
    with native_server(tmp_path, 'agent_server.py') as client:
        agents = client.agents(expected_space='alpha')
        assert client.capabilities().features['agents.text_stream'] is True
        saved = agents.save_plan(
            TaskPlan('answers', (ModelTask('answer', 'worker', 'careful', 'Answer'),)), expected_revision=0
        )
        agents.start('answer-run', plan=saved, question='PRIVATE request')
        events = []
        with agents.stream_answer('answer-run', 'answer') as stream:
            for event in stream:
                events.append(event)
                if isinstance(event, (Terminal, Ended)):
                    break
        kinds = [type(event) for event in events]
        assert kinds[-1] is Terminal and events[-1].read_receipt is True and events[-1].status == 'completed'
        assert Withdrawn not in kinds and Gap not in kinds
        deltas = [event for event in events if isinstance(event, TextDelta)]
        assert len(deltas) == 2, 'two pieces, written apart; the reader saw them as two'
        assert [delta.sequence for delta in deltas] == [1, 2]
        streamed = ''.join(delta.text for delta in deltas)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = agents.status('answer-run')
            if status.status == 'completed' and not status.active_local:
                break
            time.sleep(0.01)
        result = agents.result('answer-run')
        assert result.results['answer'].text == streamed == 'careful completed the task.'
        assert 'PRIVATE' not in streamed

        # After the run, the window is closed: a reader is pointed at the
        # receipt and given no provisional text.
        with agents.stream_answer('answer-run', 'answer') as later:
            after = list(later)
        assert [type(event) for event in after] == [Terminal]

        # A cursor resumes after what it has: from 1, only the second piece
        # would follow -- and here the window is gone, so only the receipt.
        with agents.stream_answer('answer-run', 'answer', after=1) as resumed:
            assert [type(event) for event in resumed] == [Terminal]
