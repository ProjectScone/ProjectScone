"""A conversation against the real service: open, ask, watch the reply arrive, read the receipt, stop, delete."""

import time

from scone import ConversationConflict, ReplyDelta, ReplyTerminal, Scone
from native_server import native_server


def test_a_session_runs_a_turn_streams_its_reply_and_is_closed_and_deleted(tmp_path):
    with native_server(tmp_path, 'conversation_server.py') as served:
        client = Scone(served.base_url, 'conversation-fixture')
        conversations = client.conversations(expected_space='alpha')
        caps = conversations.capabilities()
        assert caps.text_configured and caps.streaming and caps.turn_cancellation and caps.session_deletion
        session = conversations.create(request_id='open-1')
        assert session.state == 'running' and conversations.create(request_id='open-1').session_id == session.session_id
        receipt = conversations.submit(session.session_id, request_id='ask-1', text_='hello there', expected_revision=session.revision)
        assert receipt.status == 'pending'
        with conversations.stream_reply(session.session_id, 'ask-1') as stream:
            events = list(stream)
        deltas = [event.text for event in events if isinstance(event, ReplyDelta)]
        assert ''.join(deltas) == 'Scripted reply to hello there' and len(deltas) >= 2, ('the reply arrived as it was written', events)
        assert isinstance(events[-1], ReplyTerminal) and events[-1].read_receipt
        settled = conversations.wait(session.session_id, 'ask-1', timeout=10)
        assert settled.status == 'completed' and settled.text == 'Scripted reply to hello there'
        assert conversations.turns(session.session_id).turns[0].request_id == 'ask-1'
        transcript = conversations.transcript(session.session_id)
        assert any('Scripted reply' in episode.content for episode in transcript.episodes)
        history = conversations.events(session.session_id)
        assert [event.action for event in history.events][:2] == ['create', 'start']
        current = conversations.session(session.session_id)
        try:
            conversations.stop(session.session_id, request_id='stop-1', expected_revision=current.revision + 7)
        except ConversationConflict as conflict:
            assert conflict.revision == current.revision
        else:
            raise AssertionError('a stale revision must be a conflict')
        stopped = conversations.stop(session.session_id, request_id='stop-1', expected_revision=current.revision)
        assert stopped.state in ('stopping', 'ended')
        final = conversations.session(session.session_id)
        for _ in range(50):
            if final.closed:
                break
            time.sleep(0.1)
            final = conversations.session(session.session_id)
        assert final.closed
        assert conversations.delete(session.session_id) is None
        client.close()
