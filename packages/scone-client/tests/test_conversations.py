"""Conversations from the client, against a scripted server: every command named, every read checked."""

import json

import pytest

from scone import ConversationConflict, ReplyDelta, ReplyEnded, ReplyGap, ReplyTerminal, SconeError
from stub_server import StubScone
from test_agent_history_stream import frame

CAPS = {'schema_version': 1, 'implementation': 'stub', 'features': {'conversations': True}}
CONVERSATION_CAPS = {'schema_version': 1, 'text_configured': True, 'personas': False, 'streaming': True,
                     'text_stream': {'transport': 'sse', 'replay': 'active_window', 'max_bytes': 65536, 'max_chunks': 256},
                     'turn_cancellation': True, 'session_deletion': True, 'transcript_pagination': True,
                     'max_sessions': 100, 'max_turns': 100, 'reply_transport': 'poll', 'something_new': {'x': 1}}
SESSION = {'space': 'alpha', 'session_id': 's1', 'mode': 'text', 'state': 'running', 'revision': 2,
           'created_at': '2026-09-13T20:00:00Z', 'updated_at': '2026-09-13T20:00:01Z', 'recall_scope': {}, 'persona': None,
           'active_request_id': None, 'latest_request_id': None}
PENDING = {'request_id': 't1', 'status': 'pending', 'result': None, 'result_state': None, 'error': None}
DONE = {'request_id': 't1', 'status': 'completed', 'result': {'text': 'Hello back.', 'assistant_episode_id': 4},
        'result_state': 'available', 'error': None}


@pytest.fixture
def fixture():
    with StubScone() as server, __import__('scone').Scone(server.base_url, 'key') as client:
        server.route('GET', '/v1/capabilities', 200, CAPS)
        server.route('GET', '/v1/status', 200, {'space': 'alpha'})
        server.route('GET', '/v1/conversations/capabilities', 200, CONVERSATION_CAPS)
        yield server, client.conversations(expected_space='alpha')


def test_capabilities_are_read_for_what_they_promise_and_tolerate_growth(fixture):
    server, conversations = fixture
    caps = conversations.capabilities()
    assert caps.streaming and caps.turn_cancellation and caps.session_deletion and caps.max_turns == 100
    server.route('GET', '/v1/conversations/capabilities', 200, {**CONVERSATION_CAPS, 'text_stream': {'transport': 'ws'}})
    with pytest.raises(SconeError, match='transport'):
        conversations.capabilities()


def test_create_names_the_request_and_returns_the_session_of_the_expected_space(fixture):
    server, conversations = fixture
    server.route('POST', '/v1/conversations', 200, SESSION)
    session = conversations.create(request_id='new-1')
    assert session.session_id == 's1' and session.revision == 2 and session.state == 'running' and not session.closed
    body = server.requests[-1].json
    assert body == {'request_id': 'new-1', 'capture': True, 'mode': 'text'}
    server.route('POST', '/v1/conversations', 200, {**SESSION, 'space': 'beta'})
    with pytest.raises(SconeError, match='expected space'):
        conversations.create(request_id='new-2')


def test_a_stale_revision_is_a_conflict_carrying_the_revision_the_server_holds(fixture):
    server, conversations = fixture
    server.route('POST', '/v1/conversations/s1/turns', 409, {'error': 'stale revision', 'revision': 5})
    with pytest.raises(ConversationConflict) as refused:
        conversations.submit('s1', request_id='t1', text_='hello', expected_revision=2)
    assert refused.value.revision == 5 and refused.value.status == 409 and refused.value.code is None
    server.route('POST', '/v1/conversations', 409, {'error': 'persona changed', 'code': 'persona_selection_stale', 'fingerprint': 'abc'})
    with pytest.raises(ConversationConflict) as stale:
        conversations.create(request_id='new-3', persona='p1', persona_fingerprint='abc')
    assert stale.value.code == 'persona_selection_stale' and stale.value.fingerprint == 'abc' and stale.value.revision is None


def test_a_turn_is_submitted_read_and_waited_for(fixture):
    server, conversations = fixture
    server.route('POST', '/v1/conversations/s1/turns', 202, PENDING)
    receipt = conversations.submit('s1', request_id='t1', text_='hello', expected_revision=2)
    assert receipt.status == 'pending' and not receipt.settled and receipt.text is None
    assert server.requests[-1].json == {'request_id': 't1', 'expected_revision': 2, 'text': 'hello'}
    server.route('GET', '/v1/conversations/s1/turns/t1', 200, DONE)
    done = conversations.wait('s1', 't1', timeout=2, interval=0.05)
    assert done.settled and done.text == 'Hello back.' and done.result_state == 'available'
    server.route('GET', '/v1/conversations/s1/turns/t1', 200, {**DONE, 'request_id': 't2'})
    with pytest.raises(SconeError, match='identity'):
        conversations.turn('s1', 't1')
    server.route('GET', '/v1/conversations/s1/turns/t1', 200, {**PENDING, 'result': {'text': 'x'}})
    with pytest.raises(SconeError, match='pending turn'):
        conversations.turn('s1', 't1')
    server.route('GET', '/v1/conversations/s1/turns/t1', 200, PENDING)
    with pytest.raises(SconeError, match='still pending'):
        conversations.wait('s1', 't1', timeout=0.1, interval=0.05)


def test_pages_of_sessions_turns_events_and_transcript(fixture):
    server, conversations = fixture
    server.route('GET', '/v1/conversations', 200, {'items': [SESSION], 'next_after': 's1', 'has_more': True})
    page = conversations.sessions(limit=1)
    assert page.items[0].session_id == 's1' and page.next_after == 's1' and page.has_more
    server.route('GET', '/v1/conversations/s1/turns', 200, {'turns': [DONE, PENDING], 'next_after': '', 'has_more': False})
    with pytest.raises(SconeError, match='turn page'):
        conversations.turns('s1')
    server.route('GET', '/v1/conversations/s1/turns', 200, {'turns': [DONE], 'next_after': '', 'has_more': False})
    assert conversations.turns('s1').turns[0].text == 'Hello back.'
    events = [{'session_id': 's1', 'request_id': 'new-1', 'revision': 1, 'action': 'create', 'previous_state': None,
               'state': 'created', 'recorded_at': '2026-09-13T20:00:00Z'},
              {'session_id': 's1', 'request_id': 'new-1', 'revision': 2, 'action': 'start', 'previous_state': 'created',
               'state': 'running', 'recorded_at': '2026-09-13T20:00:01Z'}]
    server.route('GET', '/v1/conversations/s1/events', 200, {'events': events, 'next_after': 2, 'has_more': False})
    history = conversations.events('s1')
    assert [e.action for e in history.events] == ['create', 'start'] and history.next_after == 2
    server.route('GET', '/v1/conversations/s1/events', 200, {'events': events[::-1], 'next_after': 2, 'has_more': False})
    with pytest.raises(SconeError, match='event page'):
        conversations.events('s1')
    episode = {'episode_id': 4, 'kind': 'conversation', 'content': 'Hello back.', 'source': None, 'tags': [],
               'metadata': {'session_id': 's1'}, 'created_at': '2026-09-13T20:00:02Z', 'ingested_at': '2026-09-13T20:00:02Z',
               'attachments': []}
    server.route('GET', '/v1/conversations/s1/transcript', 200, {'episodes': [episode], 'has_more': True, 'next_before': 'b64cursor'})
    transcript = conversations.transcript('s1', limit=1)
    assert transcript.episodes[0].content == 'Hello back.' and transcript.next_before == 'b64cursor'
    conversations.transcript('s1', before='b64cursor')
    assert server.requests[-1].query.get('before') == ['b64cursor']


def test_cancel_stop_and_delete_are_acknowledged_or_refused(fixture):
    server, conversations = fixture
    server.route('POST', '/v1/conversations/s1/turns/t1/cancel', 200, {**PENDING, 'status': 'cancelled'})
    assert conversations.cancel('s1', 't1').status == 'cancelled'
    server.route('POST', '/v1/conversations/s1/turns/t1/cancel', 200, DONE)
    with pytest.raises(SconeError, match='cancel acknowledgement'):
        conversations.cancel('s1', 't1')
    server.route('POST', '/v1/conversations/s1/stop', 200, {**SESSION, 'state': 'stopping', 'revision': 3})
    stopped = conversations.stop('s1', request_id='stop-1', expected_revision=2)
    assert stopped.state == 'stopping' and server.requests[-1].json == {'request_id': 'stop-1', 'expected_revision': 2}
    server.route('DELETE', '/v1/conversations/s1', 200, {})
    assert conversations.delete('s1') is None
    server.route('DELETE', '/v1/conversations/s1', 409, {'error': 'session is running'})
    with pytest.raises(ConversationConflict):
        conversations.delete('s1')
    server.route('GET', '/v1/conversations/capabilities', 200, {**CONVERSATION_CAPS, 'session_deletion': False})
    with pytest.raises(SconeError, match='does not delete'):
        conversations.delete('s1')


def reply_frames():
    return [frame('text', {'sequence': 1, 'text': 'Hello ', 'provisional': True}, '1'),
            b': keep-alive\n\n',
            frame('text', {'sequence': 2, 'text': 'back.', 'provisional': True}, '2'),
            frame('terminal', {'request_id': 't1', 'status': 'completed', 'read_receipt': True})]


def test_the_reply_arrives_in_sequence_and_the_terminal_points_at_the_receipt(fixture):
    server, conversations = fixture
    server.stream('GET', '/v1/conversations/s1/turns/t1/stream', reply_frames())
    with conversations.stream_reply('s1', 't1') as stream:
        events = list(stream)
    assert [e.text for e in events if isinstance(e, ReplyDelta)] == ['Hello ', 'back.']
    assert isinstance(events[-1], ReplyTerminal) and events[-1].status == 'completed' and events[-1].read_receipt
    assert stream.cursor == 2 and 'after' not in server.requests[-1].query
    server.stream('GET', '/v1/conversations/s1/turns/t1/stream', [frame('gap', {'after': 2, 'next_sequence': 5}),
                                                                    frame('text', {'sequence': 5, 'text': 'x', 'provisional': True}, '5'),
                                                                    frame('end', {'request_id': 't1', 'reason': 'service_shutdown', 'read_receipt': True})])
    with conversations.stream_reply('s1', 't1', after=2) as resumed:
        events = list(resumed)
    assert isinstance(events[0], ReplyGap) and events[1].sequence == 5 and isinstance(events[2], ReplyEnded)
    sent = server.requests[-1]
    assert sent.query.get('after') == ['2']
    assert {k.lower(): v for k, v in sent.headers.items()}.get('last-event-id') == '2'


@pytest.mark.parametrize('frames, reason', [
    ([frame('text', {'sequence': 2, 'text': 'x', 'provisional': True}, '2')], 'sequence'),
    ([frame('text', {'sequence': 1, 'text': 'x', 'provisional': True}, '9')], 'name its sequence'),
    ([frame('text', {'sequence': 1, 'text': 'x', 'provisional': False}, '1')], 'final'),
    ([frame('terminal', {'request_id': 't2', 'status': 'completed', 'read_receipt': True})], 'for this turn'),
    ([frame('terminal', {'request_id': 't1', 'status': 'completed', 'read_receipt': False})], 'read_receipt'),
    ([frame('gap', {'after': 1, 'next_sequence': 3})], 'gap sequence'),
    ([frame('error', {'reason': 'window_unavailable'})], 'refused'),
    ([b'event: text\ndata: {"sequence": 1'], 'inside a frame'),
])
def test_a_frame_that_breaks_the_grammar_is_refused(fixture, frames, reason):
    server, conversations = fixture
    server.stream('GET', '/v1/conversations/s1/turns/t1/stream', frames)
    with pytest.raises(SconeError, match=reason):
        with conversations.stream_reply('s1', 't1') as stream:
            list(stream)


def test_streaming_needs_the_server_to_stream(fixture):
    server, conversations = fixture
    server.route('GET', '/v1/conversations/capabilities', 200, {**CONVERSATION_CAPS, 'text_stream': None, 'streaming': False})
    with pytest.raises(SconeError, match='does not stream'):
        conversations.stream_reply('s1', 't1')
