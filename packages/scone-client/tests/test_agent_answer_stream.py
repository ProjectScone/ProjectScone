"""An agent's answer, read as it is written, and never mistaken for the receipt.

The host publishes the text a running step is writing as SSE: `text`
frames with their sequence as the id, `withdraw` when text streamed before
a tool turn was not the answer, `gap` when the reader fell behind the
host's bounded window, `terminal` once the run has a receipt, and `end`
when the observation window closed. The client reads them as events, in
sequence, bound to the run it asked about; an `error` frame is a refusal;
a frame whose id does not name its sequence, a sequence that skips without
a gap, or a stream cut mid-frame is refused. What arrives is provisional:
the terminal says `read_receipt`, and the receipt is what `result()`
returns.
"""
import time

import pytest

from scone import SconeError
from test_agent_history_stream import HISTORY_CAPS, frame
from test_agents import fixture  # noqa: F401

ANSWER_CAPS = {**HISTORY_CAPS, 'features': {**HISTORY_CAPS['features'], 'agents.text_stream': True}}
PATH = '/v1/agent-runs/one/steps/find/text/stream'


def with_answers(server, caps=ANSWER_CAPS):
    server.route('GET', '/v1/capabilities', 200, caps)
    server.route('GET', '/v1/agent-runs/one/request', 200, __import__('test_agent_models').REQUEST)


def text(sequence, value):
    return frame('text', {'sequence': sequence, 'text': value}, str(sequence))


def test_text_arrives_in_sequence_and_a_terminal_points_at_the_receipt(fixture):
    server, agents = fixture
    with_answers(server)
    server.stream('GET', PATH, [
        text(1, 'We '), b': keep-alive\n\n', text(2, 'decided.'),
        frame('terminal', {'status': 'completed', 'read_receipt': True}),
    ], delay=0.02)
    seen = []
    with agents.stream_answer('one', 'find') as stream:
        for event in stream:
            seen.append(event)
    assert [event.kind for event in seen] == ['text', 'text', 'terminal']
    assert ''.join(event.text for event in seen if event.kind == 'text') == 'We decided.'
    assert seen[-1].status == 'completed' and seen[-1].read_receipt is True
    assert stream.cursor == 2
    request = server.requests[-1]
    assert request.path == PATH and request.headers.get('accept') == 'text/event-stream'
    assert not (request.query or {}).get('after')


def test_withdraw_gap_and_end_are_events_the_reader_can_act_on(fixture):
    server, agents = fixture
    with_answers(server)
    server.stream('GET', PATH, [
        text(1, 'Looking that up.'), frame('withdraw', {'sequence': 2}, '2'),
        frame('gap', {'after': 2, 'next_sequence': 7}), text(7, 'We decided.'),
        frame('end', {'reason': 'observation_window_ended'}),
    ])
    with agents.stream_answer('one', 'find') as stream:
        kinds = [(event.kind, getattr(event, 'sequence', None)) for event in stream]
    assert kinds == [('text', 1), ('withdraw', 2), ('gap', None), ('text', 7), ('end', None)]
    assert stream.cursor == 7


def test_resuming_sends_the_cursor_both_ways_and_requires_continuity(fixture):
    server, agents = fixture
    with_answers(server)
    server.stream('GET', PATH, [text(4, 'more'), frame('end', {'reason': 'observation_window_ended'})])
    with agents.stream_answer('one', 'find', after=3) as stream:
        events = list(stream)
    assert [event.kind for event in events] == ['text', 'end']
    request = server.requests[-1]
    assert request.query.get('after') == ['3'] and request.headers.get('last-event-id') == '3'
    server.stream('GET', PATH, [text(5, 'skipped one'), frame('end', {'reason': 'observation_window_ended'})])
    with pytest.raises(SconeError, match='sequence'):
        with agents.stream_answer('one', 'find', after=3) as stream:
            list(stream)


@pytest.mark.parametrize('frames, match', [
    ([frame('text', {'sequence': 1, 'text': 'x'}, '2')], 'id'),
    ([text(1, 'x'), text(1, 'again')], 'sequence'),
    ([frame('error', {'reason': 'text_unavailable'})], 'text_unavailable'),
    ([b'event: text\nid: 1\n'], 'inside a frame'),
    ([frame('other', {'sequence': 1})], 'kind'),
    ([frame('text', {'sequence': 1, 'text': 'x', 'reasoning': 'private'}, '1')], 'fields'),
    ([frame('terminal', {'status': 'completed', 'read_receipt': False})], 'receipt'),
])
def test_a_frame_that_is_not_one_of_ours_is_refused(fixture, frames, match):
    server, agents = fixture
    with_answers(server)
    server.stream('GET', PATH, frames)
    with pytest.raises(SconeError, match=match):
        with agents.stream_answer('one', 'find') as stream:
            list(stream)


def test_a_host_that_goes_quiet_cannot_hold_the_reader_past_its_timeout(fixture):
    server, agents = fixture
    with_answers(server)
    server.stream('GET', PATH, [text(1, 'x'), b'unused'], hang_after=1)
    agents._client.timeout = 0.5
    started = time.monotonic()
    with pytest.raises(SconeError):
        with agents.stream_answer('one', 'find') as stream:
            list(stream)
    assert time.monotonic() - started < 3


def test_the_capability_is_required_and_the_step_is_an_identifier(fixture):
    server, agents = fixture
    with_answers(server, HISTORY_CAPS)
    with pytest.raises(SconeError, match='agents.text_stream'):
        agents.stream_answer('one', 'find')
    with_answers(server)
    with pytest.raises(SconeError):
        agents.stream_answer('one', '../etc')
    with pytest.raises(SconeError):
        agents.stream_answer('one', 'find', after=-1)
