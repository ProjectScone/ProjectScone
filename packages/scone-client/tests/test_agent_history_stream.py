"""Typed live delivery: frames arrive one at a time, bound to the run, and stop honestly.

The native host publishes verified history pages as SSE. The client reads
them as it would read a page -- the same `HistoryPage` validation, the
same cursor continuity -- with three things a page read never had to
care about: a frame is bounded before it is parsed, a server that goes
quiet cannot hold the client past its read timeout, and an `error` frame
is a refusal to be raised, not a page to be shown.

No frame is ever an answer. This is past execution metadata delivered as
it is written, and nothing here calls it anything else.
"""
import json
import time

import pytest

from scone import SconeError
from test_agent_history import cursor, page
from test_agents import fixture, CAPS

HISTORY_CAPS = {**CAPS, 'features': {**CAPS['features'], 'agents.history': True}}


def with_history(server):
    server.route('GET', '/v1/capabilities', 200, HISTORY_CAPS)
    server.route('GET', '/v1/agent-runs/one/request', 200, __import__('test_agent_models').REQUEST)


def frame(kind, data, identifier=None):
    head = 'event: ' + kind + '\n'
    if identifier is not None:
        head += 'id: ' + identifier + '\n'
    return (head + 'data: ' + json.dumps(data, separators=(',', ':')) + '\n\n').encode()


def page_at(first_position, count, next_position=None):
    """A page whose items run from `first_position`, cursored at its last."""
    value = page()
    items = []
    for offset in range(count):
        item = dict(value['items'][0])
        item['position'] = first_position + offset
        items.append(item)
    value['items'] = items
    value['next_after'] = cursor(first_position + count - 1)
    return value


def test_pages_arrive_frame_by_frame_and_stop_at_end(fixture):
    server, agents = fixture
    with_history(server)
    first, second = page_at(1, 1), page_at(2, 1)
    server.stream('GET', '/v1/agent-runs/one/history/stream', [
        frame('history', first, first['next_after']),
        b': keep-alive\n\n',
        frame('history', second, second['next_after']),
        frame('end', {'reason': 'observation_window_ended'}),
    ], delay=0.05)
    seen = []
    with agents.stream_history('one', limit=1) as pages:
        for received in pages:
            seen.append(received)
    assert [p.items[0].position for p in seen] == [1, 2]
    assert seen[-1].next_after == second['next_after']
    request = next(r for r in server.requests if r.path.startswith('/v1/agent-runs/one/history/stream'))
    assert request.headers.get('accept', '').startswith('text/event-stream')
    assert request.query.get('limit') == ['1']


def test_an_error_frame_is_a_refusal_and_nothing_after_it_is_read(fixture):
    server, agents = fixture
    with_history(server)
    first = page_at(1, 1)
    server.stream('GET', '/v1/agent-runs/one/history/stream', [
        frame('history', first, first['next_after']),
        frame('error', {'reason': 'history_unavailable'}),
        frame('history', page_at(2, 1), cursor(2)),
    ])
    seen = []
    with pytest.raises(SconeError) as error:
        with agents.stream_history('one', limit=1) as pages:
            for received in pages:
                seen.append(received)
    assert 'history_unavailable' in str(error.value)
    assert len(seen) == 1


def test_a_frame_whose_id_disagrees_with_its_page_is_refused(fixture):
    """The SSE `id` is what a reconnect will send back. If it does not
    name the page it rode with, resuming from it would skip or repeat."""
    server, agents = fixture
    with_history(server)
    first = page_at(1, 1)
    server.stream('GET', '/v1/agent-runs/one/history/stream', [frame('history', first, cursor(7))])
    with pytest.raises(SconeError):
        with agents.stream_history('one', limit=1) as pages:
            list(pages)


def test_resume_sends_the_cursor_both_ways_and_checks_continuity(fixture):
    server, agents = fixture
    with_history(server)
    resumed = page_at(2, 1)
    server.stream('GET', '/v1/agent-runs/one/history/stream', [
        frame('history', resumed, resumed['next_after']),
        frame('end', {'reason': 'observation_window_ended'}),
    ])
    with agents.stream_history('one', limit=1, after=cursor(1)) as pages:
        received = list(pages)
    assert [p.items[0].position for p in received] == [2]
    request = next(r for r in server.requests if r.path.startswith('/v1/agent-runs/one/history/stream'))
    assert request.headers.get('last-event-id') == cursor(1)
    assert request.query.get('after') == [cursor(1)]

    # A page that starts before the cursor is a repeat, and is refused.
    server.stream('GET', '/v1/agent-runs/one/history/stream', [
        frame('history', page_at(1, 1), cursor(1)),
    ])
    with pytest.raises(SconeError):
        with agents.stream_history('one', limit=1, after=cursor(1)) as pages:
            list(pages)


def test_a_server_that_goes_quiet_is_bounded_by_the_read_timeout(fixture):
    """The client's promise is about time. A stalled server must produce
    a refusal within the timeout, measured on the wall clock -- never a
    hang waiting for a frame that will not come."""
    server, agents = fixture
    with_history(server)
    first = page_at(1, 1)
    server.stream('GET', '/v1/agent-runs/one/history/stream',
                  [frame('history', first, first['next_after']), b'unused'], hang_after=1)
    agents._client.timeout = 0.5
    began = time.monotonic()
    with pytest.raises(SconeError):
        with agents.stream_history('one', limit=1) as pages:
            list(pages)
    assert time.monotonic() - began < 3.0


def test_an_oversized_frame_is_refused_before_it_is_parsed(fixture):
    server, agents = fixture
    with_history(server)
    huge = b'event: history\ndata: ' + b'x' * (2 * 1024 * 1024) + b'\n\n'
    server.stream('GET', '/v1/agent-runs/one/history/stream', [huge])
    # The total-body bound would refuse this too. Lift it out of the way so
    # that what is proven here is the per-line bound -- a single frame that
    # is too long is refused before it is buffered, whatever the total
    # allowance is.
    agents._client.max_response_bytes = 64 * 1024 * 1024
    with pytest.raises(SconeError) as error:
        with agents.stream_history('one', limit=1) as pages:
            list(pages)
    # Refused *for its size*, not for failing to parse afterwards. Without
    # this the test could not tell the bound from the JSON decoder.
    assert 'limit' in str(error.value), str(error.value)


def test_streaming_requires_the_history_capability(fixture):
    server, agents = fixture
    without = json.loads(json.dumps(CAPS))
    without['features'].pop('agents.history', None)
    server.route('GET', '/v1/capabilities', 200, without)
    with pytest.raises(SconeError):
        with agents.stream_history('one') as pages:
            list(pages)
