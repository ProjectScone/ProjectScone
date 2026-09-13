"""Read-only live observations keep source and caller checks per publication."""

import asyncio
import json

import pytest

from tests.api.test_agent_runs_api import auth, body, setup
from tests.api.test_agent_history_api import evidence_setup, start


async def collect(app, *, query='', token='reader', on_frame=None, on_start=None):
    messages = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.start' and on_start is not None:
            await on_start()
        if message['type'] == 'http.response.body' and message.get('body') and on_frame is not None:
            await on_frame(message['body'])

    await app(
        {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'}, 'http_version': '1.1',
         'method': 'GET', 'scheme': 'http', 'path': '/v1/agent-runs/one/history/stream',
         'raw_path': b'/v1/agent-runs/one/history/stream', 'query_string': query.encode(),
         'headers': [(b'authorization', ('Bearer ' + token).encode())],
         'client': ('127.0.0.1', 1234), 'server': ('scone.test', 80)}, receive, send,
    )
    return messages


def frames(messages):
    body = b''.join(message.get('body', b'') for message in messages)
    result = []
    for frame in body.decode().split('\n\n'):
        values = dict(line.split(': ', 1) for line in frame.splitlines() if ': ' in line)
        if 'data' in values:
            result.append((values.get('event'), values.get('id'), json.loads(values['data'])))
    return result


@pytest.fixture
def short_window(monkeypatch):
    from scone_memory.api import agent_history
    monkeypatch.setattr(agent_history, 'STREAM_SECONDS', 0.1, raising=False)
    monkeypatch.setattr(agent_history, 'POLL_SECONDS', 0.005, raising=False)


async def test_stream_replays_pages_and_finishes_observation_without_reexecution(setup, short_window):
    _, app, _, _, _, _, calls = setup
    await start(setup)
    messages = await collect(app, query='limit=1')
    assert messages[0]['status'] == 200
    headers = dict(messages[0]['headers'])
    assert headers[b'content-type'].startswith(b'text/event-stream')
    assert headers[b'cache-control'] == b'no-store'
    events = frames(messages)
    pages = [data for kind, _, data in events if kind == 'history']
    entries = [entry for page in pages for entry in page['items']]
    assert entries[0]['event']['kind'] == 'collection_started'
    assert entries[-1]['event']['kind'] == 'collection_finished'
    assert [entry['position'] for entry in entries] == list(range(1, len(entries) + 1))
    assert all(cursor == data['next_after'] for kind, cursor, data in events if kind == 'history')
    assert events[-1] == ('end', None, {'reason': 'observation_window_ended'})
    assert len(calls) == 1
    assert b'Selected model answer' not in b''.join(m.get('body', b'') for m in messages)


@pytest.mark.parametrize('query', ['limit=0', 'limit=101', 'limit=1&limit=2', 'after=', 'unknown=x'])
async def test_stream_rejects_invalid_query_before_headers(setup, query):
    response = await setup[0].get('/v1/agent-runs/one/history/stream?' + query, headers=auth())
    assert response.status_code == 422


async def test_stream_rechecks_authority_after_response_headers(setup, short_window):
    _, app, _, _, _, _, calls = setup
    await start(setup)

    async def revoke():
        app.state.keys.pop('reader')

    events = frames(await collect(app, on_start=revoke))
    assert events == [('error', None, {'reason': 'history_unavailable'})]
    assert len(calls) == 1


async def test_stream_revocation_between_pages_withholds_next_page(setup, short_window):
    _, app, _, _, _, _, calls = setup
    await start(setup)

    async def revoke(frame):
        if b'event: history' in frame:
            app.state.keys.pop('reader')

    events = frames(await collect(app, query='limit=1', on_frame=revoke))
    assert [kind for kind, _, _ in events] == ['history', 'error']
    assert len(events[0][2]['items']) == 1 and len(calls) == 1


async def test_stream_rechecks_forgotten_sources_between_pages(evidence_setup, short_window):
    _, app, _, memory, source, calls = evidence_setup

    async def forget(frame):
        if b'event: history' in frame:
            await memory.forget('alpha', source.episode_id)

    events = frames(await collect(app, query='limit=1', token='writer', on_frame=forget))
    assert [kind for kind, _, _ in events] == ['history', 'error']
    assert events[-1][2] == {'reason': 'history_unavailable'}
    assert len(calls) == 1


async def test_stream_observes_new_events_while_model_is_already_running(setup, short_window):
    client, app, service, _, release, entered, calls = setup
    assert (await client.post('/v1/agent-runs', json=body(), headers=auth())).status_code == 202
    await entered.wait()

    async def finish(frame):
        if b'event: history' in frame:
            release.set()
            await service.wait('alpha', 'one')

    events = frames(await collect(app, on_frame=finish))
    pages = [data for kind, _, data in events if kind == 'history']
    assert len(pages) >= 2
    assert pages[0]['items'][-1]['event']['kind'] != 'collection_finished'
    assert pages[-1]['items'][-1]['event']['kind'] == 'collection_finished'
    assert len(calls) == 1
