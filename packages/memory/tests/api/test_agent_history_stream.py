"""Read-only live observations keep source and caller checks per publication."""

import asyncio
import json

import pytest

from tests.api.test_agent_runs_api import auth, body, setup
from tests.api.test_agent_history_api import evidence_setup, start


async def collect(app, *, query='', token='reader', on_frame=None, on_start=None, headers=(),
                  send_body=None):
    """Drive the app over ASGI and keep every message it sent.

    ``send_body`` replaces the wait for a body message when given, so a
    client that stops reading can be played back exactly.
    """
    messages = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.start' and on_start is not None:
            await on_start()
        if message['type'] == 'http.response.body' and message.get('body'):
            if send_body is not None:
                await send_body(message['body'])
            if on_frame is not None:
                await on_frame(message['body'])

    await app(
        {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'}, 'http_version': '1.1',
         'method': 'GET', 'scheme': 'http', 'path': '/v1/agent-runs/one/history/stream',
         'raw_path': b'/v1/agent-runs/one/history/stream', 'query_string': query.encode(),
         'headers': [(b'authorization', ('Bearer ' + token).encode()), *headers],
         'client': ('127.0.0.1', 1234), 'server': ('scone.test', 80)}, receive, send,
    )
    return messages


def parse(body: bytes):
    """SSE frames from raw bytes, as (event, id, data)."""
    result = []
    for frame in body.decode().split('\n\n'):
        values = dict(line.split(': ', 1) for line in frame.splitlines() if ': ' in line)
        if 'data' in values:
            result.append((values.get('event'), values.get('id'), json.loads(values['data'])))
    return result


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


# --- The bounds the stream claims, exercised where they can actually fail ---

def counting(service, monkeypatch):
    """Count deliveries, so a closed observer can be shown to stop asking."""
    seen = []
    original = service.history_for_delivery

    async def counted(*args, **kwargs):
        seen.append(None)
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, 'history_for_delivery', counted)
    return seen


@pytest.mark.parametrize('blocked_call', [1, 2])
async def test_a_delivery_that_overruns_its_bound_is_refused_not_published(setup, short_window,
                                                                          monkeypatch, blocked_call):
    """The reviewer's hypothesis, reproduced rather than argued about.

    `asyncio.timeout` cancels at an `await`. `history_for_delivery` does
    its `stat`s, opens the journal and reads the tail **synchronously**,
    so a stalled disk cannot be interrupted -- the call returns whenever
    the disk does, and nothing then compares that moment to the bound
    the route claimed. Blocking the first call (before headers) must
    refuse admission; blocking a later one must end the stream with the
    fixed refusal, never a healthy frame that arrived late.
    """
    import time
    from scone_memory.api import agent_history

    _, app, service, _, _, _, _ = setup
    await start(setup)
    monkeypatch.setattr(agent_history, 'VERIFY_SECONDS', 0.05, raising=False)
    original = service.history_for_delivery
    calls = []

    async def stalled(*args, **kwargs):
        calls.append(None)
        page = await original(*args, **kwargs)
        if len(calls) == blocked_call:
            time.sleep(0.2)   # synchronous: the loop cannot run the timeout's callback
        return page

    monkeypatch.setattr(service, 'history_for_delivery', stalled)
    messages = await collect(app)
    if blocked_call == 1:
        assert messages[0]['status'] != 200, messages[0]
        assert b'history_unavailable' in b''.join(m.get('body', b'') for m in messages)
    else:
        assert messages[0]['status'] == 200
        kinds = [kind for kind, _, _ in frames(messages)]
        assert 'history' not in kinds, kinds
        assert kinds == ['error'], kinds


async def test_a_client_that_disconnects_releases_the_observer(setup, short_window, monkeypatch):
    """Cancellation is the disconnect signal an ASGI server delivers. It
    must propagate -- the route catches `Exception` after headers, and
    `CancelledError` is not one -- and once it has, nothing is read."""
    _, app, service, _, _, _, _ = setup
    await start(setup)
    seen = counting(service, monkeypatch)
    first = asyncio.Event()

    async def note(frame):
        if b'event: history' in frame:
            first.set()

    task = asyncio.create_task(collect(app, query='limit=1', on_frame=note))
    await asyncio.wait_for(first.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    settled = len(seen)
    await asyncio.sleep(0.05)
    assert len(seen) == settled, "a cancelled observer kept delivering"


async def test_a_cursor_from_before_the_retention_floor_reconnects_with_the_gap_disclosed(
        setup, short_window):
    """Reconnect after purge. The retained window moved on while the
    client was away, so its `Last-Event-ID` points below the floor. That
    is not refused -- the reader gets what remains -- but the page says
    where retention now starts and which positions it will never see."""
    client, app, service, _, release, entered, _ = setup
    service._history._capacity = 2
    assert (await client.post('/v1/agent-runs', json=body(), headers=auth())).status_code == 202
    await entered.wait()
    early = frames(await collect(app, query='limit=1'))
    cursor = next(cursor for kind, cursor, _ in early if kind == 'history')
    held = int(cursor.split('.')[1], 16)
    release.set()
    await service.wait('alpha', 'one')

    later = await collect(app, headers=[(b'last-event-id', cursor.encode())])
    assert later[0]['status'] == 200
    page = next(data for kind, _, data in frames(later) if kind == 'history')
    assert page['retained_from'] is not None and page['retained_from'] > held + 1
    assert page['omitted'] is not None, page
    assert all(item['position'] >= page['retained_from'] for item in page['items'])


async def test_the_stream_is_readable_off_a_real_socket(setup, short_window):
    """Everything above drives the app through ASGI callbacks. This puts
    uvicorn on a loopback port and reads the bytes back through a real
    HTTP client, so the framing, headers and end-of-stream are the ones
    a browser or SDK would actually see."""
    import httpx
    import uvicorn

    _, app, _, _, _, _, _ = setup
    await start(setup)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='error',
                                           lifespan='off'))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}') as client:
            async with client.stream('GET', '/v1/agent-runs/one/history/stream?limit=1',
                                     headers={**auth(), 'Connection': 'close'}) as response:
                assert response.status_code == 200
                assert response.headers['content-type'].startswith('text/event-stream')
                assert response.headers['cache-control'] == 'no-store'
                raw = b''.join([chunk async for chunk in response.aiter_raw()])
    finally:
        server.should_exit = True
        await serving
    events = parse(raw)
    pages = [data for kind, _, data in events if kind == 'history']
    entries = [entry for page in pages for entry in page['items']]
    assert entries and entries[-1]['event']['kind'] == 'collection_finished'
    assert [entry['position'] for entry in entries] == list(range(1, len(entries) + 1))
    assert events[-1] == ('end', None, {'reason': 'observation_window_ended'})


# --- Boundaries the first ten tests did not reach ---------------------------

import time as _time


@pytest.fixture
def short_bounds(monkeypatch, short_window):
    from scone_memory.api import agent_history
    monkeypatch.setattr(agent_history, 'SEND_SECONDS', 0.05, raising=False)
    monkeypatch.setattr(agent_history, 'VERIFY_SECONDS', 0.5, raising=False)


async def test_reconnect_with_last_event_id_resumes_after_that_position(setup, short_window):
    """A dropped connection is resumed from the cursor the client last saw,
    and nothing before it is sent twice."""
    _, app, _, _, _, _, calls = setup
    await start(setup)
    first = frames(await collect(app, query='limit=1'))
    pages = [(cursor, data) for kind, cursor, data in first if kind == 'history']
    assert len(pages) >= 2
    cursor, page = pages[0]
    seen = page['items'][-1]['position']

    async def resume(app, cursor):
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'GET',
                   'scheme': 'http', 'path': '/v1/agent-runs/one/history/stream',
                   'raw_path': b'/v1/agent-runs/one/history/stream', 'query_string': b'limit=1',
                   'headers': [(b'authorization', b'Bearer reader'), (b'last-event-id', cursor.encode())],
                   'client': ('127.0.0.1', 1), 'server': ('scone.test', 80)}, receive, send)
        return messages

    resumed = frames(await resume(app, cursor))
    positions = [e['position'] for k, _, d in resumed if k == 'history' for e in d['items']]
    assert positions and positions[0] == seen + 1, positions
    assert positions == sorted(set(positions)), 'no duplicate or reordered positions'
    assert len(calls) == 1


@pytest.mark.parametrize('cursor', ['not-a-cursor', 'a' * 32 + '.' + '0' * 16 + '.' + 'b' * 64, ''])
async def test_a_forged_or_empty_last_event_id_is_refused_before_headers(setup, cursor):
    """The cursor is signed against the run; a forged, foreign or empty
    one is refused as a 422 with nothing streamed."""
    await start(setup)
    response = await setup[0].get('/v1/agent-runs/one/history/stream',
                                  headers={**auth(), 'Last-Event-ID': cursor})
    assert response.status_code == 422, response.text


async def test_a_conflicting_after_and_last_event_id_is_refused(setup):
    await start(setup)
    first = frames(await collect(setup[1], query='limit=1'))
    cursor = next(c for k, c, _ in first if k == 'history')
    response = await setup[0].get('/v1/agent-runs/one/history/stream?after=' + cursor[:-1] + '0',
                                  headers={**auth(), 'Last-Event-ID': cursor})
    assert response.status_code == 422


async def test_a_client_that_stops_reading_is_released_within_the_send_bound(setup, short_bounds):
    """The slow-client case. A `send` that does not complete inside
    SEND_SECONDS must not retain the observer: the bounded send raises,
    the response ends, and the generator is closed -- measured on the
    wall clock, because the promise is about time."""
    _, app, _, _, _, _, _ = setup
    await start(setup)
    stalled = asyncio.Event()

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        if message['type'] == 'http.response.body' and message.get('body'):
            stalled.set()
            await asyncio.sleep(10)   # far past SEND_SECONDS

    began = _time.monotonic()
    with pytest.raises(TimeoutError):
        await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'GET',
                   'scheme': 'http', 'path': '/v1/agent-runs/one/history/stream',
                   'raw_path': b'/v1/agent-runs/one/history/stream', 'query_string': b'',
                   'headers': [(b'authorization', b'Bearer reader')],
                   'client': ('127.0.0.1', 1), 'server': ('scone.test', 80)}, receive, send)
    assert stalled.is_set()
    assert _time.monotonic() - began < 2.0, 'the send bound did not hold'


async def test_cancellation_propagates_and_can_be_repeated(setup, short_window):
    """Cancellation is not swallowed into an error frame -- it reaches the
    caller as CancelledError -- and a second observer after a cancelled
    one starts clean, with no state left behind by the first."""
    _, app, _, _, _, _, calls = setup
    await start(setup)
    for _ in range(2):
        seen = asyncio.Event()

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message['type'] == 'http.response.body' and message.get('body'):
                seen.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(app(
            {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'GET',
             'scheme': 'http', 'path': '/v1/agent-runs/one/history/stream',
             'raw_path': b'/v1/agent-runs/one/history/stream', 'query_string': b'',
             'headers': [(b'authorization', b'Bearer reader')],
             'client': ('127.0.0.1', 1), 'server': ('scone.test', 80)}, receive, send))
        await asyncio.wait_for(seen.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    events = frames(await collect(app))
    assert events[-1][0] == 'end'
    assert len(calls) == 1


async def test_the_observation_window_is_a_wall_clock_bound(setup, short_window):
    """STREAM_SECONDS is a promise about time, not about frame count."""
    _, app, _, _, _, _, _ = setup
    await start(setup)
    began = _time.monotonic()
    events = frames(await collect(app))
    elapsed = _time.monotonic() - began
    assert events[-1] == ('end', None, {'reason': 'observation_window_ended'})
    assert elapsed < 0.1 + 1.0, elapsed


async def test_service_shutdown_mid_stream_ends_with_a_fixed_refusal(setup, short_window):
    """Closing the service while a stream is open must not leak a
    traceback into the body; the observer sees history_unavailable."""
    _, app, service, _, _, _, _ = setup
    await start(setup)

    async def close(frame):
        if b'event: history' in frame:
            await service.aclose()

    events = frames(await collect(app, query='limit=1', on_frame=close))
    kinds = [k for k, _, _ in events]
    assert kinds[0] == 'history' and kinds[-1] in ('error', 'end'), kinds
    assert all(d == {'reason': 'history_unavailable'} for k, _, d in events if k == 'error')
    body = b''.join(m.get('body', b'') for m in await collect(app, query='limit=1'))
    assert b'Traceback' not in body and b'Exception' not in body


async def test_journal_loss_between_pages_is_a_fixed_refusal(setup, short_window):
    """The journal's identity is re-checked before every frame. Replace
    the file underneath and the next frame is a refusal, not private
    exception text."""
    _, app, service, _, _, _, _ = setup
    await start(setup)
    request = service._runs.get('alpha', 'one')
    path = service._path(request)

    async def replace(frame):
        if b'event: history' in frame and path.exists():
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)   # same bytes, new inode

    events = frames(await collect(app, query='limit=1', on_frame=replace))
    kinds = [k for k, _, _ in events]
    assert kinds[0] == 'history' and 'error' in kinds, kinds
    assert events[kinds.index('error')][2] == {'reason': 'history_unavailable'}


async def test_frames_arrive_over_a_real_tcp_socket_before_the_window_ends(setup, short_window, monkeypatch):
    """The proof the ASGI-callback tests cannot give: bytes on a socket,
    delivered incrementally rather than buffered until the end, and a
    server that returns to idle once the client hangs up."""
    import uvicorn
    from scone_memory.api import agent_history

    monkeypatch.setattr(agent_history, 'STREAM_SECONDS', 2.0, raising=False)
    _, app, _, _, _, _, _ = setup
    await start(setup)
    config = uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning', lifespan='off')
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        assert server.started
        port = server.servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        try:
            writer.write(b'GET /v1/agent-runs/one/history/stream?limit=1 HTTP/1.1\r\n'
                         b'Host: 127.0.0.1\r\nAuthorization: Bearer reader\r\n\r\n')
            await writer.drain()
            began = _time.monotonic()
            head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 5)
            assert head.startswith(b'HTTP/1.1 200'), head
            assert b'text/event-stream' in head.lower()
            first = await asyncio.wait_for(reader.readuntil(b'\n\n'), 5)
            # Chunked transfer: the first line is the chunk length in hex,
            # then CRLF, then the frame. Split on the CRLF -- stripping "hex
            # digits" also strips the `e` of `event`.
            frame = first.split(b'\r\n', 1)[1]
            assert frame.startswith(b'event: history'), first
            assert _time.monotonic() - began < 1.5, 'first frame was not delivered incrementally'
        finally:
            # A failed assertion must not leave the client transport to the
            # garbage collector: under `-W error` that warning lands on
            # whichever later test is running when it fires.
            writer.close()
            await writer.wait_closed()
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 10)


async def test_outlasting_the_window_is_the_window_ending_not_a_failure(setup, monkeypatch):
    """Near the end of the observation window a delivery has milliseconds
    left. One that is well inside its verification bound but outlasts
    that remainder has not failed -- the clock ran down -- and the reader
    must be told `observation_window_ended`, never `history_unavailable`.
    Before the two bounds were separated this was reported as a refusal,
    and a repeated-cancellation test flaked on exactly that."""
    import time
    from scone_memory.api import agent_history

    _, app, service, _, _, _, _ = setup
    await start(setup)
    monkeypatch.setattr(agent_history, 'STREAM_SECONDS', 0.03, raising=False)
    monkeypatch.setattr(agent_history, 'POLL_SECONDS', 0.001, raising=False)
    monkeypatch.setattr(agent_history, 'VERIFY_SECONDS', 5.0, raising=False)
    original = service.history_for_delivery
    calls = []

    async def slow_but_sound(*args, **kwargs):
        calls.append(None)
        page = await original(*args, **kwargs)
        if len(calls) == 2:
            time.sleep(0.05)   # outlasts the window, nowhere near the verify bound
        return page

    monkeypatch.setattr(service, 'history_for_delivery', slow_but_sound)
    events = frames(await collect(app))
    assert events[-1] == ('end', None, {'reason': 'observation_window_ended'}), events
    assert not any(kind == 'error' for kind, _, _ in events), events


async def test_a_delivery_still_awaiting_when_the_window_closes_ends_cleanly(setup, monkeypatch):
    """The other way to outlast the window: not a synchronous stall but an
    honest await that the window's timeout can actually interrupt. That
    interruption is the same `TimeoutError` a missed verification bound
    would raise, and it must be read as the window ending -- the delivery
    was well inside its verification bound."""
    from scone_memory.api import agent_history

    _, app, service, _, _, _, _ = setup
    await start(setup)
    monkeypatch.setattr(agent_history, 'STREAM_SECONDS', 0.03, raising=False)
    monkeypatch.setattr(agent_history, 'POLL_SECONDS', 0.001, raising=False)
    monkeypatch.setattr(agent_history, 'VERIFY_SECONDS', 5.0, raising=False)
    original = service.history_for_delivery
    calls = []

    async def awaiting(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            await asyncio.sleep(0.2)   # cancellable, and far past the window
        return await original(*args, **kwargs)

    monkeypatch.setattr(service, 'history_for_delivery', awaiting)
    events = frames(await collect(app))
    assert events[-1] == ('end', None, {'reason': 'observation_window_ended'}), events
    assert not any(kind == 'error' for kind, _, _ in events), events


async def test_a_cursor_survives_a_server_restart_and_resumes_exactly(tmp_path, short_window):
    """The restart case. The cursor is signed against the run and its
    on-disk journal, not against anything held in a process -- so a
    client that reconnects to a freshly built service with the cursor it
    last saw resumes from exactly that position, with nothing repeated
    and nothing skipped, and without any model call."""
    import httpx
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.evidence_loop import ToolStep
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.run_service import AgentRunService
    from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
    from scone_memory.api.app import create_app
    from scone_memory.retrieval.recall_scope import RecallScope

    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(messages)
            return ToolStep(content='Selected model answer')

    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', Model)], agents=[AgentDefinition(
        agent_id='research', instructions='Use evidence.', models=('local',), default_model='local',
        initial_search=False)])

    lives = 0

    async def build():
        nonlocal lives
        lives += 1
        memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
        plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
        if lives == 1:
            plans.save('alpha', AgentTaskPlan(workflow_id='report', tasks=(
                AgentTask(task_id='find', agent_id='research', prompt='Find evidence.'),)),
                catalog=catalog, expected_revision=0)
        service = AgentRunService(tmp_path / 'runs', key=b'k' * 32, catalog=catalog, plans=plans,
                                  memory=memory, scope_for=lambda space: RecallScope.validated(),
                                  max_active=1, max_parallel_tasks=1)
        app = create_app(memory, {'writer': 'alpha', 'reader': 'alpha'}, roles={'writer': 'write', 'reader': 'read'}, agent_catalog=catalog,
                         agent_plan_store=plans, agent_run_service=service)
        return memory, plans, service, app

    # First life: run to completion and stream one page.
    memory, plans, service, app = await build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        assert (await client.post('/v1/agent-runs', json=body(), headers=auth())).status_code == 202
        await service.wait('alpha', 'one')
    first = frames(await collect(app, query='limit=1'))
    cursor, page = next((c, d) for k, c, d in first if k == 'history')
    seen = page['items'][-1]['position']
    await service.aclose(); plans.close(); await memory.close()

    # Second life: nothing in memory survives; only the journal on disk.
    memory, plans, service, app = await build()
    try:
        resumed = await collect(app, query='limit=1', headers=[(b'last-event-id', cursor.encode())])
        assert resumed[0]['status'] == 200, resumed[0]
        positions = [e['position'] for k, _, d in frames(resumed) if k == 'history' for e in d['items']]
        assert positions and positions[0] == seen + 1, positions
        assert positions == sorted(set(positions))
        assert len(calls) == 1, 'a restart must not re-run the model'
    finally:
        await service.aclose(); plans.close(); await memory.close()
