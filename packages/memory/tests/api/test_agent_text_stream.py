"""A step's answer, delivered to an authorized reader as it is written.

Metadata streams over `/history/stream`; this is the text. The route reads
the process-local window the run service holds for a running step and
publishes it the way conversations publish a reply: `text` frames with
their sequence as the SSE id, `withdraw` when text streamed before a tool
turn was not the answer, `gap` when a reader fell behind the window,
`terminal` once the run has a receipt, and `end` when the window is gone.
Every publication is to the current recipient of the current run, and
what streams is provisional: the receipt is what `/result` returns.
"""

import asyncio
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolCall, ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope

pytestmark = pytest.mark.asyncio

KEY = b'k' * 32
PATH = '/v1/agent-runs/one/steps/find/text/stream'


def auth(key='writer'):
    return {'Authorization': 'Bearer ' + key}


def body(run_id='one'):
    return {'run_id': run_id, 'workflow_id': 'report', 'plan_revision': 1, 'question': 'Question'}


class Streaming:
    def __init__(self, turns, entered, release, hold_turn=0, hold_index=0):
        self.turns, self.entered, self.release = list(turns), entered, release
        self.hold_turn, self.hold_index = hold_turn, hold_index
        self.turn = 0

    async def complete(self, messages, tools, *, on_public_text=None):
        pieces, step = self.turns.pop(0)
        for index, piece in enumerate(pieces):
            if on_public_text is not None:
                await on_public_text(piece)
            if self.turn == self.hold_turn and index == self.hold_index:
                self.entered.set()
                await self.release.wait()
        self.turn += 1
        return step


async def build(tmp_path, model_factory, *, public_text=True):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', model_factory)], agents=[AgentDefinition(
        agent_id='research', instructions='Use evidence.', models=('local',), default_model='local', initial_search=False)])
    plans = AgentPlanStore(tmp_path / 'plans.db', key=KEY)
    plans.save('alpha', AgentTaskPlan(workflow_id='report', tasks=(AgentTask(task_id='find', agent_id='research', prompt='Find evidence.'),)),
               catalog=catalog, expected_revision=0)
    service = AgentRunService(tmp_path / 'runs', key=KEY, catalog=catalog, plans=plans, memory=memory,
                              scope_for=lambda space: RecallScope.validated(), max_active=1, public_text=public_text)
    keys = {'writer': 'alpha', 'reader': 'alpha', 'other': 'bravo'}
    app = create_app(memory, keys, roles={'writer': 'write', 'reader': 'read'},
                     agent_catalog=catalog, agent_plan_store=plans, agent_run_service=service)
    return app, service, plans, memory, keys


@pytest.fixture
async def setup(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    model = Streaming([(['Answer ', 'from model.'], ToolStep(content='Answer from model.'))], entered, release)
    app, service, plans, memory, keys = await build(tmp_path, lambda: model)
    try:
        yield app, service, entered, release, keys, tmp_path, plans, memory
    finally:
        release.set()
        await service.aclose()
        plans.close()
        await memory.close()


async def collect(app, *, path=PATH, query='', token='reader', on_start=None, headers=()):
    """Drive the app over ASGI and keep every message it sent."""
    messages = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.start' and on_start is not None:
            await on_start()

    await app({'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'}, 'http_version': '1.1',
               'method': 'GET', 'scheme': 'http', 'path': path, 'raw_path': path.encode(), 'query_string': query.encode(),
               'headers': [(b'authorization', ('Bearer ' + token).encode()), *headers],
               'client': ('127.0.0.1', 1234), 'server': ('scone.test', 80)}, receive, send)
    return messages


def status_of(messages):
    return next(message['status'] for message in messages if message['type'] == 'http.response.start')


def frames(messages):
    raw = b''.join(message.get('body', b'') for message in messages)
    result = []
    for frame in raw.decode().split('\n\n'):
        values = dict(line.split(': ', 1) for line in frame.splitlines() if ': ' in line)
        if 'data' in values:
            result.append((values.get('event'), values.get('id'), json.loads(values['data'])))
    return result


@pytest.fixture
def short_window(monkeypatch):
    from scone_memory.api import agent_text
    monkeypatch.setattr(agent_text, 'STREAM_SECONDS', 0.3, raising=False)


async def started(app, service, entered):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        response = await client.post('/v1/agent-runs', json=body(), headers=auth())
        assert response.status_code == 202, response.text
    await entered.wait()


async def test_text_frames_arrive_in_sequence_and_a_terminal_follows_the_receipt(setup):
    app, service, entered, release, *_ = setup
    await started(app, service, entered)

    async def let_it_finish():
        await asyncio.sleep(0.05)
        release.set()

    finishing = asyncio.create_task(let_it_finish())
    messages = await collect(app)
    await finishing
    assert status_of(messages) == 200
    events = frames(messages)
    assert [(event, identifier) for event, identifier, _ in events][:2] == [('text', '1'), ('text', '2')]
    assert ''.join(data['text'] for event, _, data in events if event == 'text') == 'Answer from model.'
    assert events[-1][0] == 'terminal' and events[-1][2] == {'status': 'completed', 'read_receipt': True}
    start = next(message for message in messages if message['type'] == 'http.response.start')
    assert dict(start['headers'])[b'content-type'].startswith(b'text/event-stream')


async def test_a_reader_resumes_after_its_cursor_and_the_cursor_must_agree_with_last_event_id(setup):
    app, service, entered, release, *_ = setup
    await started(app, service, entered)
    assert status_of(await collect(app, query='after=1', headers=[(b'last-event-id', b'2')])) == 422
    assert status_of(await collect(app, query='after=-1')) == 422
    assert status_of(await collect(app, query='after=9')) == 409, 'a cursor ahead of what was observed'

    async def let_it_finish():
        await asyncio.sleep(0.05)
        release.set()

    finishing = asyncio.create_task(let_it_finish())
    events = frames(await collect(app, headers=[(b'last-event-id', b'1')]))
    await finishing
    assert [(event, data.get('text')) for event, _, data in events if event == 'text'] == [('text', 'from model.')]


async def test_the_stream_is_scoped_to_the_run_the_space_and_the_recipient(setup):
    app, service, entered, *_ = setup
    await started(app, service, entered)
    assert status_of(await collect(app, path='/v1/agent-runs/two/steps/find/text/stream')) == 404
    assert status_of(await collect(app, path='/v1/agent-runs/one/steps/other/text/stream')) == 404
    assert status_of(await collect(app, token='other')) == 404
    assert status_of(await collect(app, token='nobody')) == 401


async def test_a_host_without_public_text_says_so_and_advertises_nothing(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    release.set()
    model = Streaming([(['x'], ToolStep(content='x'))], entered, release)
    app, service, plans, memory, _ = await build(tmp_path, lambda: model, public_text=False)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
            caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
            assert caps['agents.text_stream'] is False
            assert (await client.post('/v1/agent-runs', json=body(), headers=auth())).status_code == 202
        await service.wait('alpha', 'one')
        assert status_of(await collect(app)) == 501
    finally:
        await service.aclose(); plans.close(); await memory.close()


async def test_the_capability_is_advertised_when_the_host_asked_for_public_text(setup):
    app, *_ = setup
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test') as client:
        assert (await client.get('/v1/capabilities', headers=auth())).json()['features']['agents.text_stream'] is True


async def test_text_withdrawn_before_a_tool_turn_is_a_withdraw_frame(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    search = ToolStep(calls=(ToolCall(id='s1', name='search_memory', arguments={'query': 'decision'}),))
    model = Streaming([(['Looking that up.'], search), (['We decided.'], ToolStep(content='We decided.'))], entered, release, hold_turn=1)
    app, service, plans, memory, _ = await build(tmp_path, lambda: model)
    try:
        await started(app, service, entered)

        async def let_it_finish():
            await asyncio.sleep(0.05)
            release.set()

        finishing = asyncio.create_task(let_it_finish())
        events = frames(await collect(app))
        await finishing
        assert [(event, identifier) for event, identifier, _ in events][:3] == [('text', '1'), ('withdraw', '2'), ('text', '3')]
        assert events[1][2] == {'sequence': 2}
        assert events[2][2]['text'] == 'We decided.'
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()


async def test_a_window_that_did_not_survive_the_process_ends_the_stream_and_points_at_the_receipt(setup):
    app, service, entered, release, keys, tmp_path, plans, memory = setup
    await started(app, service, entered)
    release.set()
    await service.wait('alpha', 'one')
    await service.aclose()
    reopened = AgentRunService(tmp_path / 'runs', key=KEY, catalog=service._catalog, plans=plans, memory=memory,
                               scope_for=lambda space: RecallScope.validated(), public_text=True)
    app2 = create_app(memory, keys, roles={'writer': 'write', 'reader': 'read'},
                      agent_catalog=service._catalog, agent_plan_store=plans, agent_run_service=reopened)
    try:
        events = frames(await collect(app2))
        assert events == [('terminal', None, {'status': 'completed', 'read_receipt': True})], \
            'a completed run has a receipt; the provisional text is gone with the process'
    finally:
        await reopened.aclose()


async def test_a_key_revoked_after_the_headers_delivers_no_text(setup, short_window):
    app, service, entered, release, keys, *_ = setup
    await started(app, service, entered)

    async def revoke():
        app.state.keys.pop('reader')
        await asyncio.sleep(0.05)
        release.set()

    events = frames(await collect(app, on_start=revoke))
    assert not [event for event in events if event[0] == 'text'], events
    assert events and events[-1][0] == 'error' and events[-1][2] == {'reason': 'text_unavailable'}


async def test_the_observation_window_ends_by_the_clock_while_the_step_still_runs(setup, short_window):
    app, service, entered, *_ = setup
    await started(app, service, entered)
    events = frames(await collect(app))
    assert events[-1] == ('end', None, {'reason': 'observation_window_ended'})
    assert [event for event, _, _ in events if event == 'text'] == ['text'], 'the one delta written so far'


async def test_a_reader_that_fell_behind_the_window_is_told_the_gap_not_handed_stale_text(tmp_path):
    """The window keeps a bounded tail. A reader asking from before it gets
    a gap naming the next sequence it can read, then that text."""
    from scone_memory.api import text_stream

    entered, release = asyncio.Event(), asyncio.Event()
    pieces = [f'w{index} ' for index in range(text_stream.MAX_CHUNKS + 40)]
    model = Streaming([(pieces, ToolStep(content=''.join(pieces)))], entered, release, hold_index=len(pieces) - 1)
    app, service, plans, memory, _ = await build(tmp_path, lambda: model)
    try:
        await started(app, service, entered)

        async def let_it_finish():
            await asyncio.sleep(0.05)
            release.set()

        finishing = asyncio.create_task(let_it_finish())
        events = frames(await collect(app))
        await finishing
        first = events[0]
        assert first[0] == 'gap' and first[2]['after'] == 0 and first[2]['next_sequence'] == len(pieces) - text_stream.MAX_CHUNKS + 1
        texts = [data['sequence'] for event, _, data in events if event == 'text']
        assert texts[0] == first[2]['next_sequence'] and texts == list(range(texts[0], len(pieces) + 1))
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()
