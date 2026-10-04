"""Explicit agent writes retain public sources without implying host capture."""
from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.bearer_keys import KeyHolder
from scone_memory.observability.events import InMemoryEventLog, SqliteEventLog
from scone_memory.runtime.mcp import create_server, http_app
from scone_memory.testing import Clock

from .test_mcp import call
from .test_mcp_http import PROTOCOL, Session, opened, served


@pytest.fixture(params=['memory', 'sqlite'])
async def agent_memory(request, tmp_path):
    log = InMemoryEventLog() if request.param == 'memory' else SqliteEventLog(tmp_path / 'agent-events.db')
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock(), events=log).open()
    yield engine, create_server(engine, 'alpha')
    await engine.close()


async def receipt(server, tool, **arguments):
    error, text = await call(server, tool, **arguments)
    assert not error, text
    return json.loads(text)


async def test_arbitrary_harness_registration_is_observation_not_configuration(agent_memory):
    engine, server = agent_memory
    assert {'memory_agent_connect', 'memory_record_interaction'} <= {tool.name for tool in await server.list_tools()}
    registered = await receipt(server, 'memory_agent_connect', session_id='session-a', project='orchard', agent_name='Research harness')
    assert registered['status'] == 'registered'
    assert registered['agent'] == 'other'
    assert registered['identity_basis'] == 'self_declared'
    assert registered['host_capture_configured'] is False
    source = await engine.episode('alpha', registered['episode_id'])
    assert source.metadata['agent_name'] == 'Research harness'
    assert source.metadata['project'] == 'orchard'
    event = await engine.events.get('alpha', registered['event_id'])
    assert event.payload['event'] == 'session_start'
    assert event.payload['episode_id'] == source.episode_id
    assert 'client_name' not in source.metadata
    replay = await receipt(server, 'memory_agent_connect', session_id='session-a', project='orchard', agent_name='Research harness')
    assert (replay['episode_id'], replay['event_id']) == (registered['episode_id'], registered['event_id'])


async def test_explicit_public_interactions_are_linked_redacted_and_idempotent(agent_memory):
    engine, server = agent_memory
    arguments = dict(session_id='session-a', request_id='turn-1', role='user', public_text='Keep the plan. api_key=abcdefghijklmnop1234', project='orchard', agent_name='codex')
    first = await receipt(server, 'memory_record_interaction', **arguments)
    replay = await receipt(server, 'memory_record_interaction', **arguments)
    assert (first['episode_id'], first['event_id']) == (replay['episode_id'], replay['event_id'])
    source = await engine.episode('alpha', first['episode_id'])
    assert source.content == 'Keep the plan. api_key=[redacted]'
    assert source.kind == 'conversation' and source.metadata['role'] == 'user'
    event = await engine.events.get('alpha', first['event_id'])
    assert event.payload['agent'] == 'codex' and event.payload['event'] == 'prompt'
    assert event.payload['episode_id'] == source.episode_id
    assert 'text' not in event.payload
    assert 'abcdefghijklmnop' not in json.dumps(first)
    reply = await receipt(server, 'memory_record_interaction', **{**arguments, 'role': 'assistant', 'public_text': 'The plan is retained.'})
    assert reply['episode_id'] != first['episode_id']
    assert (await engine.events.get('alpha', reply['event_id'])).payload['event'] == 'response'


async def test_changed_replay_conflicts_without_creating_another_source(agent_memory):
    engine, server = agent_memory
    arguments = dict(session_id='session-a', request_id='request-a', role='assistant', public_text='First public answer')
    await receipt(server, 'memory_record_interaction', **arguments)
    before = await engine.documents.recent_episodes('alpha', 20)
    for change in [{'public_text': 'Changed answer'}, {'project': 'different'}, {'agent_name': 'different'}]:
        error, text = await call(server, 'memory_record_interaction', **{**arguments, **change})
        assert error and 'different' in text
    assert await engine.documents.recent_episodes('alpha', 20) == before


async def test_failed_event_write_can_retry_the_already_retained_source(agent_memory, monkeypatch):
    engine, server = agent_memory
    original = engine.record
    calls = 0

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('event sink unavailable')
        return await original(*args, **kwargs)

    monkeypatch.setattr(engine, 'record', fail_once)
    arguments = dict(session_id='session-a', request_id='request-a', role='assistant', public_text='Explicit public answer')
    error, text = await call(server, 'memory_record_interaction', **arguments)
    assert error and 'retry' in text.lower()
    retry = await receipt(server, 'memory_record_interaction', **arguments)
    assert len(await engine.documents.recent_episodes('alpha', 20)) == 1
    assert (await engine.events.get('alpha', retry['event_id'])).payload['episode_id'] == retry['episode_id']


async def test_concurrent_identical_requests_share_source_and_receipt(agent_memory):
    engine, server = agent_memory
    arguments = dict(session_id='session-a', request_id='request-a', role='user', public_text='One repeated question')
    results = await asyncio.gather(*(receipt(server, 'memory_record_interaction', **arguments) for _ in range(4)))
    assert len({(r['episode_id'], r['event_id']) for r in results}) == 1
    assert len(await engine.documents.recent_episodes('alpha', 20)) == 1


async def test_cancelled_event_write_can_retry_without_duplicate_source(agent_memory, monkeypatch):
    engine, server = agent_memory
    original = engine.record
    entered = asyncio.Event()

    async def interrupted(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(engine, 'record', interrupted)
    arguments = dict(session_id='session-a', request_id='request-a', role='user', public_text='A cancellable public question')
    task = asyncio.create_task(call(server, 'memory_record_interaction', **arguments))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(engine, 'record', original)
    result = await receipt(server, 'memory_record_interaction', **arguments)
    assert len(await engine.documents.recent_episodes('alpha', 20)) == 1
    assert (await engine.events.get('alpha', result['event_id'])).payload['episode_id'] == result['episode_id']


async def test_read_only_and_cross_space_calls_cannot_write(agent_memory):
    engine, _ = agent_memory
    for role in ['read', 'review']:
        server = create_server(engine, 'alpha', holder=KeyHolder('alpha', role))
        for name, arguments in [('memory_agent_connect', dict(session_id='session-a')), ('memory_record_interaction', dict(session_id='session-a', request_id='a', role='user', public_text='Not retained'))]:
            error, text = await call(server, name, **arguments)
            assert error and 'cannot write' in text
    server = create_server(engine, 'alpha', holder=KeyHolder('alpha', 'write'))
    error, text = await call(server, 'memory_agent_connect', session_id='session-a', space='beta')
    assert error and 'space' in text
    assert await engine.documents.recent_episodes('alpha', 20) == []
    assert await engine.documents.recent_episodes('beta', 20) == []


async def test_forgotten_interaction_is_not_recreated_by_replay(agent_memory):
    engine, server = agent_memory
    arguments = dict(session_id='session-a', request_id='request-a', role='user', public_text='Forget this public text')
    first = await receipt(server, 'memory_record_interaction', **arguments)
    await engine.forget('alpha', first['episode_id'])
    error, text = await call(server, 'memory_record_interaction', **arguments)
    assert error and 'forgotten' in text
    assert await engine.documents.recent_episodes('alpha', 20) == []


async def test_missing_event_log_refuses_before_source_retention():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        error, text = await call(create_server(engine), 'memory_agent_connect', session_id='session-a')
        assert error and 'event log' in text
        assert await engine.documents.recent_episodes('default', 20) == []
    finally:
        await engine.close()


async def test_input_bounds_are_checked_before_embedding(agent_memory, monkeypatch):
    engine, server = agent_memory

    async def forbidden(*args, **kwargs):
        pytest.fail('invalid arguments must not reach the embedder')

    monkeypatch.setattr(engine.embedder, 'embed', forbidden)
    cases = [dict(session_id=''), dict(session_id=' x '), dict(session_id='x\nforged'), dict(session_id='x' * 129), dict(session_id='s', project='😀' * 121), dict(session_id='s', agent_name='x' * 121)]
    for arguments in cases:
        error, _ = await call(server, 'memory_agent_connect', **arguments)
        assert error
    for text in ['', '  ', '😀' * 15001]:
        error, _ = await call(server, 'memory_record_interaction', session_id='s', request_id='r', role='user', public_text=text)
        assert error
    assert await engine.documents.recent_episodes('alpha', 20) == []


async def test_sdk_client_info_is_annotation_not_authority(agent_memory):
    engine, _ = agent_memory
    async with served(http_app(engine, 'default', {'alpha-key': 'alpha'})) as client:
        session = await opened(client, 'alpha-key')
        error, text = await session.call('memory_agent_connect', session_id='sdk-session')
        assert not error, text
        value = json.loads(text)
        source = await engine.episode('alpha', value['episode_id'])
        assert source.metadata['client_name'] == 'test'
        assert source.metadata['client_version'] == '0'
        assert source.metadata['identity_basis'] == 'self_declared'
        assert value['space'] == 'alpha' and value['agent'] == 'other'


async def test_untrusted_sdk_client_info_is_bounded_and_does_not_select_space(agent_memory):
    engine, _ = agent_memory
    async with served(http_app(engine, 'default', {'alpha-key': 'alpha'})) as client:
        session = Session(client, 'alpha-key')
        response = await session.post({'jsonrpc': '2.0', 'id': 0, 'method': 'initialize', 'params': {
            'protocolVersion': PROTOCOL, 'capabilities': {},
            'clientInfo': {'name': 'beta\nforged-admin', 'version': 'x' * 65}}})
        assert response.status_code == 200
        session.session_id = response.headers['mcp-session-id']
        assert (await session.post({'jsonrpc': '2.0', 'method': 'notifications/initialized'})).status_code == 202
        error, text = await session.call('memory_agent_connect', session_id='sdk-untrusted')
        assert not error, text
        value = json.loads(text)
        source = await engine.episode('alpha', value['episode_id'])
        assert 'client_name' not in source.metadata and 'client_version' not in source.metadata
        assert value['space'] == 'alpha' and value['agent'] == 'other'
        assert await engine.documents.recent_episodes('beta', 20) == []
