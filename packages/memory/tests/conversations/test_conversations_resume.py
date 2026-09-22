"""Explicit resume restores retained completed turns without replaying work."""
import asyncio
import copy

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from scone_memory.realtime.text import TextConversation
from ..api.test_conversations_api import client_for


async def finished(client, url, request):
    async with asyncio.timeout(3):
        while True:
            receipt = (await client.get(url + '/turns/' + request)).json()
            if receipt['status'] != 'pending':
                return receipt
            await asyncio.sleep(.005)


@pytest.mark.parametrize('ending', ['restart', 'stop', 'failure'])
async def test_resume_keeps_identity_scope_and_history_without_replaying_old_turns(tmp_path, ending):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    requests = []
    class Model:
        async def respond(self, messages):
            requests.append(copy.deepcopy(messages))
            if messages[-1]['content'] == 'Fail this reply':
                raise RuntimeError('provider failed')
            yield TextDelta('Recorded answer.')
            yield ReplyCompleted()
        async def aclose(self): pass
    def app():
        return create_conversation_app(memory, {'alpha-key': 'alpha', 'beta-key': 'beta'}, tmp_path / 'sessions.db', None, text_resumption=True,
            scoped_runtime_factory=lambda space, sid, scope, **kwargs: TextConversation(memory, space, sid, Model, **scope.kwargs(), **kwargs))
    scope = {'source_prefix': 'manuals/'}
    try:
        async with client_for(app()) as client:
            started = (await client.post('/v1/conversations', json={'request_id': 'create', 'capture': True, 'recall_scope': scope})).json()
            sid, revision = started['session_id'], started['revision']
            url = '/v1/conversations/' + sid
            original = {'request_id': 'first', 'text': 'My chosen codename is Juniper.', 'expected_revision': revision}
            assert (await client.post(url + '/turns', json=original)).status_code == 202
            assert (await finished(client, url, 'first'))['status'] == 'completed'
            if ending == 'failure':
                await client.post(url + '/turns', json={'request_id': 'failed', 'text': 'Fail this reply', 'expected_revision': revision})
                assert (await finished(client, url, 'failed'))['status'] == 'failed'
            if ending == 'stop':
                assert (await client.post(url + '/stop', json={'request_id': 'stop', 'expected_revision': revision})).status_code == 200
        async with client_for(app()) as client:
            prior = (await client.get(url)).json()
            count = len(requests)
            body = {'request_id': 'resume', 'expected_revision': prior['revision']}
            resumed = await client.post(url + '/resume', json=body)
            assert resumed.status_code == 200, resumed.text
            current = resumed.json()
            assert current['session_id'] == sid and current['state'] == 'running'
            assert current['recall_scope'] == scope and current['revision'] == prior['revision'] + 1
            assert len(requests) == count
            assert (await client.post(url + '/resume', json=body)).json()['revision'] == current['revision']
            assert (await client.post(url + '/resume', json=body, headers={'Authorization': 'Bearer beta-key'})).status_code == 404
            assert (await client.post(url + '/turns', json=original)).json()['status'] == 'completed'
            assert len(requests) == count
            await client.post(url + '/turns', json={'request_id': 'next', 'text': 'What codename did I choose?', 'expected_revision': current['revision']})
            assert (await finished(client, url, 'next'))['status'] == 'completed'
            assert {'role': 'user', 'content': original['text']} in requests[-1]
            assert {'role': 'assistant', 'content': 'Recorded answer.'} in requests[-1]
            assert all(m['content'] != 'Fail this reply' for m in requests[-1])
    finally:
        await memory.close()


@pytest.mark.parametrize('forgotten_role', ['user', 'assistant', None])
async def test_restore_excludes_forgotten_pairs_and_keeps_newest_whole_turns(forgotten_role):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    class Model:
        async def respond(self, messages):
            yield TextDelta('Answer ' + messages[-1]['content'])
            yield ReplyCompleted()
        async def aclose(self): pass
    runtime = TextConversation(memory, 'alpha', 'restore', Model)
    try:
        replies = [await runtime.reply(f'Message {n} ' + 'x' * 150) for n in range(3)]
        if forgotten_role:
            await memory.forget('alpha', replies[-1][forgotten_role + '_episode_id'])
        restored = TextConversation(memory, 'alpha', 'restore', Model, system_prompt='Use this history.', max_history_bytes=2400)
        receipt = await restored.restore(tuple(reply['assistant_episode_id'] for reply in replies))
        assert 0 < receipt['restored_turns'] < 3
        contents = [m['content'] for m in restored._history]
        assert not forgotten_role or all('Message 2' not in text for text in contents)
        roles = [m['role'] for m in restored._history]
        assert roles == ['system'] + ['user', 'assistant'] * receipt['restored_turns']
        await restored.reply('Continue.')
        await restored.close()
    finally:
        await runtime.close()
        await memory.close()


async def test_same_process_resume_is_serialized_and_deleted_history_cannot_reappear(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    entered, release = asyncio.Event(), asyncio.Event()
    runtimes = []
    class Model:
        async def respond(self, messages):
            yield TextDelta('Done.')
            yield ReplyCompleted()
        async def aclose(self): pass
    class Restoring(TextConversation):
        async def restore(self, ids):
            entered.set()
            await release.wait()
            return await super().restore(ids)
    def factory(space, sid):
        runtime = Restoring(memory, space, sid, Model)
        runtimes.append(runtime)
        return runtime
    app = create_conversation_app(memory, {'alpha-key': 'alpha'}, tmp_path / 'sessions.db', factory, text_resumption=True)
    try:
        async with client_for(app) as client:
            current = (await client.post('/v1/conversations', json={'request_id': 'new', 'capture': True})).json()
            url = '/v1/conversations/' + current['session_id']
            current = (await client.post(url + '/stop', json={'request_id': 'stop', 'expected_revision': current['revision']})).json()
            body = {'request_id': 'resume', 'expected_revision': current['revision']}
            pending = asyncio.create_task(client.post(url + '/resume', json=body))
            await entered.wait()
            assert (await client.post(url + '/resume', json={**body, 'request_id': 'other'})).status_code == 409
            # Deletion wins while restoration is waiting; resume must not recreate it.
            assert (await client.delete(url)).status_code == 204
            release.set()
            assert (await pending).status_code == 404
            assert runtimes[-1].closed
            assert (await client.get(url)).status_code == 404
    finally:
        await memory.close()


async def test_failed_turn_can_resume_in_same_process_and_stale_commands_are_rejected(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    requests = []
    class Model:
        async def respond(self, messages):
            requests.append(messages)
            if len(requests) == 1:
                raise RuntimeError('failed provider')
            yield TextDelta('Recovered.')
            yield ReplyCompleted()
        async def aclose(self): pass
    app = create_conversation_app(memory, {'alpha-key': 'alpha'}, tmp_path / 'sessions.db',
        lambda space, sid: TextConversation(memory, space, sid, Model), text_resumption=True)
    try:
        async with client_for(app) as client:
            current = (await client.post('/v1/conversations', json={'request_id': 'new', 'capture': True})).json()
            url = '/v1/conversations/' + current['session_id']
            original = {'request_id': 'failed', 'text': 'Fail.', 'expected_revision': current['revision']}
            await client.post(url + '/turns', json=original)
            assert (await finished(client, url, 'failed'))['status'] == 'failed'
            assert (await client.post(url + '/resume', json={'request_id': 'resume', 'expected_revision': current['revision']})).status_code == 409
            closed = (await client.get(url)).json()
            body = {'request_id': 'resume', 'expected_revision': closed['revision']}
            current = (await client.post(url + '/resume', json=body)).json()
            assert current['state'] == 'running'
            assert (await client.post(url + '/turns', json=original)).json()['status'] == 'failed'
            assert len(requests) == 1
            await client.post(url + '/turns', json={'request_id': 'new-turn', 'text': 'Continue.', 'expected_revision': current['revision']})
            assert (await finished(client, url, 'new-turn'))['status'] == 'completed'
            assert len(requests) == 2
            stopped = (await client.post(url + '/stop', json={'request_id': 'stop', 'expected_revision': current['revision']})).json()
            assert (await client.post(url + '/resume', json=body)).json()['state'] == stopped['state'] == 'ended'
    finally:
        await memory.close()


@pytest.mark.parametrize('mismatch', ['missing', 'changed', 'unrecorded'])
async def test_resume_refuses_unavailable_or_changed_saved_persona(tmp_path, mismatch):
    from scone_memory.realtime.catalog import bind_catalog
    from scone_memory.realtime.persona import Persona
    from scone_memory.realtime.session_journal import SessionJournal
    from .test_persona_catalog import persona, registry
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    catalog = bind_catalog([Persona.model_validate(persona())], registry())
    path = tmp_path / 'sessions.db'
    with SessionJournal(path) as journal:
        saved = journal.create('alpha', 'create', persona='missing' if mismatch == 'missing' else 'helper',
            persona_fingerprint=None if mismatch == 'unrecorded' else '0' * 16)
        journal.transition('alpha', saved['session_id'], 'stop', 'stop', saved['revision'])
    app = create_conversation_app(memory, {'alpha-key': 'alpha'}, path, None, catalog=catalog, text_resumption=True)
    try:
        async with client_for(app) as client:
            url = '/v1/conversations/' + saved['session_id']
            current = (await client.get(url)).json()
            response = await client.post(url + '/resume', json={'request_id': 'resume', 'expected_revision': current['revision']})
            assert response.status_code == 409
            assert (await client.get(url)).json()['state'] == 'ended'
    finally:
        await memory.close()


def test_journal_resume_refuses_voice_and_unsettled_turns(tmp_path):
    from scone_memory.realtime.session_journal import SessionJournal
    from scone_memory.core.errors import Conflict
    with SessionJournal(tmp_path / 'sessions.db') as journal:
        for mode in ['voice', 'text']:
            saved = journal.create('alpha', mode, mode)
            sid = saved['session_id']
            if mode == 'text':
                journal.start_turn('alpha', sid, 'unfinished', {'text': 'accepted'})
            closed = journal.transition('alpha', sid, 'close', 'interrupt', saved['revision'])
            with pytest.raises(Conflict):
                journal.transition('alpha', sid, 'resume', 'resume', closed['revision'])
