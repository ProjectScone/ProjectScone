"""Operator-selected tool inference reaches the composed text service."""
import asyncio
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings
from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore

AUTH = {'Authorization':'Bearer fixture-key'}


def environment(tmp_path, mode='structured'):
    return {'SCONE_API_KEY':'fixture-key', 'SCONE_MODEL_CONNECTIONS':str(tmp_path / 'models.json'),
            'SCONE_CONVERSATIONS_JOURNAL':str(tmp_path / 'sessions.db'),
            'SCONE_CONVERSATIONS_TOOL_MODE':mode}


@pytest.mark.parametrize('changes', [
    {'SCONE_CONVERSATIONS_TOOL_MODE':'auto'}, {'SCONE_CONVERSATIONS_TOOL_MODE':''},
    {'SCONE_CONVERSATIONS_JOURNAL':''}, {'SCONE_MODEL_CONNECTIONS':''},
    {'SCONE_CONVERSATIONS_MODEL_FACTORY':'somewhere:create'},
    {'SCONE_CONVERSATIONS_PERSONAS':'personas.json', 'SCONE_CONVERSATIONS_REGISTRY':'registry:create'},
    {'SCONE_ADAPTIVE_RETRIEVAL':'1', 'SCONE_ADAPTIVE_URL':'http://127.0.0.1:11434/v1', 'SCONE_ADAPTIVE_MODEL':'judge'},
    {'SCONE_ANSWER_REVIEW_POLICY':'report', 'SCONE_ANSWER_REVIEW_URL':'http://127.0.0.1:11434/v1', 'SCONE_ANSWER_REVIEW_MODEL':'judge'},
    {'SCONE_CONVERSATIONS_TOOL_MAX_CALLS':'0'}, {'SCONE_CONVERSATIONS_TOOL_MAX_ROUNDS':'17'},
    {'SCONE_CONVERSATIONS_TOOL_TIMEOUT':'nan'}, {'SCONE_CONVERSATIONS_TOOL_TIMEOUT':'601'},
    {'SCONE_CONVERSATIONS_TOOL_MODE':'off', 'SCONE_CONVERSATIONS_TOOL_MAX_CALLS':'2'},
])
def test_incompatible_tool_settings_fail_at_startup(tmp_path, changes):
    with pytest.raises(InvalidInput):
        Settings.from_env(environment(tmp_path) | changes)


def test_tools_are_off_by_default_and_limits_are_explicit(tmp_path):
    assert Settings.from_env({}).conversations_tool_mode == 'off'
    settings = Settings.from_env(environment(tmp_path) | {'SCONE_CONVERSATIONS_TOOL_MAX_CALLS':'3',
        'SCONE_CONVERSATIONS_TOOL_MAX_ROUNDS':'2', 'SCONE_CONVERSATIONS_TOOL_TIMEOUT':'45'})
    from scone_memory.runtime.conversation_tools import build_conversation_tools
    configured = build_conversation_tools(settings)
    assert configured.mode == 'structured'
    assert configured.limits.max_tool_calls == 3 and configured.limits.max_tool_rounds == 2
    assert configured.limits.timeout_s == 45
    assert build_conversation_tools(Settings.from_env({})) is None


@pytest.mark.parametrize('mode', ['native', 'structured'])
@pytest.mark.parametrize('persona', [False, True])
@pytest.mark.parametrize('outcome', ['success', 'provider_failure', 'deleted_source'])
async def test_composed_tool_reply_preserves_scope_and_receipts(tmp_path, monkeypatch, mode, persona, outcome):
    from scone_memory.agents.evidence_loop import ToolCall, ToolStep
    from scone_memory.providers import structured_tool_chat, tool_chat

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    source = await engine.remember('default', 'Juniper uses Polaris.', metadata={'team':'blue'})
    await engine.remember('default', 'PRIVATE_OTHER_TEAM', metadata={'team':'red'})
    await engine.remember('foreign', 'PRIVATE_OTHER_SPACE', metadata={'team':'blue'})
    model_instances, requests = [], []

    class Model:
        def __init__(self, *args, **kwargs):
            model_instances.append((args, kwargs))
            self.steps = 0

        async def complete(self, messages, tools):
            requests.append(messages)
            self.steps += 1
            if self.steps == 1:
                return ToolStep(calls=(ToolCall(id='find', name='search_memory',
                    arguments={'query':'Juniper', 'limit':1}),))
            assert 'PRIVATE_' not in json.dumps(messages)
            if outcome == 'provider_failure':
                raise RuntimeError('PRIVATE_PROVIDER_SECRET')
            if outcome == 'deleted_source':
                await engine.forget('default', source.episode_id)
            return ToolStep(content='Juniper uses Polaris.')

    provider = structured_tool_chat if mode == 'structured' else tool_chat
    monkeypatch.setattr(provider, 'SelfHostedStructuredToolChat' if mode == 'structured' else 'SelfHostedToolChat', Model)
    env = environment(tmp_path, mode) | {'SCONE_CHAT_THINK':'false'}
    chat = ModelConnection(base_url='http://127.0.0.1:11434/v1', model='installed:model', timeout_s=37)
    connections = {'chat':chat}
    if persona:
        connections.update(transcription=chat, speech=chat.model_copy(update={'voice':'voice', 'sample_rate':24000}))
    store = ModelConnectionStore(tmp_path / 'models.json', {})
    for revision, (role, configured) in enumerate(connections.items()):
        store.replace(role, configured, expected_revision=revision)
    app = build_app(Settings.from_env(env), engine)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1', headers=AUTH) as client:
                caps = (await client.get('/v1/conversations/capabilities')).json()
                assert caps['tool_retrieval']['protocol'] == mode
                assert caps['tool_retrieval']['configured'] is True
                assert caps['tool_retrieval']['available'] is True
                payload = {'request_id':'session', 'capture':True, 'recall_scope':{'where':{'team':'blue'}}}
                if persona:
                    payload['persona'] = 'self-hosted-voice'
                created = await client.post('/v1/conversations', json=payload)
                assert created.status_code == 200, created.text
                session = created.json()
                url = '/v1/conversations/' + session['session_id']
                response = await client.post(url + '/turns', json={'request_id':'turn', 'text':'What does Juniper use?',
                    'expected_revision':session['revision']})
                assert response.status_code in (200, 202), response.text
                async with asyncio.timeout(5):
                    while response.json()['status'] == 'pending':
                        await asyncio.sleep(0)
                        response = await client.get(url + '/turns/turn')
                result = response.json()
                if outcome != 'success':
                    assert result['status'] == 'failed', result
                    assert 'PRIVATE_' not in response.text
                    saved = await engine.episodes('default', {'session_id':session['session_id']})
                    assert [row.metadata['role'] for row in saved] == ['user']
                    assert (await client.get('/v1/capabilities')).status_code == 200
                    assert (await client.get('/v1/conversations/capabilities')).json()['tool_retrieval']['available'] is True
                    return
                assert result['status'] == 'completed', result
                context = result['result']['memory_context']
                assert result['result']['text'] == 'Juniper uses Polaris.'
                assert context['tool_retrieval']['source_status'] == 'retained'
                assert context['tool_retrieval']['tool_calls'] == 1
                assert context['evidence_graph_status'] == 'prepared'
                assert len(model_instances) == 1 and len(requests) == 2
                assert model_instances[0][0] == (chat.base_url, chat.model)
                assert model_instances[0][1]['timeout_s'] == 37
                assert model_instances[0][1]['think'] is False
                saved = await engine.episodes('default', {'session_id':session['session_id']})
                assert saved[-1].metadata['completion_evidence'] == 'source_checked_tool_answer'
    finally:
        await engine.close()


async def test_enabled_tool_mode_without_chat_is_configured_but_unavailable(tmp_path):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = build_app(Settings.from_env(environment(tmp_path)), engine)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1', headers=AUTH) as client:
                caps = (await client.get('/v1/conversations/capabilities')).json()
                assert caps['tool_retrieval']['configured'] is True
                assert caps['tool_retrieval']['available'] is False and caps['text_configured'] is False
                result = await client.post('/v1/conversations', json={'request_id':'no-model', 'capture':True})
                assert result.status_code == 503
    finally:
        await engine.close()


@pytest.mark.parametrize('persona', [False, True])
async def test_tool_sessions_keep_selected_connection_across_model_edits(tmp_path, persona):
    from scone_memory.runtime.conversation_tools import build_conversation_tools
    from scone_memory.runtime.model_runtime import DynamicSelfHostedCatalog, self_hosted_text_runtime
    from scone_memory.retrieval.recall_scope import RecallScope

    first = ModelConnection(base_url='http://127.0.0.1:11434/v1', model='first', timeout_s=31)
    store = ModelConnectionStore(tmp_path / 'models.json', {'chat':first, 'transcription':first,
        'speech':first.model_copy(update={'voice':'voice'})})
    configured = build_conversation_tools(Settings.from_env(environment(tmp_path)))
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    catalog = DynamicSelfHostedCatalog(store, tools=configured)
    runtime = self_hosted_text_runtime(engine, store, tools=configured)

    def create(sid):
        return (catalog.get('self-hosted-voice').text(engine, 'default', sid) if persona
                else runtime('default', sid, RecallScope.validated()))

    old = create('old')
    store.replace('chat', first.model_copy(update={'model':'second', 'timeout_s':53}), expected_revision=0)
    new = create('new')
    store.replace('chat', None, expected_revision=1)
    try:
        assert old._tool_factory()._model == 'first' and old._tool_factory()._timeout == 31
        assert new._tool_factory()._model == 'second' and new._tool_factory()._timeout == 53
        assert catalog.personas == ()
        with pytest.raises(InvalidInput, match='No self-hosted chat'):
            runtime('default', 'removed', RecallScope.validated())
    finally:
        await old.close()
        await new.close()
        await engine.close()


def test_self_hosted_voice_keeps_plain_pipeline_when_text_tools_are_enabled(tmp_path, monkeypatch):
    from scone_memory.realtime import voice
    from scone_memory.runtime.conversation_tools import build_conversation_tools
    from scone_memory.runtime.model_runtime import DynamicSelfHostedCatalog

    chat = ModelConnection(base_url='http://127.0.0.1:11434/v1', model='voice-chat')
    store = ModelConnectionStore(tmp_path / 'models.json', {'chat':chat, 'transcription':chat,
        'speech':chat.model_copy(update={'voice':'voice'})})
    catalog = DynamicSelfHostedCatalog(store, tools=build_conversation_tools(Settings.from_env(environment(tmp_path))))
    captured = []
    monkeypatch.setattr(voice, 'VoiceSession', lambda *args, **kwargs: captured.append(kwargs))
    chosen = catalog.get('self-hosted-voice')
    chosen.voice(None, 'default', 'voice-session', transport_factory=lambda:None, capture=True)
    assert captured[0]['model_factory'] is chosen.model_factory
    assert 'tool_model_factory' not in captured[0] and 'tool_limits' not in captured[0]
