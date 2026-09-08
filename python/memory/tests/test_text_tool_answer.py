"""Native tool turns preserve conversation publication/capture boundaries."""
import asyncio
import json

import pytest

from scone_memory.agents.evidence_loop import ToolCall, ToolLoopLimits, ToolStep
from scone_memory.api.conversations import _without_cached_graph
from scone_memory.realtime.text import TextConversation


class Script:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    async def complete(self, messages, tools):
        self.requests.append(messages)
        step = self.steps.pop(0)
        return await step() if callable(step) else step


def search():
    return ToolStep(content='Provisional text must not escape.', calls=(ToolCall(
        id='search', name='search_memory', arguments={'query': 'Juniper'}),))


def unused_factory():
    raise AssertionError('tool mode must not create the ordinary text provider')


async def test_neighbor_read_sources_survive_conversation_receipt_and_capture(engine):
    from test_chunk_window import document

    added, chunks = await document(engine, metadata={'team':'blue'})
    model = Script()

    async def read_neighbors():
        packet = json.loads(model.requests[-1][-1]['content'])
        seed_id = packet['items'][0]['chunk_id']
        return ToolStep(calls=(ToolCall(id='read', name='read_memory',
            arguments={'chunk_id':seed_id, 'before':4, 'after':4}),))

    model.steps = [ToolStep(calls=(ToolCall(id='search', name='search_memory',
        arguments={'query':'calibration', 'limit':1}),)), read_neighbors,
        ToolStep(content='The manual records calibration samples.')]
    conversation = TextConversation(engine, 'alpha', 'neighbor-receipt',
        where={'team':'blue'}, tool_model_factory=lambda:model)
    observed = []
    async def observe(text):
        observed.append(text)

    try:
        result = await conversation.reply('What do the surrounding sections say?', on_text=observe)
        receipt = result['memory_context']['tool_retrieval']
        window = json.loads(model.requests[-1][-1]['content'])
        returned_ids = {row['chunk_id'] for row in window['items']}
        assert len(returned_ids) > 1
        assert returned_ids <= {row.chunk_id for row in chunks}
        assert {f'chunk:{key}' for key in returned_ids} <= set(receipt['evidence_ids'])
        assert receipt['outcomes'][-1]['name'] == 'read_memory'
        assert receipt['source_status'] == 'retained'
        assert all(row['episode_id'] == added.episode_id for row in window['items'])
        assert observed == [result['text']]
        episodes = await engine.episodes('alpha', {'session_id':'neighbor-receipt'})
        assert episodes[-1].content == result['text']
        assert episodes[-1].metadata['completion_evidence'] == 'source_checked_tool_answer'
    finally:
        await conversation.close()


async def test_native_tools_publish_only_final_and_preserve_scoped_receipt_history(engine):
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    await engine.remember('alpha', 'Juniper PRIVATE_MARKER', metadata={'team': 'red'})
    model = Script(search(), ToolStep(content='Juniper uses Polaris.'))
    conversation = TextConversation(engine, 'alpha', 'tool-session', unused_factory,
        where={'team': 'blue'}, tool_model_factory=lambda: model)
    observed = []
    async def observe(text): observed.append(text)
    result = await conversation.reply('What does Juniper use?', on_text=observe)
    assert observed == [result['text']] == ['Juniper uses Polaris.']
    receipt = result['memory_context']['tool_retrieval']
    assert receipt['source_status'] == 'retained' and receipt['verified_accuracy'] is False
    assert receipt['model_calls'] == 2 and receipt['tool_calls'] == 1
    assert receipt['packets'] and receipt['evidence_ids']
    assert result['memory_context']['references']
    assert 'PRIVATE_MARKER' not in json.dumps(model.requests)
    assert all('Scone retrieved source material:' not in row['content'] for row in model.requests[0])
    episodes = await engine.episodes('alpha', {'session_id': 'tool-session'})
    assert [row.content for row in episodes] == ['What does Juniper use?', result['text']]
    assert episodes[-1].metadata['completion_evidence'] == 'source_checked_tool_answer'
    assert conversation._history[-1] == {'role': 'assistant', 'content': result['text']}
    await conversation.close()


async def test_stale_tool_source_prevents_callback_and_assistant_capture(engine):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.')
    async def delete_then_answer():
        await engine.documents.delete_episode('alpha', episode.episode_id)
        return ToolStep(content='Must never be published.')
    model = Script(search(), delete_then_answer)
    conversation = TextConversation(engine, 'alpha', 'stale-tools', unused_factory, tool_model_factory=lambda: model)
    observed = []
    async def observe(text): observed.append(text)
    with pytest.raises(RuntimeError, match='evidence changed'):
        await conversation.reply('Juniper?', on_text=observe)
    assert observed == [] and conversation.closed
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id': 'stale-tools'})] == ['user']


async def test_history_budget_is_checked_before_tool_answer_callback_capture(engine):
    model = Script(ToolStep(content='answer ' * 70))
    conversation = TextConversation(engine, 'alpha', 'budget-tools', unused_factory,
        system_prompt='Answer.', max_history_bytes=512, tool_model_factory=lambda: model)
    observed = []
    async def observe(text): observed.append(text)
    with pytest.raises(RuntimeError, match='conversation history byte limit'):
        await conversation.reply('Hello?', on_text=observe)
    assert observed == []
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id': 'budget-tools'})] == ['user']


async def test_current_session_is_excluded_from_search_not_from_conversation_history(engine):
    models = [Script(ToolStep(content='Hello.')), Script(search(), ToolStep(content='No stored evidence.'))]
    second = models[1]
    conversation = TextConversation(engine, 'alpha', 'excluded-tools', unused_factory,
        tool_model_factory=lambda: models.pop(0))
    await conversation.reply('Juniper?')
    result = await conversation.reply('What did I ask?')
    assert any(row['content'] == 'Juniper?' for row in second.requests[0])
    assert result['memory_context']['tool_retrieval']['source_status'] == 'none'
    assert json.loads(second.requests[1][-1]['content'])['status'] == 'empty'
    await conversation.close()


async def test_factory_failure_has_no_raw_error_or_assistant_capture(engine):
    def fail(): raise ValueError('PRIVATE_MODEL_KEY')
    conversation = TextConversation(engine, 'alpha', 'failed-tools', unused_factory, tool_model_factory=fail)
    with pytest.raises(RuntimeError, match='tool model unavailable') as error:
        await conversation.reply('Hello?')
    assert 'PRIVATE_MODEL_KEY' not in str(error.value)
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id': 'failed-tools'})] == ['user']


async def test_turn_deadline_prevents_late_tool_answer_even_when_model_suppresses_cancel(engine):
    async def suppress():
        try: await asyncio.Event().wait()
        except asyncio.CancelledError: return ToolStep(content='late reply')
    conversation = TextConversation(engine, 'alpha', 'timeout-tools', unused_factory,
        tool_model_factory=lambda: Script(suppress), turn_timeout=0.5)
    observed = []
    async def observe(text): observed.append(text)
    with pytest.raises(TimeoutError):
        await conversation.reply('Hello?', on_text=observe)
    assert observed == []
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id': 'timeout-tools'})] == ['user']


def test_tool_receipt_cache_drops_source_packets_without_mutating_live_result():
    receipt = {'source_status': 'retained', 'packets': [{'items': [{'text': 'PRIVATE_SOURCE'}]}],
               'evidence_ids': ['chunk:1'], 'verified_accuracy': False}
    result = {'text': 'Public reply', 'memory_context': {'tool_retrieval': receipt}}
    cached = _without_cached_graph(result)
    assert 'PRIVATE_SOURCE' not in json.dumps(cached)
    assert cached['memory_context']['tool_retrieval']['packets_status'] == 'unavailable'
    assert cached['memory_context']['tool_retrieval']['evidence_ids'] == ['chunk:1']
    assert result['memory_context']['tool_retrieval']['packets']


@pytest.mark.parametrize('kwargs', [
    {'tool_limits': ToolLoopLimits()},
    {'tool_model_factory': 'invalid'},
    {'tool_model_factory': lambda: None, 'tool_limits': {}},
    {'tool_model_factory': lambda: None, 'evidence_selector': type('Selector', (), {'select': lambda *a: None})()},
])
def test_incompatible_tool_settings_fail_before_runtime(kwargs):
    with pytest.raises(ValueError, match='tool'):
        TextConversation(None, 'alpha', 'validation', unused_factory, **kwargs)


async def test_observer_deleting_source_prevents_assistant_capture(engine):
    episode = await engine.remember('alpha', 'Juniper uses Polaris.')
    model = Script(search(), ToolStep(content='Juniper uses Polaris.'))
    conversation = TextConversation(engine, 'alpha', 'observer-delete', unused_factory, tool_model_factory=lambda: model)
    observed = []
    async def observe(text):
        observed.append(text)
        await engine.documents.delete_episode('alpha', episode.episode_id)
    with pytest.raises(RuntimeError, match='evidence changed before capture'):
        await conversation.reply('Juniper?', on_text=observe)
    assert observed == ['Juniper uses Polaris.']
    assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id': 'observer-delete'})] == ['user']


async def test_http_runtime_retains_fingerprints_and_refreshes_graph_without_caching_source_text(tmp_path):
    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
    from scone_memory.api.conversations import create_conversation_app
    from test_conversations_api import client_for, create
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    source = await engine.remember('alpha', 'Juniper uses Polaris. UNIQUE_SOURCE_MARKER')
    models = []
    def factory(space, sid):
        model = Script(search(), ToolStep(content='Juniper uses Polaris.'))
        models.append(model)
        return TextConversation(engine, space, sid, unused_factory, tool_model_factory=lambda: model)
    app = create_conversation_app(engine, {'alpha-key': 'alpha'}, tmp_path / 'tools.db', factory)
    try:
        async with client_for(app) as client:
            session = await create(client)
            url = '/v1/conversations/' + session['session_id']
            response = await client.post(url + '/turns', json={'request_id':'reply', 'text':'Juniper?',
                                                              'expected_revision':session['revision']})
            assert response.status_code in (200, 202), response.text
            async with asyncio.timeout(2):
                while response.json()['status'] == 'pending':
                    await asyncio.sleep(0)
                    response = await client.get(url + '/turns/reply')
            assert response.status_code == 200, response.text
            completed = response.json()
            assert completed['status'] == 'completed'
            context = completed['result']['memory_context']
            assert context['tool_retrieval']['source_status'] == 'retained'
            assert context['tool_retrieval']['packets_status'] == 'unavailable'
            assert 'packets' not in context['tool_retrieval']
            assert context['evidence_graph_status'] == 'prepared'
            assert 'UNIQUE_SOURCE_MARKER' in json.dumps(context['evidence_graph'])
            await engine.forget('alpha', source.episode_id)
            refreshed = await client.get(url + '/turns/reply')
            assert refreshed.status_code == 200, refreshed.text
            assert 'UNIQUE_SOURCE_MARKER' not in refreshed.text
            assert len(models[0].requests) == 2  # replay does not generate again
    finally:
        await engine.close()


@pytest.mark.parametrize('cancel', [False, True])
async def test_post_observer_validation_deadline_or_cancel_prevents_capture(monkeypatch, cancel):
    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember('alpha', 'Juniper uses Polaris.')
    original_get = engine.documents.get_episode
    entered = asyncio.Event()
    async def slow_get(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await original_get(*args)
    async def observe(text):
        monkeypatch.setattr(engine.documents, 'get_episode', slow_get)
    conversation = TextConversation(engine, 'alpha', 'post-validation', unused_factory,
        tool_model_factory=lambda: Script(search(), ToolStep(content='Juniper uses Polaris.')),
        tool_limits=ToolLoopLimits(timeout_s=1.0 if cancel else 0.1))
    try:
        task = asyncio.create_task(conversation.reply('Juniper?', on_text=observe))
        async with asyncio.timeout(2):
            await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        monkeypatch.setattr(engine.documents, 'get_episode', original_get)
        episodes = await engine.episodes('alpha', {'session_id': 'post-validation'})
        assert [row.metadata['role'] for row in episodes] == ['user']
    finally:
        await conversation.close()
        await engine.close()


async def test_changed_fact_is_removed_from_replayed_http_graph(tmp_path):
    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
    from scone_memory.core.ports import NewFact
    from scone_memory.api.conversations import create_conversation_app
    from test_conversations_api import client_for, create
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    source = await engine.remember('alpha', 'Juniper uses Polaris.')
    fact = await engine.documents.insert_fact(NewFact(space='alpha', subject='Juniper', predicate='uses', object='Polaris',
        valid_from='2025-01-01T00:00:00Z', source_episode_id=source.episode_id, quote='Juniper uses Polaris.'))
    def factory(space, sid):
        model = Script(search(), ToolStep(content='Juniper uses Polaris.'))
        return TextConversation(engine, space, sid, unused_factory, tool_model_factory=lambda: model)
    app = create_conversation_app(engine, {'alpha-key':'alpha'}, tmp_path / 'fact-tools.db', factory)
    try:
        async with client_for(app) as client:
            session = await create(client)
            url = '/v1/conversations/' + session['session_id']
            response = await client.post(url + '/turns', json={'request_id':'reply', 'text':'Juniper?',
                'expected_revision':session['revision']})
            async with asyncio.timeout(2):
                while response.json()['status'] == 'pending':
                    await asyncio.sleep(0)
                    response = await client.get(url + '/turns/reply')
            context = response.json()['result']['memory_context']
            assert str(fact.fact_id) in context['claim_fingerprints']
            assert any(row['kind'] == 'claim' for row in context['evidence_graph']['nodes'])
            await engine.documents.update_fact(fact.model_copy(update={'object':'Altered'}))
            replay = await client.get(url + '/turns/reply')
            graph = replay.json()['result']['memory_context']['evidence_graph']
            assert not any(row['kind'] == 'claim' for row in graph['nodes'])
            assert 'Altered' not in json.dumps(graph)
    finally:
        await engine.close()


async def test_failed_tool_arguments_remain_visible_in_receipt_without_private_query(engine):
    invalid = ToolStep(calls=(ToolCall(id='search', name='search_memory',
        arguments={'query':'PRIVATE_QUERY', 'limit':999}),))
    model = Script(invalid, ToolStep(content='The memory request was invalid.'))
    conversation = TextConversation(engine, 'alpha', 'tool-error', unused_factory, tool_model_factory=lambda: model)
    result = await conversation.reply('Juniper?')
    receipt = result['memory_context']['tool_retrieval']
    assert receipt['source_status'] == 'none'
    assert receipt['outcomes'][0]['status'] == 'unavailable'
    assert receipt['outcomes'][0]['error'] == 'invalid_arguments'
    assert 'PRIVATE_QUERY' not in json.dumps(receipt)
    assert [row.metadata['completion_evidence'] for row in await engine.episodes('alpha', {'session_id':'tool-error'})
            if row.metadata['role'] == 'assistant'] == ['native_tool_answer']
    await conversation.close()


async def test_tool_only_conversation_does_not_require_unused_plain_factory(engine):
    conversation = TextConversation(engine, 'alpha', 'tool-only',
        tool_model_factory=lambda: Script(ToolStep(content='Hello.')))
    assert (await conversation.reply('Hello!'))['text'] == 'Hello.'
    await conversation.close()


async def test_cached_tool_outcomes_do_not_retain_model_authored_ids_or_unknown_names(engine):
    model = Script(ToolStep(calls=(ToolCall(id='PRIVATE_ID_MARKER', name='PRIVATE_SOURCE_MARKER', arguments={}),)),
                   ToolStep(content='Unavailable.'))
    conversation = TextConversation(engine, 'alpha', 'opaque-tool-errors', tool_model_factory=lambda: model)
    result = _without_cached_graph(await conversation.reply('Juniper?'))
    assert 'PRIVATE_' not in json.dumps(result)
    outcome = result['memory_context']['tool_retrieval']['outcomes'][0]
    assert outcome['call_id'] == 'tool-1' and outcome['name'] == 'unknown_tool'
    assert outcome['error'] == 'unknown_tool'
    await conversation.close()
