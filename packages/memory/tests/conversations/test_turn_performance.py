import asyncio
import pytest

from scone_memory.observability.turn_performance import TurnPerformance, observe


async def test_concurrent_turns_keep_separate_bounded_content_free_timelines():
    async def run(model):
        with TurnPerformance() as trace:
            await asyncio.sleep(0)
            observe('embedding', elapsed_ms=12, outcome='completed', model=model,
                    text='private input', api_key='secret')
            trace.first_text()
            await asyncio.sleep(0)
            trace.first_text()
            return trace.snapshot()
    one, two = await asyncio.gather(run('one'), run('two'))
    assert one['spans'][0]['model'] == 'one'
    assert two['spans'][0]['model'] == 'two'
    assert 'private' not in str(one) and 'secret' not in str(one)
    assert 0 <= one['first_text_ms'] <= one['total_ms']
    with TurnPerformance() as trace:
        for _ in range(100):
            observe('generation', elapsed_ms=1, outcome='failed')
        result = trace.snapshot()
    assert len(result['spans']) == 64 and result['truncated'] is True
    assert result['first_text_ms'] is None


def test_invalid_metrics_and_sensitive_model_values_are_not_published():
    with TurnPerformance() as trace:
        observe('arbitrary private text', elapsed_ms=1, outcome='completed')
        observe('embedding', elapsed_ms=float('nan'), outcome='completed')
        observe('embedding', elapsed_ms=-2, outcome='completed')
        observe('embedding', elapsed_ms=2, outcome='exception contains secret',
                model='https://user:secret@host/model?key=secret', provider='private host')
        result = trace.snapshot()
    assert len(result['spans']) == 1
    assert result['spans'][0]['outcome'] == 'unknown'
    assert 'secret' not in str(result) and 'private' not in str(result)
    observe('embedding', elapsed_ms=2, outcome='completed')
    assert len(trace.snapshot()['spans']) == 1


def test_first_public_text_and_stage_offsets_use_one_monotonic_clock(monkeypatch):
    from scone_memory.observability import turn_performance as module
    now = [10.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    with TurnPerformance() as trace:
        now[0] = 10.4
        observe('capture_user', elapsed_ms=400, outcome='completed')
        now[0] = 11.0
        trace.first_text()
        now[0] = 11.5
        observe('generation', elapsed_ms=1100, outcome='completed', first_text_ms=600)
        trace.first_text()
        result = trace.snapshot()
    assert result['first_text_ms'] == 1000
    assert result['total_ms'] == 1500
    assert result['spans'][1]['start_ms'] == 400
    assert result['spans'][1]['first_text_ms'] == 600


@pytest.mark.parametrize('failed', [False, True])
async def test_api_publishes_timings_for_completed_and_failed_turns(tmp_path, failed):
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api.conversations import create_conversation_app
    from scone_memory.realtime.text import TextConversation
    from scone_memory.realtime.events import TextDelta, ReplyCompleted
    from ..api.test_conversations_api import client_for
    from .test_conversations_resume import finished

    class Model:
        async def respond(self, messages):
            if failed:
                raise RuntimeError('private failure')
            yield TextDelta('Hello.')
            yield ReplyCompleted()
        async def aclose(self): pass

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_conversation_app(memory, {'alpha-key':'alpha'}, tmp_path/'sessions.db',
        lambda space, sid: TextConversation(memory, space, sid, Model), public_text_streaming=True)
    try:
        async with client_for(app) as client:
            created = (await client.post('/v1/conversations', json={'request_id':'create','capture':True})).json()
            url = '/v1/conversations/' + created['session_id']
            await client.post(url+'/turns', json={'request_id':'turn','text':'hello','expected_revision':created['revision']})
            receipt = await finished(client, url, 'turn')
            assert receipt['status'] == ('failed' if failed else 'completed')
            performance = receipt['performance']
            stages = [span['stage'] for span in performance['spans']]
            assert 'capture_user' in stages and 'recall' in stages
            assert ('capture_assistant' in stages) is not failed
            assert (performance['first_text_ms'] is None) is failed
            assert 'private failure' not in str(performance)
            assert (await client.get(url+'/turns/turn', headers={'Authorization':'Bearer unknown'})).status_code == 401
    finally:
        await memory.close()
