"""The native controller must withhold unsupported answers before streaming."""
import asyncio
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from scone_memory.realtime.text import TextConversation


@pytest.mark.parametrize('private,supported,allowed', [(0.99, .01, False), (.01, .01, True), (.99, .99, True)])
async def test_grounding_decision_controls_generation_and_public_stream(private, supported, allowed):
    from scone_memory.providers.typesafe_grounding import TypeSafeAnswerGrounder
    state = []
    def respond(request):
        body = json.loads(request.content)
        state.append(body['state'])
        return httpx.Response(200, json={'model': 'jev-test', 'usage': {'input_tokens': 2, 'output_tokens': 2},
            'answers': {'private': {'type': 'noul', 'noul': private}, 'supported': {'type': 'noul', 'noul': supported}}})
    grounder = TypeSafeAnswerGrounder('https://api.typesafe.ai', 'jev-latest', api_key='test',
        workspace='Lyra, our launch project', transport=httpx.MockTransport(respond))
    calls, streamed = [], []
    class Model:
        async def respond(self, messages):
            calls.append(messages)
            yield TextDelta('Generated answer.')
            yield ReplyCompleted()
        async def aclose(self): pass
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    conversation = TextConversation(memory, 'alpha', 'one', Model, answer_grounder=grounder)
    async def on_text(text): streamed.append(text)
    try:
        result = await conversation.reply('What is our project goal?', on_text=on_text)
        assert bool(calls) is allowed
        assert ''.join(streamed) == result['text']
        assert (result['text'] == 'Generated answer.') is allowed
        assert state[0]['question'] == 'What is our project goal?'
        assert state[0]['evidence'] is None
        assert state[0]['workspace'] == 'Lyra, our launch project'
        assert result['memory_context']['answer_grounding']['status'] == ('general' if private < .2 else 'supported' if allowed else 'insufficient')
    finally:
        await conversation.close()
        await memory.close()


async def test_grounding_failure_does_not_release_unchecked_model_output():
    from scone_memory.providers.typesafe_grounding import TypeSafeAnswerGrounder
    grounder = TypeSafeAnswerGrounder('https://api.typesafe.ai', 'jev-latest', api_key='test',
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text='secret error')))
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    def forbidden(): raise AssertionError('unchecked generation')
    conversation = TextConversation(memory, 'alpha', 'one', forbidden, answer_grounder=grounder)
    try:
        result = await conversation.reply('What is our project goal?')
        assert result['memory_context']['answer_grounding']['status'] == 'unavailable'
        assert 'secret' not in json.dumps(result)
    finally:
        await conversation.close()
        await memory.close()


async def test_grounding_provider_cancellation_propagates():
    from scone_memory.providers.typesafe_grounding import TypeSafeAnswerGrounder
    entered = asyncio.Event()
    async def wait(request):
        entered.set()
        await asyncio.Event().wait()
    grounder = TypeSafeAnswerGrounder('https://api.typesafe.ai', 'jev-latest', api_key='test', transport=httpx.MockTransport(wait))
    task = asyncio.create_task(grounder.assess('What is our goal?', [], None))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task


async def test_source_deleted_during_judgment_withholds_generation():
    from scone_memory.realtime.grounding import GroundingJudgment
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    source = await memory.remember('alpha', 'Lyra launch owner is Leo.', source='brief://lyra')
    class Grounder:
        async def assess(self, question, history, evidence):
            assert evidence and 'Leo' in evidence
            await memory.forget('alpha', source.episode_id)
            return GroundingJudgment(private=.99, supported=.99)
    def forbidden(): raise AssertionError('changed source must prevent generation')
    conversation = TextConversation(memory, 'alpha', 'one', forbidden, answer_grounder=Grounder())
    try:
        result = await conversation.reply('Who owns the Lyra launch?')
        assert result['memory_context']['answer_grounding']['status'] == 'unavailable'
        assert 'Leo' not in result['text']
    finally:
        await conversation.close()
        await memory.close()


async def test_grounding_accepts_revalidated_user_facts_evicted_from_history():
    from scone_memory.realtime.grounding import GroundingJudgment
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceDecision
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    states = []
    class Assessor:
        async def assess(self, question, candidates):
            return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))
    class Grounder:
        async def assess(self, question, history, evidence):
            states.append((history, evidence))
            return GroundingJudgment(private=.99, supported=.99)
    class Model:
        async def respond(self, messages):
            yield TextDelta('Noted. ' + 'x' * 200)
            yield ReplyCompleted()
        async def aclose(self): pass
    conversation = TextConversation(memory, 'alpha', 'one', Model, system_prompt='Answer.',
        max_history_bytes=2000, history_policy='window', answer_grounder=Grounder(),
        adaptive_retriever=AdaptiveRetriever(memory, Assessor(), limits=AdaptiveLimits(timeout_s=1)))
    try:
        first = await conversation.reply('The gate code for the yard is 4471, remember it.')
        for index in range(6):
            await conversation.reply(f'Filler {index}. ' + 'y' * 300)
        result = await conversation.reply('What is the gate code for the yard?')
        history, evidence = states[-1]
        assert all('4471' not in message['content'] for message in history)
        assert evidence and '4471' in evidence
        assert result['memory_context']['answer_grounding']['status'] == 'supported'
        assert result['memory_context']['answer_grounding']['source_checked'] is True
        assert any(ref['episode_id'] == first['user_episode_id'] for ref in result['memory_context']['references'])
    finally:
        await conversation.close()
        await memory.close()
