"""OpenAI-compatible requests must use the endpoint's thinking control field."""
import json

import httpx
import pytest

from scone_memory.providers.llm import OpenAICompatibleChat, OpenAICompatibleTextModel
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.realtime.events import ReplyCompleted, TextDelta


@pytest.mark.parametrize('mode', ['chat', 'schema', 'stream', 'tools', 'structured_tools'])
@pytest.mark.parametrize('think', [None, False, True])
async def test_thinking_control_uses_compatible_wire_field(mode, think):
    requests=[]
    async def serve(request):
        body=json.loads(request.content)
        requests.append(body)
        assert 'think' not in body
        if think is None:
            assert 'reasoning_effort' not in body
        else:
            assert body['reasoning_effort']==('medium' if think else 'none')
        choice={'message':{'role':'assistant','content':'Answer.','reasoning_content':'PRIVATE_REASONING'},
                'finish_reason':'stop'}
        return httpx.Response(200,json={'choices':[choice]})

    kwargs={'think':think,'transport':httpx.MockTransport(serve)}
    if mode in ('chat','schema'):
        model=OpenAICompatibleChat('http://127.0.0.1:11434/v1','fixture',**kwargs)
        answer=await model.complete('System.','Question.') if mode=='chat' else await model.complete_structured(
            'System.','Question.',{'type':'object'})
        assert answer=='Answer.'
    elif mode=='stream':
        model=OpenAICompatibleTextModel('http://127.0.0.1:11434/v1','fixture',**kwargs)
        try:
            events=[event async for event in model.respond([{'role':'user','content':'Question.'}])]
            assert [event.text for event in events if isinstance(event,TextDelta)]==['Answer.']
            assert isinstance(events[-1],ReplyCompleted)
        finally:
            await model.aclose()
    else:
        provider=SelfHostedToolChat if mode=='tools' else SelfHostedStructuredToolChat
        model=provider('http://127.0.0.1:11434/v1','fixture',**kwargs)
        answer=await model.complete([{'role':'user','content':'Question.'}],[])
        assert answer.content=='Answer.'
    assert len(requests)==1
