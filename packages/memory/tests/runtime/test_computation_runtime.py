"""The optional calculator must reach the served native and structured sessions."""
import asyncio
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.evidence_loop import ToolCall, ToolStep
from scone_memory.api.__main__ import build_app
from scone_memory.core.errors import InvalidInput
from scone_memory.realtime.text import TextConversation
from scone_memory.runtime.config import Settings
from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore
from ..runtime.test_conversation_tools_config import AUTH, environment


def test_compute_default_is_off_and_config_requires_tool_mode(tmp_path):
    assert Settings.from_env({}).conversations_tool_compute is False
    for changes in ({'SCONE_CONVERSATIONS_TOOL_COMPUTE':'maybe'},
                    {'SCONE_CONVERSATIONS_TOOL_COMPUTE':'1','SCONE_CONVERSATIONS_TOOL_MODE':'off'}):
        with pytest.raises(InvalidInput): Settings.from_env(environment(tmp_path) | changes)


@pytest.mark.parametrize('value',[True,'true',1,None])
def test_plain_text_runtime_refuses_compute_without_a_tool_model(value):
    with pytest.raises(ValueError,match='tool_compute'):
        TextConversation(None,'alpha','session',lambda:None,tool_compute=value)


@pytest.mark.parametrize('mode',['native','structured'])
@pytest.mark.parametrize('persona',[False,True])
async def test_served_session_calculates_from_retrieved_sources(tmp_path,monkeypatch,mode,persona):
    from scone_memory.providers import structured_tool_chat, tool_chat
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    await engine.remember('default','Recorded value: 12',metadata={'team':'blue'})
    await engine.remember('default','PRIVATE_OTHER_TEAM value: 99',metadata={'team':'red'})
    observed=[]
    class Model:
        def __init__(self,*args,**kwargs): pass
        async def complete(self,messages,tools):
            assert 'PRIVATE_' not in json.dumps(messages)
            packet=json.loads(messages[-1]['content'])
            observed.append(packet)
            if 'computation' not in packet:
                assert 'compute_memory' in [row['function']['name'] for row in tools]
                return ToolStep(calls=(ToolCall(id='calculate',name='compute_memory',arguments={
                    'operation':'sum','left':[{'chunk_id':packet['items'][0]['chunk_id'],'quote':'12'}]}),))
            assert packet['computation']['value']=='12'
            return ToolStep(content='12')
    provider=structured_tool_chat if mode=='structured' else tool_chat
    monkeypatch.setattr(provider,'SelfHostedStructuredToolChat' if mode=='structured' else 'SelfHostedToolChat',Model)
    env=environment(tmp_path,mode) | {'SCONE_CONVERSATIONS_TOOL_COMPUTE':'1','SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH':'1'}
    connection=ModelConnection(base_url='http://127.0.0.1:11434/v1',model='fixture')
    connections={'chat':connection}
    if persona: connections.update(transcription=connection,speech=connection.model_copy(update={'voice':'fixture'}))
    store=ModelConnectionStore(tmp_path/'models.json',{})
    for revision,(role,value) in enumerate(connections.items()):
        store.replace(role,value,expected_revision=revision)
    app=build_app(Settings.from_env(env),engine)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://127.0.0.1',headers=AUTH) as client:
                assert (await client.get('/v1/conversations/capabilities')).json()['tool_retrieval']['computation'] is True
                payload={'request_id':'session','capture':True,'recall_scope':{'where':{'team':'blue'}}}
                if persona: payload['persona']='self-hosted-voice'
                created=await client.post('/v1/conversations',json=payload)
                assert created.status_code==200,created.text
                session=created.json()
                url='/v1/conversations/'+session['session_id']
                response=await client.post(url+'/turns',json={'request_id':'turn','text':'What is the recorded value?',
                    'expected_revision':session['revision']})
                assert response.status_code in (200,202),response.text
                async with asyncio.timeout(5):
                    while response.json()['status']=='pending':
                        await asyncio.sleep(0)
                        response=await client.get(url+'/turns/turn')
                result=response.json()
                assert result['status']=='completed',result
                assert result['result']['text']=='12'
                tools=result['result']['memory_context']['tool_retrieval']
                assert [row['name'] for row in tools['outcomes']]==['search_memory','compute_memory']
                assert tools['source_status']=='retained' and len(observed)==2
                saved=await engine.episodes('default',{'session_id':session['session_id']})
                assert [row.metadata['role'] for row in saved]==['user','assistant']
    finally:
        await engine.close()
