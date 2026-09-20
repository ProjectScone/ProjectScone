"""Explicit cloud inference must not weaken local service boundaries."""
import json

import httpx
import pytest

from scone_memory.providers.self_hosted import validate_self_hosted_endpoint
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore
from scone_memory.runtime.agent_runtime import LocalAgentModel
from scone_memory.runtime.conversation_tools import ConversationTools
from scone_memory.agents.evidence_loop import ToolLoopLimits

URL = 'https://openrouter.ai/api/v1'
MODEL = 'google/gemma-4-31b-it'


def connection(**changes):
    return ModelConnection(**{'provider': 'openrouter', 'base_url': URL, 'model': MODEL,
                              'api_key_env': 'TEST_ROUTER_KEY', **changes})


def test_explicit_cloud_connection_survives_restart_without_persisting_key(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_ROUTER_KEY', 'private-test-key')
    store = ModelConnectionStore(tmp_path / 'models.json', {})
    saved = store.replace('chat', connection(), expected_revision=0)
    assert saved['connections']['chat']['provider'] == 'openrouter'
    assert 'private-test-key' not in (tmp_path / 'models.json').read_text()
    reopened = ModelConnectionStore(tmp_path / 'models.json', {})
    assert reopened.get('chat') == connection()
    assert reopened.get('chat').base_url == URL + '/'


@pytest.mark.parametrize('changes', [
    {'provider':'self_hosted'}, {'provider':'unknown'}, {'model':'invalid'},
    {'model':'google/gemma-4-31b-it:free'}, {'api_key_env':None},
    {'model':'../gemma'}, {'model':'./gemma'},
    {'base_url':'http://openrouter.ai/api/v1'}, {'base_url':URL+'?key=x'},
    {'base_url':'https://openrouter.ai.evil.test/api/v1'},
    {'base_url':'https://secret@openrouter.ai/api/v1'},
    {'base_url':URL+'/../v1'}, {'base_url':'https://openrouter.ai:444/api/v1'},
])
def test_cloud_requires_explicit_mode_exact_tls_endpoint_model_and_key_reference(changes):
    with pytest.raises(ValueError):
        connection(**changes)


@pytest.mark.parametrize('role', ['extraction','vision','transcription','speech'])
def test_cloud_mode_is_only_supported_for_chat_connections(tmp_path, role):
    selected = connection(voice='voice')
    with pytest.raises(ValueError):
        ModelConnectionStore(tmp_path / 'models.json', {role:selected})
    store = ModelConnectionStore(tmp_path / 'models.json', {})
    with pytest.raises(ValueError):
        store.replace(role, selected, expected_revision=0)
    assert not (tmp_path / 'models.json').exists()


def test_local_service_boundary_still_rejects_cloud():
    with pytest.raises(ValueError):
        validate_self_hosted_endpoint(URL)
    with pytest.raises(ValueError):
        SelfHostedToolChat(URL, MODEL, api_key='test')


async def test_cloud_tool_call_round_trip_pins_model_and_never_falls_back():
    requests = []
    async def serve(request):
        requests.append(request)
        return httpx.Response(200, json={'choices':[{'finish_reason':'tool_calls','message':{
            'role':'assistant','content':None,'tool_calls':[{'id':'one','type':'function',
            'function':{'name':'search_memory','arguments':'{"query":"test"}'}}]}}]})
    model = SelfHostedToolChat(URL, MODEL, provider='openrouter', api_key='private-test-key',
                              transport=httpx.MockTransport(serve))
    step = await model.complete([{'role':'user','content':'Find test'}], [{'type':'function'}])
    assert step.calls[0].name == 'search_memory'
    assert step.calls[0].arguments == {'query':'test'}
    assert len(requests) == 1
    assert str(requests[0].url) == URL + '/chat/completions'
    assert requests[0].headers['Authorization'] == 'Bearer private-test-key'
    body = json.loads(requests[0].content)
    assert body['model'] == MODEL and 'models' not in body and 'route' not in body


async def test_upstream_rate_limit_has_no_retry_or_model_fallback():
    calls = []
    def serve(request):
        calls.append(request)
        return httpx.Response(429,json={'error':{'message':'private-provider-detail'}})
    model = SelfHostedToolChat(URL, MODEL, provider='openrouter', api_key='test',
                              transport=httpx.MockTransport(serve))
    with pytest.raises(RuntimeError, match='tool model unavailable'):
        await model.complete([{'role':'user','content':'Hi'}], [])
    assert len(calls) == 1


@pytest.mark.parametrize('protocol', ['native','structured'])
def test_cloud_factory_configuration_reaches_conversation_and_agent_adapters(monkeypatch, protocol):
    monkeypatch.setenv('TEST_ROUTER_KEY','test')
    selected = connection()
    conversation = ConversationTools(protocol, ToolLoopLimits()).factory(selected)()
    assert conversation._endpoint == URL + '/chat/completions'
    assert conversation._key == 'test'
    configured = LocalAgentModel(model_id='gemma',label='Gemma',revision='1',
        provider='openrouter',base_url=URL,model=MODEL,api_key_env='TEST_ROUTER_KEY',protocol=protocol)
    model = configured.create()
    assert model._endpoint == URL + '/chat/completions'
    assert model._model == MODEL and model._key == 'test'


def test_existing_local_configuration_wire_and_agent_identity_do_not_change():
    import hashlib
    local = ModelConnection(base_url='http://localhost:11434/v1',model='local')
    assert 'provider' not in local.model_dump()
    assert 'provider' not in json.loads(local.model_dump_json())
    model = LocalAgentModel(model_id='local',label='Local',revision='1',
        base_url='http://localhost:11434/v1',model='local')
    legacy = {'model_id':'local','label':'Local','revision':'1','base_url':'http://localhost:11434/v1/',
        'model':'local','protocol':'native','timeout_s':120.0,'max_tokens':2048,
        'max_response_bytes':128000,'think':None,'api_key_env':None}
    expected = hashlib.sha256(json.dumps(legacy,separators=(',',':')).encode()).hexdigest()
    assert model.registered().revision == expected


@pytest.mark.parametrize('listed', [True,False])
async def test_cloud_discovery_reads_only_selected_model_and_never_runs_inference(monkeypatch, listed):
    from scone_memory.api.model_connections import probe_model_connection
    monkeypatch.setenv('TEST_ROUTER_KEY','test')
    requests=[]
    def serve(request):
        requests.append(request)
        return httpx.Response(200,json={'data':{'id':MODEL,'endpoints':[{'provider_name':'Google AI Studio'}] if listed else []}})
    result=await probe_model_connection(connection(),transport=httpx.MockTransport(serve))
    assert result == {'models':[MODEL] if listed else [],'model_available':listed}
    assert len(requests)==1 and requests[0].method=='GET'
    assert str(requests[0].url)==URL+'/models/google/gemma-4-31b-it/endpoints'
