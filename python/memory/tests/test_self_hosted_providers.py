"""Canonical self-hosted names keep endpoint and saved-config compatibility."""
import pytest

from scone_memory.providers.self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from scone_memory.providers.local import validate_local_endpoint, validate_local_identifier
from scone_memory.providers.speech import SelfHostedOpenAISpeech, LocalOpenAISpeech
from scone_memory.providers.transcription import SelfHostedOpenAITranscription, LocalOpenAITranscription
from scone_memory.providers.vision import SelfHostedOpenAIVision, OpenAICompatibleVision
from scone_memory.realtime.catalog import fingerprint_of
from scone_memory.realtime.persona import Persona, ModelChoice, VoiceChoice
from scone_memory.realtime.providers import ProviderRegistry
from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore
from scone_memory.runtime.model_runtime import DynamicSelfHostedCatalog, DynamicLocalCatalog


@pytest.mark.parametrize('url', ['http://localhost/v1', 'http://10.10.2.4:8000/v1',
    'https://192.168.1.8/v1', 'http://[fd00::2]/v1', 'https://models.home.arpa/v1'])
def test_self_hosted_and_legacy_endpoint_names_accept_same_private_services(url):
    assert validate_self_hosted_endpoint(url) == validate_local_endpoint(url) == url + '/'
    assert validate_self_hosted_identifier('operator/model') == validate_local_identifier('operator/model')


@pytest.mark.parametrize('url', ['https://api.openai.com/v1', 'http://169.254.169.254/v1',
    'http://10.0.0.1/../v1', 'http://user:token@10.0.0.1/v1', 'http://10.0.0.1/v1?token=secret'])
def test_self_hosted_rename_does_not_broaden_endpoint_policy(url):
    with pytest.raises(ValueError, match='self-hosted service URL'):
        validate_self_hosted_endpoint(url)
    with pytest.raises(ValueError, match='local service URL'):
        validate_local_endpoint(url)


def test_canonical_provider_classes_and_legacy_imports_share_implementation():
    from scone_memory.providers.speech.self_hosted_openai import SelfHostedOpenAISpeech as Speech
    from scone_memory.providers.transcription.self_hosted_openai import SelfHostedOpenAITranscription as Transcription
    from scone_memory.providers.speech.local_openai import LocalOpenAISpeech as LegacySpeech
    from scone_memory.providers.transcription.local_openai import LocalOpenAITranscription as LegacyTranscription
    assert Speech is SelfHostedOpenAISpeech is LocalOpenAISpeech is LegacySpeech
    assert Transcription is SelfHostedOpenAITranscription is LocalOpenAITranscription is LegacyTranscription
    assert SelfHostedOpenAIVision is OpenAICompatibleVision
    speech = Speech(base_url='http://10.2.3.4/v1', model='operator/tts', voice='voice')
    stt = Transcription(base_url='http://10.2.3.4/v1', model='operator/stt')
    assert speech.provider == stt.provider == 'self-hosted-openai'
    assert speech._request('hello')[0] == 'http://10.2.3.4/v1/audio/speech'
    assert speech._request('hello')[2]['model'] == 'operator/tts'
    assert stt._request()[0] == 'http://10.2.3.4/v1/audio/transcriptions'
    assert SelfHostedOpenAIVision('http://10.2.3.4/v1', 'operator/vision').base_url == 'http://10.2.3.4/v1/'


def test_self_hosted_catalog_advertises_canonical_names_and_resumes_legacy_identity(tmp_path):
    store = ModelConnectionStore(tmp_path / 'connections.json', {
        'chat': ModelConnection(base_url='http://10.1.2.3/v1', model='chat'),
        'transcription': ModelConnection(base_url='http://10.1.2.3/v1', model='stt'),
        'speech': ModelConnection(base_url='http://10.1.2.3/v1', model='tts', voice='voice'),
    })
    assert DynamicLocalCatalog is DynamicSelfHostedCatalog
    catalog = DynamicSelfHostedCatalog(store)
    public = catalog.public(voice=True)
    assert len(public) == 1
    assert public[0]['id'] == 'self-hosted-voice'
    assert public[0]['name'] == 'Self-hosted voice'
    assert public[0]['reply']['provider'] == 'self-hosted-openai'
    canonical = catalog.get('self-hosted-voice').persona
    legacy = Persona(id='local-voice', name='Local voice', instructions=canonical.instructions,
        reply=ModelChoice(provider='local-openai', model=canonical.reply.model),
        transcription=ModelChoice(provider='local-openai', model=canonical.transcription.model),
        speech=VoiceChoice(provider='local-openai', model=canonical.speech.model, voice=canonical.speech.voice))
    assert catalog.get('local-voice').persona == legacy
    assert catalog.fingerprint('local-voice') == fingerprint_of(legacy)
    assert catalog.name('local-voice') == 'Self-hosted voice'
    assert catalog.get('unknown') is None


@pytest.mark.parametrize('registered,requested', [('self-hosted-openai', 'local-openai'), ('local-openai', 'self-hosted-openai')])
def test_registry_accepts_legacy_provider_alias_only_for_exact_authorized_choices(registered, requested):
    factory = lambda: object()
    registry = ProviderRegistry(reply={(registered,'chat'): factory},
        transcription={(registered,'stt'): factory}, speech={(registered,'tts','voice'): factory})
    persona = Persona(id='voice', name='Voice', instructions='Hello',
        reply=ModelChoice(provider=requested, model='chat'),
        transcription=ModelChoice(provider=requested, model='stt'),
        speech=VoiceChoice(provider=requested, model='tts', voice='voice'))
    assert registry.resolve(persona).model_factory is factory
    changed = persona.model_copy(update={'reply': ModelChoice(provider=requested, model='unregistered')})
    with pytest.raises(ValueError, match='not registered'):
        registry.resolve(changed)
    exact = lambda: 'explicit'
    registry = ProviderRegistry(reply={(registered,'chat'): factory, (requested,'chat'): exact},
        transcription={(registered,'stt'): factory}, speech={(registered,'tts','voice'): factory})
    assert registry.resolve(persona).model_factory is exact


async def test_model_connection_api_names_self_hosted_endpoints(tmp_path):
    import httpx
    from fastapi import FastAPI
    from scone_memory.api.model_connections import mount_model_connection_routes
    app = FastAPI()
    async def authorize(): return None
    mount_model_connection_routes(app, ModelConnectionStore(tmp_path / 'models.json', {}), authorize)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
        invalid = {'base_url': 'https://public.invalid/v1', 'model': 'test'}
        saved = await client.put('/v1/model-connections/chat', json={'expected_revision': 0, 'connection': invalid})
        probed = await client.post('/v1/model-connections/probe', json={'connection': invalid})
    assert saved.status_code == probed.status_code == 400
    assert 'self-hosted endpoint' in saved.json()['error']
    assert 'self-hosted endpoint' in probed.json()['error']


def test_runtime_legacy_symbols_remain_aliases_to_canonical_names():
    from scone_memory.runtime import model_runtime as runtime
    assert runtime.local_admin_enabled is runtime.host_admin_enabled
    assert runtime.authorize_local_admin is runtime.authorize_host_admin
    assert runtime.local_vision_factory is runtime.self_hosted_vision_factory
    assert runtime.local_text_runtime is runtime.self_hosted_text_runtime
    assert runtime.LocalModelWorker is runtime.SelfHostedModelWorker
