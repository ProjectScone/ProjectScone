"""Observed timestamps from an explicit local service; never invent them from text."""
import asyncio
import io
import json
import wave

import httpx
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.providers.transcription.document import LocalDocumentTranscriber


def wav_bytes(seconds=1):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b'\0' * int(32000 * seconds))
    return output.getvalue()


class Body(httpx.AsyncByteStream):
    def __init__(self, raw):
        self.raw, self.closed = raw, False
    async def __aiter__(self):
        yield self.raw
    async def aclose(self):
        self.closed = True


def response(payload, *, status=200, headers=None):
    return httpx.Response(status, stream=Body(json.dumps(payload).encode()), headers=headers or {'content-type':'application/json'})


def payload():
    return {'text':'Café launch.', 'segments':[{'id':0, 'start':0.1, 'end':0.7, 'text':'Café launch.'}]}


async def test_explicit_model_and_timestamp_contract_preserve_observed_segments():
    requests=[]
    async def serve(request):
        requests.append(request)
        raw=await request.aread()
        assert request.url == 'http://127.0.0.1:9876/v1/audio/transcriptions'
        assert request.headers['authorization']=='Bearer local-fixture'
        assert request.headers['accept-encoding']=='identity'
        for value in (b'name="model"\r\n\r\nmy-local-model', b'name="response_format"\r\n\r\nverbose_json',
                      b'name="timestamp_granularities[]"\r\n\r\nsegment', wav_bytes()):
            assert value in raw
        return response(payload())
    provider=LocalDocumentTranscriber(base_url='http://127.0.0.1:9876/v1',model='my-local-model',
        api_key='local-fixture',transport=httpx.MockTransport(serve))
    assert requests==[]
    segments=await provider.transcribe(wav_bytes())
    assert [(s.text,s.start_seconds,s.end_seconds) for s in segments]==[('Café launch.',0.1,0.7)]
    assert len(requests)==1


@pytest.mark.parametrize('url',['https://api.openai.com/v1','http://10.0.0.1/v1','http://service.local/v1',
    'http://127.0.0.1:9/v1?key=x','http://name:secret@localhost/v1','http://localhost/v1/../outside'])
def test_only_explicit_loopback_endpoints_are_accepted(url):
    with pytest.raises(ValueError):
        LocalDocumentTranscriber(base_url=url,model='local')


@pytest.mark.parametrize('options',[{'model':''},{'timeout':True},{'timeout':float('inf')},
    {'max_response_bytes':0},{'max_segments':True},{'max_segments':10001},{'api_key':'bad\nkey'}])
def test_configuration_is_validated_without_network(options):
    with pytest.raises(ValueError):
        LocalDocumentTranscriber(base_url='http://localhost:9/v1',**{'model':'local',**options})


@pytest.mark.parametrize('data',[b'',b'not wav',wav_bytes()[:-2],wav_bytes()+b'trailing'])
async def test_invalid_wav_is_refused_before_upload(data):
    def unexpected(request): raise AssertionError('invalid input reached transport')
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',transport=httpx.MockTransport(unexpected))
    with pytest.raises(InvalidInput,match='WAV'):
        await provider.transcribe(data)


@pytest.mark.parametrize('value',[
    {'text':'No times'}, {'segments':[]}, {'segments':[{'start':0,'end':1,'text':' '}]},
    {'segments':[{'start':True,'end':1,'text':'x'}]},
    {'segments':[{'start':0,'end':float('nan'),'text':'x'}]},
    {'segments':[{'start':0,'end':2,'text':'x'}]},
    {'segments':[{'start':0.5,'end':0.9,'text':'x'},{'start':0.1,'end':0.2,'text':'y'}]},
    {'segments':[{'start':0,'end':0.5,'text':'\ud800'}]},
])
async def test_malformed_or_unobserved_timestamps_are_refused(value):
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',transport=httpx.MockTransport(lambda r:response(value)))
    with pytest.raises(InvalidInput): await provider.transcribe(wav_bytes())


@pytest.mark.parametrize('mode',['status','redirect','encoding','size','json','count'])
async def test_response_limits_close_without_retry_or_remote_error_text(mode):
    calls=[]
    body=Body(b'secret-remote-body')
    def serve(request):
        calls.append(request)
        if mode=='status': return httpx.Response(500,stream=body)
        if mode=='redirect': return httpx.Response(307,stream=body,headers={'location':'http://localhost:9/elsewhere'})
        if mode=='encoding': return httpx.Response(200,stream=body,headers={'content-encoding':'gzip'})
        if mode=='size': return httpx.Response(200,stream=body,headers={'content-length':'9999999','content-type':'application/json'})
        if mode=='json': return httpx.Response(200,stream=body,headers={'content-type':'application/json'})
        return response({'segments':payload()['segments']*2})
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',max_segments=1,transport=httpx.MockTransport(serve))
    expected={'status':'refused','redirect':'refused','encoding':'encoded','size':'byte limit','json':'invalid JSON','count':'bounded observed'}[mode]
    with pytest.raises(InvalidInput,match=expected) as error: await provider.transcribe(wav_bytes())
    assert 'secret-remote-body' not in str(error.value)
    assert len(calls)==1
    if mode!='count': assert body.closed


async def test_cancellation_closes_the_response_and_allows_a_later_explicit_call():
    started=asyncio.Event()
    class Held(Body):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b''
    held=Held(b'')
    replies=iter([httpx.Response(200,stream=held,headers={'content-type':'application/json'}),response(payload())])
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',transport=httpx.MockTransport(lambda r:next(replies)))
    task=asyncio.create_task(provider.transcribe(wav_bytes()))
    await started.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert held.closed
    assert (await provider.transcribe(wav_bytes()))[0].text=='Café launch.'


async def test_actual_stream_bytes_are_bounded_even_without_length_header():
    body=Body(b' ' * 1025)
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',max_response_bytes=1024,
        transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=body,headers={'content-type':'application/json'})))
    with pytest.raises(InvalidInput,match='byte limit'): await provider.transcribe(wav_bytes())
    assert body.closed


async def test_whole_deadline_closes_a_stalled_response():
    class Slow(Body):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b''
    body=Slow(b'')
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',timeout=0.02,
        transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=body,headers={'content-type':'application/json'})))
    with pytest.raises(InvalidInput,match='time limit'): await provider.transcribe(wav_bytes())
    assert body.closed


async def test_transport_errors_do_not_expose_endpoint_keys_or_response_details():
    def serve(request):
        raise httpx.ConnectError('sensitive transport diagnostics',request=request)
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',transport=httpx.MockTransport(serve))
    with pytest.raises(InvalidInput,match='transport failed') as error: await provider.transcribe(wav_bytes())
    assert 'sensitive' not in str(error.value)


async def test_overlapping_observed_segments_and_negative_zero_are_preserved():
    value={'segments':[{'start':-0.0,'end':0.8,'text':'left'}, {'start':0.2,'end':0.9,'text':'right'}]}
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',transport=httpx.MockTransport(lambda r:response(value)))
    segments=await provider.transcribe(wav_bytes())
    assert [s.start_seconds for s in segments]==[-0.0,0.2]
    assert [s.end_seconds for s in segments]==[0.8,0.9]


@pytest.mark.parametrize('options',[{'model':'bad\ud800'}, {'base_url':'http://localhost/v1/\ud800'}, {'timeout':10**500}])
def test_invalid_unicode_and_huge_timeout_are_configuration_errors(options):
    with pytest.raises(ValueError):
        LocalDocumentTranscriber(**{'base_url':'http://localhost:9/v1','model':'local',**options})


async def test_response_parsing_consumes_the_same_deadline(monkeypatch):
    from scone_memory.providers.transcription import document
    clock=[10.0]
    monkeypatch.setattr(document,'monotonic',lambda:clock[0])
    original_loads=document.json.loads
    def slow_parse(raw):
        clock[0]+=2
        return original_loads(raw)
    monkeypatch.setattr(document.json,'loads',slow_parse)
    provider=LocalDocumentTranscriber(base_url='http://localhost:9/v1',model='local',timeout=1,
        transport=httpx.MockTransport(lambda r:response(payload())))
    with pytest.raises(InvalidInput,match='time limit'):
        await provider.transcribe(wav_bytes())


async def test_empty_observations_require_explicit_opt_in_and_an_empty_text_reply():
    options = {'base_url': 'http://localhost:9/v1', 'model': 'local', 'allow_empty': True}
    provider = LocalDocumentTranscriber(**options, transport=httpx.MockTransport(
        lambda request: response({'text': '', 'segments': []})))
    assert await provider.transcribe(wav_bytes()) == ()
    for value in ({'segments': []}, {'text': 'Unlocated words', 'segments': []},
                  {'text': None, 'segments': []}, {'text': '', 'segments': None}):
        provider = LocalDocumentTranscriber(**options, transport=httpx.MockTransport(
            lambda request: response(value)))
        with pytest.raises(InvalidInput):
            await provider.transcribe(wav_bytes())


@pytest.mark.parametrize('allow_empty', [1, 'yes', None])
def test_empty_observation_opt_in_requires_a_boolean(allow_empty):
    with pytest.raises(ValueError, match='allow_empty'):
        LocalDocumentTranscriber(base_url='http://localhost:9/v1', model='local', allow_empty=allow_empty)
