"""Local HTTP contracts and lifecycle, without running or contacting a service."""

import asyncio
from contextlib import aclosing
import io
import json
import wave

import httpx
import pytest

from scone_memory.providers.local import validate_local_endpoint
from scone_memory.providers.speech import LocalOpenAISpeech, SpeechProviderError
from scone_memory.providers.transcription import LocalOpenAITranscription, TranscriptionProviderError
from scone_memory.realtime.audio import AudioChunk, SpeechStarted, Transcript


@pytest.mark.parametrize('url', [
    'http://127.0.0.1:8880/v1', 'http://localhost/v1/', 'http://voice.local/v1',
    'https://voice.home.arpa/v1', 'http://stt.localhost/v1', 'http://10.0.0.4:8000',
    'http://172.16.2.4', 'http://192.168.1.10', 'http://[::1]:8000/v1', 'https://[fd00::1]/v1',
])
def test_local_endpoint_admission(url):
    assert validate_local_endpoint(url) == url.rstrip('/') + '/'


@pytest.mark.parametrize('url', [
    '', None, 'https://api.openai.com/v1', 'http://8.8.8.8', 'http://169.254.169.254',
    'http://0.0.0.0', 'http://[::]', 'http://[ff00::1]', 'http://[fe80::1%25en0]',
    'http://user:secret@localhost/v1', 'http://@localhost', 'http://localhost/v1?key=secret',
    'http://localhost/v1#fragment', 'http://localhost?', 'file://localhost/v1',
    'http://localhost:bad', 'http://localhost:0', 'http://localhost:65536',
    'http://localhost.evil.com', 'http://evil-local', ' http://localhost',
    'http://localhost/\n', 'http://localhost/a/../v1', 'http://localhost/%2e%2e/v1',
    'http://localhost/%0a', 'http://localhost\\@evil.com',
])
def test_nonlocal_or_ambiguous_endpoints_are_rejected(url):
    with pytest.raises(ValueError, match='local service URL'):
        validate_local_endpoint(url)


class Stream(httpx.AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts
        self.closed = False
        self.reads = 0

    async def __aiter__(self):
        for part in self.parts:
            self.reads += 1
            yield part

    async def aclose(self):
        self.closed = True


def speech(stream, *, status=200, headers=None, **options):
    requests = []

    async def remote(request):
        requests.append(request)
        return httpx.Response(status, stream=stream, headers=headers or {'content-type': 'audio/pcm'})

    adapter = LocalOpenAISpeech(base_url='http://127.0.0.1:8880/v1', model='local/kokoro',
                               voice='af_heart', transport=httpx.MockTransport(remote), **options)
    return adapter, requests


async def test_local_tts_streams_exact_model_voice_and_pcm_without_vendor_credentials():
    stream = Stream([b'\x01', b'\x02\x03\x04', b'\x05\x06'])
    adapter, requests = speech(stream, sample_rate=22050)
    async with aclosing(adapter), aclosing(adapter.synthesize('Hello.')) as output:
        first = await anext(output)
        assert first.pcm == b'\x01\x02\x03\x04' and stream.reads == 2
        chunks = [first] + [chunk async for chunk in output]
    assert b''.join(chunk.pcm for chunk in chunks) == b'\x01\x02\x03\x04\x05\x06'
    assert {(chunk.sample_rate, chunk.channels) for chunk in chunks} == {(22050, 1)}
    assert stream.closed and len(requests) == 1
    request = requests[0]
    assert str(request.url) == 'http://127.0.0.1:8880/v1/audio/speech'
    assert 'authorization' not in request.headers
    assert json.loads(request.content) == {'model': 'local/kokoro', 'voice': 'af_heart',
                                          'input': 'Hello.', 'response_format': 'pcm'}


@pytest.mark.parametrize('headers,parts', [
    ({'content-type': 'audio/wav'}, [b'RIFF']),
    ({'content-type': 'audio/mpeg'}, [b'MP3!']),
    ({'content-type': 'audio/pcm', 'content-encoding': 'gzip'}, [b'00']),
    ({'content-type': 'audio/pcm'}, []),
    ({'content-type': 'audio/pcm'}, [b'0']),
])
async def test_local_tts_requires_complete_raw_pcm(headers, parts):
    stream = Stream(parts)
    adapter, _ = speech(stream, headers=headers)
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError):
            _ = [chunk async for chunk in adapter.synthesize('Hello.')]
    assert stream.closed


async def test_tts_redirect_is_not_followed_and_optional_local_token_is_explicit():
    stream = Stream([b'secret remote body'])
    adapter, requests = speech(stream, status=307, headers={'location': 'https://cloud.invalid'}, api_key='local-token')
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError) as raised:
            _ = [chunk async for chunk in adapter.synthesize('private text')]
    assert len(requests) == 1 and requests[0].headers['authorization'] == 'Bearer local-token'
    assert 'secret' not in str(raised.value) and 'private' not in str(raised.value)
    assert stream.closed and stream.reads == 0


async def test_local_tts_early_close_reuse_and_cancellation():
    entered = asyncio.Event()

    class Waiting(Stream):
        async def __aiter__(self):
            yield b'00'
            entered.set()
            await asyncio.Event().wait()

    stream = Waiting([])
    adapter, requests = speech(stream)
    async with aclosing(adapter):
        async with aclosing(adapter.synthesize('First')) as output:
            await anext(output)
        assert stream.closed
        output = adapter.synthesize('Second')
        await anext(output)
        task = asyncio.create_task(anext(output))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await output.aclose()
    assert stream.closed and len(requests) == 2
    with pytest.raises(SpeechProviderError, match='closed'):
        await anext(adapter.synthesize('Third'))


async def feed():
    yield AudioChunk(b'\x00\x40' * 3200, 16000)


async def test_local_stt_uploads_mono_wav_and_returns_final_transcript_without_token():
    requests = []

    async def remote(request):
        requests.append(request)
        return httpx.Response(200, json={'text': ' Heard locally. '})

    adapter = LocalOpenAITranscription(base_url='http://localhost:8000/v1/', model='local/whisper',
                                       transport=httpx.MockTransport(remote))
    async with aclosing(adapter):
        events = [event async for event in adapter.transcribe(feed())]
    assert isinstance(events[0], SpeechStarted)
    assert events[1] == Transcript('Heard locally.', final=True)
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == 'http://localhost:8000/v1/audio/transcriptions'
    assert 'authorization' not in request.headers
    body = request.content
    assert b'local/whisper' in body and b'name="response_format"\r\n\r\njson' in body
    start = body.index(b'RIFF')
    with wave.open(io.BytesIO(body[start:body.index(b'\r\n--', start)]), 'rb') as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.readframes(audio.getnframes()) == b'\x00\x40' * 3200
    with pytest.raises(TranscriptionProviderError, match='closed'):
        _ = [event async for event in adapter.transcribe(feed())]


async def test_local_stt_cancellation_closes_active_response():
    entered = asyncio.Event()

    class Waiting(Stream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b'{}'

    stream = Waiting([])
    adapter = LocalOpenAITranscription(base_url='http://localhost:8000/v1', model='whisper',
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)))
    async with aclosing(adapter), aclosing(adapter.transcribe(feed())) as events:
        assert isinstance(await anext(events), SpeechStarted)
        task = asyncio.create_task(anext(events))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed


@pytest.mark.parametrize('status,payload', [(302, b'secret'), (200, b'{}'), (200, b'not-json'), (200, b'X' * 2048)])
async def test_stt_rejects_redirect_invalid_or_unbounded_transcript(status, payload):
    stream = Stream([payload])
    requests = []

    async def remote(request):
        requests.append(request)
        return httpx.Response(status, stream=stream, headers={'location': 'https://cloud.invalid'})

    adapter = LocalOpenAITranscription(base_url='http://localhost:8000/v1', model='whisper',
        max_transcript_bytes=1024, transport=httpx.MockTransport(remote))
    async with aclosing(adapter):
        with pytest.raises(TranscriptionProviderError) as raised:
            _ = [event async for event in adapter.transcribe(feed())]
    assert 'secret' not in str(raised.value)
    assert stream.closed and len(requests) == 1


@pytest.mark.parametrize('provider,extra', [(LocalOpenAISpeech, {'voice': 'af_heart'}), (LocalOpenAITranscription, {})])
@pytest.mark.parametrize('override', [{'base_url': 'https://api.openai.com/v1'}, {'model': ''},
                                    {'api_key': 'key\nheader'}, {'api_key': ''}, {'timeout': float('inf')}])
def test_invalid_local_config_fails_before_a_client_is_created(provider, extra, override):
    with pytest.raises(ValueError):
        provider(**{'base_url': 'http://localhost/v1', 'model': 'local-model', **extra, **override})


@pytest.mark.parametrize('chunk', [AudioChunk(b'0000', 16000, 2), AudioChunk(b'00', 24000)])
async def test_stt_rejects_audio_formats_it_cannot_label_as_mono_wav(chunk):
    requests = []

    async def audio():
        yield chunk

    adapter = LocalOpenAITranscription(base_url='http://localhost/v1', model='whisper',
        transport=httpx.MockTransport(lambda request: requests.append(request)))
    async with aclosing(adapter):
        with pytest.raises(TranscriptionProviderError, match='mono PCM at 16000'):
            _ = [event async for event in adapter.transcribe(audio())]
    assert requests == []


async def test_stt_rejects_concurrent_listener_and_accepts_new_listener_after_close():
    requests = []

    async def remote(request):
        requests.append(request)
        return httpx.Response(200, json={'text': 'Heard.'})

    adapter = LocalOpenAITranscription(base_url='http://localhost/v1', model='whisper',
                                       api_key='local-token', transport=httpx.MockTransport(remote))
    async with aclosing(adapter):
        async with aclosing(adapter.transcribe(feed())) as events:
            assert isinstance(await anext(events), SpeechStarted)
            with pytest.raises(TranscriptionProviderError, match='active listener'):
                await anext(adapter.transcribe(feed()))
        assert not requests
        events = [event async for event in adapter.transcribe(feed())]
        assert events[-1] == Transcript('Heard.')
    assert len(requests) == 1 and requests[0].headers['authorization'] == 'Bearer local-token'
    assert requests[0].headers['accept-encoding'] == 'identity'


async def test_stt_rejects_encoded_response_before_reading():
    stream = Stream([b'not a gzip body'])
    adapter = LocalOpenAITranscription(base_url='http://localhost/v1', model='whisper',
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream,
            headers={'content-encoding': 'gzip'})))
    async with aclosing(adapter):
        with pytest.raises(TranscriptionProviderError, match='Encoded'):
            _ = [event async for event in adapter.transcribe(feed())]
    assert stream.closed and stream.reads == 0


async def test_local_clients_disable_environment_proxy_and_redirects(monkeypatch):
    original = httpx.AsyncClient
    options = []

    def create_client(**kwargs):
        options.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', create_client)
    tts, _ = speech(Stream([b'00']))
    stt = LocalOpenAITranscription(base_url='http://localhost/v1', model='whisper',
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'text': 'Hello.'})))
    assert not options, 'construction must not open clients'
    async with aclosing(tts), aclosing(stt):
        _ = [chunk async for chunk in tts.synthesize('Hello.')]
        _ = [event async for event in stt.transcribe(feed())]
    assert len(options) == 2
    assert all(option['trust_env'] is False and option['follow_redirects'] is False for option in options)
